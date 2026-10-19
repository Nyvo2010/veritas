"""
Synthesis Agent - Identifies patterns and synthesizes knowledge
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from typing import List, Dict

from models.research_state import (
    SubQuestion,
    SearchResult,
    KnowledgeFinding,
    ResearchGap,
    ResearchState
)
from utils.llm_client import LLMClient
from utils.text import clip, string_list

log = logging.getLogger("veritas.synthesis")


class SynthesisAgent:
    """
    Agent responsible for analyzing search results and synthesizing knowledge
    Uses ReAct-style reasoning to determine next actions
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def analyze_results(
        self,
        question: SubQuestion,
        search_results: List[SearchResult],
        state: ResearchState,
        ui_manager=None
    ) -> Dict:
        """
        Analyze search results and determine if question is sufficiently answered
        
        Args:
            question: The question being analyzed
            search_results: Search results for this question
            state: Current research state
            ui_manager: Optional UI manager for progress updates
        
        Returns:
            Analysis dict with answer, confidence, gaps, follow-up questions
        """
        # Suppress analysis start messages
        
        # Get previous findings for context
        previous_findings = [
            f.content for f in state.get_findings_for_question(question.id)
        ]
        
        # Suppress previous findings message
        
        # Format results for LLM
        results_data = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.full_content or r.content or "",
                "score": r.score,
                "category": r.category,
                "quality": getattr(r, "quality", 0.0),
                "published_date": getattr(r, "published_date", None),
            }
            for r in search_results
        ]
        
        # Suppress LLM analysis message
        
        # Use the configured LLM to analyze with ReAct reasoning
        analysis = self.llm.analyze_search_results(
            question=question.question,
            search_results=results_data,
            previous_findings=previous_findings
        )
        
        # Suppress detailed analysis output
        
        # Extract and store knowledge finding. Model output is untrusted: a
        # field requested as a list can arrive as one string, which would
        # otherwise be iterated character by character.
        answer = str(analysis.get('answer') or '').strip()
        if answer:
            try:
                confidence = float(analysis.get('confidence', 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            # The analysis names the result indices that support its answer.
            # Keeping those URLs (rather than a blind slice of the first five)
            # is what makes a finding's citations traceable later.
            cited_urls = []
            for raw_index in analysis.get('source_indices', []) or []:
                try:
                    index = int(raw_index) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(search_results):
                    url = search_results[index].url
                    if url and url not in cited_urls:
                        cited_urls.append(url)
            if not cited_urls:
                cited_urls = [r.url for r in search_results[:5]]

            finding = KnowledgeFinding(
                content=answer,
                sources=cited_urls[:5],
                confidence=confidence,
                question_id=question.id,
                contradictions=string_list(analysis.get('contradictions')),
            )
            state.findings.append(finding)
            
            # Update question
            question.answer = answer
            
            # Show the answer, clipped: the panel keeps one line per entry.
            if ui_manager:
                ui_manager.add_message(f"Insight: {clip(answer)}", style="white")
        
        # Store identified gaps
        for gap_desc in string_list(analysis.get('gaps')):
            gap = ResearchGap(
                description=gap_desc,
                question_id=question.id,
                severity="moderate",
                suggested_searches=[]
            )
            state.gaps.append(gap)
        
        return analysis
    
    def cross_reference_findings(
        self,
        state: ResearchState,
        ui_manager=None
    ) -> Dict:
        """
        Cross-reference all findings to identify consensus and contradictions
        
        Args:
            state: Current research state
            ui_manager: Optional UI manager for progress updates
        
        Returns:
            Dict with consensus points, contradictions, and confidence assessments
        """
        if len(state.findings) < 2:
            return {"consensus": [], "contradictions": []}
        
        # Suppress cross-reference start messages
        
        system_prompt = """You are an expert at analyzing research findings for consistency.

Compare all findings and identify:
1. Points where multiple sources agree (consensus)
2. Points where sources contradict each other
3. Overall confidence in conclusions

Return JSON:
{
    "consensus_points": [
        {
            "point": "The agreed-upon fact",
            "supporting_findings": [indices of supporting findings],
            "confidence": 0.0-1.0
        }
    ],
    "contradictions": [
        {
            "topic": "What's contradicted",
            "viewpoint_a": "First perspective",
            "viewpoint_b": "Second perspective",
            "finding_indices": [indices of conflicting findings]
        }
    ],
    "overall_confidence": 0.0-1.0,
    "reliability_assessment": "Your assessment of source reliability"
}"""
        
        findings_text = "\n\n".join([
            f"Finding {i}: {f.content}\n"
            f"Confidence: {f.confidence}\n"
            f"Sources: {', '.join(f.sources[:2])}"
            for i, f in enumerate(state.findings)
        ])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research findings to analyze:\n\n{findings_text}"}
        ]
        
        # Suppress LLM cross-analysis message
        
        try:
            cross_ref = self.llm.chat_json(messages)
            
            if ui_manager:
                # Show only: Analyzing findings found (x) conflicts/agreements, (x) uncertainties
                consensus_count = len(cross_ref.get('consensus_points', []))
                contradiction_count = len(cross_ref.get('contradictions', []))
                
                # Count uncertainties from low confidence findings
                uncertainty_count = sum(1 for f in state.findings if f.confidence < 0.6)
                
                ui_manager.add_message(
                    f"Analyzing Findings: {consensus_count} agreements, "
                    f"{contradiction_count} conflicts, {uncertainty_count} uncertainties",
                    style="white"
                )
            
            return cross_ref
        
        except Exception as e:
            log.warning("cross-reference failed: %s", e, exc_info=True)
            if ui_manager:
                ui_manager.add_message(f"Cross-reference error: {clip(str(e))}", style="yellow")

            return {"consensus": [], "contradictions": []}
