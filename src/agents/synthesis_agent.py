"""
Synthesis Agent - Identifies patterns and synthesizes knowledge
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict

from models.research_state import (
    SubQuestion,
    SearchResult,
    KnowledgeFinding,
    ResearchGap,
    ResearchState
)
from utils.nvidia_client import NvidiaClient


class SynthesisAgent:
    """
    Agent responsible for analyzing search results and synthesizing knowledge
    Uses ReAct-style reasoning to determine next actions
    """
    
    def __init__(self, nvidia_client: NvidiaClient):
        self.nvidia = nvidia_client
    
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
                "content": r.full_content if r.full_content else r.content,
                "score": r.score,
                "category": r.category,
            }
            for r in search_results
        ]
        
        # Suppress LLM analysis message
        
        # Use NVIDIA NIM to analyze with ReAct reasoning
        analysis = self.nvidia.analyze_search_results(
            question=question.question,
            search_results=results_data,
            previous_findings=previous_findings
        )
        
        # Suppress detailed analysis output
        
        # Extract and store knowledge finding
        if analysis.get('answer'):
            finding = KnowledgeFinding(
                content=analysis['answer'],
                sources=[r.url for r in search_results[:5]],
                confidence=analysis.get('confidence', 0.5),
                question_id=question.id,
                contradictions=analysis.get('contradictions', []),
                supporting_facts=[r.content for r in search_results[:3]]
            )
            state.findings.append(finding)
            
            # Update question
            question.answer = analysis['answer']
            question.sources = [r.url for r in search_results[:10]]
            
            # Show key insight
            if ui_manager:
                answer_preview = analysis['answer'][:150] + "..." if len(analysis['answer']) > 150 else analysis['answer']
                confidence = analysis.get('confidence', 0.5)
                ui_manager.add_message(
                    f"Key Insight: {answer_preview}",
                    style="white"
                )
                ui_manager.add_message(
                    f"Confidence: {confidence:.0%}",
                    style="white"
                )
        
        # Store identified gaps
        for gap_desc in analysis.get('gaps', []):
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
            cross_ref = self.nvidia.chat_json(messages)
            
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
            error_msg = f"Cross-reference analysis failed: {str(e)}"
            if ui_manager:
                ui_manager.add_message(f"Cross-reference error: {str(e)}", style="red bold")

            return {"consensus": [], "contradictions": []}
    
    def synthesize_answer(
        self,
        question: str,
        findings: List[KnowledgeFinding],
        cross_ref: Dict
    ) -> str:
        """
        Synthesize a comprehensive answer from multiple findings
        
        Args:
            question: The original question
            findings: List of relevant findings
            cross_ref: Cross-reference analysis
        
        Returns:
            Synthesized comprehensive answer
        """
        system_prompt = """You are an expert research synthesizer. Create a comprehensive, nuanced answer.

Guidelines:
- Integrate information from all findings
- Note where sources agree (consensus)
- Acknowledge contradictions or uncertainties
- Cite sources inline [Source X]
- Be clear and concise
- Present balanced perspective

Format:
1. Direct answer to question
2. Supporting evidence
3. Important caveats or contradictions
4. Confidence level"""
        
        findings_text = "\n\n".join([
            f"Finding {i+1}: {f.content}\n"
            f"Confidence: {f.confidence}\n"
            f"Sources: {', '.join(f.sources[:2])}"
            for i, f in enumerate(findings)
        ])
        
        consensus_text = "\n".join([
            f"- {cp.get('point', '')}"
            for cp in cross_ref.get('consensus_points', [])[:5]
        ])
        
        contradictions_text = "\n".join([
            f"- {c.get('topic', '')}: {c.get('viewpoint_a', '')} vs {c.get('viewpoint_b', '')}"
            for c in cross_ref.get('contradictions', [])[:3]
        ])
        
        user_prompt = f"""Question: {question}

Findings:
{findings_text}

Consensus Points:
{consensus_text}

Contradictions:
{contradictions_text}

Synthesize a comprehensive answer."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.nvidia.chat_completion(messages, max_tokens=1500)
