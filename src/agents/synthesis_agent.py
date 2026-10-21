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
from utils.sources import source_type
from utils.text import as_int, clip, is_near_duplicate, string_list
from utils.ui_manager import notify

log = logging.getLogger("veritas.synthesis")

# One analysis pass describes a question; the distinct points inside it become
# separate findings, because the report writes one section per finding. Four
# covers the usual "several things are true here" answer without turning a
# single search into a report of its own.
MAX_FINDINGS_PER_ANALYSIS = 4


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
                "source_type": source_type(r.url),
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
        
        # Anything handed to the analysis model counts as analyzed, even when
        # this pass produces no finding. The bibliography uses the flag to
        # stop presenting untouched search hits as evidence.
        for result in search_results:
            if hasattr(result, "analyzed"):
                result.analyzed = True

        # Extract and store knowledge findings. Model output is untrusted: a
        # field requested as a list can arrive as one string, which would
        # otherwise be iterated character by character.
        answer = str(analysis.get('answer') or '').strip()
        try:
            confidence = float(analysis.get('confidence', 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # Findings are extracted even when ``answer`` is empty: a reply can
        # carry a populated ``findings`` array and no summary sentence, and
        # gating on the answer would silently throw every point away.
        findings = self._extract_findings(
            analysis, question, search_results, state, confidence
        )
        state.findings.extend(findings)

        # Update question
        if answer:
            question.answer = answer

        if findings and not answer:
            log.info(
                "analysis for %r returned %s findings but no answer",
                clip(question.question, 60), len(findings),
            )

        # Show the answer, clipped: the panel keeps one line per entry.
        if ui_manager and (answer or findings):
            label = f"{len(findings)} insights" if len(findings) > 1 else "Insight"
            shown = answer or findings[0].content
            notify(ui_manager, "insight", f"{label}: {clip(shown)}")
        
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

    @staticmethod
    def _cited_urls(raw_indices, search_results, limit: int = 5) -> List[str]:
        """URLs behind a model-supplied list of 1-based result indices."""
        urls: List[str] = []
        if isinstance(raw_indices, (str, int)):
            raw_indices = [raw_indices]
        if not isinstance(raw_indices, list):
            return urls
        for raw_index in raw_indices:
            index = as_int(raw_index, default=-1) - 1
            if 0 <= index < len(search_results):
                url = search_results[index].url
                if url and url not in urls:
                    urls.append(url)
        return urls[:limit]

    def _extract_findings(
        self,
        analysis: Dict,
        question: SubQuestion,
        search_results: List[SearchResult],
        state: ResearchState,
        answer_confidence: float,
    ) -> List[KnowledgeFinding]:
        """One finding per distinct point the analysis named.

        Models that ignore the ``findings`` array still answer with ``answer``;
        that is one finding, exactly as before. A point that names no sources
        keeps an empty list: presenting the first results as its support made
        every claim look grounded when it was not.
        """
        answer = str(analysis.get("answer") or "").strip()
        answer_urls = self._cited_urls(analysis.get("source_indices"), search_results)
        contradictions = string_list(analysis.get("contradictions"))

        raw_findings = analysis.get("findings")
        if not isinstance(raw_findings, list):
            raw_findings = []

        candidates = []
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            claim = str(
                item.get("claim") or item.get("content") or item.get("answer") or ""
            ).strip()
            if not claim:
                continue
            # A point with no indices field of its own inherits the answer's
            # support (the same model named it for the same search); an
            # explicit empty list means "nothing supports this".
            if "source_indices" in item:
                urls = self._cited_urls(item.get("source_indices"), search_results)
            else:
                urls = list(answer_urls)
            try:
                item_confidence = float(item.get("confidence", answer_confidence))
            except (TypeError, ValueError):
                item_confidence = answer_confidence
            candidates.append((claim, urls, max(0.0, min(1.0, item_confidence))))

        if not candidates and answer:
            candidates = [(answer, answer_urls, answer_confidence)]

        existing = [f.content for f in state.get_findings_for_question(question.id)]
        findings: List[KnowledgeFinding] = []
        kept_text = []
        for claim, urls, item_confidence in candidates:
            if len(findings) >= MAX_FINDINGS_PER_ANALYSIS:
                log.info(
                    "analysis for %r returned more than %s findings; extra points dropped",
                    clip(question.question, 60), MAX_FINDINGS_PER_ANALYSIS,
                )
                break
            if any(is_near_duplicate(claim, text) for text in existing + kept_text):
                log.debug("dropping duplicate finding: %r", clip(claim, 80))
                continue
            kept_text.append(claim)
            findings.append(KnowledgeFinding(
                content=claim,
                sources=urls,
                confidence=item_confidence,
                question_id=question.id,
                contradictions=list(contradictions),
            ))
        return findings

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
}

Write reliability_assessment for a reader of the final report. Refer to the
sources and what they say, never to internal finding numbers ("Finding 3"):
the reader has not seen them."""
        
        # Bounded: one analysis pass now yields several findings, and the
        # cross-reference prompt used to serialise every one of them in full.
        max_findings = 30
        findings_text = "\n\n".join([
            f"Finding {i}: {clip(f.content, 500)}\n"
            f"Confidence: {f.confidence}\n"
            f"Sources: {', '.join(f.sources[:2])}"
            for i, f in enumerate(state.findings[:max_findings])
        ])
        if len(state.findings) > max_findings:
            log.info(
                "cross-reference: analysing the first %s of %s findings",
                max_findings, len(state.findings),
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research findings to analyze:\n\n{findings_text}"}
        ]
        
        # Suppress LLM cross-analysis message
        
        try:
            cross_ref = self.llm.chat_json(messages)
            
            if ui_manager:
                # Show only: Analyzing findings found (x) conflicts/agreements, (x) uncertainties
                # A model can answer a requested list with null; `len(None)`
                # would discard the whole cross-reference result after it was
                # paid for.
                consensus_count = len(cross_ref.get('consensus_points') or [])
                contradiction_count = len(cross_ref.get('contradictions') or [])
                
                # Count uncertainties from low confidence findings
                uncertainty_count = sum(1 for f in state.findings if f.confidence < 0.6)
                
                notify(
                    ui_manager, "info",
                    f"Analyzing Findings: {consensus_count} agreements, "
                    f"{contradiction_count} conflicts, {uncertainty_count} uncertainties",
                )
            
            return cross_ref
        
        except Exception as e:
            log.warning("cross-reference failed: %s", e, exc_info=True)
            if ui_manager:
                notify(ui_manager, "warn", f"Cross-reference error: {clip(str(e))}")

            return {"consensus": [], "contradictions": []}
