"""
Query Analyzer Agent - Breaks down research questions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict

import logging
import uuid

from models.research_state import SubQuestion, ResearchState, QuestionStatus
from utils.llm_client import LLMClient
from utils.text import as_bool, as_int, is_near_duplicate, string_list

log = logging.getLogger("veritas.query")

class QueryAnalyzerAgent:
    """
    Agent responsible for breaking down queries into sub-questions
    and managing the question hierarchy using ReAct-style reasoning
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    @staticmethod
    def _is_duplicate(state: ResearchState, question_text: str) -> bool:
        """True when *question_text* paraphrases a question already queued.

        Decomposition and follow-ups both restate the main query in different
        words, and each duplicate becomes its own finding and report section
        talking about the same thing. The check is word overlap, not exact
        text, so a reworded repeat is caught too.
        """
        return any(
            is_near_duplicate(question_text, existing.question)
            for existing in state.questions
        )
    
    def validate_query(self, query: str) -> bool:
        """
        Validate if the query is a proper research query.
        
        Args:
            query: The user's original query.
        
        Returns:
            True if the query is valid, False otherwise.
        """
        # Simple validation: reject queries that are too short or generic
        if len(query.strip()) < 3:
            return False
        return True

    def decompose_main_query(
        self,
        main_query: str,
        state: ResearchState,
        ui_manager=None
    ) -> List[SubQuestion]:
        """
        Decompose the main query into prioritized sub-questions
        
        Args:
            main_query: The user's original query
            state: Current research state
        
        Returns:
            List of SubQuestion objects
        """
        # Validate the query before proceeding
        if not self.validate_query(main_query):
            if ui_manager:
                ui_manager.add_message(f"Query '{main_query}' is not valid for research.", style="red bold")
            return []

        # Use LLM to decompose query
        decomposition = self.llm.decompose_query(main_query)
        raw_questions = (
            decomposition.get("sub_questions")
            if isinstance(decomposition, dict) else None
        )
        if not isinstance(raw_questions, list):
            raw_questions = []

        # Convert to SubQuestion objects
        sub_questions = []
        
        for sq_data in raw_questions:
            if not isinstance(sq_data, dict):
                continue  # skip malformed entries rather than crash later
            question_text = str(sq_data.get("question") or "").strip()
            if not question_text:
                continue
            if self._is_duplicate(state, question_text):
                log.debug("skipping duplicate question: %r", question_text)
                continue
            requested_depth = str(
                sq_data.get("search_depth") or ""
            ).strip().lower()
            sub_q = SubQuestion(
                id=str(uuid.uuid4()),
                question=question_text,
                priority=as_int(sq_data.get("priority"), default=5, low=0, high=10),
                depth=0,  # Top-level questions
                # Only the two depths Tavily accepts; anything else is left to
                # the deterministic policy in SearchAgent.
                search_depth=(
                    requested_depth
                    if requested_depth in ("basic", "advanced") else None
                ),
                status=QuestionStatus.PENDING,
            )
            sub_questions.append(sub_q)
            
            # Add to state
            state.add_question(sub_q)

        return sub_questions
    
    def generate_follow_up_questions(
        self,
        parent_question: SubQuestion,
        analysis_result: Dict,
        state: ResearchState,
        *,
        max_depth: int = 3,
        ui_manager=None
    ) -> List[SubQuestion]:
        """
        Generate follow-up questions based on analysis results
        
        Args:
            parent_question: The question that was just analyzed
            analysis_result: Results from analyzing search results
            state: Current research state
            max_depth: Maximum depth of question tree
            ui_manager: Optional UI manager for progress updates
        
        Returns:
            List of new SubQuestion objects
        """
        # Don't generate if we've reached max depth
        if parent_question.depth >= max_depth:
            if ui_manager:
                ui_manager.add_message("Max depth reached - no follow-ups", style="dim yellow")
            return []
        
        # Check if more research is needed. Models sometimes answer with the
        # string "false", which is truthy in Python.
        if not as_bool(analysis_result.get("needs_more_research")):
            return []
        
        follow_ups = string_list(analysis_result.get("follow_up_questions"))
        gaps = string_list(analysis_result.get("gaps"))
        
        if not follow_ups and not gaps:
            if ui_manager:
                ui_manager.add_message("No follow-up questions needed", style="dim white")
            return []
        
        new_questions = []

        # Create SubQuestions from follow-ups
        for i, fq in enumerate(follow_ups[:3]):  # Limit to 3 follow-ups
            if self._is_duplicate(state, fq):
                log.debug("skipping duplicate follow-up: %r", fq)
                continue
            sub_q = SubQuestion(
                id=str(uuid.uuid4()),
                question=fq,
                priority=parent_question.priority - 1,  # Slightly lower priority
                depth=parent_question.depth + 1,
                status=QuestionStatus.PENDING,
            )
            new_questions.append(sub_q)
            state.add_question(sub_q)

        return new_questions
    
    def identify_knowledge_gaps(
        self,
        state: ResearchState,
        ui_manager=None
    ) -> List[Dict]:
        """
        Meta-analysis to identify overall knowledge gaps across all findings
        
        Args:
            state: Current research state
        
        Returns:
            Gaps as ``{"description": str, "severity": str}`` dicts, so the
            caller can keep critical and moderate gaps distinct.
        """
        if not state.findings:
            return []
        
        # Use LLM to analyze all findings for gaps
        system_prompt = """You are an expert research analyst. Review all research findings and identify critical knowledge gaps.

Focus on:
- What questions remain unanswered?
- Where is evidence weak or contradictory?
- What important perspectives are missing?
- What data would strengthen conclusions?

Return JSON:
{
    "critical_gaps": ["list of critical gaps"],
    "moderate_gaps": ["list of moderate gaps"],
    "suggested_research": ["specific searches to fill gaps"]
}"""
        
        findings_summary = "\n".join([
            f"- {f.content[:200]}..." 
            for f in state.findings[:20]
        ])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research findings:\n{findings_summary}\n\nMain query: {state.main_query}"}
        ]
        
        try:
            gaps_data = self.llm.chat_json(messages)
            
            gaps = [
                {"description": description, "severity": severity}
                for severity, key in (("critical", "critical_gaps"),
                                      ("moderate", "moderate_gaps"))
                for description in string_list(
                    gaps_data.get(key) if isinstance(gaps_data, dict) else None
                )
            ]
            return gaps

        except Exception as e:
            if ui_manager:
                ui_manager.add_message(f"Gap analysis failed: {str(e)}", style="yellow")

            return []
