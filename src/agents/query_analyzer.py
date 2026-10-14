"""
Query Analyzer Agent - Breaks down research questions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict

from models.research_state import SubQuestion, ResearchState, QuestionStatus
from utils.llm_client import LLMClient
import uuid

class QueryAnalyzerAgent:
    """
    Agent responsible for breaking down queries into sub-questions
    and managing the question hierarchy using ReAct-style reasoning
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
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
        
        # Convert to SubQuestion objects
        sub_questions = []
        
        # Suppress query analysis summary
        
        # Add header for subquestions
        if ui_manager and decomposition.get("sub_questions"):
            ui_manager.add_message("Generated Subquestions:", style="white")
        
        for i, sq_data in enumerate(decomposition.get("sub_questions", [])):
            question_text = str(sq_data.get("question") or "").strip()
            if not question_text:
                continue  # skip malformed entries rather than crash later
            sub_q = SubQuestion(
                id=str(uuid.uuid4()),
                question=question_text,
                priority=sq_data.get("priority", 5),
                depth=0,  # Top-level questions
                status=QuestionStatus.PENDING,
            )
            sub_questions.append(sub_q)
            
            # Add to state
            state.add_question(sub_q)
            
            if ui_manager:
                # Show generated subquestions
                ui_manager.add_message(f"{i+1}. {question_text}", style="white")

        return sub_questions
    
    def generate_follow_up_questions(
        self,
        parent_question: SubQuestion,
        analysis_result: Dict,
        state: ResearchState,
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
        
        # Check if more research is needed
        if not analysis_result.get("needs_more_research", False):
            return []
        
        follow_ups = analysis_result.get("follow_up_questions", [])
        gaps = analysis_result.get("gaps", [])
        
        if not follow_ups and not gaps:
            if ui_manager:
                ui_manager.add_message("No follow-up questions needed", style="dim white")
            return []
        
        # Suppress follow-up generation messages
        
        new_questions = []
        
        # Add header for follow-up questions
        if ui_manager and follow_ups:
            ui_manager.add_message("Generated Follow-up Questions:", style="white")
        
        # Create SubQuestions from follow-ups
        for i, fq in enumerate(follow_ups[:3]):  # Limit to 3 follow-ups
            sub_q = SubQuestion(
                id=str(uuid.uuid4()),
                question=fq,
                parent_id=parent_question.id,
                priority=parent_question.priority - 1,  # Slightly lower priority
                depth=parent_question.depth + 1,
                status=QuestionStatus.PENDING,
            )
            new_questions.append(sub_q)
            state.add_question(sub_q)
            
            if ui_manager:
                # Show generated follow-up subquestions
                ui_manager.add_message(f"{i+1}. {fq}", style="white")

        return new_questions
    
    def identify_knowledge_gaps(
        self,
        state: ResearchState,
        ui_manager=None
    ) -> List[str]:
        """
        Meta-analysis to identify overall knowledge gaps across all findings
        
        Args:
            state: Current research state
        
        Returns:
            List of identified gaps
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
            
            all_gaps = (
                gaps_data.get("critical_gaps", []) + 
                gaps_data.get("moderate_gaps", [])
            )
            
            if all_gaps and ui_manager:
                ui_manager.add_message(
                    f"Knowledge Gaps Identified: {len(all_gaps)} total",
                    style="white"
                )
                for gap in all_gaps[:5]:
                    ui_manager.add_message(f"- {gap}", style="white")
                if len(all_gaps) > 5:
                    ui_manager.add_message(f"... and {len(all_gaps) - 5} more", style="dim white")

            return all_gaps

        except Exception as e:
            if ui_manager:
                ui_manager.add_message(f"Gap analysis failed: {str(e)}", style="yellow")

            return []
