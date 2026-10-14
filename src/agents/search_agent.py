"""
Search Agent - Manages web searches and processes results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from typing import List, Optional

from models.research_state import (
    SubQuestion,
    SearchResult,
    ResearchState
)
from utils.tavily_client import TavilySearchClient
from utils.llm_client import LLMClient
from config import MAX_SEARCH_RESULTS


class SearchAgent:
    """
    Agent responsible for executing searches and enriching results with full content
    """
    
    def __init__(self, tavily_client: TavilySearchClient, llm_client: LLMClient):
        self.tavily = tavily_client
        self.llm = llm_client
    
    async def execute_search(
        self,
        question: SubQuestion,
        state: ResearchState,
        search_depth: str = "basic",
        max_results: int = MAX_SEARCH_RESULTS,
        include_full_content: bool = False,
        ui_manager=None
    ) -> List[SearchResult]:
        """
        Execute a search for a question (async version)
        
        Args:
            question: The question to search foroptimized for search engines
            state: Current research state
            search_depth: "basic" or "advanced"
            max_results: Maximum results to return
            include_full_content: Whether to fetch full webpage content
            ui_manager: Optional UI manager for progress updates
        
        Returns:
            List of SearchResult objects
        """
        # Use the question directly as the search query - no conversion
        search_query = question.question
        
        # Determine topic based on query
        topic = self._determine_topic(search_query)
        time_range = self._determine_time_range(search_query)
        
        # Suppress initial search announcement
        
        try:
            # Execute Tavily search off the event loop so the TUI spinner
            # keeps animating during network waits.
            response = await asyncio.to_thread(
                lambda: self.tavily.search(
                    query=search_query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_raw_content=True,
                    include_answer=True,
                    topic=topic,
                    time_range=time_range,
                    category_toggles=state.category_toggles,
                )
            )
            
            state.total_searches += 1

            # Convert to SearchResult objects
            results = []
            raw_results = response.get("results", [])
            
            if ui_manager:
                ui_manager.add_message(
                    f'Searching: "{search_query}"',
                    style="white"
                )
                ui_manager.add_message(
                    f"Results: {len(raw_results)} found",
                    style="white"
                )
            
            for result_data in raw_results:
                result = SearchResult(
                    url=result_data.get("url"),
                    title=result_data.get("title"),
                    content=result_data.get("content", ""),
                    score=result_data.get("score", 0.0),
                    raw_content=result_data.get("raw_content"),
                    favicon=result_data.get("favicon"),
                    category=result_data.get("category", "general"),
                    question_id=question.id,
                )
                results.append(result)
                state.search_results.append(result)
            
            return results
        
        except Exception as e:
            error_msg = f"Search failed for '{search_query[:50]}...': {str(e)}"
            if ui_manager:
                ui_manager.add_message(f"Search error: {str(e)}", style="red bold")
                ui_manager.add_message(
                    "This could be due to API timeout, rate limiting, or network issues",
                    style="yellow",
                )
            state.reasoning_steps.append(f"[SEARCH ERROR] {error_msg}")

            return []
    
    def _determine_topic(self, question: str) -> str:
        """Determine search topic based on question content"""
        question_lower = question.lower()
        
        # Keywords for different topics
        news_keywords = [
            "recent", "latest", "current", "today", "yesterday",
            "breaking", "news", "2024", "2025", "update"
        ]
        finance_keywords = [
            "stock", "market", "price", "economy", "financial",
            "investment", "trading", "crypto", "bitcoin"
        ]
        
        if any(kw in question_lower for kw in news_keywords):
            return "news"
        elif any(kw in question_lower for kw in finance_keywords):
            return "finance"
        
        return "general"
    
    def _determine_time_range(self, question: str) -> Optional[str]:
        """Determine if time filtering is needed"""
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in ["today", "yesterday"]):
            return "day"
        elif any(kw in question_lower for kw in ["this week", "recent"]):
            return "week"
        elif any(kw in question_lower for kw in ["this month", "lately"]):
            return "month"
        elif any(kw in question_lower for kw in ["this year", "2025", "2024"]):
            return "year"
        
        return None
    
    def adaptive_search_depth(
        self,
        question: SubQuestion,
        iteration: int,
        previous_confidence: float = 0.0
    ) -> str:
        """
        Determine search depth adaptively
        
        Args:
            question: The question being searched
            iteration: Current iteration number
            previous_confidence: Confidence from previous search
        
        Returns:
            "basic" or "advanced"
        """
        # Use advanced for:
        # - High priority questions
        # - Low confidence from previous search
        # - Later iterations (dive deeper)
        
        if question.priority >= 8:
            return "advanced"
        
        if previous_confidence < 0.5:
            return "advanced"
        
        if iteration > 2 and previous_confidence < 0.7:
            return "advanced"
        
        return "basic"
