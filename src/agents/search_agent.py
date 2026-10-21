"""
Search Agent - Manages web searches and processes results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
import logging
from collections import Counter
from typing import List, Optional

from models.research_state import (
    SubQuestion,
    SearchResult,
    ResearchState
)
from utils.tavily_client import TavilySearchClient
from utils.llm_client import LLMClient
from utils.sources import (
    canonical_url,
    clean_page_text,
    duplicate_of,
    merge_result,
    prepare_results,
    select_for_analysis,
)
from utils.text import clip
from utils.ui_manager import notify
from config import MAX_ANALYSIS_RESULTS, MAX_RESULTS_PER_DOMAIN, MAX_SEARCH_RESULTS

log = logging.getLogger("veritas.search")

# A search whose best result is this weakly related did not answer the query;
# the engine filled the list with loose word matches. Analysing them produces
# a confident-looking section built on the wrong subject, so the search is
# treated as empty instead. Genuine matches score far higher (0.7-0.95); the
# one run that motivated this had a best score of 0.24 and six film pages.
MIN_USEFUL_RELEVANCE = 0.30


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
        ui_manager=None
    ) -> List[SearchResult]:
        """
        Execute a search for a question (async version)
        
        Args:
            question: The question to search foroptimized for search engines
            state: Current research state
            search_depth: "basic" or "advanced"
            max_results: Maximum results to return
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
        
        # Count the attempt, not just the successes: a run that fails every
        # search still made searches, and the report's stats say so.
        state.total_searches += 1

        try:
            # Execute Tavily search off the event loop so the TUI spinner
            # keeps animating during network waits.
            response = await asyncio.to_thread(
                lambda: self.tavily.search(
                    query=search_query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_raw_content=True,
                    # The provider's generated answer was requested and then
                    # never read (only its "results" are used), so it is no
                    # longer paid for.
                    include_answer=False,
                    topic=topic,
                    time_range=time_range,
                    category_toggles=state.category_toggles,
                )
            )
            
            # Convert to SearchResult objects
            candidates = []
            raw_results = response.get("results", [])
            
            for result_data in raw_results:
                candidates.append(SearchResult(
                    url=result_data.get("url"),
                    title=result_data.get("title") or "",
                    # A null field from the API must not reach slicing in the
                    # analysis prompt. Fetched pages are markdown/HTML soup;
                    # stripping images, link URLs and nav boilerplate here
                    # means every later prompt spends its character budget on
                    # text that could actually support a claim.
                    content=clean_page_text(result_data.get("content") or ""),
                    score=result_data.get("score", 0.0),
                    full_content=clean_page_text(result_data.get("raw_content") or ""),
                    category=result_data.get("category", "general"),
                    question_id=question.id,
                    published_date=(
                        result_data.get("published_date")
                        or result_data.get("publishedDate")
                    ),
                ))

            # Drop pages that carry no evidence and fold duplicates into one
            # record. Repeating the same URL across questions must not inflate
            # the source count or create two bibliography entries.
            usable, rejected = prepare_results(candidates)
            if rejected:
                reasons = Counter(reason for _, reason in rejected)
                log.info(
                    "filtered %s of %s results: %s",
                    len(rejected), len(candidates),
                    ", ".join(f"{reason} x{count}" for reason, count in reasons.items()),
                )

            if usable and max(float(result.score or 0.0) for result in usable) < MIN_USEFUL_RELEVANCE:
                best = max(float(result.score or 0.0) for result in usable)
                log.warning(
                    "search results for %r look unrelated (best relevance %.2f); treating as empty",
                    search_query, best,
                )
                state.reasoning_steps.append(
                    f"[SEARCH WARNING] results did not match the question "
                    f"(best relevance {best:.2f})"
                )
                if ui_manager:
                    notify(
                        ui_manager, "warn",
                        f"Results look unrelated (best match {best:.0%}) - skipping",
                    )
                return []

            by_url = {
                canonical_url(existing.url): existing
                for existing in state.search_results
            }

            results = []
            for result in usable:
                key = canonical_url(result.url)
                # Same URL, or the same article republished elsewhere (the
                # site chrome differs, the text does not). Both are one source
                # in the bibliography, not two that agree.
                existing = by_url.get(key) or duplicate_of(
                    result, state.search_results
                )

                if existing is not None:
                    merge_result(existing, result)
                    results.append(existing)
                    continue

                state.search_results.append(result)
                by_url[key] = result
                results.append(result)

            # The state keeps every usable source for the bibliography; the
            # analysis model gets only the strongest, most independent subset
            # so quality actually shapes the answer instead of just decorating
            # it.
            return select_for_analysis(
                results,
                limit=MAX_ANALYSIS_RESULTS,
                per_domain=MAX_RESULTS_PER_DOMAIN,
            )
        
        except Exception as e:
            error_msg = f"Search failed for '{search_query[:50]}...': {str(e)}"
            log.warning("search failed for %r: %s", search_query, e, exc_info=True)
            if ui_manager:
                notify(
                    ui_manager, "warn",
                    f"Search error: {clip(str(e))} (timeout, rate limit, or network)",
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
        previous_confidence: Optional[float] = None
    ) -> str:
        """
        Choose the cheapest search depth that can still answer the question.

        Tavily charges twice as much for ``advanced``, so it is reserved for
        questions that need it: the planner's explicit request, explicit
        freshness or finance questions, long multi-clause questions, and a
        re-search after a low-confidence answer. Priority only orders the
        queue; decomposition models rate most questions highly, so treating a
        high priority as "needs advanced" made every search advanced. A
        question searched for the first time has no previous confidence, which
        must not be mistaken for zero.

        Args:
            question: The question being searched
            iteration: Current iteration number (kept for callers that track it)
            previous_confidence: Confidence from an earlier search of this
                question, or None when there was none

        Returns:
            "basic" or "advanced"
        """
        if str(getattr(question, "search_depth", "") or "").lower() == "advanced":
            return "advanced"

        if previous_confidence is not None and previous_confidence < 0.5:
            return "advanced"

        text = question.question or ""
        if len(text.split()) >= 15:
            return "advanced"

        if self._determine_topic(text) in ("news", "finance"):
            return "advanced"

        if self._determine_time_range(text):
            return "advanced"

        return "basic"
