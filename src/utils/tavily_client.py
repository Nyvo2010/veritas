"""
Tavily Client - Handles all interactions with Tavily Search API
"""

import sys
import os
import time
import inspect
import requests
from typing import Dict, Optional
from urllib.parse import urlparse
import logging
from tavily import TavilyClient

log = logging.getLogger("veritas.tavily")

# SDK errors that repeat identically on every attempt. Matched by class name so
# the check keeps working across tavily-python releases that add or rename
# error types.
_NON_RETRYABLE_ERRORS = {
    "InvalidAPIKeyError",
    "MissingAPIKeyError",
    "ForbiddenError",
    "BadRequestError",
    "KeylessUnsupportedEndpointError",
    "UsageLimitExceededError",
    "TavilyKeylessLimitError",
}

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    TAVILY_API_KEY,
    WEBSITE_CATEGORIES,
    MAX_RETRIES,
    RETRY_DELAY,
    EXPONENTIAL_BACKOFF,
    SEARCH_TIMEOUT
)


class TavilySearchClient:
    """Wrapper for Tavily Search API with advanced filtering"""
    
    def __init__(self, api_key: str = TAVILY_API_KEY):
        if not api_key:
            raise ValueError("No API key provided. Please provide the api_key attribute or set the TAVILY_API_KEY environment variable.")
        self.client = TavilyClient(api_key=api_key)
        self.api_key = api_key
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self.timeout = SEARCH_TIMEOUT  # Timeout for search operations
    
    def _handle_retry(self, attempt: int, error_msg: str) -> bool:
        """
        Handle retry logic with exponential backoff
        
        Args:
            attempt: Current retry attempt
            error_msg: Error message to display
            
        Returns:
            True if should retry, False otherwise
        """
        # ``attempt`` is zero-based: the call just made was attempt + 1. Stop
        # once that was the last one, instead of sleeping first and then
        # raising a generic "failed after N attempts".
        if attempt + 1 >= self.max_retries:
            return False
        
        if EXPONENTIAL_BACKOFF:
            delay = self.retry_delay * (2 ** attempt)
        else:
            delay = self.retry_delay

        log.warning("%s: retrying in %ss (%s/%s)", error_msg, delay, attempt + 1, self.max_retries)
        time.sleep(delay)
        return True
    
    def _categorize_url(self, url: str) -> Optional[str]:
        """Determine which category a URL belongs to.

        Hosts are matched exactly or on a dot boundary. Substring matching
        would let ``x.com`` claim ``netflix.com`` and ``github.com`` claim
        ``notgithub.com``, which corrupts both the quality priors and the
        exclude list built from the source toggles.
        """
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            return "general"
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return "general"

        for category, domains in WEBSITE_CATEGORIES.items():
            for cat_domain in domains:
                listed = str(cat_domain).lower().strip(".")
                if listed and (host == listed or host.endswith("." + listed)):
                    return category

        return "general"

    def _search_accepts_timeout(self) -> bool:
        """Whether the installed SDK lets a call set its own timeout.

        ``timeout`` was added to ``TavilyClient.search`` after the minimum
        supported release, so older SDKs would treat it as an unexpected
        argument. Skip it there and let the SDK's own default apply.
        """
        try:
            return "timeout" in inspect.signature(self.client.search).parameters
        except (TypeError, ValueError):
            return False

    def _is_retryable(self, error: Exception) -> bool:
        """Whether another attempt could plausibly succeed.

        Auth, configuration, and usage-limit failures fail the same way every
        time, so retrying them only burns the backoff budget before the user
        sees the real reason.
        """
        if type(error).__name__ in _NON_RETRYABLE_ERRORS:
            return False

        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(status, int):
            if status == 429 or status >= 500:
                return True
            if 400 <= status < 500:
                return False

        # Bad arguments are a caller bug, not a network blip.
        return not isinstance(error, (ValueError, TypeError, KeyError, AttributeError))
    
    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 10,
        include_raw_content: bool = True,
        include_answer: bool = True,
        topic: str = "general",
        time_range: Optional[str] = None,
        category_toggles: Optional[Dict[str, bool]] = None,
    ) -> Dict:
        """
        Perform a Tavily search with filtering and retry logic
        
        Args:
            query: Search query
            search_depth: "basic" (1 credit) or "advanced" (2 credits)
            max_results: Maximum number of results (0-20)
            include_raw_content: Include full page content
            include_answer: Include LLM-generated answer
            topic: "general", "news", or "finance"
            time_range: Optional time filter (day, week, month, year)
            category_toggles: Dict of category filters
        """
        attempt = 0
        started = time.perf_counter()
        log.info(
            "search -> %r | depth=%s, max_results=%s, topic=%s, time_range=%s, raw=%s, answer=%s",
            query, search_depth, max_results, topic, time_range,
            include_raw_content, include_answer,
        )

        while attempt < self.max_retries:
            try:
                # Build exclude domains list from category toggles
                exclude_domains = []
                if category_toggles:
                    for category, enabled in category_toggles.items():
                        if not enabled and category in WEBSITE_CATEGORIES:
                            exclude_domains.extend(WEBSITE_CATEGORIES[category])
                
                # Prepare search parameters
                search_params = {
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "include_answer": include_answer,
                    "include_raw_content": include_raw_content,
                    "topic": topic,
                }
                
                if time_range:
                    search_params["time_range"] = time_range

                # Bound every attempt when the SDK supports it, so a stalled
                # connection cannot hold a search for the SDK's full default.
                if self._search_accepts_timeout():
                    search_params["timeout"] = self.timeout
                
                if exclude_domains:
                    search_params["exclude_domains"] = exclude_domains[:150]  # Max 150
                
                # Perform search
                log.debug("search params: %s", search_params)
                response = self.client.search(**search_params)
                
                # Add categories to results
                if "results" in response:
                    for result in response["results"]:
                        result["category"] = self._categorize_url(result.get("url", ""))

                results = response.get("results", []) if isinstance(response, dict) else []
                log.info(
                    "search <- %r in %.2fs | %s results, %s categories%s",
                    query, time.perf_counter() - started, len(results),
                    len({r.get("category") for r in results}),
                    f", answer: {str(response.get('answer'))[:120]}" if response.get("answer") else "",
                )
                for rank, result in enumerate(results[:10], 1):
                    log.debug(
                        "  result %s: score=%s category=%s %s | %s",
                        rank, result.get("score"), result.get("category"),
                        result.get("url"), str(result.get("title"))[:80],
                    )

                return response
            
            except requests.exceptions.Timeout as e:
                log.warning("tavily timeout (attempt %s/%s): %s", attempt + 1, self.max_retries, e)
                if not self._handle_retry(attempt, "Request timeout"):
                    raise Exception(f"Tavily search timeout after {self.max_retries} attempts")
                attempt += 1
            
            except requests.exceptions.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", "?")
                if status == 429:  # Rate limit
                    log.warning("tavily rate limit (attempt %s/%s)", attempt + 1, self.max_retries)
                    if not self._handle_retry(attempt, "Rate limit"):
                        raise Exception(f"Tavily rate limit exceeded after {self.max_retries} attempts")
                    attempt += 1
                elif isinstance(status, int) and status >= 500:  # Server error
                    log.warning("tavily server error %s (attempt %s/%s)",
                                status, attempt + 1, self.max_retries)
                    if not self._handle_retry(attempt, "Server error"):
                        raise Exception(f"Tavily server error after {self.max_retries} attempts")
                    attempt += 1
                else:
                    log.error("tavily HTTP %s for %r: %s", status, query, e)
                    raise Exception(f"Tavily API error: {e}")
            
            except requests.exceptions.RequestException as e:
                log.warning("tavily request failed (attempt %s/%s): %s",
                            attempt + 1, self.max_retries, e)
                if not self._handle_retry(attempt, "Request failed"):
                    raise Exception(f"Tavily request failed after {self.max_retries} attempts")
                attempt += 1
            
            except Exception as e:
                if not self._is_retryable(e):
                    log.error("tavily error is not retryable (%s): %s", type(e).__name__, e)
                    raise Exception(f"Tavily search failed ({type(e).__name__}): {e}")
                log.warning("unexpected tavily error (attempt %s/%s): %s",
                            attempt + 1, self.max_retries, e)
                if attempt >= self.max_retries - 1:
                    raise Exception(f"Tavily search failed: {e}")
                time.sleep(self.retry_delay)
                attempt += 1
        
        raise Exception(f"Failed to complete Tavily search after {self.max_retries} attempts")
