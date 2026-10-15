"""
Tavily Client - Handles all interactions with Tavily Search API
"""

import sys
import os
import time
import requests
from typing import Dict, Optional
from urllib.parse import urlparse
import logging
from tavily import TavilyClient

log = logging.getLogger(__name__)

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
        if attempt >= self.max_retries:
            return False
        
        if EXPONENTIAL_BACKOFF:
            delay = self.retry_delay * (2 ** attempt)
        else:
            delay = self.retry_delay
        
        log.debug(f"   {error_msg}. Retrying in {delay}s... ({attempt + 1}/{self.max_retries})")
        time.sleep(delay)
        return True
    
    def _categorize_url(self, url: str) -> Optional[str]:
        """Determine which category a URL belongs to"""
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        domain = domain.replace("www.", "")
        
        for category, domains in WEBSITE_CATEGORIES.items():
            for cat_domain in domains:
                if cat_domain in domain or domain in cat_domain:
                    return category
        
        return "general"
    
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
                
                if exclude_domains:
                    search_params["exclude_domains"] = exclude_domains[:150]  # Max 150
                
                # Perform search
                response = self.client.search(**search_params)
                
                # Add categories to results
                if "results" in response:
                    for result in response["results"]:
                        result["category"] = self._categorize_url(result.get("url", ""))
                
                return response
            
            except requests.exceptions.Timeout as e:
                log.debug(f"   Tavily timeout error: {e}")
                if not self._handle_retry(attempt, "Request timeout"):
                    raise Exception(f"Tavily search timeout after {self.max_retries} attempts")
                attempt += 1
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    log.debug("   Tavily rate limit hit")
                    if not self._handle_retry(attempt, "Rate limit"):
                        raise Exception(f"Tavily rate limit exceeded after {self.max_retries} attempts")
                    attempt += 1
                elif e.response.status_code >= 500:  # Server error
                    log.debug(f"   Tavily server error: {e.response.status_code}")
                    if not self._handle_retry(attempt, "Server error"):
                        raise Exception(f"Tavily server error after {self.max_retries} attempts")
                    attempt += 1
                else:
                    log.debug(f"   Tavily HTTP error: {e}")
                    raise Exception(f"Tavily API error: {e}")
            
            except requests.exceptions.RequestException as e:
                log.debug(f"   Tavily request error: {e}")
                if not self._handle_retry(attempt, "Request failed"):
                    raise Exception(f"Tavily request failed after {self.max_retries} attempts")
                attempt += 1
            
            except Exception as e:
                log.debug(f"   Unexpected error in Tavily search: {e}")
                if attempt >= self.max_retries - 1:
                    raise Exception(f"Tavily search failed: {e}")
                time.sleep(self.retry_delay)
                attempt += 1
        
        raise Exception(f"Failed to complete Tavily search after {self.max_retries} attempts")
