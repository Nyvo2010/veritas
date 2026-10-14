"""
Utility module for interacting with any OpenAI-compatible LLM endpoint.
Defaults to NVIDIA NIM; override the base URL, key, and model per install.
"""

import sys
import os

# Ensure project root is on sys.path so local modules (like config) can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import re
import time
import asyncio
from typing import List, Dict
from urllib.parse import urlparse

import httpx
import logging
from openai import OpenAI, AsyncOpenAI

log = logging.getLogger(__name__)

try:
    from config import (
        NVIDIA_API_KEY,
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL,
        TEMPERATURE_ANALYSIS,
        TEMPERATURE_CREATIVE,
        TOP_P,
        MAX_RETRIES,
        RETRY_DELAY,
        EXPONENTIAL_BACKOFF,
        REPORT_SECTION_MAX_TOKENS,
        REPORT_SUMMARY_MAX_TOKENS,
        API_REQUEST_TIMEOUT,
    )
except ImportError as e:
    raise ImportError("The 'config' module is missing or not found. Ensure it exists and is in the Python path.") from e


def _tolerant_loads(candidate: str) -> Dict:
    """Parse a JSON candidate, tolerating common model output mistakes.

    Tries strict parse first, then lenient parse (literal control chars),
    then trailing-comma repair. Raises the original JSONDecodeError if all fail.
    """
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
        pass
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(repaired, strict=False)
    except json.JSONDecodeError:
        pass
    return json.loads(candidate)  # Re-raise the informative strict error


def _extract_json(text: str) -> Dict:
    """Parse JSON from model output, tolerating markdown code fences.

    Reasoning models (like Nemotron Lightning) sometimes wrap JSON in
    ```json ... ``` blocks or prepend reasoning text. This helper strips
    fences and extracts the first balanced {...} object.
    """
    cleaned = text.strip()
    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        return _tolerant_loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to first balanced {...} block
    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return _tolerant_loads(cleaned[start:i + 1])
    # Last resort: try raw parse and let it raise
    return _tolerant_loads(cleaned)


def fetch_models(api_base_url: str, api_key: str, timeout: float = 15.0) -> List[str]:
    """List model ids from any OpenAI-compatible /models endpoint.

    Raises RuntimeError with a human-friendly message when the endpoint
    cannot be reached, rejects the key, or answers unexpectedly.
    """
    normalized = (api_base_url or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("The endpoint URL is empty.")
    url = f"{normalized}/models"
    headers = {"Authorization": f"Bearer {(api_key or '').strip()}"} if (api_key or "").strip() else {}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise RuntimeError(f"Cannot reach {normalized}. Check the URL and your connection.") from e
    except httpx.TimeoutException as e:
        raise RuntimeError(f"{normalized} took too long to answer.") from e
    except Exception as e:
        raise RuntimeError(f"Could not query {normalized}: {e}") from e
    if response.status_code == 401:
        raise RuntimeError("The endpoint rejected the API key (401).")
    if response.status_code >= 400:
        raise RuntimeError(f"The endpoint answered {response.status_code}: {response.text[:160]}")
    try:
        payload = response.json()
    except ValueError as e:
        raise RuntimeError("The endpoint did not return JSON for /models.") from e
    data = payload.get("data", []) if isinstance(payload, dict) else []
    ids = sorted({str(item.get("id", "")) for item in data if isinstance(item, dict) and item.get("id")})
    if not ids:
        raise RuntimeError("The endpoint returned no models.")
    return ids


class LLMClient:
    """Wrapper for any OpenAI-compatible chat API with research helpers."""

    def __init__(
        self,
        api_key: str = None,
        api_base_url: str = None,
        model: str = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the endpoint (default from config/env)
            api_base_url: Base URL of the OpenAI-compatible endpoint
            model: Model id to use for all inference (default from config/env)
        """
        self.api_base_url = self._normalize_base_url(api_base_url or LLM_BASE_URL)
        timeout = API_REQUEST_TIMEOUT if API_REQUEST_TIMEOUT else 120
        resolved_key = api_key if api_key is not None else LLM_API_KEY
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=self.api_base_url,
            timeout=timeout,
        )
        self.async_client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=self.api_base_url,
            timeout=timeout,
        )
        self.model = (model or LLM_MODEL).strip() if (model or LLM_MODEL) else ""
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self._last_api_call = 0  # Track last API call time for rate limiting
        self._rate_limit_delay = 1.0  # Minimum 1 second between API calls
        self._last_finish_reason = None  # finish_reason of the most recent completion

    def _normalize_base_url(self, api_base_url: str) -> str:
        """Validate and normalize the API base URL so requests never omit a host."""
        normalized = (api_base_url or "").strip()
        if not normalized:
            raise ValueError(
                "LLM API base URL is empty. Set it in Settings or the LLM_BASE_URL variable."
            )

        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                "LLM API base URL must include a scheme and hostname, e.g. https://integrate.api.nvidia.com/v1"
            )

        rebuilt = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            rebuilt += parsed.path.rstrip("/")

        return rebuilt

    def _raise_connection_error(self, error: Exception) -> None:
        """Raise a human-friendly error when DNS or network settings block the API."""
        message = (
            f"Unable to reach the LLM API at {self.api_base_url}. "
            "Check your internet connection, VPN/DNS/proxy settings, or update "
            "the endpoint URL in Settings if you route through a custom endpoint."
        )
        raise ConnectionError(message) from error

    def list_models(self, timeout: float = 15.0) -> List[str]:
        """List model ids advertised by this client's endpoint."""
        return fetch_models(self.api_base_url, getattr(self.client, "api_key", "") or "", timeout=timeout)

    def _handle_rate_limit_sync(self, attempt: int) -> bool:
        """
        Handle rate limit with exponential backoff

        Args:
            attempt: Current retry attempt number

        Returns:
            True if should retry, False if max retries exceeded
        """
        if attempt >= self.max_retries:
            return False

        if EXPONENTIAL_BACKOFF:
            delay = self.retry_delay * (2 ** attempt)
        else:
            delay = self.retry_delay

        log.debug(f"   Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{self.max_retries}...")
        time.sleep(delay)
        return True

    async def _handle_rate_limit_async(self, attempt: int) -> bool:
        """Async version of rate limit handler"""
        if attempt >= self.max_retries:
            return False

        if EXPONENTIAL_BACKOFF:
            delay = self.retry_delay * (2 ** attempt)
        else:
            delay = self.retry_delay

        log.debug(f"   Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{self.max_retries}...")
        await asyncio.sleep(delay)
        return True

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = TEMPERATURE_ANALYSIS,
        max_tokens: int = 4096,
        stream: bool = False,
        json_mode: bool = False,
        top_p: float = TOP_P,
        reasoning_effort: str = None,  # Kept for compatibility, ignored by most endpoints
    ) -> str:
        """
        Send a chat completion request with retry logic

        Args:
            messages: Chat messages as dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            json_mode: Enable JSON mode for structured output
            top_p: Nucleus sampling parameter
            reasoning_effort: Not used (kept for compatibility)

        Returns:
            Response text (or stream if stream=True)
        """
        # Rate limiting: ensure at least 1 second between API calls
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        if time_since_last_call < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - time_since_last_call)

        attempt = 0

        while attempt < self.max_retries:
            try:
                self._last_api_call = time.time()  # Update last call time

                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }

                if json_mode:
                    params["response_format"] = {"type": "json_object"}

                if stream:
                    response = self.client.chat.completions.create(**params, stream=True)
                    return response
                else:
                    response = self.client.chat.completions.create(**params)
                    self._last_finish_reason = response.choices[0].finish_reason
                    content = response.choices[0].message.content
                    return content or ""

            except httpx.ConnectError as e:
                self._raise_connection_error(e)
            except Exception as e:
                error_str = str(e).lower()

                # Handle rate limiting
                if "rate" in error_str or "429" in error_str:
                    log.debug(f"   Rate limit error: {e}")
                    if not self._handle_rate_limit_sync(attempt):
                        raise Exception(f"Max retries ({self.max_retries}) exceeded due to rate limiting")
                    attempt += 1

                # Handle timeout errors
                elif "timeout" in error_str or "timed out" in error_str:
                    log.debug(f"   Timeout error: {e}")
                    if attempt >= self.max_retries - 1:
                        raise Exception(f"API timeout after {self.max_retries} attempts")
                    time.sleep(self.retry_delay)
                    attempt += 1

                # Handle bad request errors
                elif "400" in error_str or "bad request" in error_str:
                    log.debug(f"   Bad request error: {e}")
                    raise Exception(f"Bad request to the LLM API: {e}")

                # Handle auth errors (don't retry)
                elif "401" in error_str or "403" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
                    log.debug(f"   Auth error: {e}")
                    raise Exception(f"LLM API authentication failed (check the API key): {e}")

                # Handle general API errors
                elif "api" in error_str or "500" in error_str or "503" in error_str:
                    log.debug(f"   API error: {e}")
                    if attempt >= self.max_retries - 1:
                        raise Exception(f"API error after {self.max_retries} attempts: {e}")
                    time.sleep(self.retry_delay)
                    attempt += 1

                else:
                    log.debug(f"   Unexpected error in LLM chat completion: {e}")
                    raise

        raise Exception(f"Failed to complete request after {self.max_retries} attempts")

    # Nudge sent to the model when its previous reply was not parseable JSON.
    _JSON_REPAIR_NUDGE = (
        "Your previous response was not valid JSON. "
        "Reply with ONLY the corrected JSON object and no other text."
    )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = TEMPERATURE_ANALYSIS,
        max_tokens: int = 4096,
        top_p: float = TOP_P,
        max_parse_retries: int = 3,
    ) -> Dict:
        """Request structured JSON output with automatic repair retries.

        Handles the two common failure modes of reasoning models:
        - Truncated output (finish_reason="length"): retries with a larger
          max_tokens budget.
        - Malformed JSON: re-sends the bad output with a repair nudge.

        Args:
            messages: Chat messages as dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Initial maximum tokens in response
            top_p: Nucleus sampling parameter
            max_parse_retries: How many parse attempts before giving up

        Returns:
            Parsed JSON dict

        Raises:
            Exception: If no valid JSON is produced after all retries
        """
        attempt_messages = list(messages)
        current_max_tokens = max_tokens
        last_error = None

        for parse_attempt in range(1, max_parse_retries + 1):
            content = self.chat_completion(
                attempt_messages,
                temperature=temperature,
                max_tokens=current_max_tokens,
                json_mode=True,
                top_p=top_p,
            )
            try:
                return _extract_json(content)
            except json.JSONDecodeError as e:
                last_error = e
                if parse_attempt >= max_parse_retries:
                    break
                if self._last_finish_reason == "length":
                    current_max_tokens = min(int(current_max_tokens * 1.5) + 512, 16000)
                    log.debug(
                        f"   [!] JSON response truncated "
                        f"(attempt {parse_attempt}/{max_parse_retries}); "
                        f"retrying with max_tokens={current_max_tokens}..."
                    )
                    # Fresh retry with the original messages and a bigger budget
                    attempt_messages = list(messages)
                else:
                    log.debug(
                        f"   [!] JSON parse failed "
                        f"(attempt {parse_attempt}/{max_parse_retries}): {e}. "
                        f"Asking model to repair..."
                    )
                    attempt_messages = attempt_messages + [
                        {"role": "assistant", "content": content[:4000]},
                        {"role": "user", "content": self._JSON_REPAIR_NUDGE},
                    ]

        raise Exception(
            f"Failed to get valid JSON after {max_parse_retries} attempts: {last_error}"
        )

    def decompose_query(self, main_query: str) -> Dict:
        """
        Decompose a main query into sub-questions using ReAct-style reasoning

        Returns dict with: reasoning, sub_questions, priorities, search_strategies
        """
        system_prompt = """You are an expert research planner. Your task is to break down complex queries into comprehensive sub-questions.

Use ReAct-style reasoning:
1. THOUGHT: Analyze what aspects need to be researched
2. REASONING: Explain why each sub-question is important
3. ACTION: List the sub-questions with priorities

CRITICAL: Stay 100% focused on the EXACT topic in the user's query. Do NOT drift to related topics.
- If the query is about AI, keep ALL sub-questions about AI
- If the query is about healthcare, keep ALL sub-questions about healthcare
- Do NOT mix topics unless explicitly mentioned in the original query

IMPORTANT: Generate questions that will be used for ONLINE SEARCHES. Keep questions:
- Concise and focused (under 15 words when possible)
- Using natural search keywords, not overly formal language
- Specific but not too verbose
- Action-oriented (what, how, which) rather than philosophical
- DIRECTLY related to the main query topic

Consider:
- What background knowledge is needed ABOUT THIS SPECIFIC TOPIC?
- What are the key components of THIS PARTICULAR question?
- What perspectives should be explored WITHIN THIS TOPIC?
- What data or evidence is required FOR THIS SPECIFIC SUBJECT?

Return a JSON with this structure:
{
    "reasoning": "Your analysis of the query",
    "sub_questions": [
        {
            "question": "The sub-question",
            "priority": 0-10 (higher = more important),
            "rationale": "Why this question matters",
            "search_depth": "basic" or "advanced",
            "suggested_topic": "general" or "news" or "finance"
        }
    ],
    "estimated_complexity": "low/medium/high",
    "key_themes": ["theme1", "theme2"]
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query to decompose: {main_query}\n\nReminder: Keep ALL sub-questions focused on the EXACT topic in this query. Do not drift to other topics."}
        ]

        return self.chat_json(messages)

    def analyze_search_results(
        self,
        question: str,
        search_results: List[Dict],
        previous_findings: List[str] = None
    ) -> Dict:
        """
        Analyze search results and determine next actions using ReAct reasoning

        Returns dict with: analysis, answer, confidence, gaps, follow_up_questions, needs_more_research
        """
        system_prompt = """You are an expert research analyst. Analyze search results and determine if the question is sufficiently answered.

Use ReAct-style reasoning:
1. OBSERVATION: What information do the sources provide?
2. THOUGHT: Is this information sufficient and reliable?
3. REASONING: What's missing or uncertain?
4. ACTION: What should be done next?

CRITICAL: Stay 100% focused on answering the SPECIFIC question asked.
- If the question is about AI, answer ONLY about AI
- If the question is about healthcare, answer ONLY about healthcare
- Do NOT drift to tangentially related topics
- Base your answer ONLY on what's directly relevant to the question

Consider:
- Do multiple sources agree ON THIS SPECIFIC QUESTION?
- Are there contradictions ABOUT THIS TOPIC?
- What information is missing TO FULLY ANSWER THIS QUESTION?
- What follow-up questions arise WITHIN THIS TOPIC?
- How confident are you in the answer TO THIS SPECIFIC QUESTION?

Return JSON:
{
    "observation": "Summary of what sources say ABOUT THIS SPECIFIC QUESTION",
    "analysis": "Your critical analysis OF THE QUESTION ASKED",
    "answer": "Best answer based on available info FOR THIS SPECIFIC QUESTION",
    "confidence": 0.0-1.0,
    "source_agreement": "high/medium/low",
    "contradictions": ["list any contradictions found"],
    "gaps": ["list knowledge gaps DIRECTLY RELATED TO THIS QUESTION"],
    "follow_up_questions": ["questions that need more research WITHIN THIS TOPIC"],
    "needs_more_research": true/false,
    "reasoning": "Why you need/don't need more research FOR THIS SPECIFIC QUESTION"
}"""

        # Format search results
        results_text = "\n\n".join([
            f"Source {i+1}: {r.get('title', 'No title')}\n"
            f"URL: {r.get('url', 'No URL')}\n"
            f"Content: {r.get('content', 'No content')[:1000]}..."
            for i, r in enumerate(search_results[:10])
        ])

        context = f"Question: {question}\n\n"
        if previous_findings:
            context += "Previous findings:\n" + "\n".join(previous_findings[:5]) + "\n\n"
        context += f"Search Results:\n{results_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context + "\n\nReminder: Answer ONLY the specific question asked. Do not drift to other topics."}
        ]

        # 4096-token budget: analysis JSON is long and must not be truncated mid-string
        return self.chat_json(messages, max_tokens=4096)

    def identify_report_themes(self, all_findings: List[Dict]) -> Dict:
        """
        Meta-analysis to identify major themes for parallel report generation

        Returns: themes with their related findings
        """
        system_prompt = """You are an expert at organizing research findings into SPECIFIC, DISTINCT themes that avoid redundancy.

Analyze all findings and identify 6-10 SHARPLY FOCUSED themes for a comprehensive report.

CRITICAL REQUIREMENTS:
- Each theme must be SPECIFIC and DISTINCT - no overlap between themes
- Focus on CONCRETE aspects, not generic categories
- Each theme should explore a UNIQUE angle or dimension of the research
- Avoid repetitive patterns like "X in healthcare", "X in education", "X in finance"
- Instead, identify CROSS-CUTTING insights, mechanisms, or dynamics

Return JSON:
{
    "themes": [
        {
            "title": "Specific, unique theme focusing on a particular mechanism or insight",
            "description": "Detailed description emphasizing what makes this theme DISTINCT from others (100+ words)",
            "importance": 0-10,
            "related_findings": [0, 1, 2],
            "key_points": ["highly specific point 1", "highly specific point 2", "highly specific point 3"],
            "sub_topics": ["specific sub-topic 1", "specific sub-topic 2"],
            "unique_angle": "What makes this theme different from others"
        }
    ],
    "narrative_flow": "How themes build on each other logically without repeating information",
    "cross_cutting_issues": ["issues that connect multiple themes in specific ways"]
}

AVOID: Generic themes like "Overview of X", "Impact on Y", "Introduction to Z"
PREFER: Specific mechanisms, dynamics, patterns, trade-offs, or insights"""

        findings_text = "\n\n".join([
            f"Finding {i}: {f.get('content', '')[:600]}"
            for i, f in enumerate(all_findings)
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Findings to analyze:\n\n{findings_text}\n\nIdentify 6-10 SPECIFIC, NON-OVERLAPPING themes. Focus on unique mechanisms, dynamics, or insights."}
        ]

        return self.chat_json(messages, max_tokens=3000)

    async def generate_section_async(
        self,
        section_title: str,
        theme_description: str,
        relevant_findings: List[Dict],
        main_query: str,
        style: str = "comprehensive"
    ) -> str:
        """
        Generate a single report section asynchronously

        Args:
            section_title: Title of the section
            theme_description: Description of what to cover
            relevant_findings: Findings relevant to this section
            main_query: Original research query
            style: Writing style (comprehensive, concise, technical)
        """
        system_prompt = f"""You are an expert analyst writing a SPECIFIC, FOCUSED section for a research report.

CRITICAL: This section must ONLY address the main research query: "{main_query}"
- Stay 100% focused on this specific topic
- Do NOT drift to tangentially related subjects
- Base ALL analysis on findings directly relevant to this query

CRITICAL REQUIREMENTS:
- Write 1000-1800 words of ORIGINAL, SPECIFIC analysis
- Focus on the UNIQUE aspects of this particular theme
- Use CONCRETE examples, data, and evidence - no generic statements
- Cite specific sources frequently [Source X]
- Avoid generic introductions or conclusions
- NO repetition of information that would appear in other sections
- Dive deep into the specific mechanisms, dynamics, or patterns
- Include counterarguments, nuances, and limitations

Style: {style}
- Clear, analytical prose with specific claims
- Evidence-based arguments with proper citations
- Logical flow between paragraphs
- Focus on INSIGHTS, not just description

DO NOT include the section title as a heading (it will be added automatically).
Start directly with substantive content."""

        findings_text = "\n\n".join([
            f"[Source {i+1}] {f.get('content', '')[:600]}\n"
            f"Confidence: {f.get('confidence', 0.5):.2f}"
            for i, f in enumerate(relevant_findings[:15])
        ])

        user_prompt = f"""Research Query: {main_query}

Section Focus: {section_title}
Unique Angle: {theme_description}

Relevant Evidence:
{findings_text}

Write a SPECIFIC, IN-DEPTH section (1000-1800 words) exploring this unique theme:

STRUCTURE:
1. Open with the most important insight or finding (no generic intro)
2. Develop 3-4 distinct sub-points with concrete evidence
3. Include specific examples, data points, and mechanisms
4. Address nuances, limitations, or counterarguments
5. Synthesize how these points connect to the broader research question

Focus on DEPTH over BREADTH. This section should reveal something specific and insightful, not generic observations. Use extensive citations."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use report model with high token limit
        # Rate limiting: ensure at least 1 second between API calls
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        if time_since_last_call < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last_call)

        attempt = 0
        while True:
            try:
                self._last_api_call = time.time()  # Update last call time

                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": TEMPERATURE_CREATIVE,
                    "max_tokens": REPORT_SECTION_MAX_TOKENS,
                    "top_p": TOP_P,
                }

                # Add timeout to prevent hanging
                response = await asyncio.wait_for(
                    self.async_client.chat.completions.create(**params),
                    timeout=180.0  # 3 minute timeout per section
                )
                content = response.choices[0].message.content
                return content or ""

            except asyncio.TimeoutError:
                log.debug(f"  [!] Report generation timeout for section (attempt {attempt + 1}/{self.max_retries})")
                if attempt >= self.max_retries - 1:
                    raise Exception(f"Report generation timed out after {self.max_retries} attempts")
                await asyncio.sleep(self.retry_delay)
                attempt += 1

            except Exception as e:
                error_str = str(e).lower()
                log.debug(f"  [!] Report generation error: {e}")

                # Handle rate limiting
                if "rate" in error_str or "429" in error_str or "capacity" in error_str:
                    if not await self._handle_rate_limit_async(attempt):
                        raise Exception(f"Rate limit exceeded after {self.max_retries} attempts")
                    attempt += 1
                    continue

                if attempt >= self.max_retries - 1:
                    raise Exception(f"Report generation failed: {str(e)}")
                await asyncio.sleep(self.retry_delay)
                attempt += 1

        raise Exception(f"Failed to generate section after {self.max_retries} attempts")

    def generate_executive_summary(
        self,
        main_query: str,
        all_sections: List[str],
        key_findings: List[str]
    ) -> str:
        """Generate executive summary after all sections are complete"""
        system_prompt = """You are an expert at writing CONCISE, INSIGHTFUL executive summaries.

Create a focused executive summary (600-900 words) that:
- Clearly states the research question and its importance
- Highlights 5-7 KEY findings with specific details (not generic statements)
- Identifies major insights and patterns that emerged
- Discusses practical implications
- Notes important limitations or uncertainties
- Maintains a clear, accessible tone

AVOID:
- Generic statements or platitudes
- Repeating section content verbatim
- Long-winded explanations
- Academic jargon without explanation

FOCUS ON:
- Specific, actionable insights
- Concrete findings backed by evidence
- Clear, direct language
- What's surprising or important about the findings"""

        sections_preview = "\n\n".join([
            f"Section {i+1} key points:\n{s[:500]}..."
            for i, s in enumerate(all_sections[:8])
        ])

        findings_text = "\n".join([f"- {f[:150]}" for f in key_findings[:10]])

        user_prompt = f"""Research Query: {main_query}

Key Findings:
{findings_text}

Section Previews:
{sections_preview}

Write a FOCUSED executive summary (600-900 words) that captures the most important insights and findings. Be specific and actionable, avoiding generic statements."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use report model with higher token limit
        attempt = 0
        while attempt < self.max_retries:
            try:
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": TEMPERATURE_CREATIVE,
                    "max_tokens": REPORT_SUMMARY_MAX_TOKENS,
                    "top_p": TOP_P,
                }

                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content
                return content or ""

            except Exception as e:
                log.debug(f"  [!] Summary generation error: {e}")
                if attempt >= self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
                attempt += 1

        raise Exception(f"Failed to generate summary after {self.max_retries} attempts")


# Backward-compatible alias: the client used to be NVIDIA-only.
NvidiaClient = LLMClient
