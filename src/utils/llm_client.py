"""
Utility module for interacting with any OpenAI-compatible LLM endpoint.
The base URL, key, and model are supplied per install (env, .env, or the
settings file); nothing is bundled.
"""

import sys
import os

# Ensure project root is on sys.path so local modules (like config) can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import re
import time
import asyncio
import threading
from typing import List, Dict
from urllib.parse import urlparse

import httpx
import logging
from openai import OpenAI

from utils.text import clip

log = logging.getLogger("veritas.llm")


# Voice and format contract shared by every long-form report prompt, so the
# sections and the executive summary read as one document instead of a set of
# unrelated essays. Keep this in one place and inject it, rather than letting
# each prompt drift.
#
# The document is assembled by ReportGeneratorAgent._assemble_report, which
# adds the report title, the "EXECUTIVE SUMMARY" / "UNCERTAINTIES & KNOWLEDGE
# GAPS" / "SOURCES" headings, the ═ rule lines, the upper-cased section
# headings, and the global source numbering. The model must never reproduce
# any of that structure itself.
REPORT_STYLE_GUIDE = """VOICE
- Clear third-person analytic prose. No first person ("I", "we"), no
  meta-commentary about the research process, no "as an AI".
- Normal sentence case. Do NOT use ALL-CAPS for emphasis.
- Lead with the finding, then back it with evidence. No generic introductions,
  no restating the question, no empty conclusions or "in summary" filler.
- Prefer specific, verifiable claims (numbers, names, mechanisms) over
  adjectives; state uncertainty plainly instead of hedging with filler.

DOCUMENT FORMAT (Markdown)
- Return only the body text of this part. Do NOT write a document title, a
  top-level "#"/"##" heading, the section's own name, "EXECUTIVE SUMMARY",
  "SOURCES", or rule lines such as "════"; the report assembler adds all of
  those around your text.
- Use "### Subheading" (sentence-style, no trailing colon) to break the text
  into 2-4 subsections when the material has distinct parts. Do not use "#" or
  "##". Do not nest headings.
- Write flowing paragraphs of roughly 3-6 sentences. Use "-" bullet lists only
  when the content is genuinely enumerable, keep them at a single level with no
  nested bullets, and never let a list replace the analysis.
- Use **bold** sparingly, for a key term at most, never for whole sentences.
  Avoid italics for emphasis. Do not use emoji. Do not use tables.
- Match the section's expected length; do not pad to reach it.

CITATIONS
- Cite supplied evidence inline as [Source N] immediately after the claim it
  supports, using the numbers given in the evidence list, e.g. "memory
  bandwidth caps throughput [Source 3]".
- Multiple sources for one claim: [Source 1][Source 4].
- Never invent, renumber, merge, or cite a source number that was not in the
  evidence list you were given."""


def _prompt_size(messages) -> tuple:
    """(characters, message count) for a request, without dumping it all."""
    total = 0
    for message in messages or []:
        total += len(str(message.get("content", "")))
    return total, len(messages or [])


try:
    from config import (
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL,
        LLM_FALLBACK_MODEL,
        TEMPERATURE_ANALYSIS,
        TEMPERATURE_CREATIVE,
        TOP_P,
        MAX_RETRIES,
        RETRY_DELAY,
        EXPONENTIAL_BACKOFF,
        REPORT_SECTION_MAX_TOKENS,
        REPORT_SUMMARY_MAX_TOKENS,
        API_REQUEST_TIMEOUT,
        ANALYSIS_CONTENT_CHARS,
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
    except json.JSONDecodeError:
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


def _close_open_json(text: str) -> str:
    """Close a truncated JSON document so it parses as far as it got.

    Walks the text tracking string/escape state and the stack of open
    containers, then terminates the dangling string (if any), drops a
    trailing separator, and appends the matching closers. Best effort: the
    caller still validates the result.
    """
    in_string = False
    escape = False
    stack: List[str] = []
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()

    out = text
    if in_string:
        out += '"'
    out = out.rstrip()
    while out and out[-1] in ",:":
        out = out[:-1].rstrip()
    return out + "".join(reversed(stack))


def _repair_truncated_json(text: str, max_cuts: int = 8) -> str:
    """Salvage a document cut off mid-value by closing it, backtracking if needed."""
    candidate = text
    for _ in range(max_cuts + 1):
        closed = _close_open_json(candidate)
        try:
            json.loads(closed, strict=False)
            return closed
        except json.JSONDecodeError:
            cut = max(candidate.rfind(","), candidate.rfind("{"), candidate.rfind("["))
            if cut <= 0:
                break
            candidate = candidate[:cut]
    return _close_open_json(text)


def _looks_truncated(error: json.JSONDecodeError, content: str) -> bool:
    """Heuristic: an error at the very end of the reply means it ran out of tokens."""
    if not content:
        return False
    if "Unterminated string" in str(error):
        return True
    return error.pos >= max(0, len(content) - 2)


def _extract_json(text: str) -> Dict:
    """Parse JSON from model output, tolerating fences, prose, and truncation.

    Reasoning models sometimes wrap JSON in ```json``` blocks, prepend
    reasoning text, or run out of tokens mid-document. This strips fences,
    extracts the first balanced {...} object, and finally salvages a
    truncated document so a cut-off reply still yields partial data.
    """
    cleaned = text.strip()
    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    candidates = [cleaned]
    start = cleaned.find("{")
    if start != -1 and start > 0:
        candidates.append(cleaned[start:])

    last_error = None
    for candidate in candidates:
        try:
            return _tolerant_loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e

    # Fall back to the first balanced {...} block.
    if start != -1:
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
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _tolerant_loads(cleaned[start:i + 1])
                    except json.JSONDecodeError as e:
                        last_error = e
                    break

    # Last resort: the reply was cut off — salvage what completed.
    for candidate in candidates:
        try:
            return _tolerant_loads(_repair_truncated_json(candidate))
        except json.JSONDecodeError as e:
            last_error = e

    raise last_error if last_error else json.JSONDecodeError("No JSON object found", text, 0)



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
    started = time.perf_counter()
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
    log.debug("listed %s models from %s in %.2fs", len(ids), normalized, time.perf_counter() - started)
    return ids


class LLMClient:
    """Wrapper for any OpenAI-compatible chat API with research helpers."""

    def __init__(
        self,
        api_key: str = None,
        api_base_url: str = None,
        model: str = None,
        fallback_model: str = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the endpoint (default from config/env)
            api_base_url: Base URL of the OpenAI-compatible endpoint
            model: Model id to use for all inference (default from config/env)
            fallback_model: Optional model to retry with when the primary one
                keeps failing (default from config/env). Empty disables it.
        """
        self.api_base_url = self._normalize_base_url(api_base_url or LLM_BASE_URL)
        timeout = API_REQUEST_TIMEOUT if API_REQUEST_TIMEOUT else 120
        resolved_key = api_key if api_key is not None else LLM_API_KEY
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=self.api_base_url,
            timeout=timeout,
        )
        self.model = ((model if model is not None else LLM_MODEL) or "").strip()
        if not self.model:
            raise ValueError(
                "No LLM model id configured. Set one in Settings or the LLM_MODEL variable."
            )
        default_fallback = fallback_model if fallback_model is not None else LLM_FALLBACK_MODEL
        self.fallback_model = (default_fallback or "").strip()
        if self.fallback_model == self.model:
            self.fallback_model = ""
        self.using_fallback = False
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self._last_api_call = 0  # Track last API call time for rate limiting
        # Minimum seconds between API calls. Widen it on rate-limited tiers
        # (e.g. a free Gemini key) with VERITAS_LLM_RATE_LIMIT_DELAY.
        try:
            self._rate_limit_delay = float(os.environ.get("VERITAS_LLM_RATE_LIMIT_DELAY") or "1.0")
        except ValueError:
            self._rate_limit_delay = 1.0
        self._sync_lock = threading.Lock()  # serialises threaded section calls
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
                "LLM API base URL must include a scheme and hostname, "
                "e.g. https://your-endpoint.example/v1"
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

    def _with_fallback(self, call, *args, **kwargs):
        """Run *call*, retrying once on the fallback model if one is set.

        The primary model gets its full retry budget first; only when it
        exhausts that do we switch. Once switched we stay on the fallback for
        the rest of this client's life (a fresh client is built per run).
        """
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            fallback = (self.fallback_model or "").strip()
            if not fallback or fallback == self.model:
                raise
            log.warning(
                "model %r failed after %s attempts (%s); switching to fallback %r",
                self.model, self.max_retries, exc, fallback,
            )
            self.model = fallback
            self.using_fallback = True
            return call(*args, **kwargs)

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

    def _throttle_sync(self) -> None:
        """Space threaded section calls at least ``_rate_limit_delay`` apart.

        Report sections run concurrently on a thread pool; without a lock they
        all read the same ``_last_api_call`` and fire in a burst, which trips
        free-tier rate limits. Holding one lock across the wait-then-stamp
        serialises their request start times.
        """
        with self._sync_lock:
            wait = self._rate_limit_delay - (time.time() - self._last_api_call)
            if wait > 0:
                time.sleep(wait)
            self._last_api_call = time.time()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = TEMPERATURE_ANALYSIS,
        max_tokens: int = 4096,
        json_mode: bool = False,
        top_p: float = TOP_P,
    ) -> str:
        """Chat completion with the primary model, falling back once if it fails."""
        return self._with_fallback(
            self._chat_completion,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            top_p=top_p,
        )

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = TEMPERATURE_ANALYSIS,
        max_tokens: int = 4096,
        json_mode: bool = False,
        top_p: float = TOP_P,
    ) -> str:
        """
        Send a chat completion request with retry logic

        Args:
            messages: Chat messages as dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            json_mode: Enable JSON mode for structured output
            top_p: Nucleus sampling parameter

        Returns:
            Response text
        """
        # Rate limiting: ensure at least 1 second between API calls
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        if time_since_last_call < self._rate_limit_delay:
            waited = self._rate_limit_delay - time_since_last_call
            log.debug("rate limiter: waiting %.2fs before the next call", waited)
            time.sleep(waited)

        chars, count = _prompt_size(messages)
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            started = time.perf_counter()
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

                log.debug(
                    "chat completion -> %s | %s msgs, %s chars, temp=%s, max_tokens=%s, "
                    "json=%s, attempt=%s/%s | last user msg: %s",
                    self.model, count, chars, temperature, max_tokens, json_mode,
                    attempt + 1, self.max_retries, clip(messages[-1].get("content") if messages else "", 400),
                )

                response = self.client.chat.completions.create(**params)
                self._last_finish_reason = response.choices[0].finish_reason
                content = response.choices[0].message.content
                usage = getattr(response, "usage", None)
                log.info(
                    "chat completion <- %s in %.2fs | %s chars, finish=%s, "
                    "tokens in/out=%s/%s | %s",
                    self.model, time.perf_counter() - started, len(content or ""),
                    self._last_finish_reason,
                    getattr(usage, "prompt_tokens", "?") if usage else "?",
                    getattr(usage, "completion_tokens", "?") if usage else "?",
                    clip(content, 200),
                )
                return content or ""

            except httpx.ConnectError as e:
                log.error("cannot reach %s: %s", self.api_base_url, e)
                self._raise_connection_error(e)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                log.warning(
                    "chat completion failed after %.2fs (attempt %s/%s): %s",
                    time.perf_counter() - started, attempt + 1, self.max_retries, e,
                )

                # Handle rate limiting (match real limit signals only — a bare
                # "rate" substring also matches echoed params like "temperature")
                if ("429" in error_str or "rate limit" in error_str
                        or "rate_limit" in error_str or "quota" in error_str
                        or "resource_exhausted" in error_str
                        or "too many requests" in error_str):
                    if attempt >= self.max_retries - 1:
                        raise Exception(
                            f"Rate limited by the LLM API after {self.max_retries} attempts. "
                            "Raise VERITAS_LLM_RATE_LIMIT_DELAY, use a lighter model, or check "
                            f"your quota: {e}"
                        )
                    if not self._handle_rate_limit_sync(attempt):
                        raise Exception(f"Max retries ({self.max_retries}) exceeded due to rate limiting")
                    attempt += 1

                # Handle timeout errors
                elif "timeout" in error_str or "timed out" in error_str:
                    if attempt >= self.max_retries - 1:
                        raise Exception(f"API timeout after {self.max_retries} attempts")
                    time.sleep(self.retry_delay)
                    attempt += 1

                # Handle bad request errors
                elif "400" in error_str or "bad request" in error_str:
                    raise Exception(f"Bad request to the LLM API: {e}")

                # Handle unknown-model / bad-endpoint errors (don't retry)
                elif "404" in error_str or "not found" in error_str:
                    raise Exception(
                        f"Model or endpoint not found by the LLM API (check the model id "
                        f"in Settings): {e}"
                    )

                # Handle auth errors (don't retry)
                elif "401" in error_str or "403" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
                    raise Exception(f"LLM API authentication failed (check the API key): {e}")

                # Handle general API / transient network errors
                elif ("api" in error_str or "500" in error_str or "503" in error_str
                      or "connection" in error_str or "connect" in error_str
                      or "reset" in error_str or "temporarily" in error_str):
                    if attempt >= self.max_retries - 1:
                        raise Exception(f"API error after {self.max_retries} attempts: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    attempt += 1

                else:
                    raise

        raise Exception(
            f"Failed to complete request after {self.max_retries} attempts"
            + (f": {last_error}" if last_error else "")
        )

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
        chars, count = _prompt_size(messages)

        for parse_attempt in range(1, max_parse_retries + 1):
            content = self.chat_completion(
                attempt_messages,
                temperature=temperature,
                max_tokens=current_max_tokens,
                json_mode=True,
                top_p=top_p,
            )
            try:
                parsed = _extract_json(content)
                log.debug("json parse ok (attempt %s/%s, %s msgs, %s chars in)",
                          parse_attempt, max_parse_retries, count, chars)
                return parsed
            except json.JSONDecodeError as e:
                last_error = e
                if parse_attempt >= max_parse_retries:
                    break
                truncated = self._last_finish_reason == "length" or _looks_truncated(e, content)
                if truncated:
                    current_max_tokens = min(int(current_max_tokens * 1.5) + 512, 16000)
                    log.warning(
                        "json truncated (attempt %s/%s); retrying with max_tokens=%s",
                        parse_attempt, max_parse_retries, current_max_tokens,
                    )
                    # Fresh retry with the original messages and a bigger budget
                    attempt_messages = list(messages)
                else:
                    log.warning(
                        "json parse failed (attempt %s/%s): %s; asking model to repair: %s",
                        parse_attempt, max_parse_retries, e, clip(content, 200),
                    )
                    attempt_messages = attempt_messages + [
                        {"role": "assistant", "content": content[:4000]},
                        {"role": "user", "content": self._JSON_REPAIR_NUDGE},
                    ]

        raise Exception(
            f"Failed to get valid JSON after {max_parse_retries} attempts: {last_error}"
        )

    def decompose_query(self, main_query: str) -> Dict:
        """Decompose a main query into prioritised sub-questions.

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
        """Analyse search results and decide whether the question is answered.

        Returns dict with: analysis, answer, confidence, gaps, follow_up_questions,
            needs_more_research
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
- Weight high-quality sources over thin, promotional, or user-generated pages.
  If only low-quality sources support a claim, say so and lower the confidence.

Return JSON:
{
    "observation": "Summary of what sources say ABOUT THIS SPECIFIC QUESTION",
    "analysis": "Your critical analysis OF THE QUESTION ASKED",
    "answer": "Best answer based on available info FOR THIS SPECIFIC QUESTION",
    "confidence": 0.0-1.0,
    "source_agreement": "high/medium/low",
    "source_indices": [1, 3],
    "contradictions": ["list any contradictions found"],
    "gaps": ["list knowledge gaps DIRECTLY RELATED TO THIS QUESTION"],
    "follow_up_questions": ["questions that need more research WITHIN THIS TOPIC"],
    "needs_more_research": true/false,
    "reasoning": "Why you need/don't need more research FOR THIS SPECIFIC QUESTION"
}

"source_indices" must be 1-based indices of the Search Results that directly
support the answer. Do not list a source you did not use."""

        # Format search results. The pipeline already selected the strongest
        # subset; spend a real text budget on each one instead of reading only
        # the opening paragraph.
        results_text = "\n\n".join([
            f"Source {i+1}: {r.get('title', 'No title')}\n"
            f"URL: {r.get('url', 'No URL')}\n"
            f"Category: {r.get('category', 'general')} · "
            f"evidence quality: {float(r.get('quality', 0.0)):.2f}"
            + (f" · published: {r['published_date']}" if r.get('published_date') else "")
            + f"\nContent: {(r.get('content') or 'No content')[:ANALYSIS_CONTENT_CHARS]}..."
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
        """Meta-analysis: cluster all findings into distinct report themes.

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

    def generate_section(
        self,
        section_title: str,
        theme_description: str,
        relevant_findings: List[Dict],
        main_query: str,
        style: str = "comprehensive",
    ) -> str:
        """Generate one report section (sync) with primary -> fallback retry."""
        return self._with_fallback(
            self._generate_section_sync,
            section_title,
            theme_description,
            relevant_findings,
            main_query,
            style,
        )

    async def generate_section_async(
        self,
        section_title: str,
        theme_description: str,
        relevant_findings: List[Dict],
        main_query: str,
        style: str = "comprehensive",
    ) -> str:
        """Generate one report section without blocking the event loop.

        Sections used to go through the async OpenAI client, but its async TLS
        stack (anyio/httpcore) fails with a connection reset on some platforms
        even though the sync client works fine. Running the sync path on a
        worker thread keeps sections overlapping without that failure mode.
        """
        return await asyncio.to_thread(
            self.generate_section,
            section_title,
            theme_description,
            relevant_findings,
            main_query,
            style,
        )

    def _generate_section_sync(
        self,
        section_title: str,
        theme_description: str,
        relevant_findings: List[Dict],
        main_query: str,
        style: str = "comprehensive",
    ) -> str:
        """Generate a single report section.

        Args:
            section_title: Section heading (added automatically).
            theme_description: What this section should cover.
            relevant_findings: Findings fed to this section.
            main_query: The user's original question.
            style: Writing style (comprehensive, concise, technical).
        """

        system_prompt = f"""You are an expert analyst writing one section of a research report.

Scope: this section speaks only to the main research query — "{main_query}".
Target style: {style}. Stay on topic, do not drift to tangentially related
subjects, and base every claim on the evidence provided.

Write 1000-1800 words of original analysis that:
- examines the specific mechanisms, dynamics, or patterns behind this theme
- uses concrete examples, figures, and evidence rather than generalities
- includes counterarguments, nuances, and limitations where they exist
- avoids material that other sections would already cover

{REPORT_STYLE_GUIDE}

Return only the section body; its heading is added automatically."""


        def evidence_label(finding: Dict, index: int) -> str:
            """Stable, report-wide source numbers supplied by the assembler."""
            numbers = finding.get("source_numbers") or []
            if numbers:
                return "".join(f"[Source {int(number)}] " for number in numbers)
            # Standalone callers (tests, other frontends) may not have a
            # registry; fall back to local numbering for the prompt only.
            return f"[Source {index + 1}] "

        findings_text = "\n\n".join([
            f"{evidence_label(f, i)}{f.get('content', '')[:600]}\n"
            f"Confidence: {float(f.get('confidence', 0.5)):.2f}"
            for i, f in enumerate(relevant_findings[:15])
        ])

        user_prompt = f"""Research Query: {main_query}

Section Focus: {section_title}
Unique Angle: {theme_description}

Relevant Evidence:
{findings_text}

Write this section (1000-1800 words), exploring the theme in depth:

STRUCTURE:
1. Open with the most important insight or finding (no generic intro)
2. Develop 3-4 distinct sub-points, each with concrete evidence
3. Include specific examples, data points, and mechanisms
4. Address nuances, limitations, or counterarguments
5. Connect the points back to the broader research question

Prioritise depth over breadth: reveal something specific and well-evidenced,
not generic observations. Cite the evidence above as [Source N]. Use only the
numbers shown in the evidence labels; when a finding carries several numbers,
cite only the ones that actually support that sentence."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Rate limiting: keep concurrent sections spaced out on the wire.
        self._throttle_sync()

        chars, count = _prompt_size(messages)
        started = time.perf_counter()
        log.info("section %r -> %s | %s msgs, %s chars, max_tokens=%s",
                 section_title, self.model, count, chars, REPORT_SECTION_MAX_TOKENS)
        # _chat_completion already handles retries, backoff and rate-limit
        # pauses, so the section path reuses it rather than re-implementing them.
        content = self._chat_completion(
            messages=messages,
            temperature=TEMPERATURE_CREATIVE,
            max_tokens=REPORT_SECTION_MAX_TOKENS,
            top_p=TOP_P,
        )
        log.info("section %r <- %s chars in %.1fs",
                 section_title, len(content or ""), time.perf_counter() - started)
        return content or ""

    def generate_executive_summary(
        self,
        main_query: str,
        all_sections: List[str],
        key_findings: List[str]
    ) -> str:
        """Executive summary with the primary model, falling back once on failure."""
        return self._with_fallback(
            self._generate_executive_summary, main_query, all_sections, key_findings
        )

    def _generate_executive_summary(
        self,
        main_query: str,
        all_sections: List[str],
        key_findings: List[str]
    ) -> str:
        """Generate executive summary after all sections are complete"""
        system_prompt = f"""You are an expert analyst writing the executive summary of a research report.

Write 600-900 words that:
- state the research question and why it matters
- surface the 5-7 most important findings, each with concrete detail
- draw out the patterns and tensions across the findings
- note the practical implications and the important limitations or uncertainties

        {REPORT_STYLE_GUIDE}

        Unlike the sections, this summary has no numbered source list, so omit
        the [Source N] markers here — the sections carry the citations.

        The summary must stand alone: do not refer to "the report", "the sections
        above", or the research process itself."""

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

Write the executive summary (600-900 words) capturing the most important
insights. Be specific and actionable; avoid generic statements."""

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

