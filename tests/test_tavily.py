"""Tavily client decisions: category priors and retry classification.

Network-free. The client is built with ``__new__`` and a stub transport, so
these tests exercise what the wrapper decides, not what the API returns.

Run with: python -m unittest discover -s tests
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import tavily.errors as tavily_errors  # noqa: E402

from utils.tavily_client import TavilySearchClient  # noqa: E402


def _client():
    return TavilySearchClient.__new__(TavilySearchClient)


class CategoryTests(unittest.TestCase):
    def setUp(self):
        self.client = _client()

    def test_listed_hosts_and_their_subdomains_keep_their_category(self):
        for url, expected in (
            ("https://arxiv.org/abs/2401.00001", "educational"),
            ("https://www.reddit.com/r/programming", "social"),
            ("https://scholar.google.com/citations?user=x", "educational"),
        ):
            with self.subTest(url=url):
                self.assertEqual(self.client._categorize_url(url), expected)

    def test_a_lookalike_host_does_not_inherit_a_category(self):
        for url in (
            "https://netflix.com/title/1",   # "x.com" must not claim it
            "https://notgithub.com/page",    # "github.com" must not claim it
            "https://x.com.evil.example/",   # suffix, not a dot boundary
        ):
            with self.subTest(url=url):
                self.assertEqual(self.client._categorize_url(url), "general")

    def test_unknown_and_malformed_urls_are_general(self):
        self.assertEqual(self.client._categorize_url(""), "general")
        self.assertEqual(self.client._categorize_url("https://example.com/a"), "general")


class _RecordingTransport:
    """Mirrors the current SDK: search accepts an explicit timeout."""

    def __init__(self, response=None):
        self.response = response if response is not None else {"results": []}
        self.calls = []

    def search(self, query, search_depth=None, max_results=None,
               include_answer=None, include_raw_content=None, topic=None,
               timeout=60, **kwargs):
        self.calls.append({
            "query": query, "search_depth": search_depth,
            "max_results": max_results, "include_answer": include_answer,
            "include_raw_content": include_raw_content, "topic": topic,
            "timeout": timeout, **kwargs,
        })
        return self.response


class _LegacyTransport:
    """An older SDK whose search has no timeout parameter."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(kwargs)
        return {"results": []}


class _FailingTransport:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def search(self, **kwargs):
        self.calls += 1
        raise self.error


def _client_with(error, retries=3):
    client = _client()
    client.client = _FailingTransport(error)
    client.api_key = "test"
    client.max_retries = retries
    client.retry_delay = 0
    client.timeout = 10
    return client


class RetryTests(unittest.TestCase):
    def test_every_attempt_is_bounded_by_the_configured_timeout(self):
        client = _client()
        client.client = _RecordingTransport()
        client.api_key = "test"
        client.max_retries = 1
        client.retry_delay = 0
        client.timeout = 7

        client.search("q", max_results=3)

        self.assertEqual(client.client.calls[0]["timeout"], 7)

    def test_an_older_sdk_without_timeout_support_is_not_passed_one(self):
        client = _client()
        client.client = _LegacyTransport()
        client.api_key = "test"
        client.max_retries = 1
        client.retry_delay = 0
        client.timeout = 7

        client.search("q")

        self.assertNotIn("timeout", client.client.calls[0])

    def test_the_last_attempt_does_not_sleep_before_failing(self):
        client = _client_with(tavily_errors.TimeoutError(10.0), retries=2)
        with mock.patch("utils.tavily_client.time.sleep") as sleep:
            self.assertFalse(client._handle_retry(1, "timeout"))
        sleep.assert_not_called()

    def test_an_invalid_key_fails_on_the_first_attempt(self):
        client = _client_with(tavily_errors.InvalidAPIKeyError("bad key"))
        with self.assertRaises(Exception) as caught:
            client.search("q")
        self.assertEqual(client.client.calls, 1)
        self.assertIn("InvalidAPIKeyError", str(caught.exception))

    def test_an_exhausted_usage_limit_is_not_retried(self):
        client = _client_with(tavily_errors.UsageLimitExceededError("plan limit"))
        with self.assertRaises(Exception):
            client.search("q")
        self.assertEqual(client.client.calls, 1)

    def test_a_timeout_is_retried_up_to_the_limit(self):
        client = _client_with(tavily_errors.TimeoutError(10.0), retries=3)
        with self.assertRaises(Exception):
            client.search("q")
        self.assertEqual(client.client.calls, 3)

    def test_bad_arguments_are_not_retried(self):
        client = _client_with(ValueError("bad parameter"))
        with self.assertRaises(Exception):
            client.search("q")
        self.assertEqual(client.client.calls, 1)


if __name__ == "__main__":
    unittest.main()
