"""QueryAnalyzerAgent follow-up generation.

Regression for a call that passed the UI object into ``max_depth`` positionally,
producing ``'>=' not supported between instances of 'int' and 'QueueUI'`` when
the depth check ran. The extra arguments are now keyword-only.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agents.query_analyzer import QueryAnalyzerAgent  # noqa: E402
from agents.search_agent import SearchAgent  # noqa: E402
from agents.synthesis_agent import SynthesisAgent  # noqa: E402
from models.research_state import (  # noqa: E402
    ResearchState,
    SearchResult,
    SubQuestion,
)


class _FakeLLM:
    model = "fake"

    def decompose_query(self, main_query):
        return {"sub_questions": []}


class _SentinelUI:
    """Stand-in for QueueUI: not an int, so depth math would blow up on it."""

    def add_message(self, *a, **k):
        pass


def _agent():
    return QueryAnalyzerAgent(_FakeLLM())


def _question(depth=0):
    return SubQuestion(id="q1", question="Why?", priority=5, depth=depth)


class FollowUpTests(unittest.TestCase):
    def test_ui_is_accepted_as_a_keyword(self):
        agent = _agent()
        state = ResearchState(main_query="q", category_toggles={})
        analysis = {"needs_more_research": True, "gaps": ["a"], "follow_up_questions": []}
        # Must not raise: the UI object must never land in the max_depth slot.
        result = agent.generate_follow_up_questions(
            _question(), analysis, state, ui_manager=_SentinelUI()
        )
        self.assertEqual(result, [])

    def test_ui_cannot_be_passed_positionally_into_max_depth(self):
        agent = _agent()
        state = ResearchState(main_query="q", category_toggles={})
        analysis = {"needs_more_research": True, "gaps": [], "follow_up_questions": []}
        with self.assertRaises(TypeError):
            agent.generate_follow_up_questions(_question(), analysis, state, _SentinelUI())

    def test_depth_limit_stops_follow_ups(self):
        agent = _agent()
        state = ResearchState(main_query="q", category_toggles={})
        analysis = {"needs_more_research": True, "gaps": [], "follow_up_questions": ["x"]}
        self.assertEqual(
            agent.generate_follow_up_questions(_question(depth=3), analysis, state), []
        )


class AdaptiveDepthTests(unittest.TestCase):
    """Depth decides the Tavily bill, so only questions that need it escalate."""

    def setUp(self):
        self.agent = SearchAgent(tavily_client=None, llm_client=None)

    def _depth(self, text, priority=5, previous=None):
        question = SubQuestion(id="q1", question=text, priority=priority)
        return self.agent.adaptive_search_depth(question, 1, previous)

    def test_plain_short_question_uses_the_cheap_depth(self):
        self.assertEqual(self._depth("What is the capital of France?"), "basic")

    def test_high_priority_question_uses_advanced(self):
        self.assertEqual(
            self._depth("What is the capital of France?", priority=8), "advanced"
        )

    def test_long_multi_clause_question_uses_advanced(self):
        text = (
            "How do transformer attention mechanisms scale compared with recurrent "
            "networks on long context reasoning benchmarks"
        )
        self.assertEqual(self._depth(text), "advanced")

    def test_time_sensitive_question_uses_advanced(self):
        self.assertEqual(self._depth("What is the latest Python release?"), "advanced")

    def test_low_previous_confidence_escalates_a_research(self):
        self.assertEqual(
            self._depth("What is the capital of France?", previous=0.2), "advanced"
        )

    def test_unknown_previous_confidence_is_not_treated_as_low(self):
        # A first search has no prior answer. Passing 0.0 for it would mark
        # every search advanced, which is the bug this guards.
        self.assertEqual(self._depth("What is the capital of France?"), "basic")


class _DecomposeLLM:
    model = "fake"

    def __init__(self, payload):
        self.payload = payload

    def decompose_query(self, main_query):
        return self.payload


class MalformedModelOutputTests(unittest.TestCase):
    """Model JSON is untrusted: wrong types must not crash or multiply items."""

    def test_string_and_out_of_range_priorities_are_coerced(self):
        payload = {"sub_questions": [
            {"question": "First?", "priority": "8"},
            {"question": "Second?", "priority": "urgent"},
            {"question": "Third?", "priority": 90},
        ]}
        state = ResearchState(main_query="q", category_toggles={})
        agent = QueryAnalyzerAgent(_DecomposeLLM(payload))

        questions = agent.decompose_main_query("a research question", state)

        self.assertEqual([q.priority for q in questions], [8, 5, 10])
        # Priorities now share one type, so the queue can be sorted.
        self.assertIsNotNone(state.get_next_question())

    def test_non_dict_questions_are_skipped(self):
        payload = {"sub_questions": ["just a string", {"question": "Kept?"}]}
        state = ResearchState(main_query="q", category_toggles={})
        agent = QueryAnalyzerAgent(_DecomposeLLM(payload))

        questions = agent.decompose_main_query("a research question", state)

        self.assertEqual([q.question for q in questions], ["Kept?"])

    def test_a_string_follow_up_field_yields_one_question(self):
        agent = _agent()
        state = ResearchState(main_query="q", category_toggles={})
        analysis = {
            "needs_more_research": True,
            "follow_up_questions": "What next?",
            "gaps": [],
        }

        questions = agent.generate_follow_up_questions(_question(), analysis, state)

        self.assertEqual([q.question for q in questions], ["What next?"])

    def test_the_string_false_does_not_start_follow_up_research(self):
        agent = _agent()
        state = ResearchState(main_query="q", category_toggles={})
        analysis = {
            "needs_more_research": "false",
            "follow_up_questions": ["What next?"],
            "gaps": [],
        }

        self.assertEqual(
            agent.generate_follow_up_questions(_question(), analysis, state), []
        )


class _FakeTavily:
    """Returns one prepared Tavily response per call, in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def search(self, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _tavily_result(url, title="A useful source", content="x" * 600, score=0.7):
    return {"url": url, "title": title, "content": content, "score": score,
            "raw_content": content}


class SearchAgentFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicates_and_thin_results_never_reach_the_state(self):
        tavily = _FakeTavily([{"results": [
            _tavily_result("https://example.com/a"),
            _tavily_result("http://www.example.com/a/"),
            _tavily_result("https://example.com/thin", content="short", score=0.1,
                          title="x"),
        ]}])
        state = ResearchState(main_query="q", category_toggles={})
        agent = SearchAgent(tavily, llm_client=None)
        question = SubQuestion(id="q1", question="What is it?")

        results = await agent.execute_search(question, state)

        self.assertEqual(len(state.search_results), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(state.search_results[0].url, "https://example.com/a")
        self.assertEqual(state.total_searches, 1)

    async def test_a_repeated_url_is_merged_not_counted_twice(self):
        tavily = _FakeTavily([
            {"results": [_tavily_result("https://example.com/a", content="a" * 400)]},
            {"results": [_tavily_result("https://example.com/a", content="b" * 900)]},
        ])
        state = ResearchState(main_query="q", category_toggles={})
        agent = SearchAgent(tavily, llm_client=None)
        question = SubQuestion(id="q1", question="What is it?")

        await agent.execute_search(question, state)
        second = await agent.execute_search(question, state)

        self.assertEqual(len(state.search_results), 1)
        self.assertEqual(len(second), 1)
        self.assertIn("b" * 100, state.search_results[0].full_content)


class _AnalysisLLM:
    def __init__(self, source_indices):
        self.source_indices = source_indices

    def analyze_search_results(self, question, search_results, previous_findings=None):
        return {
            "answer": "A supported answer.",
            "confidence": "0.9",  # models sometimes return a string
            "source_indices": self.source_indices,
            "contradictions": [],
            "gaps": [],
            "needs_more_research": False,
        }


def _finding_state():
    state = ResearchState(main_query="q", category_toggles={})
    state.add_question(SubQuestion(id="q1", question="What is it?"))
    return state


def _search_results():
    return [
        SearchResult(url="https://one.example", title="One", content="x" * 600, score=0.8),
        SearchResult(url="https://two.example", title="Two", content="y" * 600, score=0.9),
    ]


class _MalformedAnalysisLLM:
    def analyze_search_results(self, question, search_results, previous_findings=None):
        return {
            "answer": "A supported answer.",
            "confidence": "high",  # unparseable
            "source_indices": [],
            "contradictions": "one contradiction",  # string, not a list
            "gaps": "one gap",
            "needs_more_research": False,
        }


class MalformedAnalysisTests(unittest.TestCase):
    def test_string_fields_become_single_items(self):
        state = _finding_state()
        question = state.get_question_by_id("q1")

        SynthesisAgent(_MalformedAnalysisLLM()).analyze_results(
            question, _search_results(), state
        )

        finding = state.findings[0]
        self.assertEqual(finding.contradictions, ["one contradiction"])
        self.assertEqual([gap.description for gap in state.gaps], ["one gap"])
        self.assertEqual(finding.confidence, 0.5)


class SynthesisCitationTests(unittest.TestCase):
    def test_finding_sources_follow_the_models_cited_indices(self):
        state = _finding_state()
        question = state.get_question_by_id("q1")
        agent = SynthesisAgent(_AnalysisLLM([2]))

        agent.analyze_results(question, _search_results(), state)

        self.assertEqual(state.findings[0].sources, ["https://two.example"])
        self.assertEqual(state.findings[0].confidence, 0.9)

    def test_missing_or_invalid_indices_fall_back_to_the_result_list(self):
        state = _finding_state()
        question = state.get_question_by_id("q1")
        agent = SynthesisAgent(_AnalysisLLM(["not-a-number", 99]))

        agent.analyze_results(question, _search_results(), state)

        self.assertEqual(
            state.findings[0].sources,
            ["https://one.example", "https://two.example"],
        )


if __name__ == "__main__":
    unittest.main()
