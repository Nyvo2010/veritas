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
    KnowledgeFinding,
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

    def test_priority_alone_does_not_buy_the_advanced_rate(self):
        # Decomposition models rate most questions highly; treating priority
        # as "needs advanced" made every search advanced.
        self.assertEqual(
            self._depth("What is the capital of France?", priority=10), "basic"
        )

    def test_planner_requested_advanced_depth_is_honoured(self):
        question = SubQuestion(
            id="q1", question="What is the capital of France?", search_depth="advanced"
        )
        self.assertEqual(
            self.agent.adaptive_search_depth(question, 1, None), "advanced"
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
        # No search_depth in the payload leaves the choice to the policy.
        self.assertEqual([q.search_depth for q in questions], [None, None, None])
        # Priorities now share one type, so the queue can be sorted.
        self.assertIsNotNone(state.get_next_question())

    def test_planner_search_depth_is_kept(self):
        payload = {"sub_questions": [
            {"question": "Shallow?", "search_depth": "basic"},
            {"question": "Deep?", "search_depth": "Advanced"},
            {"question": "Garbage?", "search_depth": "deepest"},
        ]}
        state = ResearchState(main_query="q", category_toggles={})
        agent = QueryAnalyzerAgent(_DecomposeLLM(payload))

        questions = agent.decompose_main_query("a research question", state)

        self.assertEqual(
            [q.search_depth for q in questions], ["basic", "advanced", None]
        )

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


class SearchFailureCountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_search_is_still_counted(self):
        class _BrokenTavily:
            def search(self, **kwargs):
                raise RuntimeError("network down")

        state = ResearchState(main_query="q", category_toggles={})
        agent = SearchAgent(_BrokenTavily(), llm_client=None)
        question = SubQuestion(id="q1", question="What is it?")

        results = await agent.execute_search(question, state)

        self.assertEqual(results, [])
        self.assertEqual(state.total_searches, 1)


class OffTopicSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_unrelated_results_are_treated_as_an_empty_search(self):
        class _MovieResults:
            def search(self, **kwargs):
                return {"results": [
                    {
                        "url": "https://en.wikipedia.org/wiki/Heat",
                        "title": "Heat - Wikipedia",
                        "content": "Heat is energy in transfer to or from a system. " * 20,
                        "score": 0.24,
                    },
                    {
                        "url": "https://www.imdb.com/title/tt0113277",
                        "title": "Heat (1995) - IMDb",
                        "content": "A crime film directed by Michael Mann. " * 20,
                        "score": 0.02,
                    },
                ]}

        state = ResearchState(main_query="q", category_toggles={})
        agent = SearchAgent(_MovieResults(), llm_client=None)
        question = SubQuestion(id="q1", question="heat pumps vs hydrogen boilers")

        results = await agent.execute_search(question, state)

        self.assertEqual(results, [])
        self.assertEqual(state.search_results, [],
                         "off-topic pages must not reach the bibliography")
        self.assertEqual(state.total_searches, 1)


class SynthesisCitationTests(unittest.TestCase):
    def test_finding_sources_follow_the_models_cited_indices(self):
        state = _finding_state()
        question = state.get_question_by_id("q1")
        agent = SynthesisAgent(_AnalysisLLM([2]))

        agent.analyze_results(question, _search_results(), state)

        self.assertEqual(state.findings[0].sources, ["https://two.example"])

    def test_no_indices_means_no_claimed_sources(self):
        state = _finding_state()
        question = state.get_question_by_id("q1")

        SynthesisAgent(_AnalysisLLM([])).analyze_results(
            question, _search_results(), state
        )

        self.assertEqual(state.findings[0].sources, [],
                         "the first results must not be presented as support")
        self.assertEqual(state.findings[0].confidence, 0.9)

    def test_invalid_indices_do_not_claim_sources(self):
        state = _finding_state()
        question = state.get_question_by_id("q1")
        agent = SynthesisAgent(_AnalysisLLM(["not-a-number", 99]))

        agent.analyze_results(question, _search_results(), state)

        self.assertEqual(state.findings[0].sources, [])


class _MultiFindingLLM:
    """Returns one analysis with several distinct points."""

    def __init__(self, findings, answer_indices=None):
        self.findings = findings
        self.answer_indices = [1] if answer_indices is None else answer_indices

    def analyze_search_results(self, question, search_results, previous_findings=None):
        return {
            "answer": "The answer covering every point.",
            "confidence": 0.8,
            "source_indices": self.answer_indices,
            "findings": self.findings,
            "contradictions": [],
            "gaps": [],
            "needs_more_research": False,
        }


def _run_analysis(llm):
    state = _finding_state()
    question = state.get_question_by_id("q1")
    SynthesisAgent(llm).analyze_results(question, _search_results(), state)
    return state


class MultiFindingTests(unittest.TestCase):
    """One answer usually holds several separately citable points.

    The report writes a section per finding, so collapsing an answer into a
    single finding is what capped a report at one section per question.
    """

    def test_each_point_becomes_its_own_finding_with_its_own_sources(self):
        state = _run_analysis(_MultiFindingLLM([
            {"claim": "4-bit weights halve memory traffic.", "source_indices": [1]},
            {"claim": "Calibration is required for outlier channels.", "source_indices": [2]},
        ]))

        self.assertEqual(len(state.findings), 2)
        self.assertEqual(state.findings[0].sources, ["https://one.example"])
        self.assertEqual(state.findings[1].sources, ["https://two.example"])
        self.assertEqual(
            [finding.content for finding in state.findings],
            ["4-bit weights halve memory traffic.",
             "Calibration is required for outlier channels."],
        )

    def test_a_point_that_names_no_sources_keeps_an_empty_list(self):
        # The answer's own indices must not be copied onto a point that
        # explicitly says nothing supports it.
        state = _run_analysis(_MultiFindingLLM([
            {"claim": "A sourced point.", "source_indices": [2]},
            {"claim": "An unsourced projection.", "source_indices": []},
        ]))

        self.assertEqual(state.findings[1].sources, [])

    def test_a_point_without_indices_inherits_the_answers_support(self):
        state = _run_analysis(_MultiFindingLLM([
            {"claim": "First point."},
            {"claim": "Second point."},
        ], answer_indices=[1, 2]))

        self.assertEqual(state.findings[0].sources, [
            "https://one.example", "https://two.example"
        ])

    def test_repeated_points_are_stored_once(self):
        state = _run_analysis(_MultiFindingLLM([
            {"claim": "4-bit weights halve memory traffic.", "source_indices": [1]},
            {"claim": "4-bit weights halve the memory traffic", "source_indices": [2]},
        ]))

        self.assertEqual(len(state.findings), 1)
        self.assertEqual(state.findings[0].sources, ["https://one.example"])

    def test_a_point_already_found_in_an_earlier_pass_is_not_duplicated(self):
        state = _finding_state()
        state.findings.append(KnowledgeFinding(
            content="4-bit weights halve memory traffic.",
            sources=["https://one.example"], confidence=0.7, question_id="q1",
        ))
        llm = _MultiFindingLLM([
            {"claim": "4-bit weights halve memory traffic", "source_indices": [1]},
            {"claim": "Calibration is required for outlier channels.",
             "source_indices": [2]},
        ])

        SynthesisAgent(llm).analyze_results(
            state.get_question_by_id("q1"), _search_results(), state
        )

        self.assertEqual(len(state.findings), 2)
        self.assertEqual(state.findings[1].content,
                         "Calibration is required for outlier channels.")

    def test_the_number_of_new_findings_per_pass_is_capped(self):
        claims = [
            {"claim": f"Distinct point number {i} about quantization.", "source_indices": [1]}
            for i in range(6)
        ]
        self.assertEqual(len(_run_analysis(_MultiFindingLLM(claims)).findings), 4)

    def test_model_that_ignores_the_findings_array_still_answers(self):
        state = _run_analysis(_AnalysisLLM([1]))
        self.assertEqual(len(state.findings), 1)
        self.assertEqual(state.findings[0].content, "A supported answer.")


class _FindingsWithoutAnswerLLM:
    """Findings array populated, answer sentence empty.

    The extraction used to sit inside ``if answer:``, so these points were
    silently thrown away even though the model had produced them.
    """

    def analyze_search_results(self, question, search_results, previous_findings=None):
        return {
            "answer": "",
            "confidence": 0.7,
            "source_indices": [1],
            "findings": [
                {"claim": "4-bit weights halve memory traffic.", "source_indices": [1]},
                {"claim": "Calibration is required for outlier channels.",
                 "source_indices": [2]},
            ],
            "contradictions": [],
            "gaps": [],
            "needs_more_research": False,
        }


class FindingsWithoutAnswerTests(unittest.TestCase):
    def test_points_survive_when_the_answer_sentence_is_empty(self):
        state = _run_analysis(_FindingsWithoutAnswerLLM())

        self.assertEqual(len(state.findings), 2)
        self.assertEqual(state.findings[0].sources, ["https://one.example"])
        self.assertEqual(state.findings[1].sources, ["https://two.example"])
        self.assertEqual(state.findings[0].confidence, 0.7)
        self.assertEqual(state.findings[0].question_id, "q1")


class _DecomposeListLLM:
    model = "fake"

    def __init__(self, questions):
        self.questions = questions

    def decompose_query(self, main_query):
        return {"sub_questions": [{"question": q} for q in self.questions]}


class QuestionDedupeTests(unittest.TestCase):
    """A restated question becomes its own finding and its own section."""

    def test_a_reworded_repeat_of_an_earlier_question_is_dropped(self):
        state = ResearchState(main_query="q", category_toggles={})
        agent = QueryAnalyzerAgent(_DecomposeListLLM([
            "What is the impact of 4-bit quantization on LLM accuracy?",
            "How does 4-bit quantization affect LLM accuracy?",
            "How much memory does 4-bit quantization save?",
        ]))

        questions = agent.decompose_main_query("quantization in LLMs", state)

        self.assertEqual(len(questions), 2)
        self.assertEqual(
            [q.question for q in questions],
            ["What is the impact of 4-bit quantization on LLM accuracy?",
             "How much memory does 4-bit quantization save?"],
        )

    def test_a_short_background_question_is_not_treated_as_a_repeat(self):
        state = ResearchState(main_query="q", category_toggles={})
        agent = QueryAnalyzerAgent(_DecomposeListLLM([
            "What is quantization?",
            "How does 4-bit quantization affect LLM accuracy?",
        ]))

        self.assertEqual(
            len(agent.decompose_main_query("quantization in LLMs", state)), 2
        )

    def test_a_follow_up_that_restates_a_queued_question_is_dropped(self):
        agent = _agent()
        state = ResearchState(main_query="q", category_toggles={})
        state.add_question(SubQuestion(
            id="q1", question="How does 4-bit quantization affect LLM accuracy?"
        ))
        analysis = {
            "needs_more_research": True,
            "follow_up_questions": [
                "What is the impact of 4-bit quantization on LLM accuracy?",
                "Which calibration datasets are used?",
            ],
            "gaps": [],
        }

        questions = agent.generate_follow_up_questions(
            state.get_question_by_id("q1"), analysis, state
        )

        self.assertEqual([q.question for q in questions],
                         ["Which calibration datasets are used?"])


if __name__ == "__main__":
    unittest.main()
