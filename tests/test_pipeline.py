"""Orchestrator behaviour that unit tests of individual agents cannot cover.

Empty search results are normal for an over-specific question. They must skip
that question, not abort the whole research run.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rich.console import Console  # noqa: E402

from agents.search_agent import SearchAgent  # noqa: E402
from main import DeepResearchAgent  # noqa: E402
from models.research_state import QuestionStatus, SubQuestion  # noqa: E402
from utils.ui_manager import UIManager  # noqa: E402


class _FakeUI:
    """Minimal UIManager-compatible surface used by research()."""

    def __init__(self):
        self.messages = []
        self.renderer = type("Renderer", (), {"completed_steps": 0})()
        self.timings = None

    def initialize(self, total, title):
        self.total = total

    def start(self):
        pass

    def set_phase(self, phase):
        self.phase = phase

    def set_status(self, status):
        self.status = status

    def set_progress(self, completed, label):
        self.renderer.completed_steps = completed

    def set_total(self, total):
        self.total = total

    def log(self, level, text):
        self.messages.append((level, text))

    def add_message(self, text, style=None):
        self.messages.append((style, text))

    def complete(self):
        pass

    def stop(self):
        pass


class _Analyzer:
    def __init__(self, count):
        self.count = count

    def decompose_main_query(self, query, state, ui):
        questions = []
        for index in range(self.count):
            question = SubQuestion(
                id=f"q{index}", question=f"Question {index}", priority=5
            )
            state.add_question(question)
            questions.append(question)
        return questions


class _Search:
    def adaptive_search_depth(self, question, iteration, previous_confidence=0.0):
        return "basic"

    async def execute_search(self, **kwargs):
        return []


class _RealDepthSearch:
    """Records the depth the orchestrator asks for, using the real policy."""

    def __init__(self):
        self.policy = SearchAgent(tavily_client=None, llm_client=None)
        self.depths = []

    def adaptive_search_depth(self, question, iteration, previous_confidence=None):
        depth = self.policy.adaptive_search_depth(
            question, iteration, previous_confidence
        )
        self.depths.append(depth)
        return depth

    async def execute_search(self, **kwargs):
        return []


class _RecordingSearch:
    """Captures the arguments the orchestrator passes to a search."""

    def __init__(self):
        self.kwargs = {}

    def adaptive_search_depth(self, question, iteration, previous_confidence=None):
        return "basic"

    async def execute_search(self, **kwargs):
        self.kwargs = kwargs
        return []


class _Synthesis:
    def cross_reference_findings(self, state, ui):
        return {"consensus_points": [], "contradictions": []}


class _Report:
    async def generate_report(self, state, ui, completed):
        # Mirrors the real generator, which owns state.final_report.
        state.final_report = "# Report\n"
        return state.final_report


class _LLM:
    model = "fake"


def _agent(search=None):
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.llm = _LLM()
    agent.query_analyzer = _Analyzer(3)
    agent.search_agent = search or _Search()
    agent.synthesis_agent = _Synthesis()
    agent.report_generator = _Report()
    agent.category_toggles = {}
    agent.last_state = None
    agent.save_error = ""
    agent._settings = {}
    return agent


class EmptySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_empty_searches_skip_questions_instead_of_aborting(self):
        state = await _agent().research(
            "an over-specific question",
            max_iterations=3,
            verbose=False,
            ui=_FakeUI(),
        )

        self.assertEqual(state.iteration_count, 3)
        self.assertEqual(len(state.completed_questions), 3)
        self.assertEqual(state.answered_questions(), [])
        self.assertEqual(len(state.failed_questions()), 3)
        self.assertTrue(
            all(q.status is QuestionStatus.FAILED for q in state.questions)
        )
        self.assertEqual(state.final_report, "# Report\n")


class RealUIManagerTests(unittest.IsolatedAsyncioTestCase):
    """The CLI path passes no UI object, so research() drives UIManager itself.

    It used to call a method UIManager did not have, and every non-TUI run died
    at the first progress update. The fake UI in the other tests defined the
    method, so the mismatch was invisible.
    """

    async def test_research_runs_against_the_real_ui_manager(self):
        console = Console(file=io.StringIO(), force_terminal=False, width=100)
        ui = UIManager(console=console)

        await _agent().research(
            "a plain question", max_iterations=1, verbose=False, ui=ui
        )

        self.assertGreater(ui.renderer.completed_steps, 0)

    def test_ui_manager_exposes_the_surface_research_calls(self):
        ui = UIManager(console=Console(file=io.StringIO()))
        for name in (
            "initialize", "start", "stop", "set_phase", "set_status",
            "set_progress", "set_total", "advance", "add_timing", "log",
            "complete",
        ):
            self.assertTrue(callable(getattr(ui, name, None)), f"UIManager.{name} is missing")
        self.assertTrue(hasattr(ui.renderer, "completed_steps"))


class SearchResultCountTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_configured_result_count_reaches_the_search(self):
        search = _RecordingSearch()
        agent = _agent(search)
        agent._settings = {"max_search_results": 4}

        await agent.research("q", max_iterations=1, verbose=False, ui=_FakeUI())

        self.assertEqual(search.kwargs.get("max_results"), 4)

    async def test_an_out_of_range_count_is_clamped(self):
        search = _RecordingSearch()
        agent = _agent(search)

        await agent.research(
            "q", max_iterations=1, verbose=False, ui=_FakeUI(), max_results=99
        )

        self.assertEqual(search.kwargs.get("max_results"), 20)


class SearchDepthWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_first_search_with_no_prior_answer_stays_cheap(self):
        # main used to pass 0.0 for "no previous answer", which the policy
        # read as low confidence and escalated every search to advanced.
        search = _RealDepthSearch()
        await _agent(search).research(
            "a plain question", max_iterations=1, verbose=False, ui=_FakeUI()
        )
        self.assertEqual(search.depths, ["basic"])


if __name__ == "__main__":
    unittest.main()
