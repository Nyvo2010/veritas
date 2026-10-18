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
from models.research_state import SubQuestion, ResearchState  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
