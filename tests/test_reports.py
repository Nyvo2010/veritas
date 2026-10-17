"""Report saving: the finished path and the partial-report path.

DeepResearchAgent is built with __new__ here on purpose: building it for real
would need live API keys, and saving is the part under test.

Run with: python -m unittest discover -s tests
"""

import logging
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from main import DeepResearchAgent  # noqa: E402

logging.getLogger("veritas").setLevel(logging.INFO)  # keep test output quiet
from models.research_state import (  # noqa: E402
    KnowledgeFinding,
    ReportSection,
    ResearchState,
)


def make_state(query="Why is the sky blue?"):
    return ResearchState(main_query=query, category_toggles={"general": True})


def make_agent(reports_dir):
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent._settings = {"reports_dir": reports_dir}
    agent.save_error = ""
    agent.last_state = None
    return agent


class SaveReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="veritas-test-")
        self.agent = make_agent(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_the_final_report(self):
        state = make_state()
        state.final_report = "# Report\n\nbody\n"
        path = self.agent.save_report(state)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(self.agent.save_error, "")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), state.final_report)

    def test_filename_is_derived_from_the_query(self):
        state = make_state("What about GPUs / NPUs: 2026?")
        state.final_report = "x"
        name = os.path.basename(self.agent.save_report(state))
        self.assertTrue(name.startswith("research_report_What_about_GPUs"))
        self.assertTrue(name.endswith(".md"))
        self.assertNotIn("/", name.split("_2026")[0].split("report_")[1])

    def test_saves_a_partial_report_when_the_run_did_not_finish(self):
        state = make_state()
        state.report_sections.append(
            ReportSection(title="Findings", content="Something specific.", sources=[], order=0)
        )
        self.agent.last_state = state
        path = self.agent.save_report(state)
        self.assertIsNotNone(path, "a stopped run must still leave a file")
        with open(path, encoding="utf-8") as handle:
            body = handle.read()
        self.assertIn("partial", body.lower())
        self.assertIn("Something specific.", body)

    def test_partial_report_falls_back_to_findings(self):
        state = make_state()
        state.findings.append(
            KnowledgeFinding(
                content="Raw finding text.",
                sources=["https://example.com"],
                confidence=0.5,
                question_id="q1",
            )
        )
        path = self.agent.save_report(state)
        self.assertIsNotNone(path)
        with open(path, encoding="utf-8") as handle:
            self.assertIn("Raw finding text.", handle.read())

    def test_reports_the_reason_when_there_is_nothing_to_save(self):
        state = make_state()
        with self.assertLogs("veritas.research", level="ERROR") as logs:
            path = self.agent.save_report(state)
        self.assertIsNone(path)
        self.assertIn("nothing to save", self.agent.save_error)
        self.assertIn("nothing to save", "\n".join(logs.output))

    def test_reports_the_reason_when_the_folder_cannot_be_used(self):
        blocker = os.path.join(self.tmp, "not-a-dir")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("x")
        agent = make_agent(os.path.join(blocker, "reports"))
        state = make_state()
        state.final_report = "body"
        with self.assertLogs("veritas.research", level="ERROR"):
            self.assertIsNone(agent.save_report(state))
        self.assertTrue(agent.save_error, "the caller needs a reason to show")

    def test_explicit_filename_is_honoured(self):
        target = os.path.join(self.tmp, "nested", "custom.md")
        state = make_state()
        state.final_report = "body"
        self.assertEqual(self.agent.save_report(state, filename=target), target)
        self.assertTrue(os.path.exists(target))

    def test_saved_report_is_timestamped(self):
        state = make_state()
        state.final_report = "x"
        stamp = datetime.now().strftime("%Y%m%d")
        self.assertIn(stamp, os.path.basename(self.agent.save_report(state)))


class PartialReportTests(unittest.TestCase):
    def test_header_states_it_is_partial(self):
        agent = make_agent(tempfile.gettempdir())
        state = make_state("How do quantization schemes differ?")
        state.report_sections.append(
            ReportSection(title="Analysis", content="Body text.", sources=[], order=0)
        )
        body = agent.build_partial_report(state)
        self.assertIn("How do quantization schemes differ?", body)
        self.assertIn("ANALYSIS", body)
        self.assertIn("Body text.", body)

    def test_empty_state_produces_nothing(self):
        agent = make_agent(tempfile.gettempdir())
        self.assertEqual(agent.build_partial_report(make_state()), "")

    def test_uses_last_state_when_no_state_is_given(self):
        agent = make_agent(tempfile.gettempdir())
        state = make_state()
        state.findings.append(KnowledgeFinding(content="kept", sources=[], confidence=0.4, question_id="q1"))
        agent.last_state = state
        self.assertIn("kept", agent.build_partial_report())


if __name__ == "__main__":
    unittest.main()
