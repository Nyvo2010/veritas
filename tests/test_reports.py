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
import threading
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import asyncio  # noqa: E402

from main import DeepResearchAgent  # noqa: E402
from agents.report_generator import ReportGeneratorAgent  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402

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


class _FakeLLM:
    """Returns each section after a per-title delay, completing out of order."""

    def __init__(self, delays):
        self.delays = delays
        self.requested = []

    def generate_section_async(self, *, section_title, theme_description,
                               relevant_findings, main_query, style):
        self.requested.append(section_title)

        async def run():
            await asyncio.sleep(self.delays[section_title])
            return f"BODY:{section_title}"

        return run()


class SectionOrderTests(unittest.IsolatedAsyncioTestCase):
    """Sections must come back in theme order even when they finish out of order.

    Regression for a NameError on the parallel gather that dropped the section
    index and then re-sorted by an undefined ``order``.
    """

    async def _run(self, themes, delays):
        state = make_state("How do quantization schemes differ?")
        state.findings.append(
            KnowledgeFinding(content="f", sources=["s"], confidence=0.9, question_id="q1")
        )
        agent = ReportGeneratorAgent(_FakeLLM(delays))
        return await agent._generate_sections_parallel(state, {"themes": themes}, "comprehensive")

    async def test_order_follows_themes_not_completion(self):
        themes = [{"title": "Alpha"}, {"title": "Beta"}, {"title": "Gamma"}]
        delays = {"Alpha": 0.03, "Beta": 0.02, "Gamma": 0.01}  # reverse completion
        sections, failed = await self._run(themes, delays)
        self.assertEqual(failed, 0)
        self.assertEqual([s.title for s in sections], ["Alpha", "Beta", "Gamma"])
        self.assertEqual([s.order for s in sections], [0, 1, 2])
        self.assertEqual([s.content for s in sections], ["BODY:Alpha", "BODY:Beta", "BODY:Gamma"])

    async def test_a_failed_section_is_counted_but_keeps_its_slot(self):
        themes = [{"title": "Alpha"}, {"title": "Beta"}]

        class _Boom(_FakeLLM):
            def generate_section_async(self, *, section_title, **kw):
                async def run():
                    if section_title == "Alpha":
                        raise RuntimeError("boom")
                    return f"BODY:{section_title}"

                return run()

        state = make_state()
        agent = ReportGeneratorAgent(_Boom({}))
        sections, failed = await agent._generate_sections_parallel(state, {"themes": themes}, "comprehensive")
        self.assertEqual(failed, 1)
        self.assertEqual([s.title for s in sections], ["Alpha", "Beta"])
        self.assertIn("failed", sections[0].content.lower())


class _SyncSectionLLM(LLMClient):
    """An LLMClient with no network setup whose sync section path is recorded."""

    def __init__(self):
        self.model = "stub-model"
        self.fallback_model = ""
        self.using_fallback = False
        self._rate_limit_delay = 0.0
        self._last_api_call = 0.0
        self._sync_lock = threading.Lock()
        self.seen = []

    def _generate_section_sync(self, section_title, theme_description,
                               relevant_findings, main_query, style="comprehensive"):
        self.seen.append((section_title, threading.current_thread().name))
        return f"BODY:{section_title}"


class SectionThreadPathTests(unittest.IsolatedAsyncioTestCase):
    """Sections must be produced by the sync client on a worker thread.

    The async OpenAI transport (anyio/httpcore TLS) resets the connection on
    some platforms even though the sync client works, so
    ``generate_section_async`` must not depend on an async client.
    """

    async def test_async_wrapper_uses_the_sync_path(self):
        llm = _SyncSectionLLM()
        out = await llm.generate_section_async("Alpha", "d", [], "q")
        self.assertEqual(out, "BODY:Alpha")
        self.assertEqual([t for t, _ in llm.seen], ["Alpha"])

    async def test_no_async_client_is_required(self):
        llm = _SyncSectionLLM()
        self.assertFalse(hasattr(llm, "async_client"))
        out = await llm.generate_section_async("Beta", "d", [], "q")
        self.assertEqual(out, "BODY:Beta")

    async def test_each_section_runs_off_the_event_loop_thread(self):
        llm = _SyncSectionLLM()
        results = await asyncio.gather(*[
            llm.generate_section_async(f"S{i}", "d", [], "q") for i in range(5)
        ])
        self.assertEqual(results, [f"BODY:S{i}" for i in range(5)])
        loop_thread = threading.current_thread().name
        self.assertTrue(all(name != loop_thread for _, name in llm.seen))


if __name__ == "__main__":
    unittest.main()