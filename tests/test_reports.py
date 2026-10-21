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
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import asyncio  # noqa: E402

from main import DeepResearchAgent  # noqa: E402
from agents.report_generator import ReportGeneratorAgent  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402
from utils.sources import build_source_registry  # noqa: E402

logging.getLogger("veritas").setLevel(logging.INFO)  # keep test output quiet
from models.research_state import (  # noqa: E402
    KnowledgeFinding,
    QuestionStatus,
    ReportSection,
    ResearchState,
    SearchResult,
    SubQuestion,
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

    def test_no_temporary_file_is_left_behind(self):
        state = make_state()
        state.final_report = "x"
        self.agent.save_report(state)
        self.assertEqual(
            [name for name in os.listdir(self.tmp) if name.startswith(".report-")],
            [],
        )

    def test_a_write_failure_leaves_the_previous_report_intact(self):
        state = make_state()
        state.final_report = "# First\n"
        path = self.agent.save_report(state, filename=os.path.join(self.tmp, "report.md"))
        with open(path, encoding="utf-8") as handle:
            original = handle.read()

        state.final_report = "# Second\n"
        with mock.patch("main.os.replace", side_effect=OSError("disk full")):
            result = self.agent.save_report(state, filename=path)

        self.assertIsNone(result)
        self.assertIn("cannot write", self.agent.save_error)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), original, "truncated the previous report")
        self.assertEqual(
            [name for name in os.listdir(self.tmp) if name.startswith(".report-")],
            [],
        )

    @unittest.skipIf(os.name == "nt", "POSIX permissions only")
    def test_replacing_a_report_keeps_its_permissions(self):
        state = make_state()
        state.final_report = "one"
        path = self.agent.save_report(state, filename=os.path.join(self.tmp, "r.md"))
        os.chmod(path, 0o640)

        state.final_report = "two"
        self.agent.save_report(state, filename=path)

        self.assertEqual(os.stat(path).st_mode & 0o777, 0o640)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "two")

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

    def test_explicit_filename_with_a_blocked_parent_reports_the_reason(self):
        blocker = os.path.join(self.tmp, "file-not-dir")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("x")
        state = make_state()
        state.final_report = "body"
        target = os.path.join(blocker, "report.md")

        with self.assertLogs("veritas.research", level="ERROR"):
            path = self.agent.save_report(state, filename=target)

        self.assertIsNone(path)
        self.assertIn("cannot create the report folder", self.agent.save_error)

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
        self.assertIn("## Analysis", body)
        self.assertIn("Body text.", body)

    def test_empty_state_produces_nothing(self):
        agent = make_agent(tempfile.gettempdir())
        self.assertEqual(agent.build_partial_report(make_state()), "")

    def test_header_separates_failed_questions_from_answered(self):
        agent = make_agent(tempfile.gettempdir())
        state = make_state()
        state.add_question(SubQuestion(id="q1", question="answered one"))
        state.add_question(SubQuestion(id="q2", question="failed one"))
        state.mark_question_completed("q1")
        state.mark_question_completed("q2")
        # The failed question stays in completed_questions (loop bookkeeping),
        # so the header must count statuses rather than that set.
        state.get_question_by_id("q2").status = QuestionStatus.FAILED
        state.findings.append(KnowledgeFinding(
            content="kept", sources=[], confidence=0.4, question_id="q1"
        ))

        body = agent.build_partial_report(state)

        self.assertIn("Questions answered: 1", body)
        self.assertIn("Questions failed: 1", body)

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
                               relevant_findings, main_query, style,
                               evidence_sources=None, other_sections=None):
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
        for i in range(len(themes)):
            state.findings.append(KnowledgeFinding(
                content=f"f{i}", sources=[], confidence=0.9, question_id=f"q{i}"
            ))
        prepared = [
            dict(theme, related_findings=theme.get("related_findings", [i]))
            for i, theme in enumerate(themes)
        ]
        agent = ReportGeneratorAgent(_FakeLLM(delays))
        return await agent._generate_sections_parallel(
            state, {"themes": prepared}, "comprehensive"
        )

    async def test_order_follows_themes_not_completion(self):
        themes = [{"title": "Alpha"}, {"title": "Beta"}, {"title": "Gamma"}]
        delays = {"Alpha": 0.03, "Beta": 0.02, "Gamma": 0.01}  # reverse completion
        sections, failed = await self._run(themes, delays)
        self.assertEqual(failed, 0)
        self.assertEqual([s.title for s in sections], ["Alpha", "Beta", "Gamma"])
        self.assertEqual([s.order for s in sections], [0, 1, 2])
        self.assertEqual([s.content for s in sections], ["BODY:Alpha", "BODY:Beta", "BODY:Gamma"])

    async def test_theme_titles_follow_finding_ids_not_positions(self):
        state = make_state()
        for i in range(2):
            state.findings.append(KnowledgeFinding(
                content=f"f{i}", sources=[], confidence=0.9, question_id=f"q{i}"
            ))
        themes = {"themes": [
            {"title": "Second theme", "related_finding_ids": ["F2"]},
            {"title": "First theme", "related_finding_ids": ["f1"]},
        ]}
        agent = ReportGeneratorAgent(_FakeLLM({"First theme": 0, "Second theme": 0}))

        sections, failed = await agent._generate_sections_parallel(
            state, themes, "comprehensive"
        )

        self.assertEqual(failed, 0)
        self.assertEqual([s.title for s in sections], ["First theme", "Second theme"])

    async def test_an_unknown_finding_id_falls_back_to_the_question(self):
        state = make_state()
        for i in range(1):
            state.findings.append(KnowledgeFinding(
                content="only", sources=[], confidence=0.9, question_id="q1"
            ))
        themes = {"themes": [{"title": "Mislabelled", "related_finding_ids": ["F9"]}]}
        agent = ReportGeneratorAgent(_FakeLLM({"Section 1": 0}))

        sections, _ = await agent._generate_sections_parallel(
            state, themes, "comprehensive"
        )

        self.assertEqual([s.title for s in sections], ["Section 1"])

    async def test_two_findings_for_one_question_get_distinct_headings(self):
        state = make_state()
        state.add_question(SubQuestion(id="q1", question="How does quantization work?"))
        for i in range(2):
            state.findings.append(KnowledgeFinding(
                content=f"f{i}", sources=[], confidence=0.9, question_id="q1"
            ))
        agent = ReportGeneratorAgent(_FakeLLM({
            "How does quantization work?": 0,
            "How does quantization work? (2)": 0,
        }))

        sections, _ = await agent._generate_sections_parallel(
            state, {"themes": []}, "comprehensive"
        )

        self.assertEqual(
            [s.title for s in sections],
            ["How does quantization work?", "How does quantization work? (2)"],
        )

    async def test_a_failed_section_is_counted_but_keeps_its_slot(self):
        themes = [
            {"title": "Alpha", "related_findings": [0]},
            {"title": "Beta", "related_findings": [1]},
        ]

        class _Boom(_FakeLLM):
            def generate_section_async(self, *, section_title, **kw):
                async def run():
                    if section_title == "Alpha":
                        raise RuntimeError("boom")
                    return f"BODY:{section_title}"

                return run()

        state = make_state()
        for i in range(2):
            state.findings.append(KnowledgeFinding(
                content=f"f{i}", sources=[], confidence=0.9, question_id=f"q{i}"
            ))
        agent = ReportGeneratorAgent(_Boom({}))
        sections, failed = await agent._generate_sections_parallel(state, {"themes": themes}, "comprehensive")
        self.assertEqual(failed, 1)
        self.assertEqual([s.title for s in sections], ["Alpha", "Beta"])
        self.assertIn("failed", sections[0].content.lower())


class _PromptLLM(LLMClient):
    """Captures the prompt the section writer would send."""

    def __init__(self):
        self.model = "stub-model"
        self.fallback_model = ""
        self.using_fallback = False
        self._rate_limit_delay = 0.0
        self._last_api_call = 0.0
        self._sync_lock = threading.Lock()
        self.messages = None

    def _chat_completion(self, messages, **kwargs):
        self.messages = messages
        return "body"


class SectionPromptTests(unittest.TestCase):
    def _prompt(self, **kwargs):
        llm = _PromptLLM()
        llm.generate_section(
            "Alpha", "what Alpha covers",
            [{"content": "a claim", "confidence": 0.8, "source_numbers": [2]}],
            "the main query", **kwargs,
        )
        system, user = llm.messages[0]["content"], llm.messages[1]["content"]
        return system + "\n" + user

    def test_other_section_headings_are_named_in_the_prompt(self):
        prompt = self._prompt(other_sections=["Beta", "Gamma"])
        self.assertIn("Beta", prompt)
        self.assertIn("Gamma", prompt)

    def test_the_prompt_requires_a_citation_in_every_subsection(self):
        self.assertIn("every subsection needs at least one sentence", self._prompt())

    def test_the_prompt_gives_the_report_wide_source_number(self):
        self.assertIn("[Source 2]", self._prompt())


class ThemeGroupingTests(unittest.IsolatedAsyncioTestCase):
    """Findings that make the same point share one section.

    Writing a section per finding repeated the same argument whenever two
    findings described one mechanism in different words.
    """

    def _state(self, count):
        state = make_state("How does quantization affect accuracy?")
        for i in range(count):
            state.findings.append(KnowledgeFinding(
                content=f"finding {i}", sources=[], confidence=0.9,
                question_id=f"q{i}",
            ))
        return state

    async def test_one_theme_covering_two_findings_writes_one_section(self):
        llm = _ReportPipelineLLM()
        themes = {"themes": [{
            "title": "Perplexity as a proxy",
            "description": "how the proxy tracks quality",
            "related_finding_ids": ["F1", "F2"],
        }]}

        sections, failed = await ReportGeneratorAgent(llm)._generate_sections_parallel(
            self._state(2), themes, "comprehensive"
        )

        self.assertEqual(failed, 0)
        self.assertEqual([s.title for s in sections], ["Perplexity as a proxy"])
        self.assertEqual([f["content"] for f in llm.seen_findings],
                         ["finding 0", "finding 1"])

    async def test_a_finding_no_theme_names_keeps_its_own_section(self):
        llm = _ReportPipelineLLM()
        themes = {"themes": [{
            "title": "Alpha", "related_finding_ids": ["F1"],
        }]}

        sections, _ = await ReportGeneratorAgent(llm)._generate_sections_parallel(
            self._state(2), themes, "comprehensive"
        )

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "Alpha")

    async def test_a_theme_naming_one_finding_twice_is_still_one_section(self):
        llm = _ReportPipelineLLM()
        themes = {"themes": [{
            "title": "Alpha", "related_finding_ids": ["F1", "F1", "f1"],
        }]}

        sections, _ = await ReportGeneratorAgent(llm)._generate_sections_parallel(
            self._state(1), themes, "comprehensive"
        )

        self.assertEqual(len(sections), 1)
        self.assertEqual(len(llm.seen_findings), 1)

    async def test_a_group_larger_than_three_findings_is_split(self):
        llm = _ReportPipelineLLM()
        themes = {"themes": [{
            "title": "One theme",
            "related_finding_ids": ["F1", "F2", "F3", "F4"],
        }]}

        sections, _ = await ReportGeneratorAgent(llm)._generate_sections_parallel(
            self._state(4), themes, "comprehensive"
        )

        self.assertEqual(len(sections), 2)
        self.assertEqual(len({s.title for s in sections}), 2)


class SectionScopeTests(unittest.IsolatedAsyncioTestCase):
    """Each section is told what its neighbours cover, so it does not repeat them."""

    async def test_each_section_receives_the_other_headings(self):
        state = make_state()
        for i in range(2):
            state.findings.append(KnowledgeFinding(
                content=f"f{i}", sources=[], confidence=0.9, question_id=f"q{i}"
            ))
        llm = _ReportPipelineLLM()
        themes = {"themes": [
            {"title": "Alpha", "related_finding_ids": ["F1"]},
            {"title": "Beta", "related_finding_ids": ["F2"]},
        ]}

        await ReportGeneratorAgent(llm)._generate_sections_parallel(
            state, themes, "comprehensive"
        )

        self.assertEqual(llm.other_by_section["Alpha"], ["Beta"])
        self.assertEqual(llm.other_by_section["Beta"], ["Alpha"])

    async def test_a_single_section_has_no_neighbours_to_avoid(self):
        state = make_state()
        state.findings.append(KnowledgeFinding(
            content="only", sources=[], confidence=0.9, question_id="q1"
        ))
        llm = _ReportPipelineLLM()

        await ReportGeneratorAgent(llm)._generate_sections_parallel(
            state, {"themes": [{"title": "Alpha", "related_finding_ids": ["F1"]}]},
            "comprehensive",
        )

        self.assertEqual(llm.other_by_section["Alpha"], [])


class MetaAnalysisRenderingTests(unittest.TestCase):
    def test_gaps_section_uses_cross_reference(self):
        state = make_state()
        state.cross_reference = {
            "consensus_points": [
                {"point": "Both sources agree on X", "confidence": 0.8},
            ],
            "contradictions": [
                {"topic": "Cost", "viewpoint_a": "Cheap", "viewpoint_b": "Expensive"},
            ],
            "reliability_assessment": "Peer-reviewed sources dominate.",
        }

        content = ReportGeneratorAgent(None)._generate_gaps_section(state)

        self.assertIn("Points of agreement", content)
        self.assertIn("Both sources agree on X", content)
        self.assertIn("**Cost**: Cheap — Expensive", content)
        self.assertIn("Peer-reviewed sources dominate.", content)




class GapRecordingTests(unittest.TestCase):
    def test_gaps_are_stored_once_and_keep_severity(self):
        agent = DeepResearchAgent.__new__(DeepResearchAgent)
        state = make_state()

        added = agent._record_gaps(state, [
            {"description": "Missing pricing data", "severity": "critical"},
            {"description": "missing PRICING data", "severity": "critical"},
            {"description": "", "severity": "critical"},
        ])

        self.assertEqual(added, 1)
        self.assertEqual(len(state.gaps), 1)
        self.assertEqual(state.gaps[0].description, "Missing pricing data")
        self.assertEqual(state.gaps[0].severity, "critical")


class _ReportPipelineLLM:
    """Deterministic stand-in for the three model calls a report makes."""

    model = "stub-model"

    def __init__(self):
        self.seen_findings = None
        self.other_sections = None
        self.other_by_section = {}

    def identify_report_themes(self, findings):
        return {"themes": [
            {"title": "Mechanism", "description": "How it works", "related_findings": [0]},
        ]}

    def generate_section_async(self, *, section_title, theme_description,
                               relevant_findings, main_query, style,
                               evidence_sources=None, other_sections=None):
        self.other_sections = other_sections
        self.other_by_section[section_title] = other_sections
        self.seen_findings = relevant_findings
        self.seen_evidence = evidence_sources

        async def run():
            return (
                "A supported claim [Source 1]. "
                "A corroborated claim [Sources 2, 1]. "
                "An invented citation [Source 99]."
            )

        return run()

    def generate_executive_summary(self, main_query, key_findings,
                                   source_numbers=None):
        self.summary_numbers = sorted(source_numbers or [])
        return "A summary that keeps a section citation [Source 1]."


class ReportCitationPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_citations_bibliography_and_summary_agree(self):
        state = make_state("How do quantization schemes differ?")
        state.search_results = [
            SearchResult(
                url="https://example.com/paper",
                title="Quantization paper",
                content="q" * 600,
                score=0.9,
                category="educational",
                published_date="2026-01-02",
            ),
            SearchResult(
                url="https://example.org/analysis",
                title="Quantization analysis",
                content="a" * 600,
                score=0.8,
                category="news",
            ),
            SearchResult(
                url="https://example.net/collected",
                title="Quantization survey",
                content="c" * 600,
                score=0.7,
                category="general",
                question_id="q1",
                analyzed=True,
            ),
        ]
        state.findings = [KnowledgeFinding(
            content="Quantization trades accuracy for speed.",
            sources=["https://example.com/paper", "https://example.org/analysis"],
            confidence=0.8,
            question_id="q1",
        )]

        llm = _ReportPipelineLLM()
        report = await ReportGeneratorAgent(llm).generate_report(state)

        # The section prompt used the report-wide numbers, not local indices.
        self.assertEqual(llm.seen_findings[0]["source_numbers"], [1, 2])
        # The finding's own pages come first, then the page the analysis read
        # for the same question but never quoted; all three are citable.
        self.assertEqual([s["number"] for s in llm.seen_evidence], [1, 2, 3])
        self.assertIn("A supported claim [Source 1].", report)
        self.assertIn("A corroborated claim [Source 2][Source 1].", report)
        self.assertNotIn("Source 99", report)
        # The summary is told which numbers it may reuse (the ones the sections
        # cited), and may not invent one beyond that list.
        self.assertEqual(llm.summary_numbers, [1, 2])
        self.assertIn("A summary that keeps a section citation [Source 1].", report)
        self.assertIn("## Executive Summary", report)
        # The bibliography lists only what the body cited, renumbered from 1,
        # and summarises the unused pages in one line rather than padding it.
        self.assertIn("Sources: 2 cited of 3 reviewed", report)
        self.assertIn("## Mechanism", report)
        self.assertIn("## Sources", report)
        self.assertIn("1. [Source 1] **Quantization paper**", report)
        self.assertIn("Published: 2026-01-02", report)
        self.assertIn("2. [Source 2] **Quantization analysis**", report)
        self.assertIn("1 additional source was reviewed but not cited", report)
        self.assertNotIn("https://example.net/collected", report)
        self.assertEqual(state.report_sections[0].sources,
                         ["https://example.com/paper", "https://example.org/analysis"])


class EvidenceSourceTests(unittest.TestCase):
    """A section gets one excerpt per page it may cite: the finding's own
    sources first, then pages the analysis read but never quoted."""

    def test_own_sources_first_then_unquoted_read_sources(self):
        state = make_state("q")
        used = SearchResult(url="https://used.example/a", title="Used",
                            content="u" * 200, score=0.4, question_id="q1")
        extra_low = SearchResult(url="https://low.example/b", title="Low",
                                 content="l" * 200, score=0.2,
                                 question_id="q1", analyzed=True)
        extra_high = SearchResult(url="https://high.example/c", title="High",
                                  content="h" * 200, score=0.8, quality=0.9,
                                  question_id="q1", analyzed=True)
        other_question = SearchResult(url="https://other.example/d", title="Other",
                                      content="o" * 200, score=0.9,
                                      question_id="q2", analyzed=True)
        unread = SearchResult(url="https://unread.example/e", title="Unread",
                              content="n" * 200, score=0.9, question_id="q1")
        state.search_results = [used, extra_low, extra_high, other_question, unread]

        numbers, _ = build_source_registry(state.search_results)
        finding = KnowledgeFinding(
            content="claim", sources=[used.url], confidence=0.7, question_id="q1"
        )
        picked = ReportGeneratorAgent._evidence_sources(
            state, [finding], numbers
        )

        self.assertEqual(
            [item["number"] for item in picked],
            [numbers[used.url], numbers[extra_high.url], numbers[extra_low.url]],
        )
        # The finding's own page leads, so its excerpt is available to cite.
        self.assertEqual(picked[0]["title"], "Used")
        included = [item["number"] for item in picked]
        self.assertNotIn(numbers[other_question.url], included)
        self.assertNotIn(numbers[unread.url], included)

    def test_excerpts_are_capped(self):
        state = make_state("q")
        state.search_results = [
            SearchResult(url=f"https://ex.example/{i}", title=f"Source {i}",
                         content=("x" * 200) + str(i), score=0.1 * i,
                         question_id="q1", analyzed=True)
            for i in range(12)
        ]
        numbers, _ = build_source_registry(state.search_results)
        finding = KnowledgeFinding(
            content="claim", sources=[state.search_results[0].url],
            confidence=0.7, question_id="q1"
        )

        picked = ReportGeneratorAgent._evidence_sources(state, [finding], numbers)

        self.assertEqual(len(picked), 6)
        self.assertEqual(picked[0]["number"], numbers[state.search_results[0].url])


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
                               relevant_findings, main_query, style="comprehensive",
                               evidence_sources=None, other_sections=None):
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