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
from agents.report_generator import (  # noqa: E402
    ReportGeneratorAgent,
    _cap_source_citations,
    _cited_claims,
    _clean_title,
    _select_excerpt,
)
from utils.llm_client import LLMClient  # noqa: E402
from utils.sources import SourceEntry, build_source_registry  # noqa: E402

logging.getLogger("veritas").setLevel(logging.INFO)  # keep test output quiet
from models.research_state import (  # noqa: E402
    KnowledgeFinding,
    QuestionStatus,
    ReportSection,
    ResearchGap,
    ResearchState,
    SearchResult,
    SubQuestion,
)


def make_state(query="Why is the sky blue?"):
    return ResearchState(main_query=query, category_toggles={"general": True})


def _gap(description, severity="moderate"):
    return ResearchGap(
        description=description, question_id="overall", severity=severity
    )


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
    def test_cross_reference_has_its_own_section_content(self):
        state = make_state()
        state.cross_reference = {
            "consensus_points": [
                {"point": "Both sources agree on X", "confidence": 0.8},
            ],
            "contradictions": [
                {"topic": "Cost", "viewpoint_a": "Cheap", "viewpoint_b": "Expensive"},
            ],
            "overall_confidence": 0.72,
            "reliability_assessment": "Peer-reviewed sources dominate.",
        }

        content = ReportGeneratorAgent(None)._generate_cross_reference_section(state)

        self.assertIn("Points of agreement", content)
        self.assertIn("Both sources agree on X", content)
        self.assertIn("**Cost**: Cheap — Expensive", content)
        # The paid-for self-rating is disclosed as the model's view.
        self.assertIn("rated its own confidence", content)
        self.assertIn("72%", content)
        self.assertIn("Peer-reviewed sources dominate.", content)

    def test_overused_citations_are_reported_by_source_title(self):
        entry = SourceEntry(
            number=4, url="https://vendor.example/post", title="A vendor page",
            category="general", quality=0.8, score=0.9, source_type="web",
        )
        section = ReportSection(
            title="Cost",
            content="A [Source 4]. B [Source 4]. C [Source 4]. D [Source 4].",
            sources=[], order=0,
        )

        overused = ReportGeneratorAgent._overused_citations([section], [entry])

        self.assertEqual(overused, [("Cost", "A vendor page", 4)])

    def test_cross_reference_is_not_buried_in_the_gaps_section(self):
        state = make_state()
        state.cross_reference = {
            "consensus_points": [{"point": "Both sources agree on X"}],
        }

        gaps = ReportGeneratorAgent(None)._generate_gaps_section(state)

        self.assertNotIn("Points of agreement", gaps)

    def test_minor_gaps_are_rendered_instead_of_dropped(self):
        state = make_state()
        state.gaps = [
            _gap("Exact per-model numbers vary by evaluation harness", "minor"),
        ]

        content = ReportGeneratorAgent(None)._generate_gaps_section(state)

        self.assertIn("### Minor gaps", content)
        self.assertIn("Exact per-model numbers vary by evaluation harness", content)

    def test_gaps_that_restate_each_other_are_printed_once(self):
        state = make_state()
        state.gaps = [
            _gap("Observed sodium cell prices contradict modeled cost corridors", "critical"),
            _gap("Modeled sodium cost corridors conflict with observed cell prices", "moderate"),
        ]

        content = ReportGeneratorAgent(None)._generate_gaps_section(state)

        self.assertEqual(content.count("Observed sodium cell prices"), 1)
        self.assertNotIn("Modeled sodium cost corridors", content)

    def test_planned_but_unresearched_questions_are_listed(self):
        state = make_state()
        state.add_question(SubQuestion(id="q1", question="answered question"))
        state.add_question(SubQuestion(id="q2", question="manufacturing scale 2026"))
        state.mark_question_completed("q1")

        content = ReportGeneratorAgent(None)._generate_gaps_section(state)

        self.assertIn("Not answered", content)
        self.assertIn("manufacturing scale 2026", content)




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
    """Deterministic stand-in for the model calls a report makes.

    ``unsupported`` feeds the citation support checker; ``support_error``
    makes it fail; ``section_content`` overrides the section prose.
    """

    model = "stub-model"

    def __init__(self, unsupported=None, support_error=False, section_content=None):
        self.seen_findings = None
        self.other_sections = None
        self.other_by_section = {}
        self.unsupported = unsupported or []
        self.support_error = support_error
        self.section_content = section_content
        self.support_claims = None
        self.support_evidence = None
        self.support_thread = None

    def check_citation_support(self, claims, evidence):
        self.support_claims = claims
        self.support_evidence = evidence
        self.support_thread = threading.current_thread().name
        if self.support_error:
            raise RuntimeError("judge unavailable")
        return list(self.unsupported)

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
            if self.section_content is not None:
                return self.section_content
            return (
                "A supported claim [Source 1]. "
                "A corroborated claim [Sources 2, 1]. "
                "An invented citation [Source 99]."
            )

        return run()

    def generate_executive_summary(self, main_query, key_findings,
                                   source_numbers=None, target_words=600):
        self.summary_numbers = sorted(source_numbers or [])
        self.summary_target_words = target_words
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
        self.assertIn("[Source 1] **Quantization paper**", report)
        self.assertIn("Published: 2026-01-02", report)
        self.assertIn("[Source 2] **Quantization analysis**", report)
        # The numbered entry is the citation target, so it is not labelled
        # twice; the page kind replaces the useless "General" category, and
        # the internal ranking scores stay out of the document.
        self.assertNotIn("1. [Source 1]", report)
        self.assertNotIn("quality heuristic", report)
        self.assertNotIn("search relevance", report)
        self.assertIn("1 additional source was reviewed but not cited", report)
        self.assertNotIn("https://example.net/collected", report)
        self.assertEqual(state.report_sections[0].sources,
                         ["https://example.com/paper", "https://example.org/analysis"])
        # A summary longer than the shortest section is a second body.
        self.assertLessEqual(llm.summary_target_words, 450)
        self.assertGreaterEqual(llm.summary_target_words, 250)

    async def test_markers_the_evidence_does_not_support_are_dropped(self):
        state = make_state("a question")
        state.search_results = [SearchResult(
            url="https://example.com/paper", title="Paper",
            content="paper " * 100, score=0.9, category="educational",
        )]
        state.findings = [KnowledgeFinding(
            content="A point.", sources=["https://example.com/paper"],
            confidence=0.8, question_id="q1",
        )]
        llm = _ReportPipelineLLM(
            unsupported=["M2"],
            section_content=(
                "A supported claim [Source 1]. "
                "A stretched claim [Source 1]."
            ),
        )

        report = await ReportGeneratorAgent(llm).generate_report(state)

        self.assertIn("A supported claim [Source 1].", report)
        self.assertIn("A stretched claim.", report)
        self.assertNotIn("A stretched claim [Source 1]", report)
        # The checker saw the marker id and the claim, not the whole section.
        self.assertEqual([claim["id"] for claim in llm.support_claims], ["M1", "M2"])
        self.assertTrue(llm.support_evidence)

    def _one_section_state(self):
        state = make_state("a question")
        state.search_results = [SearchResult(
            url="https://example.com/paper", title="Paper",
            content="paper " * 100, score=0.9, category="educational",
        )]
        state.findings = [KnowledgeFinding(
            content="A point.", sources=["https://example.com/paper"],
            confidence=0.8, question_id="q1",
        )]
        return state

    async def test_the_header_states_what_the_citation_audit_checked(self):
        llm = _ReportPipelineLLM(
            unsupported=["M2"],
            section_content=(
                "A supported claim [Source 1]. "
                "A stretched claim [Source 1]."
            ),
        )

        report = await ReportGeneratorAgent(llm).generate_report(
            self._one_section_state()
        )

        self.assertIn("Source check: 1 of 2 section citations verified", report)

    async def test_an_unavailable_judge_does_not_claim_an_audit(self):
        llm = _ReportPipelineLLM(
            support_error=True,
            section_content="A supported claim [Source 1].",
        )

        report = await ReportGeneratorAgent(llm).generate_report(
            self._one_section_state()
        )

        # Zero checked means no audit ran; claiming "1 of 1 supported" would
        # turn a judge outage into a clean bill of health.
        self.assertNotIn("Source check", report)

    async def test_a_partial_audit_is_not_summed_up_as_a_whole_report(self):
        """One section losing its judge must not leave a header that implies
        every section was checked."""

        class _FlakyJudge(_ReportPipelineLLM):
            calls = 0

            def check_citation_support(self, claims, evidence):
                type(self).calls += 1
                if type(self).calls == 2:
                    raise RuntimeError("judge went away")
                return super().check_citation_support(claims, evidence)

        state = make_state("a question")
        state.search_results = [SearchResult(
            url="https://example.com/paper", title="Paper",
            content="paper " * 100, score=0.9, category="educational",
        )]
        for i in range(2):
            state.findings.append(KnowledgeFinding(
                content=f"point {i}",
                sources=["https://example.com/paper"],
                confidence=0.8, question_id=f"q{i}",
            ))

        report = await ReportGeneratorAgent(_FlakyJudge()).generate_report(state)

        self.assertIn("## ", report)
        self.assertNotIn("Source check", report)

    async def test_the_support_check_runs_off_the_event_loop_thread(self):
        """The judge is a blocking model call; on the loop it stalls every
        other section that is still waiting for its response."""
        llm = _ReportPipelineLLM(section_content="A claim [Source 1].")

        await ReportGeneratorAgent(llm).generate_report(self._one_section_state())

        self.assertIsNotNone(llm.support_thread)
        self.assertNotEqual(llm.support_thread, threading.current_thread().name)

    async def test_a_failed_support_check_keeps_every_marker(self):
        state = make_state("a question")
        state.search_results = [SearchResult(
            url="https://example.com/paper", title="Paper",
            content="paper " * 100, score=0.9, category="educational",
        )]
        state.findings = [KnowledgeFinding(
            content="A point.", sources=["https://example.com/paper"],
            confidence=0.8, question_id="q1",
        )]
        llm = _ReportPipelineLLM(
            support_error=True,
            section_content="A supported claim [Source 1].",
        )

        report = await ReportGeneratorAgent(llm).generate_report(state)

        self.assertIn("A supported claim [Source 1].", report)

    async def test_header_reports_question_coverage(self):
        state = make_state()
        state.add_question(SubQuestion(id="q1", question="answered"))
        state.add_question(SubQuestion(id="q2", question="never reached"))
        state.mark_question_completed("q1")
        state.findings = [KnowledgeFinding(
            content="A point.", sources=[], confidence=0.8, question_id="q1"
        )]

        report = await ReportGeneratorAgent(_ReportPipelineLLM()).generate_report(state)

        self.assertIn("Questions: 1 answered of 2", report)
        self.assertIn("## Uncertainties and Knowledge Gaps", report)
        self.assertIn("never reached", report)

    async def test_a_run_with_no_findings_does_not_call_the_model(self):
        state = make_state("a question nothing answered")
        state.add_question(SubQuestion(id="q1", question="unanswerable thing"))
        state.get_question_by_id("q1").status = QuestionStatus.FAILED
        state.search_results = [SearchResult(
            url="https://example.com/movie",
            title="A 1995 film",
            content="x" * 600,
            score=0.9,
            category="general",
        )]

        report = await ReportGeneratorAgent(None).generate_report(state)

        self.assertIn("No usable findings were collected", report)
        self.assertIn("unanswerable thing", report)
        self.assertIn("No sources were cited", report)
        self.assertIn("Questions: 0 answered of 1", report)


class SupportJudgeReplyTests(unittest.TestCase):
    """A reply that is not a verdict must not be read as "all supported"."""

    class _ReplyLLM(LLMClient):
        def __init__(self, reply):
            self.reply = reply

        def chat_json(self, *args, **kwargs):
            return self.reply

    CLAIMS = [{"id": "M1", "source": 1, "sentence": "A claim."}]
    EVIDENCE = [{"number": 1, "title": "Paper", "content": "The excerpt."}]

    def test_a_reply_without_the_key_is_a_failure(self):
        llm = self._ReplyLLM({"notes": "looks fine to me"})
        with self.assertRaises(ValueError):
            llm.check_citation_support(self.CLAIMS, self.EVIDENCE)

    def test_an_empty_list_is_a_real_verdict(self):
        llm = self._ReplyLLM({"unsupported": []})
        self.assertEqual(
            llm.check_citation_support(self.CLAIMS, self.EVIDENCE), []
        )


class CitationCapTests(unittest.TestCase):
    def test_repeat_markers_past_the_cap_lose_only_the_marker(self):
        content = (
            "One [Source 2]. Two [Source 2]. Three [Source 2]. "
            "Four [Source 2]. Five [Source 3]."
        )

        capped, removed = _cap_source_citations(content)

        self.assertEqual(removed, 1)
        self.assertEqual(capped.count("[Source 2]"), 3)
        self.assertIn("Four.", capped)
        self.assertIn("Five [Source 3].", capped)

    def test_a_dropped_trailing_marker_leaves_no_stray_space(self):
        capped, removed = _cap_source_citations(
            "A [Source 1]. B [Source 1]. C [Source 1]. D [Source 1].", cap=1
        )

        self.assertEqual(removed, 3)
        self.assertEqual(capped, "A [Source 1]. B. C. D.")


class ClaimExtractionTests(unittest.TestCase):
    """The checker judges one sentence; a fragment gets rejected wrongly.

    The first version took everything after the last full stop, which lands
    inside "0.01%", "e.g." and "et al.", so the judge was shown things like
    "[Source 9] on A100." and answered unsupported.
    """

    def claim(self, text):
        claims = _cited_claims(text)
        self.assertEqual(len(claims), 1)
        return claims[0]["sentence"]

    def test_a_decimal_does_not_start_the_claim(self):
        sentence = (
            "Activation values sit between -1.0 and +1.0, but 0.01% spike as high "
            "as ±30 and crush the resolution of the rest [Source 6]."
        )
        self.assertEqual(self.claim(sentence), sentence)

    def test_an_abbreviation_does_not_start_the_claim(self):
        sentence = "INT8 doubles throughput on Ampere (e.g. 2x over FP16) [Source 5]."
        self.assertEqual(self.claim(sentence), sentence)

    def test_a_citation_never_ends_up_as_the_start_of_the_claim(self):
        sentence = "The 8-bit path is stable, as shown by Smith et al. [Source 9] on A100."
        self.assertEqual(self.claim(sentence), sentence)

    def test_a_marker_leading_a_sentence_keeps_the_claim_it_closes(self):
        sentence = (
            "[Source 2] This sentence opens with its marker and makes a long claim "
            "about throughput."
        )
        self.assertEqual(self.claim(sentence), sentence)

    def test_two_markers_in_one_sentence_share_it(self):
        claims = _cited_claims(
            "Mixed precision keeps outliers in FP16 [Source 6]. This preserves "
            "fidelity because the range stays wide [Source 4]."
        )
        self.assertEqual([c["source"] for c in claims], [6, 4])
        self.assertTrue(claims[1]["sentence"].startswith("This preserves fidelity"))

    def test_a_bullet_does_not_swallow_the_next_one(self):
        claims = _cited_claims(
            "- Bandwidth halves with INT8 [Source 3]\n- Latency drops too [Source 2]"
        )
        self.assertEqual(claims[0]["sentence"], "- Bandwidth halves with INT8 [Source 3]")
        self.assertEqual(claims[1]["sentence"], "- Latency drops too [Source 2]")

    def test_a_long_claim_is_cut_between_words(self):
        sentence = ("word " * 200) + "[Source 1]."
        claim = self.claim(sentence)

        self.assertLessEqual(len(claim), 402)
        self.assertTrue(claim.endswith("…"))
        self.assertNotIn("wor …", claim)


class SectionTitleTests(unittest.TestCase):
    def test_a_raw_question_becomes_a_heading(self):
        self.assertEqual(
            _clean_title("needle in a haystack test results for long context LLMs?"),
            "Needle in a haystack test results for long context LLMs?",
        )
        self.assertEqual(_clean_title("how quantization works."),
                         "How quantization works")

    def test_acronyms_and_proper_nouns_keep_their_casing(self):
        self.assertEqual(_clean_title("eBPF probes inside RAG pipelines"),
                         "eBPF probes inside RAG pipelines")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(_clean_title("  spaced\n\nout  "), "Spaced out")


class ExcerptSelectionTests(unittest.TestCase):
    def test_the_relevant_paragraph_beats_the_page_head(self):
        raw = (
            "Home About Contact\n\n"
            + "Botanical gardens in the city open at nine. " * 3
            + "\n\nAttention calibration and positional bias reduce the "
            "lost-in-the-middle effect in long contexts."
        )

        excerpt = _select_excerpt(raw, "attention calibration positional bias", chars=200)

        self.assertTrue(excerpt.startswith("Attention calibration"), excerpt)
        self.assertLessEqual(len(excerpt), 200)

    def test_a_page_with_no_overlap_keeps_its_head(self):
        raw = "Alpha beta gamma delta. " * 20

        excerpt = _select_excerpt(raw, "quantization", chars=200)

        self.assertTrue(excerpt.startswith("Alpha beta gamma delta."))

    def test_a_short_page_is_returned_whole(self):
        self.assertEqual(_select_excerpt("Short page.", "quantization", 200), "Short page.")


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

    def test_extras_used_by_another_section_are_taken_last(self):
        state = make_state("q")
        own = SearchResult(url="https://own.example/a", title="Own",
                           content="o" * 200, score=0.5, question_id="q1")
        high = SearchResult(url="https://high.example/b", title="High",
                            content="h" * 200, score=0.9, quality=0.9,
                            question_id="q1", analyzed=True)
        low = SearchResult(url="https://low.example/c", title="Low",
                           content="l" * 200, score=0.3, quality=0.4,
                           question_id="q1", analyzed=True)
        state.search_results = [own, high, low]

        numbers, _ = build_source_registry(state.search_results)
        finding = KnowledgeFinding(
            content="claim", sources=[own.url], confidence=0.7, question_id="q1"
        )

        picked = ReportGeneratorAgent._evidence_sources(
            state, [finding], numbers, avoid={numbers[high.url]}
        )

        # The higher-quality page is still available, but it is what the
        # previous section read, so this one takes the unclaimed page first.
        self.assertEqual(
            [item["number"] for item in picked],
            [numbers[own.url], numbers[low.url], numbers[high.url]],
        )

    def test_reused_pages_stop_at_the_floor(self):
        state = make_state("q")
        own = SearchResult(url="https://own.example/a", title="Own",
                           content="o" * 200, score=0.5, question_id="q1")
        extras = [
            SearchResult(url=f"https://shared.example/{i}", title=f"Shared {i}",
                         content=("s" * 100) + str(i), score=0.8,
                         question_id="q1", analyzed=True)
            for i in range(5)
        ]
        state.search_results = [own] + extras

        numbers, _ = build_source_registry(state.search_results)
        finding = KnowledgeFinding(
            content="claim", sources=[own.url], confidence=0.7, question_id="q1"
        )

        picked = ReportGeneratorAgent._evidence_sources(
            state, [finding], numbers, avoid={numbers[e.url] for e in extras}
        )

        # Four is enough for a section; topping it up to the six-page limit
        # with pages another section already read is how prose repeats.
        self.assertEqual(len(picked), 4)

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