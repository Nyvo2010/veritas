"""A deterministic end-to-end run that asserts what a report must never do.

The other report tests check one function at a time. This one drives the real
orchestrator, synthesis, and report generator with a stubbed model and search
back end, then checks the finished document as a reader would: every citation
resolves, the bibliography is contiguous, no two sections share a heading, no
placeholder text survives, and the summary does not cite what the body did not.

The stub is the point: it is cheap, offline, and stable enough for CI, so the
structural contract is checked on every commit instead of only on a live run.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agents.query_analyzer import QueryAnalyzerAgent  # noqa: E402
from agents.report_generator import ReportGeneratorAgent  # noqa: E402
from agents.search_agent import SearchAgent  # noqa: E402
from agents.synthesis_agent import SynthesisAgent  # noqa: E402
from main import DeepResearchAgent  # noqa: E402
from models.research_state import QuestionStatus  # noqa: E402


class _UI:
    """The UI surface research() uses, without a terminal."""

    def __init__(self):
        self.renderer = type("Renderer", (), {"completed_steps": 0})()
        self.timings = None

    def initialize(self, total, title):
        self.total = total

    def start(self):
        pass

    def stop(self):
        pass

    def complete(self):
        pass

    def set_phase(self, phase):
        self.phase = phase

    def set_status(self, status):
        self.status = status

    def set_progress(self, completed, label):
        self.renderer.completed_steps = completed

    def set_total(self, total):
        self.total = total

    def advance(self, steps, label):
        self.renderer.completed_steps += steps

    def add_timing(self, kind, seconds):
        pass

    def log(self, level, text):
        pass

    def add_message(self, text, style=None):
        pass


class _StubLLM:
    """Every model call the pipeline makes, answered deterministically."""

    model = "stub-model"
    fallback_model = ""

    def decompose_query(self, main_query):
        return {"sub_questions": [
            {"question": f"What does {main_query} mean for latency?"},
            {"question": f"What does {main_query} mean for latency"},
            {"question": f"Which measurement supports {main_query}?"},
        ]}

    def analyze_search_results(self, question, search_results, previous_findings=None):
        return {
            "answer": f"{question}: answer",
            "confidence": 0.8,
            "source_indices": [1, 2],
            "findings": [
                {"claim": f"{question} — mechanism A", "source_indices": [1, 2]},
                {"claim": f"{question} — trade-off B", "source_indices": [2, 3]},
            ],
            "contradictions": [],
            "gaps": [],
            "needs_more_research": False,
        }

    def chat_json(self, messages, **kwargs):
        system = str((messages or [{}])[0].get("content", ""))
        if "consistency" in system:
            return {
                "consensus_points": [{"point": "The sources agree on latency.",
                                      "confidence": 0.7}],
                "contradictions": [],
                "overall_confidence": 0.7,
                "reliability_assessment": "Stub sources are consistent.",
            }
        return {}

    def identify_report_themes(self, findings):
        return {"themes": [
            {
                "title": f"Theme {finding.get('id')}",
                "description": f"About {finding.get('id')}",
                "related_finding_ids": [finding.get("id")],
            }
            for finding in findings
        ]}

    def generate_section_async(self, *, section_title, theme_description,
                               relevant_findings, main_query, style,
                               evidence_sources=None, other_sections=None):
        numbers = [number for finding in relevant_findings
                   for number in finding.get("source_numbers", [])]

        async def run():
            first = numbers[0] if numbers else 1
            second = numbers[1] if len(numbers) > 1 else first
            return (
                f"{section_title} explains the mechanism [Source {first}]. "
                f"The trade-off is measurable [Source {second}]. "
                f"An invented citation is dropped [Source 99]."
            )

        return run()

    def generate_executive_summary(self, main_query, key_findings, source_numbers=None):
        numbers = sorted(int(number) for number in (source_numbers or []))
        marker = f" [Source {numbers[0]}]" if numbers else ""
        return f"The research answers the question{marker}. An invented marker [Source 99]."


class _StubTavily:
    """Stable, distinct pages: no network, same results on every run."""

    def search(self, **kwargs):
        question = str(kwargs.get("query", "query"))
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40] or "page"
        results = []
        for index in range(3):
            words = " ".join(
                f"word{(index * 41 + step * 7) % 997}" for step in range(220)
            )
            results.append({
                "url": f"https://{slug}-{index}.example/report",
                "title": f"{slug} source {index}",
                "content": (f"{question} detail {index}. " * 6) + words,
                "raw_content": (f"{question} detail {index}. " * 6) + words,
                "score": 0.9 - index * 0.1,
            })
        return {"results": results}


def _agent():
    llm = _StubLLM()
    agent = DeepResearchAgent.__new__(DeepResearchAgent)
    agent.llm = llm
    agent.query_analyzer = QueryAnalyzerAgent(llm)
    agent.search_agent = SearchAgent(_StubTavily(), llm)
    agent.synthesis_agent = SynthesisAgent(llm)
    agent.report_generator = ReportGeneratorAgent(llm)
    agent.category_toggles = {}
    agent.last_state = None
    agent.save_error = ""
    agent._settings = {"max_search_results": 3}
    return agent


def _headings(report):
    return re.findall(r"^## (.+)$", report, re.M)


def _body(report):
    return report.split("\n## Sources\n", 1)[0]


def _bibliography(report):
    tail = report.split("\n## Sources\n", 1)[-1]
    return [int(number) for number in re.findall(r"^\d+\. \[Source (\d+)\]", tail, re.M)]


def _markers(text):
    return [int(number) for number in re.findall(r"\[Source (\d+)\]", text)]


class ReportInvariantTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = await _agent().research(
            "quantization", max_iterations=2, verbose=False, ui=_UI()
        )
        self.report = self.state.final_report

    def test_the_run_reached_the_report_stage(self):
        self.assertTrue(self.report, "the stub run produced no report")
        self.assertEqual(self.state.iteration_count, 2)

    def test_every_citation_resolves_and_the_bibliography_is_contiguous(self):
        numbers = _bibliography(self.report)
        self.assertTrue(numbers, "a report with citations needs a bibliography")
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

        cited = set(_markers(_body(self.report)))
        self.assertTrue(cited, "the body cites nothing")
        self.assertLessEqual(cited, set(numbers), "a citation has no bibliography entry")

    def test_no_invented_source_number_survives(self):
        self.assertNotIn("[Source 99]", self.report)

    def test_the_summary_cites_only_what_the_sections_cited(self):
        summary = self.report.split("## Executive Summary", 1)[1].split("\n---", 1)[0]
        sections = _body(self.report).split("## Executive Summary", 1)[1]
        section_markers = set(_markers(sections.split("## Uncertainties", 1)[0]))
        self.assertLessEqual(set(_markers(summary)), section_markers)

    def test_headings_are_unique_and_placeholders_are_gone(self):
        headings = _headings(self.report)
        self.assertEqual(len(headings), len(set(headings)), headings)
        self.assertIn("Executive Summary", headings)
        self.assertIn("Uncertainties and Knowledge Gaps", headings)
        self.assertIn("Sources", headings)
        self.assertNotIn("Section generation failed", self.report)
        self.assertNotIn("timed out", self.report.lower())

    def test_one_section_per_finding_and_each_section_is_cited(self):
        body_headings = [
            heading for heading in _headings(self.report)
            if heading not in {"Executive Summary", "Uncertainties and Knowledge Gaps",
                               "Sources"}
        ]
        self.assertEqual(len(body_headings), len(self.state.findings))
        for section in self.state.report_sections:
            self.assertTrue(
                _markers(section.content),
                f"section {section.title!r} was written without a citation",
            )

    def test_a_reworded_question_is_not_researched_twice(self):
        questions = [q.question for q in self.state.questions]
        self.assertEqual(len(questions), 2, questions)
        self.assertEqual(
            len({question.rstrip("?").lower() for question in questions}), 2
        )

    def test_a_successful_run_marks_every_question_answered(self):
        self.assertTrue(all(
            question.status is QuestionStatus.COMPLETED
            for question in self.state.questions
        ))


if __name__ == "__main__":
    unittest.main()
