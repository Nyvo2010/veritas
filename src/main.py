"""
Main orchestrator for the Deep Research Agent
Coordinates all agents in an iterative research loop
"""

import asyncio
from typing import Dict, Optional
from datetime import datetime
import logging
import sys
import os
import tempfile
import time
import traceback

# Ensure src root is on sys.path for imports when running scripts from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from models.research_state import ResearchGap, ResearchState, QuestionStatus
from agents.query_analyzer import QueryAnalyzerAgent
from agents.search_agent import SearchAgent
from agents.synthesis_agent import SynthesisAgent
from agents.report_generator import ReportGeneratorAgent
from utils.llm_client import LLMClient
from utils.tavily_client import TavilySearchClient
from utils.sources import build_source_registry, number_for
from utils.text import as_bool, clip
from utils.ui_manager import UIManager
from ui.theme import format_duration
from settings import load_settings, resolve_api_keys, resolve_llm, reports_dir as resolve_reports_dir

from config import (
    LLM_API_KEY,
    TAVILY_API_KEY,
    MAX_RESEARCH_ITERATIONS,
    MAX_SEARCH_RESULTS,
)

log = logging.getLogger("veritas.research")


def _seconds(value: float) -> str:
    """Short human duration for a log line: "2.4s", "1m 12s"."""
    return f"{value:.0f}s" if value < 60 else format_duration(value)


def _ui_log(ui, level: str, text: str) -> None:
    """Push one leveled line to the UI (and mirror it into the file log)."""
    if ui is None or not text:
        return
    log.debug("ui[%s] %s", level, text)
    ui.log(level, text)


def _record_time(ui, kind: str, started: float) -> float:
    """Book one call against the UI's timing ledger; returns its duration."""
    duration = max(0.0, time.perf_counter() - started)
    add_timing = getattr(ui, "add_timing", None)
    if callable(add_timing):
        add_timing(kind, duration)
    return duration


async def _timed(awaitable, *, kind: str, label: str, ui=None):
    """Await *awaitable*, book its duration, and log the detail.

    Returns ``(result, seconds)``. Failures are booked too: time spent on a
    call that blew up is exactly the time you want to see afterwards.
    """
    started = time.perf_counter()
    try:
        result = await awaitable
    except Exception:
        duration = _record_time(ui, kind, started)
        log.exception("%s failed after %.2fs", label, duration)
        raise
    duration = _record_time(ui, kind, started)
    log.debug("%s finished in %.2fs", label, duration)
    return result, duration


class DeepResearchAgent:
    """
    Main orchestrator for deep research using iterative multi-agent approach

    Workflow:
    1. Decompose query into sub-questions
    2. Iteratively search and analyze each question
    3. Generate follow-up questions based on gaps
    4. Cross-reference findings
    5. Generate comprehensive report in parallel sections

    All blocking LLM / network calls run in worker threads so the live
    TUI spinner keeps animating. All output during a run flows through
    the UIManager event log — never print() directly while live.
    """

    def __init__(
        self,
        llm_api_key: str = LLM_API_KEY,
        tavily_api_key: str = TAVILY_API_KEY,
        category_toggles: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize the research agent

        Args:
            llm_api_key: API key for the OpenAI-compatible LLM endpoint (falls back to settings file)
            tavily_api_key: API key for Tavily Search (falls back to settings file)
            category_toggles: Dict controlling which website categories to include
        """
        self._settings = load_settings()
        file_llm, file_tavily, file_base = resolve_api_keys(self._settings)
        llm = resolve_llm(self._settings)

        # Initialize clients
        self.llm = LLMClient(
            api_key=(llm_api_key or file_llm),
            api_base_url=llm["base_url"],
            model=llm["model"],
            fallback_model=llm.get("fallback_model", ""),
        )
        self.tavily = TavilySearchClient(api_key=(tavily_api_key or file_tavily))

        # Initialize agents
        self.query_analyzer = QueryAnalyzerAgent(self.llm)
        self.search_agent = SearchAgent(self.tavily, self.llm)
        self.synthesis_agent = SynthesisAgent(self.llm)
        self.report_generator = ReportGeneratorAgent(self.llm)

        # Category toggles
        if category_toggles:
            self.category_toggles = dict(category_toggles)
        else:
            self.category_toggles = dict(self._settings.get("category_toggles", {}))

        # Rich console for UI
        self.console = Console()
        # The state of the most recent run, kept so a caller can still save a
        # partial report after research() raises.
        self.last_state: Optional[ResearchState] = None
        self.save_error: str = ""

    async def research(
        self,
        query: str,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
        verbose: bool = True,
        ui=None,
        max_results: Optional[int] = None,
    ) -> ResearchState:
        """
        Execute deep research on a query

        Args:
            query: The research query
            max_iterations: Maximum research iterations
            verbose: Whether to print progress (set False when an external
                frontend such as OpenTUI renders progress itself)
            ui: Optional UI object (UIManager-compatible). A default live
                UIManager is created when omitted.
            max_results: Results to request per search. Defaults to the
                stored setting, clamped to the provider's 1-20 range.

        Returns:
            ResearchState with all findings and final report
        """
        # Initialize state
        state = ResearchState(
            main_query=query,
            category_toggles=self.category_toggles
        )

        # Progress budget: decomposition + bounded iterations + cross-ref + report headroom.
        # Report sections vary (6-10 themes), so headroom is adjusted once known.
        estimated_steps = 1 + max_iterations + 1 + 12
        if ui is None:
            ui = UIManager(console=self.console)
        short_query = query if len(query) <= 60 else query[:57] + "..."
        ui.initialize(estimated_steps, f"Research: {short_query}")
        ui.set_status("Preparing research environment...")
        ui.start()
        self.last_state = state
        run_started = time.perf_counter()
        log.info(
            "research start: %r (max_iterations=%s, model=%s, sources=%s)",
            query, max_iterations, self.llm.model,
            ",".join(name for name, on in self.category_toggles.items() if on) or "none",
        )

        completed = 0
        # The settings screen promises "results per search"; honour it. The
        # analysis model still gets at most MAX_ANALYSIS_RESULTS, which is a
        # separate context-window budget.
        per_search = (
            max_results if max_results is not None
            else getattr(self, "_settings", {}).get("max_search_results", MAX_SEARCH_RESULTS)
        )
        try:
            per_search = max(1, min(20, int(per_search)))
        except (TypeError, ValueError):
            per_search = MAX_SEARCH_RESULTS

        try:
            # Phase 1: Query Decomposition (blocking LLM call off the event loop)
            ui.set_phase("Phase 1: Query Analysis")
            ui.set_status("Decomposing query into sub-questions...")

            state.reasoning_steps.append("Phase 1: Query Decomposition - Started")

            sub_questions, elapsed = await _timed(
                asyncio.to_thread(self.query_analyzer.decompose_main_query, query, state, ui),
                kind="llm", label="decompose_main_query", ui=ui,
            )
            _ui_log(ui, "ok", f"Query decomposed into {len(sub_questions)} questions ({_seconds(elapsed)})")

            state.reasoning_steps.append(f"Phase 1: Generated {len(sub_questions)} initial research questions")
            completed += 1
            ui.set_progress(completed, "Query decomposed")

            if not state.question_queue:
                raise Exception("Query decomposition produced no research questions; try a more specific query")

            # Phase 2: Iterative Research Loop
            ui.set_phase("Phase 2: Research and Analysis")

            state.reasoning_steps.append(f"Phase 2: Iterative Research - Started with {len(state.question_queue)} questions")

            iteration = 0
            max_iter = min(max_iterations, len(state.question_queue))
            consecutive_failures = 0

            while iteration < max_iterations and state.question_queue:
                iteration += 1
                state.iteration_count = iteration
                _ui_log(ui, "phase", f"Iteration {iteration} of {max_iter}")

                # Get next question
                question = state.get_next_question()
                if not question:
                    break

                state.reasoning_steps.append(f"\nIteration {iteration}: Processing question {iteration}/{max_iter}")
                state.reasoning_steps.append(f"  Question: {question.question}")

                # Determine search depth adaptively. A question is normally
                # searched once, so there is no earlier confidence to escalate
                # on: None means "unknown", not "low".
                previous_confidence = None
                if question.answer:
                    findings = state.get_findings_for_question(question.id)
                    if findings:
                        previous_confidence = findings[-1].confidence

                search_depth = self.search_agent.adaptive_search_depth(
                    question, iteration, previous_confidence
                )

                state.reasoning_steps.append(f"  Search depth: {search_depth}")

                # Execute search (network wait happens in a worker thread)
                ui.set_status(f"Iteration {iteration}/{max_iter} - Searching...")
                _ui_log(
                    ui, "search",
                    f"Searching ({search_depth}): {clip(question.question)}",
                )

                search_results, elapsed = await _timed(
                    self.search_agent.execute_search(
                        question=question,
                        state=state,
                        search_depth=search_depth,
                        max_results=per_search,
                        include_full_content=True,
                        ui_manager=ui,
                    ),
                    kind="search", label=f"search[{iteration}] {clip(question.question, 60)}", ui=ui,
                )
                log.info("iteration %s: search %r -> %s results in %.2fs",
                         iteration, question.question, len(search_results), elapsed)
                _ui_log(
                    ui, "ok",
                    f"{len(search_results)} results in {_seconds(elapsed)}"
                    if search_results else f"no results in {_seconds(elapsed)}",
                )

                if not search_results:
                    # Mark completion first: it removes the question from the
                    # queue and would otherwise overwrite the failed status.
                    state.mark_question_completed(question.id)
                    question.status = QuestionStatus.FAILED
                    consecutive_failures += 1

                    state.reasoning_steps.append(f"  Result: Search failed - no results found (consecutive failures: {consecutive_failures})")
                    _ui_log(ui, "warn", "No results - skipping this question")

                    # An empty result set is a normal outcome for an over-specific
                    # question, not proof the API is down. Skip it and keep the
                    # rest of the research; a genuine outage still fails when the
                    # remaining calls exhaust their retries.
                    if consecutive_failures == 3:
                        log.warning(
                            "three consecutive searches returned no results; "
                            "continuing (check API status if this persists)"
                        )

                    completed += 1
                    ui.set_progress(completed, f"Iteration {iteration}/{max_iter} - no results")
                    continue

                # Reset consecutive failures counter on successful search
                consecutive_failures = 0

                state.reasoning_steps.append(f"  Result: Found {len(search_results)} sources")

                # Analyze results (blocking LLM call off the event loop)
                ui.set_status(f"Iteration {iteration}/{max_iter} - Analyzing {len(search_results)} sources...")

                analysis, elapsed = await _timed(
                    asyncio.to_thread(
                        self.synthesis_agent.analyze_results,
                        question,
                        search_results,
                        state,
                        ui,
                    ),
                    kind="llm", label=f"analyze_results[{iteration}]", ui=ui,
                )
                log.info("iteration %s: analyzed %s sources in %.2fs",
                         iteration, len(search_results), elapsed)

                confidence = analysis.get('confidence', 0.0)
                _ui_log(
                    ui, "insight",
                    f"Answered ({confidence:.0%} confidence, {_seconds(elapsed)}): {clip(question.question)}",
                )

                state.reasoning_steps.append(f"  Analysis: {confidence:.0%} confidence")

                # Generate follow-up questions if needed
                if as_bool(analysis.get("needs_more_research")) and iteration < max_iterations - 1:
                    ui.set_status(f"Iteration {iteration}/{max_iter} - Checking for follow-ups...")
                    follow_ups, elapsed = await _timed(
                        asyncio.to_thread(
                            self.query_analyzer.generate_follow_up_questions,
                            question,
                            analysis,
                            state,
                            ui_manager=ui,
                        ),
                        kind="llm", label=f"follow_ups[{iteration}]", ui=ui,
                    )

                    if follow_ups:
                        state.reasoning_steps.append(f"  Follow-ups: Generated {len(follow_ups)} follow-up questions")
                        _ui_log(
                            ui, "ok",
                            f"{len(follow_ups)} follow-up questions ({_seconds(elapsed)})",
                        )

                # Mark question as completed
                state.mark_question_completed(question.id)
                completed += 1
                ui.set_progress(completed, f"Iteration {iteration}/{max_iter} complete")

                # Periodic gap analysis
                if iteration % 3 == 0:
                    ui.set_status(f"Iteration {iteration}/{max_iter} - Reviewing knowledge gaps...")
                    gap_items, elapsed = await _timed(
                        asyncio.to_thread(
                            self.query_analyzer.identify_knowledge_gaps, state, ui
                        ),
                        kind="llm", label=f"knowledge_gaps[{iteration}]", ui=ui,
                    )
                    added = self._record_gaps(state, gap_items)
                    _ui_log(
                        ui, "ok",
                        f"Gap review found {len(gap_items)} gaps"
                        + (f" ({added} new)" if added != len(gap_items) else "")
                        + f" ({_seconds(elapsed)})",
                    )

            # Phase 3: Cross-Reference Findings
            ui.set_phase("Phase 3: Cross-Reference")
            ui.set_status("Cross-referencing findings...")

            state.reasoning_steps.append("\nPhase 3: Cross-Reference - Starting cross-reference analysis")

            try:
                cross_ref, elapsed = await _timed(
                    asyncio.to_thread(
                        self.synthesis_agent.cross_reference_findings, state, ui
                    ),
                    kind="llm", label="cross_reference", ui=ui,
                )
                # The call is already paid for; keep its consensus and
                # contradictions for the report instead of dropping them.
                state.cross_reference = (
                    cross_ref if isinstance(cross_ref, dict) else {}
                )
                _ui_log(ui, "ok", f"Findings cross-referenced ({_seconds(elapsed)})")
            except Exception as e:
                state.reasoning_steps.append(f"Phase 3: cross-reference failed: {e}")
                log.warning("cross-reference failed: %s", e, exc_info=True)
                _ui_log(ui, "warn", "Cross-reference failed - continuing without it")

            state.reasoning_steps.append("Phase 3: Completed cross-reference analysis")
            completed += 1
            ui.set_progress(completed, "Cross-reference complete")

            # Phase 4: Generate Comprehensive Report
            ui.set_phase("Phase 4: Report Generation")
            ui.set_status("Generating comprehensive report...")

            state.reasoning_steps.append("\nPhase 4: Report Generation - Starting comprehensive report generation")

            try:
                final_report, elapsed = await _timed(
                    asyncio.wait_for(
                        self.report_generator.generate_report(state, ui, completed),
                        timeout=900.0,  # 15 minute timeout for entire report
                    ),
                    kind="llm", label="generate_report", ui=ui,
                )
                _ui_log(
                    ui, "ok",
                    f"Report written: {len(final_report):,} chars in {_seconds(elapsed)}",
                )
            except asyncio.TimeoutError:
                error_msg = "Report generation timed out after 15 minutes"
                state.reasoning_steps.append(f"Phase 4: {error_msg}")
                log.error(error_msg)
                _ui_log(ui, "err", error_msg)
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Report generation failed: {str(e)}"
                state.reasoning_steps.append(f"Phase 4: {error_msg}")
                log.error(error_msg, exc_info=True)
                _ui_log(ui, "err", error_msg)
                raise

            # Reconcile the progress total with actual steps taken.
            completed = int(ui.renderer.completed_steps)
            ui.set_total(max(completed, 1))
            ui.set_progress(completed, "Report complete")

            state.reasoning_steps.append(f"Phase 4: Completed report generation ({len(final_report):,} characters)")

            # Mark progress as complete and stop the live display
            ui.complete()

            # Finalize state
            state.end_time = datetime.now()
            duration = (state.end_time - state.start_time).total_seconds()
            ledger = getattr(ui, "timings", None)
            answered = state.answered_questions()
            failed = state.failed_questions()
            log.info(
                "research complete in %.1fs: %s iterations, %s answered, %s failed, "
                "%s searches, %s findings, %s gaps, report %s chars%s",
                duration, state.iteration_count, len(answered), len(failed),
                state.total_searches, len(state.findings), len(state.gaps),
                len(final_report),
                f"; {ledger.summary()}" if ledger is not None else "",
            )

            # Summary (printed after the live display has stopped)
            from rich.table import Table
            summary_table = Table(show_header=False, box=None, padding=(0, 2))
            summary_table.add_column("Metric", style="green bold")
            summary_table.add_column("Value", style="white", justify="right")

            summary_table.add_row("Duration:", f"{duration:.1f}s")
            summary_table.add_row("Iterations:", str(state.iteration_count))
            summary_table.add_row("Questions answered:", str(len(answered)))
            if failed:
                summary_table.add_row("Questions failed:", str(len(failed)))
            summary_table.add_row("Searches:", str(state.total_searches))
            summary_table.add_row("Findings:", str(len(state.findings)))
            contradictions = (
                state.cross_reference.get("contradictions", [])
                if isinstance(state.cross_reference, dict) else []
            )
            if contradictions:
                summary_table.add_row("Contradictions:", str(len(contradictions)))
            summary_table.add_row("Gaps Identified:", str(len(state.gaps)))
            summary_table.add_row("Report Length:", f"{len(final_report):,} chars")

            if verbose:
                summary_panel = Panel(
                    summary_table,
                    title="Research Complete",
                    title_align="left",
                    box=box.ROUNDED,
                    style="green",
                    padding=(1, 2)
                )
                self.console.print(summary_panel)
                self.console.print()

            return state

        except Exception as e:
            # Ensure UI is cleaned up even if there's an error
            log.error("research failed after %.1fs: %s", time.perf_counter() - run_started, e)
            ui.stop()

            if verbose:
                # Show error message in UI
                error_panel = Panel(
                    Text(f"Error during research: {str(e)}", style="bold red"),
                    title="Research Error",
                    title_align="left",
                    box=box.ROUNDED,
                    style="red",
                    padding=(1, 2)
                )
                self.console.print(error_panel)
                self.console.print()

            # Keep the traceback in the research record, not on stdout
            state.reasoning_steps.append("[ERROR] Full traceback:")
            state.reasoning_steps.append(traceback.format_exc())

            # Re-raise the exception
            raise

    def _record_gaps(self, state: ResearchState, gap_items) -> int:
        """Store gap-analysis output once, keeping severity when provided.

        Returns the number of newly recorded gaps. The same gap often comes
        back from two reviews, so duplicates are dropped by description.
        """
        existing = {gap.description.strip().lower() for gap in state.gaps}
        added = 0
        for item in gap_items or []:
            if isinstance(item, dict):
                description = str(item.get("description") or "").strip()
                severity = str(item.get("severity") or "moderate").strip().lower()
            else:
                description = str(item or "").strip()
                severity = "moderate"
            if not description or description.lower() in existing:
                continue
            if severity not in ("critical", "moderate", "minor"):
                severity = "moderate"
            state.gaps.append(ResearchGap(
                description=description,
                question_id="overall",
                severity=severity,
                suggested_searches=[],
            ))
            existing.add(description.lower())
            added += 1
        return added

    def save_report(self, state: ResearchState, filename: str = None) -> str:
        """
        Write the report to disk.

        A finished run writes its final report. A run that failed or was
        stopped part-way still has findings, so a clearly-labelled partial
        report is written instead of nothing at all. The reason is left in
        ``self.save_error`` whenever the write does not happen or fails.

        Args:
            state: Research state to save
            filename: Output filename (default: auto-generated in the
                configured reports directory)

        Returns:
            The path written, or None when there was nothing to save
        """
        self.save_error = ""
        body = state.final_report or self.build_partial_report(state)
        if not body:
            self.save_error = "nothing to save: the run produced no report or findings"
            log.error("save skipped: %s", self.save_error)
            return None

        if not filename:
            # Generate filename from query
            safe_query = "".join(c for c in state.main_query if c.isalnum() or c in (' ', '-', '_'))
            safe_query = safe_query[:50].strip().replace(' ', '_') or "query"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = resolve_reports_dir(self._settings)
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                self.save_error = f"cannot create the reports folder {directory}: {e}"
                log.error("save failed: %s", self.save_error)
                return None
            filename = os.path.join(directory, f"research_report_{safe_query}_{timestamp}.md")
        else:
            parent = os.path.dirname(os.path.abspath(filename))
            if parent:
                try:
                    os.makedirs(parent, exist_ok=True)
                except OSError as e:
                    self.save_error = f"cannot create the report folder {parent}: {e}"
                    log.error("save failed: %s", self.save_error)
                    return None

        # Write through a private temporary file and rename it into place: a
        # crash mid-write can never leave a truncated report, and an existing
        # file is replaced only once the new one is complete.
        try:
            directory = os.path.dirname(os.path.abspath(filename)) or "."
            try:
                mode = os.stat(filename).st_mode & 0o777
            except OSError:
                mode = 0o644
            handle, temporary = tempfile.mkstemp(
                prefix=".report-", suffix=".md", dir=directory
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as f:
                    f.write(body)
                # mkstemp is owner-only; a report is user content, so match the
                # file being replaced (or the usual default for a new one).
                try:
                    os.chmod(temporary, mode)
                except OSError:
                    pass
                os.replace(temporary, filename)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError as e:
            self.save_error = f"cannot write {filename}: {e}"
            log.error("save failed: %s", self.save_error)
            return None
        log.info("report saved to %s (%s chars%s)", filename, len(body),
                 ", partial" if not state.final_report else "")
        return filename

    def build_partial_report(self, state: ResearchState = None) -> str:
        """Assemble whatever a stopped or failed run managed to produce."""
        state = state if state is not None else self.last_state
        if state is None:
            return ""
        source_numbers, entries = build_source_registry(state.search_results)

        def cite(urls) -> str:
            numbers = []
            for url in urls:
                number = number_for(url, source_numbers)
                if number and number not in numbers:
                    numbers.append(number)
            return "".join(f"[Source {number}]" for number in numbers)

        if state.report_sections:
            body = "\n\n".join(
                f"## {section.title.strip()}\n\n{section.content.strip()}"
                for section in sorted(state.report_sections, key=lambda s: s.order)
            )
        elif state.findings:
            lines = []
            for finding in state.findings:
                marker = cite(finding.sources)
                lines.append(f"- {finding.content}" + (f" {marker}" if marker else ""))
            body = "\n".join(lines)
        else:
            return ""

        # ``completed_questions`` is "processed" bookkeeping and includes
        # questions whose search failed, so count the statuses instead.
        answered = len(state.answered_questions())
        failed = len(state.failed_questions())
        coverage = f"Questions answered: {answered}"
        if failed:
            coverage += f"  ·  Questions failed: {failed}"
        header = (
            f"# Research Report (partial): {state.main_query}\n\n"
            "> This run did not finish, so nothing was summarised or cross-checked "
            "here. It contains the raw findings collected before it stopped.\n\n"
            f"{coverage}  ·  Unique sources: {len(entries)}\n\n"
            "---\n\n"
        )
        report = header + body + "\n"
        if entries:
            source_lines = "\n".join(
                f"{entry.number}. [Source {entry.number}] {entry.title}\n   <{entry.url}>"
                for entry in entries
            )
            report += "\n---\n\n## Sources\n\n" + source_lines + "\n"
        return report

    def set_category_toggle(self, category: str, enabled: bool, silent: bool = False):
        """
        Toggle a website category on/off

        Args:
            category: Category name (social, educational, open_knowledge, news, general)
            enabled: Whether to include this category
            silent: Whether to suppress output
        """
        if category in self.category_toggles:
            self.category_toggles[category] = enabled
