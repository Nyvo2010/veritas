"""
Main orchestrator for the Deep Research Agent
Coordinates all agents in an iterative research loop
"""

import asyncio
import nest_asyncio
from typing import Dict, Optional
from datetime import datetime
import sys
import os
import traceback

# Ensure src root is on sys.path for imports when running scripts from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from models.research_state import ResearchState, QuestionStatus
from agents.query_analyzer import QueryAnalyzerAgent
from agents.search_agent import SearchAgent
from agents.synthesis_agent import SynthesisAgent
from agents.report_generator import ReportGeneratorAgent
from utils.nvidia_client import NvidiaClient
from utils.tavily_client import TavilySearchClient
from utils.ui_manager import UIManager
from settings import load_settings, resolve_api_keys, reports_dir as resolve_reports_dir

from config import (
    NVIDIA_API_KEY,
    TAVILY_API_KEY,
    MAX_RESEARCH_ITERATIONS,
)


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
        nvidia_api_key: str = NVIDIA_API_KEY,
        tavily_api_key: str = TAVILY_API_KEY,
        category_toggles: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize the research agent

        Args:
            nvidia_api_key: API key for NVIDIA NIM (falls back to settings file)
            tavily_api_key: API key for Tavily Search (falls back to settings file)
            category_toggles: Dict controlling which website categories to include
        """
        self._settings = load_settings()
        file_nvidia, file_tavily, file_base = resolve_api_keys(self._settings)

        # Initialize clients
        self.nvidia = NvidiaClient(
            api_key=(nvidia_api_key or file_nvidia),
            api_base_url=file_base,
        )
        self.tavily = TavilySearchClient(api_key=(tavily_api_key or file_tavily))

        # Initialize agents
        self.query_analyzer = QueryAnalyzerAgent(self.nvidia)
        self.search_agent = SearchAgent(self.tavily, self.nvidia)
        self.synthesis_agent = SynthesisAgent(self.nvidia)
        self.report_generator = ReportGeneratorAgent(self.nvidia)

        # Category toggles
        if category_toggles:
            self.category_toggles = dict(category_toggles)
        else:
            self.category_toggles = dict(self._settings.get("category_toggles", {}))

        # Rich console for UI
        self.console = Console()

    async def research(
        self,
        query: str,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
        verbose: bool = True
    ) -> ResearchState:
        """
        Execute deep research on a query

        Args:
            query: The research query
            max_iterations: Maximum research iterations
            verbose: Whether to print progress

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
        ui = UIManager(console=self.console)
        short_query = query if len(query) <= 60 else query[:57] + "..."
        ui.initialize(estimated_steps, f"Research: {short_query}")
        ui.set_status("Preparing research environment...")
        ui.start()

        completed = 0

        try:
            # Phase 1: Query Decomposition (blocking LLM call off the event loop)
            ui.set_phase("Phase 1: Query Analysis")
            ui.set_status("Decomposing query into sub-questions...")

            state.reasoning_steps.append("Phase 1: Query Decomposition - Started")

            sub_questions = await asyncio.to_thread(
                self.query_analyzer.decompose_main_query, query, state, ui
            )

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
            max_consecutive_failures = 3

            while iteration < max_iterations and state.question_queue:
                iteration += 1
                state.iteration_count = iteration

                # Get next question
                question = state.get_next_question()
                if not question:
                    break

                state.reasoning_steps.append(f"\nIteration {iteration}: Processing question {iteration}/{max_iter}")
                state.reasoning_steps.append(f"  Question: {question.question}")

                # Determine search depth adaptively
                previous_confidence = 0.0
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

                search_results = await self.search_agent.execute_search(
                    question=question,
                    state=state,
                    search_depth=search_depth,
                    include_full_content=True,
                    ui_manager=ui
                )

                if not search_results:
                    question.status = QuestionStatus.FAILED
                    state.mark_question_completed(question.id)
                    consecutive_failures += 1

                    state.reasoning_steps.append(f"  Result: Search failed - no results found (consecutive failures: {consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        error_msg = f"Aborting research: {max_consecutive_failures} consecutive search failures detected"
                        ui.add_message(error_msg, style="red bold")
                        ui.add_message("This usually indicates API issues, rate limiting, or network problems", style="yellow")
                        state.reasoning_steps.append(f"\nResearch aborted: {error_msg}")
                        raise Exception(error_msg)

                    completed += 1
                    ui.set_progress(completed, f"Iteration {iteration}/{max_iter} - no results")
                    continue

                # Reset consecutive failures counter on successful search
                consecutive_failures = 0

                state.reasoning_steps.append(f"  Result: Found {len(search_results)} sources")

                # Analyze results (blocking LLM call off the event loop)
                ui.set_status(f"Iteration {iteration}/{max_iter} - Analyzing {len(search_results)} sources...")

                analysis = await asyncio.to_thread(
                    self.synthesis_agent.analyze_results,
                    question,
                    search_results,
                    state,
                    ui,
                )

                confidence = analysis.get('confidence', 0.0)

                state.reasoning_steps.append(f"  Analysis: {confidence:.0%} confidence")

                # Generate follow-up questions if needed
                if analysis.get("needs_more_research") and iteration < max_iterations - 1:
                    ui.set_status(f"Iteration {iteration}/{max_iter} - Checking for follow-ups...")
                    follow_ups = await asyncio.to_thread(
                        self.query_analyzer.generate_follow_up_questions,
                        question,
                        analysis,
                        state,
                        ui,
                    )

                    if follow_ups:
                        state.reasoning_steps.append(f"  Follow-ups: Generated {len(follow_ups)} follow-up questions")

                # Mark question as completed
                state.mark_question_completed(question.id)
                completed += 1
                ui.set_progress(completed, f"Iteration {iteration}/{max_iter} complete")

                # Periodic gap analysis
                if iteration % 3 == 0:
                    ui.set_status(f"Iteration {iteration}/{max_iter} - Reviewing knowledge gaps...")
                    await asyncio.to_thread(
                        self.query_analyzer.identify_knowledge_gaps, state, ui
                    )

            # Phase 3: Cross-Reference Findings
            ui.set_phase("Phase 3: Cross-Reference")
            ui.set_status("Cross-referencing findings...")

            state.reasoning_steps.append("\nPhase 3: Cross-Reference - Starting cross-reference analysis")

            try:
                cross_ref = await asyncio.to_thread(
                    self.synthesis_agent.cross_reference_findings, state, ui
                )
            except Exception as e:
                state.reasoning_steps.append(f"Phase 3: cross-reference failed: {e}")
                ui.add_message(
                    "Warning: Cross-reference analysis failed, continuing...",
                    style="white"
                )
                cross_ref = {"consensus": [], "contradictions": []}

            state.reasoning_steps.append("Phase 3: Completed cross-reference analysis")
            completed += 1
            ui.set_progress(completed, "Cross-reference complete")

            # Phase 4: Generate Comprehensive Report
            ui.set_phase("Phase 4: Report Generation")
            ui.set_status("Generating comprehensive report...")

            state.reasoning_steps.append("\nPhase 4: Report Generation - Starting comprehensive report generation")

            try:
                final_report = await asyncio.wait_for(
                    self.report_generator.generate_report(state, ui, completed),
                    timeout=900.0  # 15 minute timeout for entire report
                )
            except asyncio.TimeoutError:
                error_msg = "Report generation timed out after 15 minutes"
                state.reasoning_steps.append(f"Phase 4: {error_msg}")
                ui.add_message(f"Error: {error_msg}", style="white")
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Report generation failed: {str(e)}"
                state.reasoning_steps.append(f"Phase 4: {error_msg}")
                ui.add_message(f"Error: {error_msg}", style="white")
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

            # Summary (printed after the live display has stopped)
            from rich.table import Table
            summary_table = Table(show_header=False, box=None, padding=(0, 2))
            summary_table.add_column("Metric", style="green bold")
            summary_table.add_column("Value", style="white", justify="right")

            summary_table.add_row("Duration:", f"{duration:.1f}s")
            summary_table.add_row("Iterations:", str(state.iteration_count))
            summary_table.add_row("Questions:", str(len(state.completed_questions)))
            summary_table.add_row("Searches:", str(state.total_searches))
            summary_table.add_row("Findings:", str(len(state.findings)))
            summary_table.add_row("Gaps Identified:", str(len(state.gaps)))
            summary_table.add_row("Report Length:", f"{len(final_report):,} chars")

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
            ui.stop()

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

    def research_sync(
        self,
        query: str,
        max_iterations: int = MAX_RESEARCH_ITERATIONS,
        verbose: bool = True
    ) -> ResearchState:
        """
        Synchronous wrapper for research method

        Args:
            query: The research query
            max_iterations: Maximum research iterations
            verbose: Whether to print progress

        Returns:
            ResearchState with all findings and final report
        """
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create and run one
            return asyncio.run(self.research(query, max_iterations, verbose))
        else:
            # Event loop already running, schedule the coroutine
            nest_asyncio.apply()
            return asyncio.run(self.research(query, max_iterations, verbose))

    def save_report(self, state: ResearchState, filename: str = None) -> str:
        """
        Save the final report to a file

        Args:
            state: Research state with final report
            filename: Output filename (default: auto-generated in the
                configured reports directory)

        Returns:
            The filename where the report was saved, or None on failure
        """
        if not state.final_report:
            return None

        if not filename:
            # Generate filename from query
            safe_query = "".join(c for c in state.main_query if c.isalnum() or c in (' ', '-', '_'))
            safe_query = safe_query[:50].strip().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = resolve_reports_dir(self._settings)
            os.makedirs(directory, exist_ok=True)
            filename = os.path.join(directory, f"research_report_{safe_query}_{timestamp}.md")
        else:
            parent = os.path.dirname(os.path.abspath(filename))
            if parent:
                os.makedirs(parent, exist_ok=True)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(state.final_report)
            return filename
        except Exception:
            return None

    def _format_toggles(self) -> str:
        """Format category toggles for display"""
        enabled = [k for k, v in self.category_toggles.items() if v]
        disabled = [k for k, v in self.category_toggles.items() if not v]

        result = ""
        if enabled:
            result += "[ON] " + ", ".join(enabled)
        if disabled:
            result += " | [OFF] " + ", ".join(disabled)
        return result or "All enabled"

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
