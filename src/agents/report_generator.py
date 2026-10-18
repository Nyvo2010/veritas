"""
Report Generator Agent - Generates comprehensive research reports
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
import logging
import time
from typing import List, Dict

from models.research_state import ResearchState, ReportSection
from utils.llm_client import LLMClient
from utils.text import clip
from config import MAX_PARALLEL_SECTIONS

log = logging.getLogger("veritas.report")

# One section may not hold the whole report hostage: a section that stalls is
# dropped and the report is still assembled from the rest.
SECTION_TIMEOUT = 420.0


SECTION_TIMEOUT_PLACEHOLDER = "[Section generation timed out after {seconds:.0f}s]"


class ReportGeneratorAgent:
    """
    Agent responsible for generating comprehensive reports
    Uses divide-and-conquer approach with parallel section generation
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def generate_report(
        self,
        state: ResearchState,
        ui_manager=None,
        base_step: int = 0,
        style: str = "comprehensive"
    ) -> str:
        """
        Generate comprehensive report using parallel section generation
        
        Args:
            state: Research state with all findings
            ui_manager: UI manager for progress tracking
            base_step: Starting step number for progress tracking
            style: Writing style (comprehensive, concise, technical)
        
        Returns:
            Complete report as markdown string
        """
        advance = None
        if ui_manager is not None and callable(getattr(ui_manager, "advance", None)):
            advance = ui_manager.advance

        def note(level: str, message: str) -> None:
            """One leveled line to the UI, plus the same line to the file log."""
            log.debug("report[%s] %s", level, message)
            log_method = getattr(ui_manager, "log", None)
            if callable(log_method):
                log_method(level, message)
            elif ui_manager is not None:
                ui_manager.add_message(message, style="white")

        started = time.perf_counter()

        # Step 1: Identify major themes through meta-analysis
        if ui_manager:
            ui_manager.set_status("Identifying report themes...")
        themes = await self._identify_themes(state, ui_manager)
        note("ok", f"Report themes identified in {time.perf_counter() - started:.0f}s")
        if advance:
            advance(1, "Report themes identified")

        # Step 2: Generate sections in parallel
        sections, failed = await self._generate_sections_parallel(
            state, themes, style, ui_manager, base_step + 2
        )
        if failed:
            note("warn", f"{failed} of {len(sections) + failed} sections did not complete")

        # Step 3: Generate executive summary
        if ui_manager:
            ui_manager.set_status("Writing executive summary...")
        summary_started = time.perf_counter()
        executive_summary = await self._generate_executive_summary(state, sections)
        note("ok", f"Executive summary written ({time.perf_counter() - summary_started:.0f}s)")
        if advance:
            advance(1, "Executive summary complete")

        # Step 4: Generate uncertainties and gaps section
        gaps_section = self._generate_gaps_section(state)

        # Step 5: Generate sources section
        sources_section = self._generate_sources_section(state)

        # Step 6: Assemble final report
        final_report = self._assemble_report(
            state,
            executive_summary,
            sections,
            gaps_section,
            sources_section
        )
        if advance:
            advance(1, "Report assembled")

        state.final_report = final_report
        log.info(
            "report assembled: %s sections, %s chars, %s failed, %.1fs total",
            len(sections), len(final_report), failed, time.perf_counter() - started,
        )

        return final_report
    
    async def _identify_themes(self, state: ResearchState, ui_manager=None) -> Dict:
        """
        Use LLM to identify major themes from all findings
        """
        findings_data = [
            {
                "content": f.content,
                "confidence": f.confidence,
                "sources": f.sources[:2]
            }
            for f in state.findings
        ]
        
        themes = await asyncio.to_thread(
            self.llm.identify_report_themes, findings_data
        )
        
        # Suppress theme analysis output
        
        return themes
    
    async def _generate_sections_parallel(
        self,
        state: ResearchState,
        themes: Dict,
        style: str,
        ui_manager=None,
        base_step: int = 0
    ) -> tuple:
        """
        Generate report sections in parallel using asyncio.

        Returns ``(sections, failed_count)``. Sections are bounded by their
        own timeout rather than one shared deadline, so a single stall costs
        one section instead of the entire report.
        """
        tasks = []
        theme_list = themes.get("themes") if isinstance(themes, dict) else None
        if not theme_list:
            # The model occasionally omits themes; never fail the whole report.
            theme_list = [{
                "title": "Analysis",
                "description": f"Comprehensive analysis of: {state.main_query}",
                "related_findings": [],
            }]

        for i, theme in enumerate(theme_list):
            # Get findings related to this theme
            related_indices = theme.get('related_findings', [])
            # Convert indices to integers (in case they're strings from JSON)
            related_indices = [int(idx) for idx in related_indices if str(idx).isdigit()]
            relevant_findings = [
                {
                    "content": state.findings[idx].content,
                    "sources": state.findings[idx].sources,
                    "confidence": state.findings[idx].confidence
                }
                for idx in related_indices if idx < len(state.findings)
            ]
            
            # If no specific findings mapped, use all findings
            if not relevant_findings:
                relevant_findings = [
                    {
                        "content": f.content,
                        "sources": f.sources,
                        "confidence": f.confidence
                    }
                    for f in state.findings[:10]  # Limit to prevent token overflow
                ]
            
            # Create async task
            title = theme.get("title") or f"Section {i + 1}"
            description = theme.get("description") or title
            task = self.llm.generate_section_async(
                section_title=title,
                theme_description=description,
                relevant_findings=relevant_findings,
                main_query=state.main_query,
                style=style
            )
            tasks.append((i, title, task))
        
        # Execute tasks with concurrency limit
        sections = []
        semaphore = asyncio.Semaphore(MAX_PARALLEL_SECTIONS)

        def note(level: str, message: str) -> None:
            log_method = getattr(ui_manager, "log", None)
            if callable(log_method):
                log_method(level, message)
            elif ui_manager is not None:
                ui_manager.add_message(message, style="white")

        async def generate_with_progress(order, title, task):
            async with semaphore:
                started = time.perf_counter()
                note("info", f"Writing section: {clip(title)}")
                try:
                    content = await asyncio.wait_for(task, timeout=SECTION_TIMEOUT)
                except asyncio.TimeoutError:
                    log.warning("section %r timed out after %.0fs", title, SECTION_TIMEOUT)
                    note("warn", f"Section timed out after {SECTION_TIMEOUT:.0f}s: {clip(title)}")
                    return order, title, SECTION_TIMEOUT_PLACEHOLDER.format(seconds=SECTION_TIMEOUT), True
                except Exception as e:
                    log.error("section %r failed: %s", title, e, exc_info=True)
                    note("err", f"Section failed: {clip(title)} ({clip(str(e), 60)})")
                    return order, title, f"[Section generation failed: {e}]", True
                duration = time.perf_counter() - started
                log.info("section %r done in %.1fs (%s chars)", title, duration, len(content))
                note("ok", f"Section ready in {duration:.0f}s: {clip(title)}")
                advance = getattr(ui_manager, "advance", None)
                if callable(advance):
                    advance(1, f"Completed section: {clip(title, 40)}")
                return order, title, content, False

        results = await asyncio.gather(*[
            generate_with_progress(order, title, task)
            for order, title, task in tasks
        ], return_exceptions=False)

        failed = 0
        for order, title, content, failed_section in sorted(results, key=lambda x: x[0]):
            failed += 1 if failed_section else 0
            section = ReportSection(
                title=title,
                content=content,
                sources=[],  # Sources are inline in content
                order=order,
            )
            sections.append(section)
            state.report_sections.append(section)

        return sections, failed
    
    async def _generate_executive_summary(
        self,
        state: ResearchState,
        sections: List[ReportSection]
    ) -> str:
        """
        Generate executive summary after all sections are complete
        """
        section_contents = [s.content for s in sections]
        
        key_findings = [
            f.content[:200] + "..."
            for f in sorted(state.findings, key=lambda x: x.confidence, reverse=True)[:10]
        ]
        
        summary = await asyncio.to_thread(
            self.llm.generate_executive_summary,
            state.main_query,
            section_contents,
            key_findings,
        )

        return summary
    
    def _generate_gaps_section(self, state: ResearchState) -> str:
        """
        Generate section about uncertainties and knowledge gaps
        """
        if not state.gaps:
            return "No significant knowledge gaps were identified during this research."
        
        content = "The following uncertainties and knowledge gaps were identified:\n\n"
        
        # Group gaps by severity
        critical_gaps = [g for g in state.gaps if g.severity == "critical"]
        moderate_gaps = [g for g in state.gaps if g.severity == "moderate"]
        
        if critical_gaps:
            content += "CRITICAL GAPS\n\n"
            for gap in critical_gaps:
                content += f"- {gap.description}\n"
                if gap.suggested_searches:
                    content += f"  - Suggested research: {', '.join(gap.suggested_searches[:2])}\n"
            content += "\n"
        
        if moderate_gaps:
            content += "MODERATE GAPS\n\n"
            for gap in moderate_gaps[:5]:  # Limit to 5
                content += f"- {gap.description}\n"
            content += "\n"
        
        # Add contradictions found
        contradictions = []
        for finding in state.findings:
            if finding.contradictions:
                contradictions.extend(finding.contradictions)
        
        if contradictions:
            content += "AREAS OF CONTRADICTION\n\n"
            content += "The research revealed contradictions in the following areas:\n\n"
            for contradiction in contradictions[:5]:
                content += f"- {contradiction}\n"
            content += "\n"
        
        return content
    
    def _generate_sources_section(self, state: ResearchState) -> str:
        """
        Generate sources section with all references
        """
        # Collect all unique sources
        sources = {}
        source_counter = 1
        
        for result in state.search_results:
            if result.url not in sources:
                sources[result.url] = {
                    "number": source_counter,
                    "title": result.title,
                    "url": result.url,
                    "category": result.category,
                    "score": result.score
                }
                source_counter += 1
        
        # Sort by source number (numeric order)
        sorted_sources = sorted(
            sources.values(),
            key=lambda x: x["number"]
        )
        # No heading here: _assemble_report writes the "SOURCES" title.
        content = ""
        for source in sorted_sources:
            content += f"{source['number']}. {source['title']}\n    {source['url']}\n"
        content += "\n"
        return content
    
    def _assemble_report(
        self,
        state: ResearchState,
        executive_summary: str,
        sections: List[ReportSection],
        gaps_section: str,
        sources_section: str
    ) -> str:
        """
        Assemble all components into final markdown report
        """
        from datetime import datetime

        model = getattr(self.llm, "model", "") or "unknown"
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        meta = "  ·  ".join([
            f"Model: {model}",
            f"Iterations: {state.iteration_count}",
            f"Sources: {len(state.search_results)}",
            f"Generated: {generated}",
        ])
        report = f"""Research Report: {state.main_query}

Generated by Veritas Deep Research Agent
{meta}

════════════════════════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY

{executive_summary}

════════════════════════════════════════════════════════════════════════════════

"""
        
        # Add main sections
        for section in sections:
            report += f"{section.title.upper()}\n\n"
            report += section.content
            report += "\n\n════════════════════════════════════════════════════════════════════════════════\n\n"
        
        # Add gaps section
        report += "UNCERTAINTIES & KNOWLEDGE GAPS\n\n"
        report += gaps_section
        report += "\n════════════════════════════════════════════════════════════════════════════════\n\n"
        
        # Add sources
        report += "SOURCES\n\n"
        report += sources_section
        
        return report
