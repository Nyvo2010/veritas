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
from utils.sources import (
    build_source_registry,
    citation_count,
    cited_numbers,
    normalize_citations,
    number_for,
)
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
        # One registry for the whole document: the same number labels evidence
        # in every section prompt and the bibliography. Building it here (not
        # per section) is what keeps citations from drifting.
        source_numbers, source_entries = build_source_registry(state.search_results)

        # Step 1: Identify major themes through meta-analysis
        if ui_manager:
            ui_manager.set_status("Identifying report themes...")
        themes = await self._identify_themes(state, ui_manager)
        note("ok", f"Report themes identified in {time.perf_counter() - started:.0f}s")
        if advance:
            advance(1, "Report themes identified")

        # Step 2: Generate sections in parallel
        sections, failed = await self._generate_sections_parallel(
            state, themes, style, ui_manager, base_step + 2, source_numbers
        )
        if failed:
            note("warn", f"{failed} of {len(sections) + failed} sections did not complete")

        # Step 3: Generate executive summary
        if ui_manager:
            ui_manager.set_status("Writing executive summary...")
        summary_started = time.perf_counter()
        # The summary sees no evidence list of its own, so it may cite only
        # numbers a section already grounded: enough to keep its claims
        # traceable, without letting it invent a source the body never used.
        summary_numbers = cited_numbers("\n".join(s.content for s in sections))
        executive_summary = await self._generate_executive_summary(
            state, sections, summary_numbers
        )
        executive_summary = normalize_citations(
            executive_summary,
            valid_numbers=set(source_numbers.values()),
            allowed_numbers=summary_numbers,
        )
        note("ok", f"Executive summary written ({time.perf_counter() - summary_started:.0f}s)")
        if advance:
            advance(1, "Executive summary complete")

        # Step 4: Generate uncertainties and gaps section
        gaps_section = self._generate_gaps_section(state)

        # Which sources the body actually cites. Recomputed here (rather than
        # inside the sources section) because the bibliography has to say what
        # the text did, not what the search engine returned.
        cited = cited_numbers("\n".join(
            [executive_summary, gaps_section] + [s.content for s in sections]
        ))

        # Step 5: Generate sources section
        sources_section = self._generate_sources_section(state, source_entries, cited)

        # Step 6: Assemble final report
        final_report = self._assemble_report(
            state,
            executive_summary,
            sections,
            gaps_section,
            sources_section,
            source_entries,
        )
        if advance:
            advance(1, "Report assembled")

        state.final_report = final_report
        log.info(
            "report assembled: %s sections, %s unique sources, %s citations, "
            "%s chars, %s failed, %.1fs total",
            len(sections), len(source_entries), citation_count(final_report),
            len(final_report), failed, time.perf_counter() - started,
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
        base_step: int = 0,
        source_numbers: dict = None,
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

        def finding_numbers(finding) -> list:
            """Report-wide numbers for a finding's supporting URLs."""
            numbers = []
            for url in finding.sources:
                number = number_for(url, source_numbers or {})
                if number and number not in numbers:
                    numbers.append(number)
            return numbers

        for i, theme in enumerate(theme_list):
            # Get findings related to this theme. Indices are model output and
            # can arrive as strings, a bare number, or not at all.
            raw_indices = theme.get('related_findings') or []
            if not isinstance(raw_indices, list):
                raw_indices = []
            related_indices = [
                int(idx) for idx in raw_indices if str(idx).isdigit()
            ]
            mapped_findings = [
                state.findings[idx]
                for idx in related_indices if idx < len(state.findings)
            ]

            # If the model mapped no findings, spread the evidence across the
            # themes instead of handing every section the same first ten.
            if not mapped_findings:
                mapped_findings = self._fallback_findings(
                    state, i, len(theme_list)
                )

            relevant_findings = [
                {
                    "content": f.content,
                    "sources": f.sources,
                    "confidence": f.confidence,
                    "source_numbers": finding_numbers(f),
                }
                for f in mapped_findings
            ]
            section_urls = list(dict.fromkeys(
                url for f in mapped_findings for url in f.sources
                if number_for(url, source_numbers or {})
            ))
            section_numbers = {
                number
                for finding in relevant_findings
                for number in finding.get("source_numbers", [])
            }
            
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
            tasks.append((i, title, task, section_numbers, section_urls))
        
        # Execute tasks with concurrency limit
        sections = []
        semaphore = asyncio.Semaphore(MAX_PARALLEL_SECTIONS)

        def note(level: str, message: str) -> None:
            log_method = getattr(ui_manager, "log", None)
            if callable(log_method):
                log_method(level, message)
            elif ui_manager is not None:
                ui_manager.add_message(message, style="white")

        async def generate_with_progress(order, title, task, section_numbers, section_urls):
            async with semaphore:
                started = time.perf_counter()
                note("info", f"Writing section: {clip(title)}")
                try:
                    content = await asyncio.wait_for(task, timeout=SECTION_TIMEOUT)
                except asyncio.TimeoutError:
                    log.warning("section %r timed out after %.0fs", title, SECTION_TIMEOUT)
                    note("warn", f"Section timed out after {SECTION_TIMEOUT:.0f}s: {clip(title)}")
                    return (order, title, SECTION_TIMEOUT_PLACEHOLDER.format(seconds=SECTION_TIMEOUT),
                            section_urls, True)
                except Exception as e:
                    log.error("section %r failed: %s", title, e, exc_info=True)
                    note("err", f"Section failed: {clip(title)} ({clip(str(e), 60)})")
                    return order, title, f"[Section generation failed: {e}]", section_urls, True
                duration = time.perf_counter() - started
                # Every marker in the body must resolve to a source that was
                # actually supplied for this section and exists in the report
                # bibliography. Malformed or invented numbers are dropped.
                content = normalize_citations(
                    content,
                    valid_numbers=set((source_numbers or {}).values()),
                    allowed_numbers=section_numbers,
                )
                log.info("section %r done in %.1fs (%s chars, %s citations)",
                         title, duration, len(content), citation_count(content))
                note("ok", f"Section ready in {duration:.0f}s: {clip(title)}")
                advance = getattr(ui_manager, "advance", None)
                if callable(advance):
                    advance(1, f"Completed section: {clip(title, 40)}")
                return order, title, content, section_urls, False

        results = await asyncio.gather(*[
            generate_with_progress(order, title, task, numbers, urls)
            for order, title, task, numbers, urls in tasks
        ], return_exceptions=False)

        failed = 0
        for order, title, content, section_urls, failed_section in sorted(results, key=lambda x: x[0]):
            failed += 1 if failed_section else 0
            section = ReportSection(
                title=title,
                content=content,
                sources=section_urls,
                order=order,
            )
            sections.append(section)
            state.report_sections.append(section)

        return sections, failed
    
    async def _generate_executive_summary(
        self,
        state: ResearchState,
        sections: List[ReportSection],
        source_numbers: set = None,
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
            source_numbers,
        )

        return summary
    
    @staticmethod
    def _fallback_findings(
        state: ResearchState,
        theme_index: int,
        theme_count: int,
        limit: int = 10,
    ) -> List:
        """Interleave findings across themes that were given no mapping.

        Theme-to-finding indices are model output and can be absent or
        unusable. Reusing ``findings[:10]`` for every such theme makes the
        sections repeat each other; a deterministic stride keeps them
        distinct while ensuring every finding still has a home.
        """
        if not state.findings:
            return []
        stride = max(1, theme_count)
        start = theme_index % len(state.findings)
        return [
            state.findings[index]
            for index in range(start, len(state.findings), stride)
        ][:limit]

    def _generate_gaps_section(self, state: ResearchState) -> str:
        """
        Generate section about uncertainties and knowledge gaps
        """
        if not state.gaps:
            content = "No significant knowledge gaps were identified during this research.\n"
        else:
            content = "The following uncertainties and knowledge gaps were identified.\n\n"

            # Group gaps by severity
            critical_gaps = [g for g in state.gaps if g.severity == "critical"]
            moderate_gaps = [g for g in state.gaps if g.severity == "moderate"]

            if critical_gaps:
                content += "### Critical gaps\n\n"
                for gap in critical_gaps:
                    content += f"- {gap.description}\n"
                    if gap.suggested_searches:
                        content += f"  - Suggested research: {', '.join(gap.suggested_searches[:2])}\n"
                content += "\n"

            if moderate_gaps:
                content += "### Moderate gaps\n\n"
                for gap in moderate_gaps[:5]:  # Limit to 5
                    content += f"- {gap.description}\n"
                content += "\n"

        cross = state.cross_reference if isinstance(state.cross_reference, dict) else {}

        # Cross-question agreements were already paid for; surface them.
        consensus = cross.get("consensus_points") or []
        if consensus:
            content += "### Points of agreement\n\n"
            for point in consensus[:5]:
                if isinstance(point, dict):
                    text = str(point.get("point") or "").strip()
                    confidence = point.get("confidence")
                else:
                    text = str(point).strip()
                    confidence = None
                if not text:
                    continue
                suffix = ""
                if isinstance(confidence, (int, float)):
                    suffix = f" (confidence {float(confidence):.0%})"
                content += f"- {text}{suffix}\n"
            content += "\n"

        # Contradictions come from two places: per-finding analysis and the
        # later cross-reference pass. Normalise both to one bullet form.
        contradictions = []
        for finding in state.findings:
            contradictions.extend(finding.contradictions or [])
        for item in cross.get("contradictions") or []:
            if isinstance(item, dict):
                topic = str(item.get("topic") or "").strip()
                viewpoint_a = str(item.get("viewpoint_a") or "").strip()
                viewpoint_b = str(item.get("viewpoint_b") or "").strip()
                if topic and viewpoint_a and viewpoint_b:
                    contradictions.append(
                        f"**{topic}**: {viewpoint_a} — {viewpoint_b}"
                    )
                elif topic:
                    contradictions.append(topic)
            elif str(item).strip():
                contradictions.append(str(item).strip())

        seen = set()
        unique_contradictions = []
        for contradiction in contradictions:
            key = str(contradiction).strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_contradictions.append(contradiction)

        if unique_contradictions:
            content += "### Areas of contradiction\n\n"
            content += "The research revealed contradictions in the following areas.\n\n"
            for contradiction in unique_contradictions[:5]:
                content += f"- {contradiction}\n"
            content += "\n"

        reliability = str(cross.get("reliability_assessment") or "").strip()
        if reliability:
            content += "### Source reliability\n\n"
            content += reliability + "\n\n"

        return content
    
    def _generate_sources_section(
        self,
        state: ResearchState,
        entries: list = None,
        cited: set = None,
    ) -> str:
        """Bibliography in the same numbering used by inline citations.

        Every entry is labelled with how far it actually got: cited in the
        body, analyzed but unused, or only collected. Without that, a long
        bibliography reads as evidence it never was.

        No heading here: _assemble_report writes the "Sources" heading.
        """
        if entries is None:
            _, entries = build_source_registry(state.search_results)
        cited = cited or set()

        if not entries:
            return "No sources were collected.\n"

        lines = []
        for source in entries:
            title = source.title or source.url
            confidence = source.category.replace("_", " ").title()
            lines.append(f"{source.number}. [Source {source.number}] **{title}**")
            lines.append(f"   <{source.url}>")
            if source.published_date:
                lines.append(f"   Published: {source.published_date}")
            lines.append(
                f"   {confidence} · evidence quality {source.quality:.2f} · "
                f"search relevance {source.score:.2f}"
            )
            if not source.analyzed:
                lines.append("   Collected only — not analyzed")
            elif source.number not in cited:
                lines.append("   Analyzed — not cited in this report")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    
    def _assemble_report(
        self,
        state: ResearchState,
        executive_summary: str,
        sections: List[ReportSection],
        gaps_section: str,
        sources_section: str,
        source_entries: list = None,
    ) -> str:
        """Assemble the components as one properly structured Markdown file."""
        from datetime import datetime

        model = getattr(self.llm, "model", "") or "unknown"
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        ordered_sections = sorted(sections, key=lambda section: section.order)
        body_for_count = "\n".join(
            [executive_summary, gaps_section]
            + [section.content for section in ordered_sections]
        )
        cited = cited_numbers(body_for_count)
        meta = "  ·  ".join([
            f"Model: `{model}`",
            f"Iterations: {state.iteration_count}",
            f"Unique sources: {len(source_entries or [])} ({len(cited)} cited)",
            f"Citations: {citation_count(body_for_count)}",
            f"Generated: {generated}",
        ])

        report = f"""# Research Report: {state.main_query}

*Generated by Veritas Deep Research Agent*

{meta}

> Inline citations use `[Source N]`, which resolves to the numbered entry under **Sources**.

---

## Executive Summary

{executive_summary.strip()}

"""
        
        # Add main sections in theme order, with the assembler owning headings.
        for section in ordered_sections:
            report += f"---\n\n## {section.title.strip()}\n\n{section.content.strip()}\n\n"
        
        # Add gaps section
        report += "---\n\n## Uncertainties and Knowledge Gaps\n\n"
        report += gaps_section.strip() + "\n\n"
        
        # Add sources
        report += "---\n\n## Sources\n\n"
        report += sources_section.strip() + "\n"
        
        return report
