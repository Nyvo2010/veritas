"""
Report Generator Agent - Generates comprehensive research reports
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from typing import List, Dict

from models.research_state import ResearchState, ReportSection
from utils.llm_client import LLMClient
from config import MAX_PARALLEL_SECTIONS


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

        # Step 1: Identify major themes through meta-analysis
        if ui_manager:
            ui_manager.set_status("Identifying report themes...")
        themes = await self._identify_themes(state, ui_manager)
        if advance:
            advance(1, "Report themes identified")

        # Step 2: Generate sections in parallel
        sections = await self._generate_sections_parallel(state, themes, style, ui_manager, base_step + 2)

        # Step 3: Generate executive summary
        if ui_manager:
            ui_manager.set_status("Writing executive summary...")
        executive_summary = await self._generate_executive_summary(state, sections)
        if advance:
            advance(1, "Executive summary complete")

        # Step 4: Generate uncertainties and gaps section
        gaps_section = self._generate_gaps_section(state)

        # Step 5: Generate sources section
        sources_section = self._generate_sources_section(state)

        # Step 6: Assemble final report
        final_report = self._assemble_report(
            state.main_query,
            executive_summary,
            sections,
            gaps_section,
            sources_section
        )
        if advance:
            advance(1, "Report assembled")

        state.final_report = final_report

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
    ) -> List[ReportSection]:
        """
        Generate report sections in parallel using asyncio
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
        
        async def generate_with_progress(order, title, task):
            async with semaphore:
                try:
                    if ui_manager:
                        advance = getattr(ui_manager, "advance", None)
                        if callable(advance):
                            advance(0, f"Generating section: {title[:40]}...")
                        else:
                            ui_manager.update_progress(base_step, f"Generating section: {title[:40]}...")
                        ui_manager.add_message(
                            f"Writing report section: {title}",
                            style="white"
                        )

                    content = await task

                    if ui_manager:
                        if callable(getattr(ui_manager, "advance", None)):
                            ui_manager.advance(1, f"Completed section: {title[:40]}")
                        ui_manager.add_message(
                            f"Completed: {title[:40]}",
                            style="white"
                        )

                    return order, title, content
                except Exception as e:
                    if ui_manager:
                        ui_manager.add_message(
                            f"Error: {title[:40]} - {str(e)[:100]}",
                            style="white"
                        )
                    # Return a placeholder instead of failing completely
                    return order, title, f"[Section generation failed: {str(e)}]"
        
        # Gather all results with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[
                    generate_with_progress(order, title, task)
                    for order, title, task in tasks
                ], return_exceptions=False),
                timeout=600.0  # 10 minute timeout for all sections
            )
        except asyncio.TimeoutError:
            error_msg = "Report section generation timed out after 10 minutes"
            if ui_manager:
                ui_manager.add_message(f"Error: {error_msg}", style="white")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Report section generation failed: {str(e)}"
            if ui_manager:
                ui_manager.add_message(f"Error: {error_msg}", style="white")
            raise
        
        # Convert to ReportSection objects
        for order, title, content in sorted(results, key=lambda x: x[0]):
            section = ReportSection(
                title=title,
                content=content,
                sources=[],  # Sources are inline in content
                order=order
            )
            sections.append(section)
            state.report_sections.append(section)
        
        # Suppress section completion summary
        
        return sections
    
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
        main_query: str,
        executive_summary: str,
        sections: List[ReportSection],
        gaps_section: str,
        sources_section: str
    ) -> str:
        """
        Assemble all components into final markdown report
        """
        report = f"""Research Report: {main_query}

Generated by Veritas Deep Research Agent

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
