"""
Report Generator Agent - Generates comprehensive research reports
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
import logging
import re
import time
from collections import Counter
from typing import List, Dict

from models.research_state import QuestionStatus, ResearchState, ReportSection
from utils.llm_client import LLMClient
from utils.sources import (
    SourceEntry,
    build_source_registry,
    citation_count,
    cited_numbers,
    normalize_citations,
    number_for,
    renumber_citations,
    source_type,
    source_type_label,
)
from utils.text import clip, is_near_duplicate
from config import MAX_PARALLEL_SECTIONS

log = logging.getLogger("veritas.report")

# One section may not hold the whole report hostage: a section that stalls is
# dropped and the report is still assembled from the rest.
SECTION_TIMEOUT = 420.0

# Findings that share a theme become one section; past three the section stops
# having a single argument and repeats the executive summary.
MAX_FINDINGS_PER_SECTION = 3

# A section may legitimately lean on one source, but past this the citation
# block claims more agreement than the evidence has. The prompt asks for the
# cap; this constant is what the post-generation check actually enforces (it
# reports a violation, it cannot invent a second source).
MAX_CITATIONS_PER_SOURCE = 3

_CITATION_MARKER_RE = re.compile(r"\[Source (\d+)\]")


def _clean_title(text: str) -> str:
    """A heading, not a transcript: one line, sentence case, no trailing mark.

    The theme pass sometimes returns a run's question as its title, so a
    report opened a section with "needle in a haystack test results for long
    context LLMs": lowercase, a question, not a heading. Only the first
    letter is touched — title-casing would break acronyms and proper nouns
    ("LLMs", "RoPE", "Ms-PoE").
    """
    title = " ".join(str(text or "").split())
    if not title:
        return ""
    first, _, rest = title.partition(" ")
    # Only an all-lowercase first word is capitalised: "eBPF" and "iPhone"
    # already carry their own casing and must survive a heading rewrite.
    if first.islower():
        first = first[0].upper() + first[1:]
        title = f"{first} {rest}" if rest else first
    # A question mark is a legitimate section heading; a trailing full stop
    # is not, and is usually punctuation copied from the question.
    return title.rstrip(" .")


def _cited_claims(content: str) -> List[Dict]:
    """One entry per citation marker, in document order.

    ``span`` is the marker's position (including the space before it, so
    removing it cannot leave "claim ."), and ``sentence`` is the surrounding
    claim shown to the support checker.
    """
    text = content or ""
    claims: List[Dict] = []
    for index, match in enumerate(_CITATION_MARKER_RE.finditer(text)):
        start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
        end = text.find(".", match.end())
        if end < 0:
            end = len(text)
        span_start = match.start() - 1 if match.start() and text[match.start() - 1] == " " else match.start()
        claims.append({
            "id": f"M{index + 1}",
            "source": int(match.group(1)),
            "sentence": clip(text[start:end + 1].strip(), 300),
            "span": (span_start, match.end()),
        })
    return claims


def _drop_marker_spans(content: str, spans: List[tuple]) -> str:
    """Remove the exact marker spans, leaving the prose untouched."""
    out = []
    cursor = 0
    for start, end in sorted(spans):
        out.append(content[cursor:start])
        cursor = end
    out.append(content[cursor:])
    return "".join(out)


def _cap_source_citations(content: str, cap: int = MAX_CITATIONS_PER_SOURCE) -> tuple:
    """Drop markers past *cap* per source, keeping the first ones.

    The prompt asks for at most three citations of one source in a section;
    the 1.0.5 run cited one blog 17 times in a single section. Reporting the
    violation (``_overused_citations``) made it visible but not true, so the
    excess markers are removed here. The prose is untouched: a claim either
    keeps its marker or loses it, it is never rewritten. Returns
    ``(content, removed_count)``.
    """
    seen = Counter()
    removed = 0
    # Consume one leading space with each marker: dropping " [Source 2]" at
    # the end of a sentence leaves "claim." rather than "claim .".
    marker = re.compile(r" ?\[Source (\d+)\]")

    def keep(match) -> str:
        nonlocal removed
        number = int(match.group(1))
        seen[number] += 1
        if seen[number] > cap:
            removed += 1
            return ""
        return match.group(0)

    return marker.sub(keep, content or ""), removed


def _strip_duplicate_heading(content: str, title: str) -> str:
    """Drop a leading H3 that just repeats the section heading.

    The assembler owns the ``##`` heading. Models sometimes open the body with
    the same title as ``###``, producing a section that says its own name
    twice before making a point.
    """
    text = (content or "").lstrip()
    first, _, rest = text.partition("\n")
    stripped = first.strip()
    if stripped.startswith("#") and stripped.lstrip("#").strip().lower() == title.strip().lower():
        return rest.lstrip()
    return content


def _unique_texts(values) -> List[str]:
    """Drop exact and paraphrased repeats, keeping first-appearance order.

    The same gap or contradiction frequently arrives from two passes with
    different wording; exact lowercase matching kept both and the report
    printed the same point twice.
    """
    kept: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if any(is_near_duplicate(text, other) for other in kept):
            continue
        kept.append(text)
    return kept


# Words that carry no evidence signal. Without this, a generic paragraph and a
# specific one match the same "this/with/that" and the wrong passage wins.
_EXCERPT_STOPWORDS = frozenset(
    """
    about above after again against also among because been before being below
    between both could does down during each from further have having here into
    itself more most must only other over same should some such than that their
    them then there these they this those through under until very were what
    when where which while with within would your
    """.split()
)

SECTION_TIMEOUT_PLACEHOLDER = "[Section generation timed out after {seconds:.0f}s]"


def _select_excerpt(text: str, focus: str, chars: int = 900) -> str:
    """The most relevant passage of a page, not its first 900 characters.

    Tavily's full text often opens with navigation, a cookie notice, or a
    generic definition; the paragraph that actually answers the question is
    further down. Taking the head made two sections quote the same opening
    line. Paragraphs are scored by how many distinct focus words they contain
    (stopwords ignored), and the best one wins only when it agrees on at
    least two words; otherwise the head is kept, which is predictable and
    never worse than the old behaviour.
    """
    raw = (text or "").strip()
    if len(raw) <= chars:
        return raw

    terms = {
        word for word in re.findall(r"[a-z]{4,}", (focus or "").lower())
        if word not in _EXCERPT_STOPWORDS
    }
    if not terms:
        return raw[:chars].strip()

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw) if part.strip()]
    best_index, best_hits = -1, 0
    for index, paragraph in enumerate(paragraphs):
        words = set(re.findall(r"[a-z]{4,}", paragraph.lower()))
        hits = len(words & terms)
        if hits > best_hits:
            best_index, best_hits = index, hits
    if best_index < 0 or best_hits < 2:
        return raw[:chars].strip()

    excerpt = paragraphs[best_index]
    for paragraph in paragraphs[best_index + 1:]:
        if len(excerpt) + len(paragraph) + 2 > chars:
            break
        excerpt = f"{excerpt}\n\n{paragraph}"
    return excerpt[:chars].strip()


def _body_only(report: str) -> str:
    """The report without its bibliography.

    ``citation_count`` matches every ``[Source N]``, and the Sources list
    prints one of those per entry, so counting the finished file reports more
    citations than the body actually makes.
    """
    head, separator, _ = report.partition("\n## Sources\n")
    return head if separator else report


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
        style: str = "comprehensive"
    ) -> str:
        """
        Generate comprehensive report using parallel section generation
        
        Args:
            state: Research state with all findings
            ui_manager: UI manager for progress tracking
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

        # Nothing to write and nothing to send a model. Every search was empty
        # or off-topic; asking for themes and a summary anyway produced a
        # confident report about the failure itself.
        if not state.findings:
            empty_summary = (
                "No usable findings were collected for this query. Every search "
                "returned no results, or results too weakly related to the "
                "question to use as evidence, so this report has nothing to "
                "summarise or cite. The questions the run could not answer are "
                "listed below.\n"
            )
            final_report = self._assemble_report(
                state, empty_summary, [], self._generate_gaps_section(state),
                source_entries,
            )
            state.final_report = final_report
            log.info(
                "report assembled with no findings: %s questions answered, "
                "%s unanswered, %s sources reviewed",
                len(state.answered_questions()), len(state.questions), len(source_entries),
            )
            if advance:
                advance(1, "Report assembled")
            return final_report

        # Step 1: Identify major themes through meta-analysis
        if ui_manager:
            ui_manager.set_status("Identifying report themes...")
        themes = await self._identify_themes(state)
        note("ok", f"Report themes identified in {time.perf_counter() - started:.0f}s")
        if advance:
            advance(1, "Report themes identified")

        # Step 2: Generate sections in parallel. ``audit`` is filled by the
        # support check; the header states what it examined, because a report
        # that silently deleted markers reads as more verified than it is.
        audit = {"checked": 0, "dropped": 0, "missed": 0}
        sections, failed = await self._generate_sections_parallel(
            state, themes, style, ui_manager, source_numbers, audit=audit
        )
        if failed:
            note("warn", f"{failed} of {len(sections) + failed} sections did not complete")
        for title, source_name, count in self._overused_citations(
            sections, source_entries
        ):
            note(
                "warn",
                f"{clip(title, 32)} cites {clip(source_name, 32)} {count} times "
                f"(cap {MAX_CITATIONS_PER_SOURCE})",
            )

        # Step 3: Generate executive summary
        if ui_manager:
            ui_manager.set_status("Writing executive summary...")
        summary_started = time.perf_counter()
        # The summary sees no evidence list of its own, so it may cite only
        # numbers a section already grounded: enough to keep its claims
        # traceable, without letting it invent a source the body never used.
        summary_numbers = cited_numbers("\n".join(s.content for s in sections))
        # Size the summary to the report instead of asking for 600-900 words
        # every time: with three short sections that request made the summary
        # as long as a section, so it restated the body rather than summarised
        # it. 450 is the ceiling because a summary that runs longer than the
        # shortest section is a second body, however well it is written.
        body_words = sum(len(s.content.split()) for s in sections)
        summary_target = max(250, min(450, int(body_words * 0.2)))
        executive_summary = await self._generate_executive_summary(
            state, summary_numbers, source_numbers, target_words=summary_target
        )
        executive_summary = normalize_citations(
            executive_summary,
            valid_numbers=set(source_numbers.values()),
            allowed_numbers=summary_numbers,
        )
        executive_summary, trimmed = _cap_source_citations(executive_summary)
        if trimmed:
            note("info", f"Summary: {trimmed} repeat citation(s) over the cap dropped")
        note("ok", f"Executive summary written ({time.perf_counter() - summary_started:.0f}s)")
        if advance:
            advance(1, "Executive summary complete")

        # Step 4: Uncertainties, gaps, and the paid-for cross-reference pass.
        # The cross-reference lives in its own section: burying agreement and
        # contradictions inside "Uncertainties and Knowledge Gaps" hid the
        # most interesting output of the run.
        gaps_section = self._generate_gaps_section(state)
        cross_reference_section = self._generate_cross_reference_section(state)

        # Step 5: Assemble the report. The bibliography is derived from the
        # body's citations after assembly, so it can never list unused pages.
        final_report = self._assemble_report(
            state, executive_summary, sections, gaps_section,
            source_entries, cross_reference_section=cross_reference_section,
            citation_audit=audit,
        )
        if advance:
            advance(1, "Report assembled")

        state.final_report = final_report
        log.info(
            "report assembled: %s sections, %s unique sources, %s citations, "
            "%s chars, %s failed, %.1fs total",
            len(sections), len(source_entries), citation_count(_body_only(final_report)),
            len(final_report), failed, time.perf_counter() - started,
        )

        return final_report
    
    async def _identify_themes(self, state: ResearchState) -> Dict:
        """
        Use LLM to identify major themes from all findings
        """
        # Stable ids travel with each finding so the theme prompt can point at
        # the finding it is about instead of a positional index. Indices are
        # what let a theme title mislabel its section: one extra or missing
        # finding shifts every number after it.
        findings_data = [
            {
                "id": f"F{index + 1}",
                "content": f.content,
                "confidence": f.confidence,
                "sources": f.sources[:2],
            }
            for index, f in enumerate(state.findings)
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
        source_numbers: dict = None,
        audit: Dict = None,
    ) -> tuple:
        """
        Generate report sections in parallel using asyncio.

        Returns ``(sections, failed_count)``. Sections are bounded by their
        own timeout rather than one shared deadline, so a single stall costs
        one section instead of the entire report. ``audit`` is an out-parameter
        the caller owns; when given, it collects ``checked`` and ``dropped``
        citation counts across sections so the report can state what the
        support check actually examined.
        """
        tasks = []
        theme_list = themes.get("themes") if isinstance(themes, dict) else None
        if not isinstance(theme_list, list):
            theme_list = []

        # A theme names the findings that make the same point. Writing one
        # section per finding repeated that point in several sections — the
        # model writes "perplexity is the proxy metric" three ways. A theme
        # covering two findings becomes one section built from both; a finding
        # no theme names keeps a section of its own.
        by_id = {f"F{index + 1}": index for index in range(len(state.findings))}
        groups = []
        assigned = set()
        for theme in theme_list:
            if not isinstance(theme, dict):
                continue
            # Preferred: the finding ids the prompt asked for. Older/creative
            # replies may still send positional indices, so accept those too.
            ids = theme.get("related_finding_ids")
            indices = []
            if isinstance(ids, list):
                for value in ids:
                    key = str(value).strip().upper()
                    index = by_id.get(key)
                    if index is not None and index not in indices \
                            and index not in assigned:
                        indices.append(index)
            raw_indices = theme.get("related_findings")
            if isinstance(raw_indices, list):
                for value in raw_indices:
                    index = int(value) if str(value).isdigit() else -1
                    if 0 <= index < len(state.findings) and index not in indices \
                            and index not in assigned:
                        indices.append(index)
            if not indices:
                continue
            assigned.update(indices)
            groups.append((theme.get("title"), theme.get("description"), indices))

        for index in range(len(state.findings)):
            if index not in assigned:
                groups.append((None, None, [index]))
        groups.sort(key=lambda group: min(group[2]))

        def finding_numbers(finding) -> list:
            """Report-wide numbers for a finding's supporting URLs."""
            numbers = []
            for url in finding.sources:
                number = number_for(url, source_numbers or {})
                if number and number not in numbers:
                    numbers.append(number)
            return numbers

        used_titles = set()

        def unique_title(preferred, fallback: str) -> str:
            """A heading no other section uses.

            Several findings can belong to the same question, and a theme
            reply may name one finding twice. Falling back to the question
            title alone then produced two sections with the same heading, so
            the question is the second choice and a number is the last.
            """
            candidates = (_clean_title(preferred), _clean_title(fallback))
            for candidate in candidates:
                if candidate and candidate not in used_titles:
                    used_titles.add(candidate)
                    return candidate
            base = candidates[0] or candidates[1] or "Section"
            suffix = 2
            while f"{base} ({suffix})" in used_titles:
                suffix += 1
            title = f"{base} ({suffix})"
            used_titles.add(title)
            return title

        # Two passes: every section needs to know the other headings before
        # any of them is written, or each one repeats what its neighbours say.
        planned = []
        for preferred, description, indices in groups:
            # A section with more than three findings stops being an argument
            # and becomes a second executive summary, so long groups split.
            for start in range(0, len(indices), MAX_FINDINGS_PER_SECTION):
                chunk = indices[start:start + MAX_FINDINGS_PER_SECTION]
                finding = state.findings[chunk[0]]
                question = (
                    state.get_question_by_id(finding.question_id)
                    if finding.question_id else None
                )
                fallback_title = (
                    question.question if question else f"Section {chunk[0] + 1}"
                )
                title = unique_title(preferred, fallback_title)
                planned.append((chunk, title, description or title))
        all_titles = [title for _, title, _ in planned]

        # The run started with a guessed total because the section count was
        # unknown. Now it is: one theme pass, one step per section, the summary
        # and assembly. Raising the total here keeps the bar honest instead of
        # leaving it at ~70% until a final jump to 100%.
        set_total = getattr(ui_manager, "set_total", None)
        if callable(set_total):
            renderer = getattr(ui_manager, "renderer", ui_manager)
            done = float(getattr(renderer, "completed_steps", 0.0) or 0.0)
            set_total(int(done) + len(planned) + 2)

        claimed_extras: set = set()
        for i, (chunk, title, description) in enumerate(planned):
            mapped_findings = [state.findings[index] for index in chunk]

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
            # Evidence the section may cite: the finding's own sources plus the
            # pages read for the same questions that the finding never quoted.
            # Without the extras a one-source finding yields a one-source
            # section, no matter what the prompt asks for.
            # Extras already handed to an earlier section are reused only
            # when there is nothing unclaimed left: two sections built on the
            # same six pages write the same paragraph twice.
            evidence_sources = self._evidence_sources(
                state, mapped_findings, source_numbers, avoid=claimed_extras
            )
            claimed_extras.update(
                source["number"] for source in evidence_sources
            )
            allowed_numbers = {
                number
                for entry in relevant_findings
                for number in entry.get("source_numbers", [])
            } | {source["number"] for source in evidence_sources}
            log.debug(
                "section %r evidence: %s findings, %s sources, numbers=%s",
                title, len(relevant_findings), len(evidence_sources),
                sorted(allowed_numbers),
            )

            task = self.llm.generate_section_async(
                section_title=title,
                theme_description=description,
                relevant_findings=relevant_findings,
                main_query=state.main_query,
                style=style,
                evidence_sources=evidence_sources,
                other_sections=[other for other in all_titles if other != title],
            )
            tasks.append((
                i, title, task, allowed_numbers, section_urls, evidence_sources
            ))
        
        # Execute tasks with concurrency limit
        sections = []
        semaphore = asyncio.Semaphore(MAX_PARALLEL_SECTIONS)

        def note(level: str, message: str) -> None:
            log_method = getattr(ui_manager, "log", None)
            if callable(log_method):
                log_method(level, message)
            elif ui_manager is not None:
                ui_manager.add_message(message, style="white")

        async def generate_with_progress(
            order, title, task, section_numbers, section_urls, evidence_sources
        ):
            async with semaphore:
                started = time.perf_counter()
                note("info", f"Writing section: {clip(title)}")
                try:
                    content = await asyncio.wait_for(task, timeout=SECTION_TIMEOUT)
                except asyncio.TimeoutError:
                    log.warning("section %r timed out after %.0fs", title, SECTION_TIMEOUT)
                    note("warn", f"Section timed out after {SECTION_TIMEOUT:.0f}s: {clip(title)}")
                    return (order, title, SECTION_TIMEOUT_PLACEHOLDER.format(seconds=SECTION_TIMEOUT),
                            section_urls, True, 0, 0)
                except Exception as e:
                    log.error("section %r failed: %s", title, e, exc_info=True)
                    note("err", f"Section failed: {clip(title)} ({clip(str(e), 60)})")
                    return order, title, f"[Section generation failed: {e}]", section_urls, True, 0, 0
                duration = time.perf_counter() - started
                # Every marker in the body must resolve to a source that was
                # actually supplied for this section and exists in the report
                # bibliography. Malformed or invented numbers are dropped.
                content = normalize_citations(
                    content,
                    valid_numbers=set((source_numbers or {}).values()),
                    allowed_numbers=section_numbers,
                )
                content = _strip_duplicate_heading(content, title)
                # The check is a blocking model call. Run it on a worker thread:
                # calling it directly stalled the event loop for its whole
                # duration, so no other section could finish meanwhile.
                content, checked, unsupported = await asyncio.to_thread(
                    self._check_support, content, title, evidence_sources
                )
                if unsupported:
                    log.info(
                        "section %r: dropped %s of %s checked citation(s) as unsupported",
                        title, unsupported, checked,
                    )
                content, trimmed = _cap_source_citations(content)
                if trimmed:
                    log.info(
                        "section %r: dropped %s citation(s) over the cap of %s per source",
                        title, trimmed, MAX_CITATIONS_PER_SOURCE,
                    )
                log.info("section %r done in %.1fs (%s chars, %s citations)",
                         title, duration, len(content), citation_count(content))
                note("ok", f"Section ready in {duration:.0f}s: {clip(title)}")
                advance = getattr(ui_manager, "advance", None)
                if callable(advance):
                    advance(1, f"Completed section: {clip(title, 40)}")
                return order, title, content, section_urls, False, checked, unsupported

        results = await asyncio.gather(*[
            generate_with_progress(order, title, task, numbers, urls, evidence)
            for order, title, task, numbers, urls, evidence in tasks
        ], return_exceptions=False)

        failed = 0
        for order, title, content, section_urls, failed_section, checked, dropped in sorted(
            results, key=lambda x: x[0]
        ):
            failed += 1 if failed_section else 0
            if audit is not None:
                # A section that could not be audited is counted, not dropped:
                # its citations left the denominator would make a partial
                # audit read as a complete one.
                if checked is None:
                    audit["missed"] = audit.get("missed", 0) + 1
                else:
                    audit["checked"] = audit.get("checked", 0) + checked
                    audit["dropped"] = audit.get("dropped", 0) + dropped
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
        allowed_numbers: set = None,
        registry: dict = None,
        target_words: int = 600,
    ) -> str:
        """
        Generate executive summary once all sections are complete.

        Written from the findings rather than the section prose: feeding the
        section openings in made the model echo them back as the summary.
        ``allowed_numbers`` is what the sections cited; ``registry`` maps URLs
        to report-wide numbers so each finding can say where it came from.
        """
        # Full-enough findings for a 600-900 word summary. The source numbers
        # travel with them, or the summary has nothing to cite.
        key_findings = []
        for finding in sorted(state.findings, key=lambda x: x.confidence, reverse=True)[:10]:
            numbers = []
            for url in finding.sources:
                number = number_for(url, registry or {})
                if number and number not in numbers:
                    numbers.append(number)
            label = "".join(f"[Source {number}]" for number in numbers)
            key_findings.append(clip(finding.content, 400) + (f" {label}" if label else ""))
        
        summary = await asyncio.to_thread(
            self.llm.generate_executive_summary,
            state.main_query,
            key_findings,
            allowed_numbers,
            target_words=target_words,
        )

        return summary
    
    def _check_support(self, content: str, title: str, evidence: List[Dict]) -> tuple:
        """Drop markers the cited excerpt does not support.

        One model call per section, and only when the section cites anything.
        The checker names marker ids (M1, M2...) so the prose is never
        rewritten. It fails open: a transport or parse error keeps every
        marker, because an unverified citation is a smaller problem than a
        silently unsupported report that looks verified. Returns
        ``(content, checked, dropped)``, where ``checked`` is None when the
        section could not be audited at all — different from 0, "there was
        nothing to check", because the report header must not add up a
        partial audit as if it were whole.
        """
        checker = getattr(self.llm, "check_citation_support", None)
        if not callable(checker):
            return content, None, 0
        claims = _cited_claims(content)
        if not claims:
            return content, 0, 0
        try:
            rejected = checker(
                [
                    {"id": c["id"], "source": c["source"], "sentence": c["sentence"]}
                    for c in claims
                ],
                evidence,
            )
        except Exception as e:  # noqa: BLE001 - fail open, log why
            log.warning("citation support check failed for %r: %s", title, e)
            return content, None, 0

        wanted = {
            str(item.get("id", "") if isinstance(item, dict) else item)
            .strip().upper().lstrip("M")
            for item in (rejected or [])
        }
        spans = [
            claim["span"] for claim in claims
            if claim["id"].lstrip("M") in wanted
        ]
        if not spans:
            return content, len(claims), 0
        return _drop_marker_spans(content, spans), len(claims), len(spans)

    @staticmethod
    def _overused_citations(sections: List[ReportSection], entries: list) -> List[tuple]:
        """Post-condition check on the citation cap.

        ``_cap_source_citations`` already drops the excess, so a hit here
        means the enforcement itself missed something (a new call path, a
        raised cap). It is kept as a tripwire, not as a report about model
        behaviour. The source is named by title because this runs before the
        bibliography is renumbered.
        """
        names = {
            entry.number: entry.title or entry.url for entry in (entries or [])
        }
        overused = []
        for section in sections:
            counts = Counter(
                int(number) for number in _CITATION_MARKER_RE.findall(section.content)
            )
            for number, count in counts.items():
                if count > MAX_CITATIONS_PER_SOURCE:
                    overused.append((
                        section.title, names.get(number, f"Source {number}"), count
                    ))
        return overused

    @staticmethod
    def _evidence_sources(
        state: ResearchState,
        findings: List,
        source_numbers: dict,
        limit: int = 6,
        excerpt_chars: int = 900,
        avoid: set = None,
        reuse_until: int = 4,
    ) -> List[Dict]:
        """Numbered page excerpts for one section, the finding's own first.

        A finding records which pages supported it as a block of numbers but
        not the text behind each number, so a model handed only the block
        pastes it onto sentence after sentence. One excerpt per number gives
        the model something to match a claim against; the rest of the list is
        pages the analysis read for the same questions but never quoted.

        *avoid* names extras another section already received. They are used
        last, and only until the section has *reuse_until* excerpts: two
        sections handed the same pages produce the same prose, but a section
        with one source cannot hold an argument either.
        """
        question_ids = {
            finding.question_id for finding in findings if finding.question_id
        }
        own_numbers = []
        for finding in findings:
            for url in finding.sources:
                number = number_for(url, source_numbers or {})
                if number and number not in own_numbers:
                    own_numbers.append(number)

        by_number = {}
        for result in state.search_results:
            number = number_for(result.url, source_numbers or {})
            if number and number not in by_number:
                by_number[number] = result

        selected = []
        seen_text = set()

        focus = " ".join(finding.content for finding in findings)

        def take(number: int, result) -> bool:
            # The relevant paragraph, not the head of the page.
            text = _select_excerpt(
                result.full_content or result.content or "", focus, excerpt_chars
            )
            if not text or text in seen_text:
                return False
            seen_text.add(text)
            selected.append({
                "number": number,
                "title": result.title or result.url,
                "type": source_type_label(source_type(getattr(result, "url", ""))),
                "content": text,
            })
            return len(selected) >= limit

        for number in sorted(own_numbers):
            result = by_number.get(number)
            if result is not None and take(number, result):
                return selected

        extras = [
            (number, result)
            for number, result in by_number.items()
            if number not in own_numbers
            and getattr(result, "analyzed", False)
            and (not question_ids or result.question_id in question_ids)
        ]
        extras.sort(
            key=lambda item: (
                getattr(item[1], "quality", 0.0),
                getattr(item[1], "score", 0.0),
            ),
            reverse=True,
        )
        avoid = set(avoid or ())
        unclaimed = [item for item in extras if item[0] not in avoid]
        reused = [item for item in extras if item[0] in avoid]
        for number, result in unclaimed:
            if take(number, result):
                return selected
        # Reuse stops at the floor, not the limit: topping every section up to
        # six shared pages is how two sections came out as the same paragraph.
        floor = min(limit, max(1, int(reuse_until)))
        for number, result in reused:
            if len(selected) >= floor:
                break
            if take(number, result):
                break
        return selected

    @staticmethod
    def _unanswered_questions(state: ResearchState) -> List:
        """Planned questions with no answer: failed, pending, or interrupted."""
        return [
            question for question in state.questions
            if question.status in (
                QuestionStatus.PENDING,
                QuestionStatus.IN_PROGRESS,
                QuestionStatus.FAILED,
            )
        ]

    def _generate_gaps_section(self, state: ResearchState) -> str:
        """Uncertainties and knowledge gaps, plus the questions never run.

        Consensus and contradictions used to be appended here, which buried
        the most useful cross-reference output under a heading that says
        "gaps". They are their own section now. A planned question that the
        iteration budget never reached is listed explicitly: otherwise a
        reader cannot tell "nothing was found" from "nothing was looked for".
        """
        critical_gaps = _unique_texts(
            gap.description for gap in state.gaps if gap.severity == "critical"
        )
        moderate_gaps = [
            text for text in _unique_texts(
                gap.description for gap in state.gaps if gap.severity == "moderate"
            )
            if not any(is_near_duplicate(text, kept) for kept in critical_gaps)
        ][:5]
        minor_gaps = [
            text for text in _unique_texts(
                gap.description for gap in state.gaps if gap.severity == "minor"
            )
            if not any(
                is_near_duplicate(text, kept)
                for kept in critical_gaps + moderate_gaps
            )
        ][:5]
        not_researched = self._unanswered_questions(state)

        if not (critical_gaps or moderate_gaps or minor_gaps or not_researched):
            return "No significant knowledge gaps were identified during this research.\n"

        content = ""
        if critical_gaps:
            content += "### Critical gaps\n\n"
            for text in critical_gaps:
                content += f"- {text}\n"
            content += "\n"

        if moderate_gaps:
            content += "### Moderate gaps\n\n"
            for text in moderate_gaps:
                content += f"- {text}\n"
            content += "\n"

        if minor_gaps:
            content += "### Minor gaps\n\n"
            for text in minor_gaps:
                content += f"- {text}\n"
            content += "\n"

        if not_researched:
            content += "### Not answered\n\n"
            content += (
                "The run did not produce an answer for these planned questions "
                "(the search failed or the iteration budget ran out), so they "
                "are not separately covered here:\n\n"
            )
            for question in not_researched[:8]:
                content += f"- {question.question}\n"
            content += "\n"

        return content

    def _generate_cross_reference_section(self, state: ResearchState) -> str:
        """Agreement, contradictions, and source reliability as one section."""
        cross = state.cross_reference if isinstance(state.cross_reference, dict) else {}
        if not cross:
            return ""

        content = ""
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
        # later cross-reference pass. Exact de-duplication missed the same
        # point phrased two ways, so the near-duplicate check collapses both.
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

        unique_contradictions = _unique_texts(contradictions)
        if unique_contradictions:
            content += "### Areas of contradiction\n\n"
            for contradiction in unique_contradictions[:5]:
                content += f"- {contradiction}\n"
            content += "\n"

        # The pass already paid for an overall confidence; show it as the
        # model's own rating, not as a property of the evidence.
        try:
            confidence = float(cross.get("overall_confidence"))
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            content += "### Overall confidence\n\n"
            content += (
                "The research pass rated its own confidence in these "
                f"conclusions at {max(0.0, min(1.0, confidence)):.0%}.\n\n"
            )

        reliability = str(cross.get("reliability_assessment") or "").strip()
        if reliability:
            content += "### Source reliability\n\n"
            content += reliability + "\n\n"

        return content
    
    def _render_sources(self, entries: List[SourceEntry], reviewed: int = 0) -> str:
        """Reference list for exactly the sources the body cites.

        A bibliography lists the works the text refers to. Pages that were
        collected or analysed but never used are summarised in one closing
        note instead of padding the list with unused entries.
        """
        if not entries:
            return "No sources were cited in this report.\n"

        lines = []
        for source in entries:
            title = source.title or source.url
            # The numbered entry *is* the citation target, so a literal
            # "[Source 1]" next to "1." only repeats itself. The kind of page
            # is what a reader can act on; the internal quality and relevance
            # scores belong in the run log, not in a finished document.
            lines.append(f"[Source {source.number}] **{title}**")
            lines.append(f"   <{source.url}>")
            if source.published_date:
                lines.append(f"   Published: {source.published_date}")
            lines.append(f"   {source_type_label(source.source_type)}")
            lines.append("")

        extra = max(0, reviewed - len(entries))
        if extra:
            noun = "source was" if extra == 1 else "sources were"
            lines.append(
                f"*{extra} additional {noun} reviewed but not cited in this report.*"
            )
        return "\n".join(lines).rstrip() + "\n"
    
    def _assemble_report(
        self,
        state: ResearchState,
        executive_summary: str,
        sections: List[ReportSection],
        gaps_section: str,
        source_entries: list = None,
        cross_reference_section: str = "",
        citation_audit: tuple = None,
    ) -> str:
        """Assemble the components as one properly structured Markdown file."""
        from datetime import datetime

        model = getattr(self.llm, "model", "") or "unknown"
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        ordered_sections = sorted(sections, key=lambda section: section.order)
        body_for_count = "\n".join(
            [executive_summary, gaps_section, cross_reference_section]
            + [section.content for section in ordered_sections]
        )
        cited = cited_numbers(body_for_count)
        reviewed = len(source_entries or [])
        meta_parts = [
            f"Model: `{model}`",
            f"Iterations: {state.iteration_count}",
        ]
        # Coverage up front: a run that planned four questions and answered
        # three should say so in the header, not leave the reader to infer it
        # from a gaps bullet.
        if state.questions:
            answered = sum(
                1 for question in state.questions
                if question.status is QuestionStatus.COMPLETED
            )
            answered_label = f"Questions: {answered} answered of {len(state.questions)}"
            failed = sum(
                1 for question in state.questions
                if question.status is QuestionStatus.FAILED
            )
            if failed:
                answered_label += f" ({failed} failed)"
            meta_parts.append(answered_label)
        meta_parts.append(f"Sources: {len(cited)} cited of {reviewed} reviewed")
        meta_parts.append(f"Citations: {citation_count(body_for_count)}")
        # Only when the check actually ran and examined something: an outage
        # reports zero checked, so the header never implies an audit it did
        # not perform. The summary is not part of this count.
        # One number per citation marker, and only when every section was
        # actually audited: a partial audit summed into one line would claim
        # more coverage than the run had. "section citations" is literal —
        # the executive summary is not part of this check.
        audit = citation_audit or {}
        checked = int(audit.get("checked") or 0)
        dropped = int(audit.get("dropped") or 0)
        missed = int(audit.get("missed") or 0)
        if checked and not missed:
            meta_parts.append(
                f"Source check: {checked - dropped} of {checked} "
                "section citations verified"
            )
        meta_parts.append(f"Generated: {generated}")
        meta = "  ·  ".join(meta_parts)

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

        # Consensus and contradictions are their own section; they were
        # previously the tail of the gaps section, where nobody looks for them.
        if cross_reference_section.strip():
            report += "---\n\n## Consensus and Contradictions\n\n"
            report += cross_reference_section.strip() + "\n\n"

        # Add gaps section
        report += "---\n\n## Uncertainties and Knowledge Gaps\n\n"
        report += gaps_section.strip() + "\n\n"

        # The bibliography lists what the body actually cites, renumbered from
        # 1 so the reference list has no gaps and no unused padding.
        report, cited_entries = renumber_citations(report, source_entries or [])

        report += "---\n\n## Sources\n\n"
        report += self._render_sources(cited_entries, reviewed).strip() + "\n"
        
        return report
