# Changelog

Notable changes per release. This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

- Cleanup pass, no behaviour change. Removed state nothing read (the unused
  `silent` argument on `set_category_toggle`, `ResearchGap.suggested_searches`
  — always empty, `QueueUI.done_ok`, `LLMClient.using_fallback`), and the
  `_seconds` wrapper, which was `theme.format_duration` plus a rounding
  difference at 59.5s.
- One source for the UI vocabulary: both frontends take log glyphs from
  `ui/theme.py`, and a legacy `add_message(text, style)` call maps to a level
  through one shared `level_for_style`, so a message cannot render red in one
  frontend and plain in the other. Section progress lines now reach the file
  log like every other report line.

- The citation support check is given the whole sentence a marker belongs to.
  Sentence starts were found by taking everything after the last full stop,
  which lands inside `0.01%`, `e.g.` and `et al.`, so the checker was asked to
  judge fragments like `[Source 9] on A100.` and rejected citations the
  sources did support — one 2026-10-21 run lost 81 of 119 markers that way.
  Claims are now cut at sentence ends and line breaks only, widen to the
  previous sentence when a marker leads its own sentence, never cross a line,
  and stop between words rather than mid-word at the 400-character cap.

- The citation support check runs on a worker thread. It is a blocking model
  call, and running it on the event loop held up every other section that was
  still waiting for its reply: a three-iteration run spent 19s in report
  generation where the same work took 53s.
- The report header states what that check examined — `Source check: 28 of 48
  section citations verified`. It is left out when any section could not be
  audited, so a partial audit is never summed up as a whole one, and a reply
  that is not a verdict counts as a failure rather than as "nothing rejected".
- The ACL Anthology, ACL, OpenReview, JMLR, NeurIPS, ICLR, AAAI and IJCAI
  hosts are classified as academic. A peer-reviewed ACL paper used to print as
  "Web page" and rank with the web prior.
- The run screen leaves one blank row between the progress card and the
  "Ctrl+C quits" hint, so the hint reads as navigation instead of as the
  card's last line.
- The run screen's three cards share one padding rule: a blank row above the
  title and below the last line, so no text touches the tinted edge. The
  progress bar gets a blank row of its own above and below, the blank line
  that separates phase groups now sits above a heading instead of below it,
  and the activity log no longer opens on a stray blank row.
- A search whose best result is only weakly related to the question is now
  treated as empty, and individual results below the relevance floor are
  dropped before analysis. One run returned six pages about the 1995 film
  *Heat* for a question about heat pumps, and the report was written from
  them; the failure is now reported and the question stays unanswered.
- Sources are classified by kind (academic, institution, news, reference,
  docs, video, forum, web) and the evidence score is multiplied by that prior.
  The bibliography shows the kind instead of `General`; the internal quality
  and relevance numbers stay in the run log. A YouTube clip and a paper no
  longer both read as 0.93.
- Fetched page text is cleaned before it reaches a prompt: image markup, link
  URLs, HTML tags and one-line cookie/navigation boilerplate are stripped, so
  the analysis budget is spent on evidence instead of page chrome.
- The report header states question coverage (`3 answered of 5`), and planned
  questions that failed or were never reached are listed under **Not
  answered** instead of being silently absent.
- Consensus and contradictions are their own section instead of the tail of
  "Uncertainties and Knowledge Gaps"; gaps and contradictions that restate
  each other are printed once.
- The executive summary is sized against the body instead of always asking
  for 600-900 words, capped at 450 words, and told to stay one piece of prose
  rather than a set of `###` subsections. A section built on thin evidence is
  asked for a shorter length rather than being padded to 1,400 words.
- Section headings are sentence-cased, so a theme pass that returns a raw
  question no longer prints a lowercase heading; acronyms and proper nouns
  keep their own casing. A trailing full stop is dropped.
- A section prefers pages no earlier section was given and only reuses shared
  pages until it has four excerpts, so two sections built on one question stop
  writing the same paragraph.
- A run that produced no findings writes a short honest report instead of
  asking the model to summarise nothing.
- A section's citations are checked against the excerpts it was given. A
  claim whose cited page does not support it loses the marker (never the
  sentence), the check fails open if the judge is unavailable, and only
  sentences that already cite something are sent. One measured run dropped 23
  unsupported markers and exposed a paragraph of vendor benchmark scores that
  had been citing a blog.
- Evidence excerpts are chosen from the most relevant passage of a page
  instead of its first 900 characters. Pages that open with navigation or a
  generic definition no longer make two sections quote the same line; a page
  with no relevant passage keeps its head.
- The cross-reference section shows the model's own confidence rating and the
  gaps list includes minor gaps, both of which the run paid for and then
  dropped. The unused `source_agreement` field is no longer requested.
- The run log no longer records HTTP transport frames and response headers
  (`httpcore2`/`httpx2`), which were most of a run's log lines. The
  three-citations-per-source cap is now enforced: markers past the cap are
  dropped and counted in the log, instead of only being reported.
- A section body that opens with an H3 repeating its own heading no longer
  prints the title twice.
- Tavily's generated answer is no longer requested (it was never read), and
  the ignored `include_full_content` and `base_step` plumbing is gone.
- The run summary lists questions that were not researched, alongside
  answered and failed ones.

- Findings survive an analysis reply that leaves `answer` empty. Extraction
  used to sit inside `if answer:`, so a model that returned a populated
  `findings` array and no summary sentence lost every point in it.
- The progress total now tracks the work actually queued. It was the iteration
  count plus a fixed 12-step report allowance, so a six-section report crawled
  to ~70% and then snapped to 100%. The theme pass raises the total once the
  section count is known, and the end-of-run reconciliation no longer papers
  over the gap.
- The run screen shows what the run is doing and how much it has found. The
  orchestrator had been publishing "Iteration 2/3 - Analyzing 9 sources..." on
  every step and the TUI never drew it; that line and a live `answers ·
  findings · sources` row now sit under the phase. The Rich CLI's progress
  panel carries the same counters.
- The Rich CLI run screen keeps its fixed height on narrow terminals: activity
  lines and the header are clipped to the panel instead of wrapping into the
  progress bar, and the Done panel is no longer printed flush against the
  retired frame.
- The run log reports body citations only. `citation_count` matched the
  `[Source N]` entries in the bibliography as well, so a 90-citation report was
  logged as 99.
- `Iteration N of M` no longer reads "3 of 2": follow-ups enlarge the queue
  mid-run, and the label was captured from its initial length while the loop
  was capped by `max_iterations`.

## [1.0.4] - 2026-10-21

- One analysis pass can now produce several findings, one per distinct point
  in the answer, instead of collapsing a question into a single finding. A
  two-iteration run that used to yield two sections now writes four to six,
  each about a different mechanism. Replies that ignore the new `findings`
  array still work as before; points that restate each other are stored once.
- Findings that make the same point share one section. The theme pass may now
  group findings, so a report no longer carries two sections on perplexity
  written from two different questions, and a group larger than three findings
  is split rather than becoming a second executive summary.
- Each section is told the other sections' headings, and every subsection is
  required to carry a citation, so neighbours stop re-arguing the same point
  and a paragraph of plausible uncited prose is not written at all.
- Questions are deduplicated before they are queued: decomposition and
  follow-ups that paraphrase a queued question are dropped instead of being
  researched and written up twice.
- The same page is one source even when it is republished elsewhere or served
  as both HTML and PDF. Duplicate detection now compares canonical URLs, the
  opening text, the article title, and the words of the whole page, so the
  bibliography no longer lists one paper twice (arXiv HTML and ACL PDF).
- Prompt sizes are bounded: theme analysis reads at most 40 findings and 500
  characters each, cross-reference at most 30 — a run with more findings can
  no longer grow those prompts without limit.
- `[Source 1], [Source 2]` is normalised to `[Source 1][Source 2]`.
- New offline end-to-end report test drives the real orchestrator, synthesis
  and report generator with a stubbed model, then asserts the structural
  contract: every citation resolves, the bibliography is contiguous, headings
  are unique, no placeholder text survives, and the summary cites only what
  the body cited. It runs in CI with the rest of the suite (179 tests).

## [1.0.3] - 2026-10-20

- Report themes point at findings by id instead of by position, so a theme title
  can no longer land on the wrong section when a finding is added or dropped
  (the earlier fusion run had a "Stellarator Geometry" section about deployment
  timelines). Replies that still send positional indices keep working.
- A finding no longer claims the first five search results as its support when
  the analysis named none. Nothing is invented; the section still receives the
  analysed pages for its question as citable evidence.
- Reports are written through a private temporary file and renamed into place,
  so a crash mid-write cannot truncate an existing report.
- A question whose search or analysis failed stays `FAILED` when the run moves
  on; `answered` and `failed` counts now agree with what happened.
- Failed searches are counted in `total_searches`, so a run that hit network
  errors no longer reports that it made no searches.
- The progress bar is one row: the completed part is a solid, gap-free,
  non-animated block, and only the remaining track keeps the drifting
  particles. Both the TUI and the Rich CLI use the same bar.
- CI runs the suite on Python 3.12, 3.13 and 3.14, and a second job builds the
  wheel, installs it into a clean venv and boots `veritas --version` — the
  packaging path the editable install never exercised.

## [1.0.2] - 2026-10-20

- Fixed the crash that broke every run outside the TUI: the orchestrator calls
  `ui.set_progress()`, but `UIManager` (the object the classic CLI passes when
  no UI is supplied) only defined `update_progress`. On Python 3.14, where
  `opentui` is not installed and the CLI is the only interface, research could
  not start at all. The CLI path is now covered by a test that drives
  `research()` with the real `UIManager`.
- Reports write one section per finding instead of re-using the same two or
  three findings across six to ten themes. A three-iteration run now produces
  three distinct sections rather than seven that paraphrase each other.
- Section evidence is widened and actually citable. Each section receives one
  excerpt per numbered page — the finding's own sources first, then pages the
  analysis read for the same questions but never quoted — and citation
  validation accepts every number it was shown. Before, any marker outside the
  finding's own block was stripped, so a one-source finding could only ever
  produce a one-source section.
- A finding's source numbers are listed after its text rather than as a block
  before it. The leading block was coming back pasted onto nearly every
  sentence; the section prompt now asks for a marker only on the sentence an
  excerpt supports, with a cap on how often one source may be cited.
- The bibliography lists only the pages the body cites, renumbered 1..k in
  first-appearance order, so it can never gap or point at an unused page. The
  header reads `Sources: X cited of Y reviewed` and one closing line accounts
  for the rest.
- The executive summary is written from the findings rather than from section
  previews, which it had been echoing back verbatim, and it still receives the
  findings' source numbers so its claims stay cited. The analysis prompt asks
  for every source that supports an answer (usually two to four) instead of
  whichever one the model named first.

## [1.0.1] - 2026-10-19

- The progress bar is a two-row braille wave: a rippling filled band with a
  rising meniscus and a denser field of drifting particles, replacing the
  single hairline row.
- The bibliography now records how far each source actually got — cited in the
  report, analyzed but unused, or collected only — and the header reports the
  number of distinct sources cited next to the citation count. A long source
  list no longer implies grounding it never provided.
- The executive summary carries `[Source N]` citations, limited to numbers the
  sections already cited, so its claims are traceable without letting it
  introduce evidence of its own.
- Section prompts spread citations across the distinct sources that support
  each point instead of leaning on one or two.
- High decomposition priority no longer forces Tavily's `advanced` depth (it
  made every search advanced); the planner's explicit `search_depth` request
  is now honoured, and the cheap depth stays the default.
- The analysis model now receives a quality-ranked, domain-diverse subset of
  the usable search results (up to 2,000 characters each) instead of the first
  ten in engine order. Every usable source still appears in the bibliography.
- Cross-reference output (consensus, contradictions, reliability) and
  knowledge-gap reviews are persisted on the research state and rendered in
  the report instead of being discarded after the call.
- An empty search skips that question and lets the run continue; three empty
  searches no longer abort the whole research. Source publication dates are
  stored and shown when the provider supplies them. Reports now separate
  answered from failed questions instead of counting processed searches as
  answers.
- Searches default to Tavily's cheap `basic` depth and escalate only for
  planner-requested, time-sensitive, finance, long, or re-researched
  questions, so easy questions no longer pay the advanced rate.
- Site categories match on a host boundary: `netflix.com` is no longer treated
  as social because `x.com` is a substring, and long paywalled or blocked pages
  no longer pass source hygiene on length alone.
- Tavily auth, configuration, and usage-limit errors fail immediately instead
  of retrying until the backoff budget runs out, search attempts honour the
  configured timeout, and the final attempt no longer sleeps before failing.
- Model output is normalised at the boundary: string priorities, string
  booleans, and single strings where lists were requested no longer crash
  sorting or turn into one item per character.
- An explicit report filename whose parent folder cannot be created now reports
  `save_error` instead of raising.
- The "max search results" setting now reaches the search call (clamped to the
  provider's 1-20 range) instead of being stored and ignored, and settings are
  written through a private temporary file so API keys are never briefly
  world-readable and a crash cannot truncate the file.
- Unmapped report themes interleave findings instead of all repeating the
  first ten, and the executive summary may reuse source numbers cited by the
  sections while invented ones are still removed.
- Reports share one source registry: `[Source N]` markers, section prompts, and
  the Sources bibliography now use the same numbers. Unknown or out-of-section
  citations are removed instead of silently pointing at the wrong source.
- Search results are deduplicated by canonical URL and syndicated article text,
  and thin, blocked, login, and search-result pages are filtered out before
  analysis. Each source carries a deterministic evidence-quality score.
- Findings record the search results the model actually cited (`source_indices`)
  instead of assuming the first five results support the answer.
- Reports are assembled as clean Markdown (`#`/`##` headings, `---` rules) for
  readable default-app previews, with a citation legend in the header.
- The results screen is now an action menu: open the report in the default app,
  reveal it in Finder/Explorer, or view the activity log. The menu keeps the
  last report reachable, and the Rich CLI offers the same open/reveal options.

## [1.0.0]

- First public release: iterative web research (Tavily) driven by any
  OpenAI-compatible endpoint, with a keyboard-driven TUI and a Rich CLI,
  a setup wizard, and Markdown report output.
- Single install: the distribution is `veritas-research`; `pipx install veritas-research`
  pulls the full TUI where opentui wheels exist (CPython 3.12/3.13) and the
  Rich CLI elsewhere — no `[tui]` extra to choose.
- `veritas --version` reports the installed version.
- Report section generation runs through the sync client on a worker thread,
  avoiding an async TLS failure that emptied sections on some platforms.
- Report header carries model, iteration count, source count and timestamp.
- Sharper LLM handling: rate-limit detection no longer matches the `rate` in
  echoed `temperature`; unknown-model 404s fail fast; truncated JSON is
  salvaged; the section and summary prompts share one voice/format guide.
- `install.sh` accepts a branch, tag or commit as `VERITAS_REF`.
