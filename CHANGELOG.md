# Changelog

Notable changes per release. This project follows [Semantic Versioning](https://semver.org/).

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
