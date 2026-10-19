# Changelog

Notable changes per release. This project follows [Semantic Versioning](https://semver.org/).

## [1.0.0]

- First public release: iterative web research (Tavily) driven by any
  OpenAI-compatible endpoint, with a keyboard-driven TUI and a Rich CLI,
  a setup wizard, and Markdown report output.
- Single install: the distribution is `veritas-ai`; `pipx install veritas-ai`
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
