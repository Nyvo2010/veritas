<p align="center"><img src="https://raw.githubusercontent.com/Nyvo2010/veritas/main/assets/logo.svg" alt="Veritas" width="560"></p>

<p align="center"><sub>Made by Nyvø w/ love · Powered by Tavily</sub></p>

# Veritas — Deep Research Agent

Veritas is a terminal app that turns a question into a sourced research report.
It decomposes your query, searches the web iteratively (Tavily), reasons over
findings with an LLM, and writes a Markdown report to your reports folder — all
behind a quiet, keyboard-driven TUI.

Any **OpenAI-compatible endpoint** works as the research brain (NVIDIA NIM is
the default). The model list is auto-detected from the endpoint's `/models`.

## Install

Requires Python 3.10+ (3.12/3.13 for the TUI).

```bash
pip install "veritas[tui]"
veritas
```

`tui` is the full keyboard-driven interface (prebuilt wheels for CPython
3.12/3.13 on macOS, Linux, Windows). Without it you get the classic Rich CLI
with the same features:

```bash
pip install veritas
```

From source:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tui]"
veritas
```

## First run

A setup wizard walks you through: AI endpoint URL → API key → model → Tavily
key → reports folder → color theme. Re-run it anytime via
Settings → Setup wizard.

Keys: `↑`/`↓` move · `Enter` select · `Esc` back · `q` quit (menu) ·
`Ctrl+C` quit (anywhere). Paste works in every input field.

Long lists scroll, with `↑ N more` / `↓ N more` markers so the highlighted row
is always visible. Model pickers also accept typing: start typing to filter the
`/models` list, or type an id that isn't listed and press `Enter` to use it.

## Settings vs environment variables

Everything can be configured in-app under Settings, stored in
`~/.config/veritas/settings.json` (mode `0600`, keys masked on screen).

- **AI provider** — endpoint, API key, model, and an optional fallback model
- **Search** — Tavily API key and the default source categories
- **Research defaults** — iterations and results per search

A `.env` file (see `.env.example`) is the alternative path for non-interactive
use — local dev, containers, CI, headless servers — or for temporarily
overriding the file without touching it. Precedence everywhere:

> explicit argument > environment variable > settings file > built-in default

Supported variables: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`,
`LLM_FALLBACK_MODEL`, `TAVILY_API_KEY` (legacy `NVIDIA_API_KEY` /
`NVIDIA_API_BASE_URL` still work as fallback). `VERITAS_REPORTS_DIR` overrides
the reports folder.

If the primary model exhausts its retries, the client switches to the fallback
model once and keeps using it for the rest of the run. Leave it empty to
disable.

## Layout

```
src/
  cli.py           entry point (`veritas`), Rich fallback CLI
  tui_app.py       OpenTUI frontend: menu, wizards, settings, live run screen
  main.py          research orchestrator (agents + clients)
  settings.py      persistent settings, key resolution, legacy migration
  config.py        env-based defaults and tuning constants
  agents/          query analysis, search, synthesis, report generation
  utils/llm_client.py    any OpenAI-compatible chat API + /models discovery
  utils/tavily_client.py web search
  ui/              theme tokens, live renderer, shared screens
```

Reports are written to `~/Veritas/reports` by default (override in Settings or
with `VERITAS_REPORTS_DIR`); nothing is written into the repository.

## License

MIT — see [LICENSE](LICENSE).
