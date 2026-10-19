<p align="center"><img src="https://raw.githubusercontent.com/Nyvo2010/veritas/main/assets/mockup.png" alt="Veritas" width="560"></p>

<p align="center">
  <a href="https://github.com/Nyvo2010/veritas/actions/workflows/tests.yml"><img src="https://github.com/Nyvo2010/veritas/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="python 3.12+">
</p>

<p align="center"><sub>Made by Nyvø w/ love · Powered by Tavily</sub></p>

# Veritas — Deep Research Agent

Veritas is a terminal app that turns a question into a sourced research report.
It decomposes your query, searches the web iteratively (Tavily), reasons over
findings with an LLM, and writes a Markdown report to your reports folder — all
behind a quiet, keyboard-driven TUI.

Any **OpenAI-compatible endpoint** works as the research brain — there is no
bundled provider. You supply the base URL, an API key, and a model id. The
model list is auto-detected from the endpoint's `/models`.

## Install

Requires Python 3.12+. The full TUI uses opentui, whose wheels cover CPython
3.12/3.13; on other Python versions Veritas runs its Rich CLI, which is the
same program with the same features. Either way it is one install command.

**pipx** (recommended) — installs Veritas in its own environment and puts
`veritas` on your PATH, on macOS, Linux and Windows:

```bash
pipx install veritas-research
veritas
```

No pipx yet? `python3 -m pip install --user pipx && pipx ensurepath` (or
`brew install pipx`, `apt install pipx`, `winget install pipx`).

**One-line install** (no pipx required) — builds an isolated virtualenv and
links the command:

```bash
curl -fsSL https://raw.githubusercontent.com/Nyvo2010/veritas/main/install.sh | sh
veritas
```

Check the installed version with `veritas --version`.

**From source**:

```bash
git clone https://github.com/Nyvo2010/veritas.git
cd veritas
python -m venv .venv && source .venv/bin/activate
pip install -e .
veritas
```

Running `./install.sh` from a checkout builds that working tree instead of the
published source and links the command for you.

## First run

A setup wizard walks you through: AI endpoint URL → API key → model → Tavily
key → reports folder → color theme. Re-run it anytime via
Settings → Setup wizard.

Keys: `↑`/`↓` move · `Enter` select · `Esc` back · `q` quit (menu) ·
`Ctrl+C` quit (anywhere). Paste works in every input field.

Long lists scroll, with `↑ N more` / `↓ N more` markers so the highlighted row
is always visible. Model pickers also accept typing: start typing to filter the
`/models` list, or type an id that isn't listed and press `Enter` to use it.

## While research runs

The run screen keeps three things on one screen: the query and current phase,
a scrolling activity log where each phase and iteration is its own heading, and
a braille progress bar — the left side fills as steps complete while particles
drift in from the right.

Below the bar, the elapsed time, a rough ETA, and a **time breakdown** show how
much of the run went to model inference versus web search, with call counts.
That is the quickest way to tell whether a slow run is the model or the search
API.

Every run keeps its activity log in memory: press `l` on the results screen, or
pick **Last run log** from the menu, to scroll back through it.

`Ctrl+C` stops the run and writes whatever is finished (see below).

## Settings vs environment variables

Everything can be configured in-app under Settings, stored in
`~/.config/veritas/settings.json` (mode `0600`, keys masked on screen).

- **AI provider** — endpoint, API key, model, and an optional fallback model
- **Search** — Tavily API key and the default source categories
- **Research defaults** — iterations and results per search

A `.env` file (see `.env.example`) is the alternative path for non-interactive
use — local dev, containers, CI, headless servers — or for temporarily
overriding the file without touching it. Precedence everywhere:

> explicit argument > environment variable > settings file

Supported variables: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`,
`LLM_FALLBACK_MODEL`, `TAVILY_API_KEY`. `VERITAS_REPORTS_DIR` overrides the
reports folder.

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
  utils/filelog.py       detailed backend log (file only)
  utils/text.py          shared text helpers
  ui/              theme tokens, braille bar, live renderer, shared screens
  ui/timings.py    where a run spends its time (model vs search)
```

## Reports

Reports are written to `~/Veritas/reports` by default (override in Settings or
with `VERITAS_REPORTS_DIR`) as `research_report_<query>_<timestamp>.md`; nothing
is written into the repository.

If a run fails or is stopped early, Veritas still writes a clearly-labelled
partial report from the findings collected so far rather than losing the work —
so a file appears even when a run does not finish. The results screen shows the
path, or exactly why the write did not happen.

## Logs

Nothing the backend logs ever reaches the screen: the TUI owns the terminal.
Detail goes to a rolling file instead, `~/.config/veritas/logs/veritas.log`
(mode `0600`, 2 MB per file, 3 files): every phase and iteration, every model
call with its model id, prompt size, duration, token usage and finish reason,
retries and backoff, fallback switches, and every search with its query, depth,
result count and duration.

```bash
VERITAS_LOG_LEVEL=INFO veritas     # quieter file log (default: DEBUG)
VERITAS_LOG_CONSOLE=1 veritas      # also stream warnings to stderr
```

The log can contain the prompts and queries you typed. It stays on your machine
and is never uploaded.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                  # editable install of the app
python -m unittest discover -s tests
```

The suite runs with no API keys: it covers report writing, the UI primitives,
and — when opentui is importable — a headless render of every screen.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow,
[RELEASING.md](RELEASING.md) for cutting a release, and
[SECURITY.md](SECURITY.md) for how to report a vulnerability.

## Roadmap

- an npm wrapper so `npx veritas` works for people without Python
- test coverage beyond the UI primitives
- screenshots in this README

## License

MIT — see [LICENSE](LICENSE). Release notes live in [CHANGELOG.md](CHANGELOG.md).
