# Contributing

Thanks for taking a look at Veritas. Issues and pull requests are welcome.

## Setup

Requires Python 3.12+.

```bash
git clone https://github.com/Nyvo2010/veritas.git
cd veritas
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Running the tests

```bash
python -m unittest discover -s tests
```

The suite needs no API keys. It exercises report writing, the UI primitives,
and a headless render of every TUI screen when opentui is installed. Please add
or update a test with any behavior change, and make sure the whole suite passes
before opening a pull request.

## Layout

- `src/cli.py` — the `veritas` entry point and the Rich CLI fallback
- `src/tui_app.py` — the OpenTUI frontend
- `src/main.py` — the research orchestrator
- `src/agents/` — query analysis, search, synthesis, report generation
- `src/utils/` — LLM, Tavily, logging, and text helpers
- `src/ui/` — theme tokens, live renderer, shared screens

## Pull requests

- Keep the change focused; one concern per pull request.
- Match the surrounding style; there is no formatter to run.
- Describe what changed and why, and how you tested it.
- Do not commit API keys, tokens, or anything from `~/.config/veritas`.

By contributing you agree that your work is licensed under the MIT License
(see [LICENSE](LICENSE)).
