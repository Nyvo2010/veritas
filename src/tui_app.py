"""Veritas OpenTUI frontend: arrow-key menus, round boxes, live run screen.

Screens (all keyboard driven):
  menu          banner + New research / Settings / Quit
  query         free-text question
  iterations    max iterations (1-20, defaults from settings)
  sources       per-run source toggles + Start research
  run           live spinner + progress bar + scrolling activity log
  done          result stats or error, back to menu
  settings      current values (merged view-config) + editors
  set-theme / set-reports / set-keys / set-key / set-defaults /
  set-default / set-sources

Research runs in a worker thread via DeepResearchAgent with a QueueUI
sink; a pump thread drains the event queue into reactive signals.
Falls back to the Rich CLI when opentui cannot be imported.
"""

import asyncio
import math
import os
import queue
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from opentui import (
    Box,
    Input,
    SelectOption,
    SelectRenderable,
    Signal,
    Text,
    component,
    render,
    use_keyboard,
    use_renderer,
)

from settings import (
    load_settings,
    mask_key,
    reports_dir,
    update_settings,
)
from ui import theme
from runner_ui import QueueUI
from main import DeepResearchAgent


# Restrained palette (one accent, saturation low, neutrals consistent,
# never pure black, nothing neon). Theme names are user-facing; each maps
# to a desaturated tone.
ACCENTS = {
    "cyan": "#5f9ea0",      # muted cadet teal
    "blue": "#6f8fb8",      # dusty blue
    "magenta": "#9a7fa8",   # muted mauve
    "green": "#7d9b76",     # sage
    "amber": "#b3945e",     # muted brass
}

LOG_GLYPH = {
    "phase": "*",
    "search": ">",
    "insight": "-",
    "ok": "+",
    "warn": "!",
    "err": "x",
    "info": ".",
}

INPUT_SCREENS = {"query", "iterations", "set-reports", "set-key", "set-default"}

BACK = {
    "query": "menu",
    "iterations": "query",
    "sources": "iterations",
    "run": None,
    "done": "menu",
    "settings": "menu",
    "set-theme": "settings",
    "set-reports": "settings",
    "set-keys": "settings",
    "set-key": "set-keys",
    "set-defaults": "settings",
    "set-default": "set-defaults",
    "set-sources": "settings",
}


def accent() -> str:
    return ACCENTS.get(str(settings_sig().get("theme", "cyan")), ACCENTS["cyan"])


def term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def banner_lines() -> list:
    if term_width() >= 96:
        return list(theme.COMPACT_LOGO)
    return ["VERITAS"]


# Flat colored panels: solid title bar + tinted body, no line borders.
PANEL_BG = "#232326"        # neutral zinc, no color cast
TITLE_FG = "#1c1c1e"        # near-black for title bars
SECONDARY = "#8e8e93"       # descriptions, values
TERTIARY = "#636366"        # hints, footnotes
DANGER = "#b86767"          # muted brick for errors


def gap(rows: int = 1) -> Text:
    """Fixed breathing room between blocks (rhythm, not decoration)."""
    return Text("\n".join([""] * rows))


def _hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in rgb])


def shade(value: str, factor: float) -> str:
    """Lighten (factor > 1) or darken (factor < 1) a hex color toward white/black."""
    r, g, b = _hex_to_rgb(value)
    if factor >= 1:
        t = min(factor - 1, 1)
        r, g, b = r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t
    else:
        r, g, b = r * factor, g * factor, b * factor
    return _rgb_to_hex((r, g, b))


def logo_lines(tick: int) -> list:
    """Banner with a slow tonal shimmer: same accent family, top lighter,
    a soft wave traveling downward. Subtle on purpose."""
    lines = banner_lines()
    total = max(len(lines) - 1, 1)
    base = accent()
    out = []
    for i, line in enumerate(lines):
        vertical = 1.20 - 0.40 * (i / total)  # lighter crown, deeper base
        wave = 0.5 + 0.5 * math.sin(tick * 0.30 - i * 0.55)
        out.append(Text(line, fg=shade(base, vertical * (0.90 + 0.20 * wave))))
    return out


def frame(title: str, *children) -> Box:
    """Quiet panel: bold text label, tinted body, no highlight bar."""
    return Box(
        Text(title, bold=True),
        gap(),
        *children,
        bg=PANEL_BG,
        padding_y=1,
        padding_x=2,
    )


def hint(text: str) -> Text:
    return Text(text, fg=TERTIARY)


def short_query(q: str, n: int = 70) -> str:
    q = " ".join(q.split())
    return q if len(q) <= n else q[: n - 3] + "..."


# --------------------------------------------------------------------------
# Global reactive state
# --------------------------------------------------------------------------

screen = Signal("menu")
settings_sig = Signal(load_settings())
log_sig = Signal(())
prog_sig = Signal((0.0, 1, "Starting..."))
phase_sig = Signal("Preparing...")
elapsed_sig = Signal("0s")
spin_sig = Signal(0)
tick_sig = Signal(0)
err_sig = Signal("")

state = {
    "query": "",
    "max_iter": 10,
    "toggles": {},
    "cursors": {},
    "key_field": "",
    "default_field": "",
    "result": None,
    "run_start": 0.0,
}

selects: dict = {}
inputs: dict = {}
renderer_ref: dict = {}
pump_stop = threading.Event()
_keyboard_bound: dict = {}


def tracked_select(name: str, options: list, **kw) -> SelectRenderable:
    """Select list with quiet styling and a cursor that survives re-renders.

    The logo ticker re-evaluates the tree several times a second; without a
    persistent cursor every rebuild would jump back to the top option.
    """
    for opt in options:
        if not opt.name.startswith("  "):
            opt.name = "  " + opt.name
    kw.setdefault("selected_background_color", accent())
    kw.setdefault("selected_text_color", TITLE_FG)
    kw.setdefault("description_color", SECONDARY)
    kw.setdefault("item_spacing", 1)
    sel = SelectRenderable(
        options=options,
        selected_index=min(state["cursors"].get(name, 0), max(len(options) - 1, 0)),
        **kw,
    )
    sel.on("selectionChanged", lambda idx, opt: state["cursors"].update({name: idx}))
    selects[name] = sel
    return sel


# --------------------------------------------------------------------------
# Research wiring
# --------------------------------------------------------------------------

def apply_event(ev: tuple) -> None:
    kind = ev[0]
    if kind == "log":
        _, level, text = ev
        glyph = LOG_GLYPH.get(level, ".")
        lines = log_sig() + ((level, f"{glyph} {text}"),)
        log_sig.set(lines[-300:])
    elif kind == "status":
        completed, total, _ = prog_sig()
        prog_sig.set((completed, total, ev[1]))
    elif kind == "phase":
        phase_sig.set(ev[1])
    elif kind == "progress":
        _, completed, total, status = ev
        prog_sig.set((float(completed), int(total), status))


def pump(ui: QueueUI) -> None:
    frames = len(theme.SPINNER_FRAMES)
    while not pump_stop.is_set():
        try:
            try:
                while True:
                    apply_event(ui.events.get_nowait())
            except queue.Empty:
                pass
            spin_sig.set((spin_sig() + 1) % frames)
            elapsed_sig.set(theme.format_duration(time.time() - state["run_start"]))
        except Exception:
            break
        time.sleep(0.09)


def worker(query: str, max_iter: int, toggles: dict) -> None:
    try:
        agent = DeepResearchAgent(category_toggles=dict(toggles))
        ui = state["ui"]
        st = asyncio.run(agent.research(query, max_iterations=max_iter, verbose=False, ui=ui))
        saved = agent.save_report(st)
        duration = (st.end_time - st.start_time).total_seconds() if st.end_time else 0.0
        state["result"] = {
            "ok": True,
            "duration": duration,
            "iterations": st.iteration_count,
            "questions": len(st.completed_questions),
            "searches": st.total_searches,
            "findings": len(st.findings),
            "gaps": len(st.gaps),
            "report_len": len(st.final_report or ""),
            "saved": saved,
        }
    except Exception as e:  # noqa: BLE001 - surfaced on the done screen
        state["result"] = {"ok": False, "error": str(e)}
    screen.set("done")


def start_run() -> None:
    settings = settings_sig()
    state["run_start"] = time.time()
    log_sig.set(())
    prog_sig.set((0.0, 1, "Starting..."))
    phase_sig.set("Preparing...")
    elapsed_sig.set("0s")
    state["result"] = None
    ui = QueueUI()
    state["ui"] = ui
    screen.set("run")
    threading.Thread(target=pump, args=(ui,), daemon=True).start()
    threading.Thread(
        target=worker,
        args=(state["query"], state["max_iter"], state["toggles"]),
        daemon=True,
    ).start()


# --------------------------------------------------------------------------
# Screen components
# --------------------------------------------------------------------------

def menu_select() -> SelectRenderable:
    def on_pick(index: int, opt) -> None:
        value = opt.value if opt else None
        if value == "new":
            state["query"] = ""
            err_sig.set("")
            screen.set("query")
        elif value == "settings":
            screen.set("settings")
        elif value == "quit":
            renderer_ref.get("r") and renderer_ref["r"].stop()

    sel = tracked_select("menu", [
        SelectOption("New research", "new", "Answer a question with deep web research"),
        SelectOption("Settings", "settings", "Theme, reports folder, API keys, defaults"),
        SelectOption("Quit", "quit", "Exit Veritas"),
    ])
    sel.on("itemSelected", on_pick)
    return sel


def sources_select(session: bool) -> SelectRenderable:
    toggles = state["toggles"] if session else settings_sig().get("category_toggles", {})

    def on_pick(index: int, opt) -> None:
        value = opt.value if opt else None
        names = list(toggles.keys())
        if value == "__start__":
            start_run()
        elif value == "__back__":
            screen.set("settings")
        elif value in names:
            toggles[value] = not toggles[value]
            if not session:
                update_settings(category_toggles={value: toggles[value]})
                settings_sig.set(load_settings())
            toggles_sig.set(dict(toggles))

    options = []
    for name in toggles:
        mark = "x" if toggles[name] else " "
        options.append(SelectOption(f"[{mark}] {name.replace('_', ' ').title()}", name))
    if session:
        options.append(SelectOption("Start research", "__start__", "Begin with these sources"))
    else:
        options.append(SelectOption("Back", "__back__"))

    sel = tracked_select("sources" if session else "set-sources", options)
    sel.on("itemSelected", on_pick)
    return sel


toggles_sig = Signal({})


def progress_bar() -> str:
    completed, total, _ = prog_sig()
    ratio = min(1.0, completed / total) if total else 0.0
    width = 36
    filled = int(width * ratio)
    return "█" * filled + "░" * (width - filled) + f" {ratio * 100:.0f}%"


def activity_lines() -> list:
    """Activity log with quiet hierarchy: phases accented, problems warm."""
    lines = log_sig()[-12:]
    if not lines:
        return [Text("Waiting for activity...", fg=SECONDARY)]
    colors = {"err": DANGER, "warn": ACCENTS["amber"], "phase": accent()}
    return [Text(text, fg=colors.get(level, SECONDARY)) for level, text in lines]


# --------------------------------------------------------------------------
# Root component + input
# --------------------------------------------------------------------------

@component
def App():
    toggles_sig()  # subscribe so source toggles rerender
    current = screen()

    def on_key(e) -> None:
        key = getattr(e, "key", "") or ""
        s = screen()
        # Esc always navigates back, even from inside a text input
        # (Input.handle_key would otherwise consume it).
        if key == "escape":
            target = BACK.get(s)
            if target:
                err_sig.set("")
                screen.set(target)
            return
        if s in selects and s not in INPUT_SCREENS:
            try:
                if selects[s].handle_key(e):
                    return
            except Exception:
                pass
        if s in inputs and s in INPUT_SCREENS:
            # Route keys to the screen's text input. Printable characters
            # bypass handle_key because this port lowercases them (which
            # would corrupt API keys, paths, and queries). The cursor is
            # pinned to the end afterwards because this port's delete_char
            # double-decrements it; caret navigation inside inputs is not
            # supported until that is fixed upstream.
            try:
                target = inputs[s]
                modifiers = (
                    getattr(e, "ctrl", False)
                    or getattr(e, "alt", False)
                    or getattr(e, "meta", False)
                )
                if len(key) == 1 and not modifiers:
                    target.insert_text(key)
                else:
                    target.handle_key(e)
                try:
                    target._cursor_position = len(target._value)
                except Exception:
                    pass
            except Exception:
                pass
            return
        if key == "q" and s == "menu":
            renderer_ref.get("r") and renderer_ref["r"].stop()
        elif key in ("return", "enter") and s == "done":
            screen.set("menu")

    # Register the global key handler exactly once per renderer. App
    # re-evaluates on every signal change, and this port stacks duplicate
    # use_keyboard handlers (stale ones then swallow keys).
    current_renderer = use_renderer()
    renderer_ref["r"] = current_renderer
    if _keyboard_bound.get("renderer") is not current_renderer:
        _keyboard_bound["renderer"] = current_renderer
        use_keyboard(on_key)

    if current == "menu":
        sel = menu_select()
        selects["menu"] = sel
        return Box(
            Box(*logo_lines(tick_sig())),
            gap(),
            frame("Main Menu", sel),
            gap(),
            hint("Up/Down to move, Enter to select, q to quit"),
        )

    if current == "query":
        def submit(value: str) -> None:
            q = value.strip()
            if len(q.split()) < 2:
                err_sig.set("Type a full question, not a single word.")
                return
            err_sig.set("")
            state["query"] = q
            state["max_iter"] = int(settings_sig().get("max_iterations", 10))
            screen.set("iterations")

        query_input = Input(
            value=state["query"],
            placeholder="Type your question...",
            focused=True,
            on_submit=submit,
        )
        inputs["query"] = query_input
        return Box(
            frame(
                "New research (1/3)",
                Text("What do you want to research?"),
                query_input,
                Text(err_sig() or "", fg=DANGER),
                hint("Enter to continue, Esc to go back"),
            )
        )

    if current == "iterations":
        def submit(value: str) -> None:
            raw = value.strip()
            if raw and (not raw.isdigit() or not 1 <= int(raw) <= 20):
                err_sig.set("Enter a number from 1 to 20.")
                return
            err_sig.set("")
            if raw:
                state["max_iter"] = int(raw)
            state["toggles"] = dict(settings_sig().get("category_toggles", {}))
            state["cursors"]["sources"] = 0
            screen.set("sources")

        iter_input = Input(
            value="",
            placeholder=f"1-20 (empty keeps {state['max_iter']})",
            focused=True,
            on_submit=submit,
        )
        inputs["iterations"] = iter_input
        return Box(
            frame(
                "New research (2/3)",
                Text(f"Query: {short_query(state['query'])}"),
                Text(f"Max iterations (default {state['max_iter']}, 1-20):"),
                iter_input,
                Text(err_sig() or "", fg=DANGER),
                hint("Enter to continue, Esc to go back"),
            )
        )

    if current == "sources":
        sel = sources_select(session=True)
        selects["sources"] = sel
        return Box(
            frame(
                "New research (3/3)",
                Text(f"Query: {short_query(state['query'])}  |  Iterations: {state['max_iter']}"),
                sel,
                hint("Up/Down to move, Enter toggles a source, Esc to go back"),
            )
        )

    if current == "run":
        frames = theme.SPINNER_FRAMES
        return Box(
            frame(
                "Research running",
                Text(f"{frames[spin_sig()]} {short_query(state['query'])}", bold=True),
                gap(),
                Text(lambda: phase_sig(), fg=SECONDARY),
                Text(lambda: progress_bar(), fg=accent()),
                Text(lambda: f"Step {prog_sig()[0]:.0f}/{prog_sig()[1]}  |  {elapsed_sig()}  |  {prog_sig()[2]}", fg=SECONDARY),
                gap(),
                frame("Activity", *activity_lines()),
                gap(),
                hint("Ctrl+C quits (partial work is lost)"),
            )
        )

    if current == "done":
        result = state["result"] or {"ok": False, "error": "No result recorded."}
        if result.get("ok"):
            saved = result.get("saved") or "not saved"
            body = "\n".join(
                [
                    f"Duration:    {theme.format_duration(result.get('duration', 0))}",
                    f"Iterations:   {result.get('iterations', 0)}",
                    f"Questions:    {result.get('questions', 0)}",
                    f"Searches:     {result.get('searches', 0)}",
                    f"Findings:     {result.get('findings', 0)}",
                    f"Gaps:         {result.get('gaps', 0)}",
                    f"Report:       {result.get('report_len', 0):,} chars",
                    f"Saved to:     {saved}",
                ]
            )
            return Box(
                frame("Research complete", Text(body)),
                gap(),
                hint("Enter to return to the menu"),
            )
        return Box(
            frame("Research failed", Text(str(result.get("error", "Unknown error")), fg=DANGER)),
            gap(),
            hint("Enter to return to the menu"),
        )

    if current == "settings":
        def on_pick(index: int, opt) -> None:
            value = opt.value if opt else None
            err_sig.set("")
            if value == "back":
                screen.set("menu")
            else:
                screen.set(value)

        s = settings_sig()
        active = s.get("category_toggles", {})
        on = sum(1 for v in active.values() if v)
        sel = tracked_select("settings", [
            SelectOption("Color theme", "set-theme", f"Current: {s.get('theme', '')}"),
            SelectOption("Reports folder", "set-reports", reports_dir(s)),
            SelectOption("API keys", "set-keys", "NVIDIA + Tavily, stored locally"),
            SelectOption("Research defaults", "set-defaults", "Iterations + results per search"),
            SelectOption("Default sources", "set-sources", f"{on}/{len(active)} enabled"),
            SelectOption("Back", "back"),
        ])
        sel.on("itemSelected", on_pick)
        return Box(
            frame("Settings", sel),
            gap(),
            hint("Up/Down to move, Enter to edit, Esc for menu"),
        )

    if current == "set-theme":
        def on_pick(index: int, opt) -> None:
            update_settings(theme=opt.value)
            settings_sig.set(load_settings())
            screen.set("settings")

        sel = tracked_select(
            "set-theme",
            [SelectOption(name, name) for name in theme.THEMES],
        )
        sel.on("itemSelected", on_pick)
        return Box(
            frame("Color theme", sel),
            gap(),
            hint("Enter to apply, Esc to go back"),
        )

    if current == "set-reports":
        def submit(value: str) -> None:
            raw = value.strip()
            if not raw:
                screen.set("settings")
                return
            expanded = os.path.expanduser(raw)
            try:
                os.makedirs(expanded, exist_ok=True)
            except OSError as e:
                err_sig.set(f"Cannot use that folder: {e}")
                return
            err_sig.set("")
            update_settings(reports_dir=raw)
            settings_sig.set(load_settings())
            screen.set("settings")

        reports_input = Input(
            value="",
            placeholder="New path (empty keeps current)",
            focused=True,
            on_submit=submit,
        )
        inputs["set-reports"] = reports_input
        return Box(
            frame(
                "Reports folder",
                Text(f"Current: {reports_dir(settings_sig())}"),
                reports_input,
                Text(err_sig() or "", fg=DANGER),
                hint("Enter to save, Esc to go back"),
            )
        )

    if current == "set-keys":
        def on_pick(index: int, opt) -> None:
            value = opt.value if opt else None
            if value == "back":
                screen.set("settings")
            else:
                state["key_field"] = value
                err_sig.set("")
                screen.set("set-key")

        s = settings_sig()
        labels = {
            "nvidia_api_key": ("NVIDIA API key", mask_key(s.get("nvidia_api_key", ""))),
            "tavily_api_key": ("Tavily API key", mask_key(s.get("tavily_api_key", ""))),
            "nvidia_api_base_url": ("Base URL", str(s.get("nvidia_api_base_url", ""))),
        }
        sel = tracked_select(
            "set-keys",
            [SelectOption(label, key, current) for key, (label, current) in labels.items()]
            + [SelectOption("Back", "back")],
        )
        sel.on("itemSelected", on_pick)
        return Box(
            frame("API keys", sel),
            gap(),
            hint("Keys are stored only in your local settings file"),
        )

    if current == "set-key":
        field = state["key_field"]
        label = {
            "nvidia_api_key": "NVIDIA API key",
            "tavily_api_key": "Tavily API key",
            "nvidia_api_base_url": "Base URL",
        }.get(field, field)

        def submit(value: str) -> None:
            raw = value.strip()
            if raw:
                update_settings(**{field: raw})
                settings_sig.set(load_settings())
            screen.set("set-keys")

        key_input = Input(
            placeholder="Paste a new value, Enter to save",
            focused=True,
            on_submit=submit,
        )
        inputs["set-key"] = key_input
        return Box(
            frame(
                label,
                key_input,
                hint("Enter to save, Esc to go back"),
            )
        )

    if current == "set-defaults":
        def on_pick(index: int, opt) -> None:
            value = opt.value if opt else None
            if value == "back":
                screen.set("settings")
            else:
                state["default_field"] = value
                err_sig.set("")
                screen.set("set-default")

        s = settings_sig()
        sel = tracked_select("set-defaults", [
            SelectOption("Max iterations", "max_iterations", f"Current: {s.get('max_iterations', '')}"),
            SelectOption("Max search results", "max_search_results", f"Current: {s.get('max_search_results', '')}"),
            SelectOption("Back", "back"),
        ])
        sel.on("itemSelected", on_pick)
        return Box(
            frame("Research defaults", sel),
            gap(),
            hint("Enter to edit, Esc to go back"),
        )

    if current == "set-default":
        field = state["default_field"]
        label = "Max iterations" if field == "max_iterations" else "Max search results"

        def submit(value: str) -> None:
            raw = value.strip()
            if not raw:
                screen.set("set-defaults")
                return
            if not raw.isdigit() or not 1 <= int(raw) <= 20:
                err_sig.set("Enter a number from 1 to 20.")
                return
            err_sig.set("")
            update_settings(**{field: int(raw)})
            settings_sig.set(load_settings())
            screen.set("set-defaults")

        default_input = Input(
            value="",
            placeholder="1-20 (empty keeps current)",
            focused=True,
            on_submit=submit,
        )
        inputs["set-default"] = default_input
        return Box(
            frame(
                label,
                Text(f"Current: {settings_sig().get(field, '')}"),
                default_input,
                Text(err_sig() or "", fg=DANGER),
                hint("Enter to save, Esc to go back"),
            )
        )

    if current == "set-sources":
        sel = sources_select(session=False)
        return Box(
            frame("Default sources", sel),
            gap(),
            hint("Enter toggles a source, Esc to go back"),
        )

    return Box(Text("Unknown screen", fg=DANGER))


def run() -> None:
    """Launch the OpenTUI app (blocking until quit)."""
    pump_stop.clear()

    def _ticker() -> None:
        # The banner shimmer lives only on the menu; ticking elsewhere would
        # rebuild inputs and selects for nothing (and reset their cursors).
        n = 0
        while not pump_stop.is_set():
            try:
                if screen() == "menu":
                    tick_sig.set(n)
                    n += 1
            except Exception:
                break
            time.sleep(0.15)

    threading.Thread(target=_ticker, daemon=True).start()
    try:
        asyncio.run(render(App))
    finally:
        pump_stop.set()
