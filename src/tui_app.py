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
    resolve_api_keys,
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

INPUT_SCREENS = {
    "query", "iterations", "set-reports", "set-key", "set-default",
    "setup-nvidia", "setup-tavily", "setup-reports",
}

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
    "setup-welcome": "menu",
    "setup-nvidia": "setup-welcome",
    "setup-tavily": "setup-nvidia",
    "setup-reports": "setup-tavily",
    "setup-theme": "setup-reports",
    "setup-done": None,
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
SUBTITLE = "#707074"        # button subtitles: dimmer than body, softer than hints
DANGER = "#b86767"          # muted brick for errors

INPUT_WIDTH = 64            # shared width for every text input
SUBTITLE_LEN = 64           # subtitles truncate past this (keeps rows tidy)


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


THEME_BLURBS = {
    "cyan": "Muted cadet teal",
    "blue": "Dusty blue",
    "magenta": "Muted mauve",
    "green": "Sage",
    "amber": "Muted brass",
}


def theme_options() -> list:
    """Every theme button carries its tone as a subtitle."""
    return [
        SelectOption(name.title(), name, THEME_BLURBS.get(name, ""))
        for name in theme.THEMES
    ]


def make_input(**kw) -> Input:
    """Every text input shares one width and focus behavior."""
    kw.setdefault("width", INPUT_WIDTH)
    kw.setdefault("focused", True)
    return Input(**kw)


def screen_box(title: str, *children, enter: str, back: str | None = None) -> Box:
    """Standard skeleton for list screens: panel, breathing room, key hint."""
    keys = f"↑↓ move · Enter {enter}"
    if back:
        keys += f" · Esc {back}"
    return Box(frame(title, *children), gap(), hint(keys))


def input_frame(title: str, lines: list, inp, enter: str, back: str, kind: str = "Type") -> Box:
    """Standard skeleton for text-entry screens: prompts, input, inline
    validation, then the key hint outside the panel."""
    return Box(
        frame(title, *lines, inp, Text(err_sig() or "", fg=DANGER)),
        gap(),
        hint(f"{kind} · Enter {enter} · Esc {back}"),
    )


def kv_rows(pairs: list) -> list:
    """Aligned stat rows: labels in one column, values in SECONDARY."""
    width = max([len(label) for label, _ in pairs] + [0])
    return [Text(f"{label:<{width}}  {value}", fg=SECONDARY) for label, value in pairs]


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
    """Button-like select list: title with a dimmer subtitle below it,
    highlight hugging the content, cursor surviving re-renders.

    Subtitles use the native description line (SUBTITLE color, TITLE_FG
    on the selected row) indented to sit under the title text. The logo
    ticker re-evaluates the tree several times a second; without a
    persistent cursor every rebuild would jump back to the top option.
    """
    for opt in options:
        opt.name = f"  {str(opt.name).strip()}  "
        subtitle = str(getattr(opt, "description", "") or "").strip()
        if len(subtitle) > SUBTITLE_LEN:
            subtitle = subtitle[: SUBTITLE_LEN - 1] + "…"
        opt.description = f"  {subtitle}" if subtitle else ""
    kw.setdefault("selected_background_color", accent())
    kw.setdefault("selected_text_color", TITLE_FG)
    kw.setdefault("selected_description_color", TITLE_FG)
    kw.setdefault("description_color", SUBTITLE)
    kw.setdefault("item_spacing", 1)
    kw.setdefault("show_description", True)
    initial = min(state["cursors"].get(name, 0), max(len(options) - 1, 0))

    def on_move(idx: int, opt) -> None:
        state["cursors"][name] = idx

    sel = SelectRenderable(
        options=options,
        selected_index=initial,
        align_self="flex-start",
        **kw,
    )
    sel.on("selectionChanged", on_move)
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
            hint("↑↓ move · Enter select · q quit"),
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

        query_input = make_input(
            value=state["query"],
            placeholder="Type your question...",
            on_submit=submit,
        )
        inputs["query"] = query_input
        return input_frame(
            "New research (1/3)",
            [Text("What do you want to research?")],
            query_input,
            enter="continue",
            back="back",
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

        iter_input = make_input(
            value="",
            placeholder=f"1-20 (empty keeps {state['max_iter']})",
            on_submit=submit,
        )
        inputs["iterations"] = iter_input
        return input_frame(
            "New research (2/3)",
            [
                Text(f"Query: {short_query(state['query'])}"),
                Text(f"Max iterations (default {state['max_iter']}, 1-20):"),
            ],
            iter_input,
            enter="continue",
            back="back",
        )

    if current == "sources":
        sel = sources_select(session=True)
        selects["sources"] = sel
        return Box(
            frame(
                "New research (3/3)",
                Text(f"Query: {short_query(state['query'])}  |  Iterations: {state['max_iter']}"),
                sel,
            ),
            gap(),
            hint("↑↓ move · Enter toggle · Esc back"),
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
            rows = kv_rows(
                [
                    ("Duration", theme.format_duration(result.get("duration", 0))),
                    ("Iterations", str(result.get("iterations", 0))),
                    ("Questions", str(result.get("questions", 0))),
                    ("Searches", str(result.get("searches", 0))),
                    ("Findings", str(result.get("findings", 0))),
                    ("Gaps", str(result.get("gaps", 0))),
                    ("Report", f"{result.get('report_len', 0):,} chars"),
                    ("Saved to", saved),
                ]
            )
            return Box(
                frame("Research complete", *rows),
                gap(),
                hint("Enter for the menu"),
            )
        return Box(
            frame("Research failed", Text(str(result.get("error", "Unknown error")), fg=DANGER)),
            gap(),
            hint("Enter for the menu"),
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
            SelectOption("Color theme", "set-theme", f"Current: {s.get('theme', '').title()}"),
            SelectOption("Reports folder", "set-reports", reports_dir(s)),
            SelectOption("API keys", "set-keys", "NVIDIA + Tavily, stored locally"),
            SelectOption("Research defaults", "set-defaults", "Iterations + results per search"),
            SelectOption("Default sources", "set-sources", f"{on}/{len(active)} enabled"),
            SelectOption("Setup wizard", "setup-welcome", "Keys, folder, theme from scratch"),
            SelectOption("Back", "back"),
        ])
        sel.on("itemSelected", on_pick)
        return screen_box("Settings", sel, enter="edit", back="menu")

    if current == "set-theme":
        def on_pick(index: int, opt) -> None:
            update_settings(theme=opt.value)
            settings_sig.set(load_settings())
            screen.set("settings")

        sel = tracked_select("set-theme", theme_options())
        sel.on("itemSelected", on_pick)
        return screen_box("Color theme", sel, enter="apply", back="back")

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

        reports_input = make_input(
            value="",
            placeholder="New path (empty keeps current)",
            on_submit=submit,
        )
        inputs["set-reports"] = reports_input
        return input_frame(
            "Reports folder",
            [Text(f"Current: {reports_dir(settings_sig())}")],
            reports_input,
            enter="save",
            back="back",
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
        return screen_box(
            "API keys",
            Text("Keys are stored only in your local settings file.", fg=SECONDARY),
            sel,
            enter="edit",
            back="back",
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

        key_input = make_input(
            placeholder="Paste a new value, Enter to save",
            on_submit=submit,
        )
        inputs["set-key"] = key_input
        return input_frame(label, [], key_input, enter="save", back="back", kind="Paste")

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
        return screen_box("Research defaults", sel, enter="edit", back="back")

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

        default_input = make_input(
            value="",
            placeholder="1-20 (empty keeps current)",
            on_submit=submit,
        )
        inputs["set-default"] = default_input
        return input_frame(
            label,
            [Text(f"Current: {settings_sig().get(field, '')}")],
            default_input,
            enter="save",
            back="back",
        )

    if current == "set-sources":
        sel = sources_select(session=False)
        toggles = settings_sig().get("category_toggles", {})
        on = sum(1 for v in toggles.values() if v)
        return screen_box(f"Default sources ({on}/{len(toggles)})", sel, enter="toggle", back="back")

    if current == "setup-welcome":
        def on_pick(index: int, opt) -> None:
            if (opt.value if opt else None) == "start":
                state["setup"] = {}
                err_sig.set("")
                screen.set("setup-nvidia")
            else:
                screen.set("menu")

        sel = tracked_select("setup-welcome", [
            SelectOption("Continue setup", "start", "Takes about a minute"),
            SelectOption("Back", "menu", "Return to the main menu"),
        ])
        sel.on("itemSelected", on_pick)
        return Box(
            Text("\n".join(banner_lines()), fg=accent()),
            gap(),
            frame(
                "Welcome to Veritas",
                Text("Three things to get going: an NVIDIA key for the research brain, a Tavily key for web search, and a folder for reports."),
                Text("Keys are stored only on this machine. Environment variables always win when set.", fg=SECONDARY),
                gap(),
                sel,
            ),
            gap(),
            hint("↑↓ move · Enter choose · Esc menu"),
        )

    if current in ("setup-nvidia", "setup-tavily"):
        is_nvidia = current == "setup-nvidia"
        field = "nvidia_api_key" if is_nvidia else "tavily_api_key"
        label = "NVIDIA API key" if is_nvidia else "Tavily API key"
        where = "build.nvidia.com" if is_nvidia else "tavily.com"
        nxt = "setup-tavily" if is_nvidia else "setup-reports"

        def submit(value: str) -> None:
            raw = value.strip()
            if not raw:
                err_sig.set("Paste a key to continue, or Esc to stop.")
                return
            err_sig.set("")
            state.setdefault("setup", {})[field] = raw
            screen.set(nxt)

        key_input = make_input(
            placeholder=f"Paste your key from {where}",
            on_submit=submit,
        )
        inputs[current] = key_input
        step = "Setup (1/4)" if is_nvidia else "Setup (2/4)"
        return input_frame(
            step,
            [Text(label, bold=True), Text(f"Get one at {where}.", fg=SECONDARY)],
            key_input,
            enter="continue",
            back="stop",
            kind="Paste",
        )

    if current == "setup-reports":
        def submit(value: str) -> None:
            raw = value.strip()
            if raw:
                expanded = os.path.expanduser(raw)
                try:
                    os.makedirs(expanded, exist_ok=True)
                except OSError as e:
                    err_sig.set(f"Cannot use that folder: {e}")
                    return
                state.setdefault("setup", {})["reports_dir"] = raw
            err_sig.set("")
            screen.set("setup-theme")

        reports_setup_input = make_input(
            value="",
            placeholder="Empty keeps the default",
            on_submit=submit,
        )
        inputs["setup-reports"] = reports_setup_input
        return input_frame(
            "Setup (3/4)",
            [
                Text("Reports folder", bold=True),
                Text(f"Default: {reports_dir(settings_sig())}", fg=SECONDARY),
            ],
            reports_setup_input,
            enter="continue",
            back="stop",
        )

    if current == "setup-theme":
        def on_pick(index: int, opt) -> None:
            update_settings(
                nvidia_api_key=state.get("setup", {}).get("nvidia_api_key", ""),
                tavily_api_key=state.get("setup", {}).get("tavily_api_key", ""),
                **({"reports_dir": state["setup"]["reports_dir"]} if "reports_dir" in state.get("setup", {}) else {}),
                theme=opt.value,
            )
            settings_sig.set(load_settings())
            screen.set("setup-done")

        sel = tracked_select("setup-theme", theme_options())
        sel.on("itemSelected", on_pick)
        return Box(
            frame("Setup (4/4)", Text("Pick a color theme", bold=True), sel),
            gap(),
            hint("↑↓ move · Enter finish · Esc back"),
        )

    if current == "setup-done":
        def on_pick(index: int, opt) -> None:
            screen.set("menu")

        s = settings_sig()
        sel = tracked_select("setup-done", [
            SelectOption("Start using Veritas", "go", "Back to the main menu"),
        ])
        sel.on("itemSelected", on_pick)
        return Box(
            frame(
                "All set",
                Text("Saved on this machine:", bold=True),
                *kv_rows(
                    [
                        ("NVIDIA key", mask_key(s.get("nvidia_api_key", ""))),
                        ("Tavily key", mask_key(s.get("tavily_api_key", ""))),
                        ("Reports", reports_dir(s)),
                        ("Theme", str(s.get("theme", "")).title()),
                    ]
                ),
                gap(),
                sel,
            ),
            gap(),
            hint("Enter for the menu"),
        )

    return Box(Text("Unknown screen", fg=DANGER))


def run() -> None:
    """Launch the OpenTUI app (blocking until quit)."""
    pump_stop.clear()
    # Fresh installs land in the setup wizard; everyone else gets the menu.
    try:
        current_settings = load_settings()
        nvidia_key, tavily_key, _ = resolve_api_keys(current_settings)
        screen.set("setup-welcome" if not (nvidia_key and tavily_key) else "menu")
    except Exception:
        screen.set("menu")

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
