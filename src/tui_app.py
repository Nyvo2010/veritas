"""Veritas OpenTUI frontend: arrow-key menus, round boxes, live run screen.

Screens (all keyboard driven):
  menu          banner + New research / Settings / Quit
  query         free-text question
  iterations    max iterations (1-20, defaults from settings)
  sources       per-run source toggles + Start research
  run           live braille progress bar + scrolling activity log
  done          result stats or error, back to menu
  settings      current values + editors
  set-theme / set-reports / set-provider / set-endpoint / set-llm-key /
  set-model / set-fallback / set-search / set-tavily / set-defaults /
  set-default / set-sources
  setup-*       first-run wizard

Lists are rendered by this module (see :func:`list_rows`) rather than by
OpenTUI's SelectRenderable, which always draws from the first option and so
cannot scroll. Ours keeps the highlight on screen, shows how many options are
above/below, and supports type-to-filter for model pickers.

Research runs in a worker thread via DeepResearchAgent with a QueueUI sink;
a pump thread drains the event queue into reactive signals. Falls back to the
Rich CLI when opentui cannot be imported.
"""

import asyncio
import os
import queue
import shutil
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from opentui import (
    Box,
    Input,
    Row,
    Signal,
    Text,
    component,
    render,
    use_keyboard,
    use_paste,
    use_renderer,
)

from settings import (
    load_settings,
    mask_key,
    reports_dir,
    resolve_api_keys,
    resolve_llm,
    update_settings,
)
from ui import theme, timings
from runner_ui import QueueUI
from utils.llm_client import fetch_models
from main import DeepResearchAgent


# --------------------------------------------------------------------------
# Palette — imported from ui.theme: one source shared by both frontends.
# --------------------------------------------------------------------------

ACCENTS = theme.ACCENTS

PANEL_BG = theme.PANEL_BG
TITLE_FG = theme.TITLE_FG
SECONDARY = theme.SECONDARY
TERTIARY = theme.TERTIARY
SUBTITLE = theme.SUBTITLE
DANGER = theme.DANGER
WARNING = theme.WARNING
INFO = theme.INFO

LOG_GLYPH = {
    "phase": "*",
    "search": ">",
    "insight": "-",
    "ok": "+",
    "warn": "!",
    "err": "x",
    "info": "·",
}

LOG_LIMIT = 300             # log lines kept in memory for a run

# One meaning per notice kind, used by every message in the app.
NOTICE_STYLE = {
    "error": (DANGER, "✗"),
    "warn": (WARNING, "!"),
    "info": (INFO, "·"),
}

THEME_BLURBS = {
    "blue": "Dusty blue (default)",
    "green": "Sage",
    "cyan": "Muted cadet teal",
    "magenta": "Muted mauve",
    "amber": "Muted brass",
    "red": "Muted brick",
    "orange": "Muted copper",
    "purple": "Muted lavender",
    "white": "Soft white",
}

# Shared geometry so every screen lines up on the same grid.
#
# Spacing rule, enforced by page()/panel()/input_frame() below: every block on
# a screen is separated by exactly one blank line, and the notice always sits
# directly above the hint so content never moves when a notice appears.
MARGIN_X = 2                # outer margin from the terminal edge
MARGIN_TOP = 1              # one blank line between the terminal's top edge and the UI
PANEL_PAD_X = 2             # padding inside every panel
INPUT_WIDTH = 64            # shared width for every text input
MAX_VISIBLE = 8             # list rows shown at once
ANIM_SECONDS = 0.09         # per animation frame: run clock + bar particles
ANIM_TRAVEL = 1.3           # bar cells a particle covers per animation frame


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------

# Screens that own a real text input (all keys are routed to it).
INPUT_SCREENS = {
    "query", "iterations", "set-reports", "set-endpoint", "set-llm-key",
    "set-tavily", "set-default",
    "setup-endpoint", "setup-llm-key", "setup-tavily", "setup-reports",
}

# Screens whose list supports type-to-search.
SEARCH_SCREENS = {"set-model", "set-fallback", "setup-model"}

BACK = {
    "query": "menu",
    "iterations": "query",
    "sources": "iterations",
    "run": None,
    "done": "menu",
    "settings": "menu",
    "set-theme": "settings",
    "set-reports": "settings",
    "set-provider": "settings",
    "set-endpoint": "set-provider",
    "set-llm-key": "set-provider",
    "set-model": "set-provider",
    "set-fallback": "set-provider",
    "set-search": "settings",
    "set-tavily": "set-search",
    "set-defaults": "settings",
    "set-default": "set-defaults",
    "set-sources": "set-defaults",
    "setup-welcome": "menu",
    "setup-endpoint": "setup-welcome",
    "setup-llm-key": "setup-endpoint",
    "setup-model": "setup-llm-key",
    "setup-tavily": "setup-model",
    "setup-reports": "setup-tavily",
    "setup-theme": "setup-reports",
    "setup-done": None,
    "logs": "menu",
}


# --------------------------------------------------------------------------
# Geometry + small formatting helpers
# --------------------------------------------------------------------------

def term_width() -> int:
    # Prefer the live renderer: it is the source of truth for the surface we
    # draw on (and the only one available under the test renderer).
    renderer = renderer_ref.get("r")
    if renderer is not None:
        try:
            return int(renderer.width)
        except Exception:
            pass
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def term_height() -> int:
    renderer = renderer_ref.get("r")
    if renderer is not None:
        try:
            return int(renderer.height)
        except Exception:
            pass
    try:
        return shutil.get_terminal_size().lines
    except Exception:
        return 24


def auto_visible() -> int:
    """How many options fit before the panel would run off screen.

    Panels here show up to two lines per option (title + subtitle); the
    remaining rows hold the banner, panel chrome, hints, and notices.
    """
    return max(3, min(MAX_VISIBLE, (term_height() - 10 - MARGIN_TOP) // 2))


def inner_width() -> int:
    """Usable width inside a panel: terminal minus the shared margins/padding."""
    return max(20, term_width() - 2 * MARGIN_X - 2 * PANEL_PAD_X)


def accent() -> str:
    return ACCENTS.get(str(settings_sig().get("theme", "blue")), ACCENTS["blue"])


def on_accent() -> str:
    """Readable text for accents used as backgrounds: saturated accents (blue,
    red, purple…) need white text, light accents (white, amber) need black."""
    hexv = accent().lstrip("#")
    r, g, b = (int(hexv[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return TITLE_FG if luminance > 140 else "#ffffff"


def log_color(level: str) -> str:
    """Body text for most levels: only phases, warnings, and errors pop."""
    if level == "phase":
        return accent()
    if level == "err":
        return DANGER
    if level == "warn":
        return WARNING
    if level == "info":
        return SUBTITLE
    return SECONDARY


def banner_lines() -> list:
    """Wordmark lines with their own blank padding stripped.

    The wide wordmark needs both room to breathe and vertical space; on
    shorter terminals the plain title keeps the menu on one screen. The
    layout adds the spacing, so the banner must not carry blanks of its own.
    """
    lines = list(theme.COMPACT_LOGO) if term_width() >= 96 and term_height() >= 32 else ["VERITAS"]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def gap(rows: int = 1) -> Text:
    """Fixed breathing room between blocks (rhythm, not decoration)."""
    return Text("\n".join([""] * rows))


def hint(text: str) -> Text:
    return Text(text, fg=TERTIARY)


def logo_lines() -> list:
    """Banner in the single accent color. Flat, no gradients, no animation."""
    return [Text(line, fg=accent()) for line in banner_lines()]


def fit(text: str, width: int) -> str:
    """Trim a string to *width* columns, ending with an ellipsis when cut."""
    text = str(text)
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def short_query(q: str, n: int = 70) -> str:
    q = " ".join(q.split())
    return q if len(q) <= n else q[: n - 3] + "..."


def endpoint_host(url: str) -> str:
    """Short host for display; falls back to the raw URL."""
    try:
        return urlparse(url or "").netloc or str(url or "")
    except Exception:
        return str(url or "")


def kv_rows(pairs: list) -> list:
    """Aligned stat rows: labels in one column, values in SECONDARY."""
    width = max([len(label) for label, _ in pairs] + [0])
    return [Text(f"{label:<{width}}  {value}", fg=SECONDARY) for label, value in pairs]


def make_input(**kw) -> Input:
    """Every text input shares one width and focus behavior.

    Width follows the live terminal: a fixed 64-wide input overflows its
    panel on narrow panes (<72 cols), which shoves the layout sideways.
    """
    kw.setdefault("width", max(20, min(INPUT_WIDTH, inner_width())))
    kw.setdefault("focused", True)
    return Input(**kw)


def text_input(name: str, **kw) -> Input:
    """One persistent Input per screen.

    OpenTUI patches a re-rendered node from the freshly built instance, so
    constructing a new, empty Input on every render would wipe whatever the
    user had typed (it happens as soon as an inline notice clears). Reusing
    the same instance makes the patch a no-op and keeps text and caret.
    """
    existing = inputs.get(name)
    if existing is None:
        existing = make_input(**kw)
        inputs[name] = existing
    return existing


def go(target: str) -> None:
    """Navigate to *target*, starting fresh on the screen we are entering."""
    clear_notice()
    if target in INPUT_SCREENS or target in SEARCH_SCREENS:
        inputs.pop(target, None)   # text-entry screens open empty
    if target in SEARCH_SCREENS:
        state["search"][target] = ""
    screen.set(target)


def provider_blurb(s=None) -> str:
    """One-line summary of the AI provider: model @ host."""
    llm = resolve_llm(s if s is not None else settings_sig())
    if not (llm["base_url"] and llm["model"]):
        return "Not configured"
    return f"{llm['model']} @ {endpoint_host(llm['base_url'])}"


# --------------------------------------------------------------------------
# Consistent panels, notices, and page margins
# --------------------------------------------------------------------------

def panel(title: str, *children) -> Box:
    """Quiet panel: bold text label, tinted body, no borders."""
    return Box(
        Text(title, bold=True),
        gap(),
        *children,
        bg=PANEL_BG,
        padding_y=1,
        padding_x=PANEL_PAD_X,
        width="100%",
    )


def page(*children, footer=None) -> Box:
    """Page skeleton: side margins, blocks, then the notice and the footer.

    The notice is rendered after the blocks so it never pushes them around:
    the panel above stays put and only the footer line moves, on every screen.
    """
    parts = list(children)
    notice = notice_row()
    if notice is not None:
        parts.append(notice)
    if footer is not None:
        parts.append(footer)
    return Box(*parts, padding_x=MARGIN_X, padding_top=MARGIN_TOP, width="100%")


def list_hint(enter: str, back: str = None, search: bool = False) -> Text:
    keys = "↑↓ move"
    if search:
        keys += " · Type to search"
    keys += f" · Enter {enter}"
    if back:
        keys += f" · Esc {back}"
    return hint(keys)


def input_frame(title: str, lines: list, inp, enter: str, back: str, kind: str = "Type") -> Box:
    """Text-entry skeleton: prompt lines, one blank line, the field, the hint."""
    interior = list(lines)
    if interior:
        interior.append(gap())   # one blank line between the prompt and the field
    interior.append(inp)
    return page(
        panel(title, *interior),
        gap(),
        footer=hint(f"{kind} · Enter {enter} · Esc {back}"),
    )


# --------------------------------------------------------------------------
# Global reactive state
# --------------------------------------------------------------------------

screen = Signal("menu")
settings_sig = Signal(load_settings())
log_sig = Signal(())
prog_sig = Signal((0.0, 1, "Starting..."))
phase_sig = Signal("Preparing...")
elapsed_sig = Signal("0s")
anim_sig = Signal(0)          # monotonic frame counter for the bar's particles
stats_sig = Signal(())        # ((kind, seconds, calls), ...) for the timing line
list_sig = Signal(0)          # bumped whenever a list selection/window changes
input_sig = Signal(0)         # bumped on every key routed to a text input
notice_sig = Signal(None)     # None | (kind, text) with kind in NOTICE_STYLE
# None = loading, ("ok", [ids]) | ("err", message) | ("manual", "")
models_sig = Signal(None)

state = {
    "query": "",
    "max_iter": 10,
    "toggles": {},
    "cursors": {},      # list name -> selected index
    "lists": {},        # list name -> [{"label", "value", "subtitle"}]
    "meta": {},         # list name -> {"activate", "visible", "searchable"}
    "search": {},       # searchable screen -> filter text
    "default_field": "",
    "result": None,
    "run_start": 0.0,
    "research_stop": False,
    "run_active": False,
    "run_done": False,
    "models_result": None,    # filled by the fetch thread, applied on-frame
    "models_applied": None,
}

inputs: dict = {}
renderer_ref: dict = {}
_keyboard_bound: dict = {}
_frame_bound: dict = {}
_frame_state = {"anim": 0.0}


def set_notice(kind: str, text: str) -> None:
    """Show one consistent notice line (kind: error | warn | info)."""
    notice_sig.set((kind, text))


def clear_notice() -> None:
    notice_sig.set(None)


def notice_row():
    """The active notice as a styled line, or None."""
    current = notice_sig()
    if not current:
        return None
    kind, text = current
    color, glyph = NOTICE_STYLE.get(kind, NOTICE_STYLE["info"])
    return Text(f"{glyph} {text}", fg=color)


def request_quit() -> None:
    """Stop any running research and ask the renderer to exit.

    The worker is blocked in a model or network call that quitting cannot
    interrupt, so what it has produced so far is written from here first —
    otherwise Ctrl+C throws away every finding collected so far.
    """
    if state.get("run_active") and not state.get("research_stop"):
        state["research_stop"] = True
        path, error = _save_partial(state.get("agent"))
        if path:
            set_notice("warn", f"Stopped early — partial report saved to {path}")
        elif error:
            set_notice("error", f"Stopped early — nothing could be saved: {error}")
    renderer = renderer_ref.get("r")
    if renderer is not None:
        try:
            renderer.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Scrollable list model
# --------------------------------------------------------------------------

def register_list(name: str, items: list, activate, visible: int = None,
                  searchable: bool = False) -> None:
    """Publish a list for the current screen and keep its cursor in range.

    visible=None sizes the window to the terminal (see auto_visible).
    """
    state["lists"][name] = items
    state["meta"][name] = {
        "activate": activate,
        "visible": visible,
        "searchable": searchable,
    }
    if items:
        idx = state["cursors"].get(name, 0)
        state["cursors"][name] = max(0, min(idx, len(items) - 1))
    else:
        state["cursors"][name] = 0


def list_cursor(name: str) -> int:
    return state["cursors"].get(name, 0)


def set_cursor(name: str, index: int) -> None:
    items = state["lists"].get(name, [])
    if not items:
        return
    state["cursors"][name] = max(0, min(index, len(items) - 1))
    list_sig.set(list_sig() + 1)


def move_cursor(name: str, delta: int) -> None:
    set_cursor(name, list_cursor(name) + delta)


def activate_current(name: str) -> None:
    meta = state["meta"].get(name)
    items = state["lists"].get(name, [])
    if not meta or not items:
        return
    idx = max(0, min(list_cursor(name), len(items) - 1))
    meta["activate"](items[idx])


def visible_rows(name: str) -> int:
    meta = state["meta"].get(name, {})
    return meta.get("visible") or auto_visible()


def list_rows(name: str, visible: int = None) -> list:
    """Render a scrollable window of *name*'s options.

    The selected row keeps the accent highlight and stays inside the window;
    rows above/below are summarised so long lists never lose the cursor.
    """
    visible = visible or visible_rows(name)
    items = state["lists"].get(name, [])
    width = inner_width()

    if not items:
        return [Text("No options available.", fg=TERTIARY)]

    index = max(0, min(list_cursor(name), len(items) - 1))
    size = max(1, min(visible, len(items)))
    start = 0
    if len(items) > size:
        start = index - size // 2
        start = max(0, min(start, len(items) - size))

    out: list = []
    if start > 0:
        out.append(Text(f"↑ {start} more", fg=TERTIARY))

    for i in range(start, min(start + size, len(items))):
        item = items[i]
        selected = i == index
        label = fit(item["label"], width)
        if selected:
            out.append(Text((" " + label).ljust(width), bg=accent(), fg=on_accent(), bold=True))
        else:
            out.append(Text(" " + label, fg=item.get("fg")))
        subtitle = item.get("subtitle")
        if subtitle:
            sub = fit(subtitle, width - 2)
            if selected:
                out.append(Text(("  " + sub).ljust(width), bg=accent(), fg=on_accent()))
            else:
                out.append(Text("  " + sub, fg=SUBTITLE))

    tail = len(items) - min(start + size, len(items))
    if tail > 0:
        out.append(Text(f"↓ {tail} more", fg=TERTIARY))
    return out


# --------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------

def kick_model_fetch(base_url: str, api_key: str) -> None:
    """Fetch /models once per endpoint+key; results land in models_sig."""
    tag = f"{base_url}\n{api_key}"
    if state.get("model_fetch_for") == tag and models_sig() is not None:
        return
    state["model_fetch_for"] = tag
    models_sig.set(None)

    def _go() -> None:
        try:
            ids = fetch_models(base_url, api_key)
            state["models_result"] = ("ok", ids)
        except Exception as e:  # noqa: BLE001 - surfaced as helper text
            state["models_result"] = ("err", str(e))

    threading.Thread(target=_go, daemon=True).start()


def _after_text_change() -> None:
    """Re-layout after the value of the focused field changed.

    A native repaint of the input alone accumulates a horizontal panel offset
    in this OpenTUI build (the text drifts right), so the whole tree has to be
    rebuilt. Typing, pasting, and deleting all go through here.
    """
    current = screen()
    target = inputs.get(current)
    if target is not None and current in SEARCH_SCREENS:
        state["search"][current] = target.value
    input_sig.set(input_sig.peek() + 1)


def _on_paste(event) -> None:
    """Bracketed paste lands here; forward it to the focused field.

    Module-level so the hook dedup keeps exactly one registration.
    Newlines collapse to spaces: every field here is single-line.
    """
    current = screen()
    text = str(getattr(event, "text", "") or "")
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return
    target = inputs.get(current)
    try:
        clear_notice()
        if target is not None:
            target.insert_text(text)
            target._cursor_position = len(target._value)
        elif current in SEARCH_SCREENS:
            # Model picker in list mode: the paste is a filter, not a value.
            state["search"][current] = (state["search"].get(current, "") + text)
            set_cursor(current, 0)
        else:
            return
        _after_text_change()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Research wiring
# --------------------------------------------------------------------------

def apply_event(ev: tuple) -> None:
    kind = ev[0]
    if kind == "log":
        _, level, text = ev
        # Entries carry the run clock so the log reads as a timeline and slow
        # steps are visible at a glance.
        elapsed = time.time() - state.get("run_start", time.time())
        entries = log_sig() + ((elapsed, level, text),)
        log_sig.set(entries[-LOG_LIMIT:])
    elif kind == "status":
        completed, total, _ = prog_sig()
        prog_sig.set((completed, total, ev[1]))
    elif kind == "phase":
        phase_sig.set(ev[1])
    elif kind == "progress":
        _, completed, total, status = ev
        prog_sig.set((float(completed), int(total), status))


def _on_frame(delta_time: float) -> None:
    """Fold worker-thread results into signals, on the render thread.

    OpenTUI signals notify subscribers synchronously, so mutating them from a
    background thread rebuilds the component tree off-thread and competes with
    input handling (key presses then get dropped). Worker threads only publish
    plain data; this once-per-renderer callback applies it at a fixed cadence.
    """
    # Model discovery result from its fetch thread.
    result = state.get("models_result")
    if result is not None and state.get("models_applied") is not result:
        state["models_applied"] = result
        models_sig.set(result)

    # Live research run: drain the event queue, refresh spinner + clock.
    ui = state.get("ui")
    if ui is not None and state.get("run_active"):
        try:
            while True:
                apply_event(ui.events.get_nowait())
        except queue.Empty:
            pass
        _frame_state["anim"] += delta_time
        if _frame_state["anim"] >= ANIM_SECONDS:
            _frame_state["anim"] = 0.0
            anim_sig.set(anim_sig.peek() + 1)
            elapsed_sig.set(theme.format_duration(time.time() - state["run_start"]))
        # Where the time is going: poll the worker's timing ledger.
        snapshot = ui.timings.snapshot()
        if snapshot != stats_sig.peek():
            stats_sig.set(snapshot)
        if state.get("run_done"):
            state["run_active"] = False
            screen.set("done")


def worker(query: str, max_iter: int, toggles: dict) -> None:
    agent = None
    try:
        agent = DeepResearchAgent(category_toggles=dict(toggles))
        # Published so quitting mid-run can still save what exists.
        state["agent"] = agent
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
            "save_error": agent.save_error,
        }
    except Exception as e:  # noqa: BLE001 - surfaced on the done screen
        # The run stopped early, but the findings are still worth keeping: try
        # to save a partial report rather than throwing the work away.
        saved, save_error = _save_partial(agent)
        state["result"] = {
            "ok": False,
            "error": str(e),
            "saved": saved,
            "save_error": save_error,
        }
    finally:
        # Hand the result back to the frame callback; never touch signals here.
        state["run_done"] = not state["research_stop"]


def _save_partial(agent) -> tuple:
    """Best-effort save of an unfinished run; returns (path, error)."""
    if agent is None or getattr(agent, "last_state", None) is None:
        return "", ""
    try:
        path = agent.save_report(agent.last_state)
    except Exception as e:  # noqa: BLE001 - reported on the done screen
        return "", str(e)
    return path or "", agent.save_error


def start_run() -> None:
    state["run_start"] = time.time()
    log_sig.set(())
    prog_sig.set((0.0, 1, "Starting..."))
    phase_sig.set("Preparing...")
    elapsed_sig.set("0s")
    anim_sig.set(0)
    stats_sig.set(())
    state["result"] = None
    state["run_done"] = False
    state["run_active"] = True
    clear_notice()
    state["ui"] = QueueUI()
    go("run")
    threading.Thread(
        target=worker,
        args=(state["query"], state["max_iter"], state["toggles"]),
        daemon=True,
    ).start()


# --------------------------------------------------------------------------
# Model picker (shared by primary + fallback + setup)
# --------------------------------------------------------------------------

def model_screen(name: str, title: str, base_url: str, api_key: str,
                 current_model: str, commit, extra_items: list = None) -> Box:
    """Model picker with autodetect and type-to-search.

    Lists /models when reachable; otherwise falls back to manual entry with
    the fetch error shown. commit(model_id) saves and navigates.
    """
    kick_model_fetch(base_url, api_key)
    res = models_sig()

    def submit(value: str) -> None:
        raw = value.strip()
        if not raw:
            set_notice("error", "Type a model id, or Esc to go back.")
            return
        clear_notice()
        commit(raw)

    if res is None:
        return page(
            panel(title, Text("Loading models…", fg=SECONDARY)),
            gap(),
            footer=hint("Esc back"),
        )

    if not isinstance(res, tuple) or res[0] in ("err", "manual"):
        manual_input = text_input(
            name,
            value="",
            placeholder="Model id (empty keeps current)",
            on_submit=submit,
        )
        lines = [Text(f"Current: {current_model}")]
        if isinstance(res, tuple) and res[0] == "err":
            lines.append(Text(f"Could not list models: {res[1]}", fg=DANGER))
        return input_frame(title, lines, manual_input, enter="save", back="back")

    inputs.pop(name, None)
    _, ids = res
    query = state["search"].get(name, "")
    needle = query.strip().lower()
    filtered = [mid for mid in ids if needle in mid.lower()] if needle else list(ids)

    host = endpoint_host(base_url)
    items: list = []
    for extra in (extra_items or []):
        items.append(extra)
    for mid in filtered:
        items.append({"label": mid, "value": mid, "subtitle": host})
    # Offer the typed text as a manual id when it is not an exact match.
    if needle and not any(mid.lower() == needle for mid in ids):
        items.append({
            "label": f'Use "{query.strip()}"',
            "value": f"__use__{query.strip()}",
            "subtitle": "Not listed — use this id",
        })

    def activate(item) -> None:
        value = item["value"]
        if isinstance(value, str) and value.startswith("__use__"):
            commit(value[len("__use__"):])
        elif value == "":
            commit("")
        else:
            commit(value)

    register_list(name, items, activate, searchable=True)
    state["meta"][name]["commit_query"] = commit if needle else None

    search_line = (
        Text(f"Search: {query}", fg=None)
        if query
        else Text("Type to search, or pick from the list…", fg=TERTIARY)
    )
    footer = f"{len(filtered)} of {len(ids)} models"
    return page(
        panel(title, search_line, gap(), *list_rows(name)),
        gap(),
        footer=hint(f"{footer} · ↑↓ move · Type to search · Enter use · Esc back"),
    )


# --------------------------------------------------------------------------
# Screen builders
# --------------------------------------------------------------------------

def screen_menu() -> Box:
    def activate(item) -> None:
        value = item["value"]
        if value == "new":
            state["query"] = ""
            state["search"] = {}
            clear_notice()
            go("query")
        elif value == "settings":
            go("settings")
        elif value == "logs":
            go("logs")
        elif value == "quit":
            request_quit()

    items = [
        {"label": "New research", "value": "new", "subtitle": "Answer a question with deep web research"},
    ]
    # Only offer the log once there is one to read.
    entries = log_sig()
    if entries:
        items.append({
            "label": "Last run log",
            "value": "logs",
            "subtitle": f"{len(entries)} lines from the previous run",
        })
    items.extend([
        {"label": "Settings", "value": "settings", "subtitle": "Theme, AI provider, search, defaults"},
        {"label": "Quit", "value": "quit", "subtitle": "Exit Veritas"},
    ])
    register_list("menu", items, activate, visible=len(items))

    return page(
        Box(*logo_lines()),
        gap(1),
        panel("Main menu", *list_rows("menu")),
        gap(),
        footer=hint("↑↓ move · Enter select · q quit"),
    )


def screen_query() -> Box:
    def submit(value: str) -> None:
        q = value.strip()
        if len(q.split()) < 2:
            set_notice("error", "Type a full question, not a single word.")
            return
        clear_notice()
        state["query"] = q
        state["max_iter"] = int(settings_sig().get("max_iterations", 10))
        go("iterations")

    query_input = text_input(
        "query",
        value=state["query"],
        placeholder="Type your question...",
        on_submit=submit,
    )
    return input_frame(
        "New research (1/3)",
        [Text("What do you want to research?")],
        query_input,
        enter="continue",
        back="back",
    )


def screen_iterations() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if raw and (not raw.isdigit() or not 1 <= int(raw) <= 20):
            set_notice("error", "Enter a number from 1 to 20.")
            return
        clear_notice()
        if raw:
            state["max_iter"] = int(raw)
        state["toggles"] = dict(settings_sig().get("category_toggles", {}))
        state["cursors"]["sources"] = 0
        go("sources")

    iter_input = text_input(
        "iterations",
        value="",
        placeholder=f"1-20 (empty keeps {state['max_iter']})",
        on_submit=submit,
    )
    return input_frame(
        "New research (2/3)",
        [
            Text(f"Query: {short_query(state['query'])}"),
            Text("Max iterations:"),
        ],
        iter_input,
        enter="continue",
        back="back",
    )


def screen_sources() -> Box:
    toggles = state["toggles"]

    def activate(item) -> None:
        value = item["value"]
        if value == "__start__":
            start_run()
        elif value in toggles:
            toggles[value] = not toggles[value]
            list_sig.set(list_sig() + 1)

    items = []
    for name, enabled in toggles.items():
        items.append({
            "label": f"{'●' if enabled else '○'} {name.replace('_', ' ').title()}",
            "value": name,
            "subtitle": "Included in this run" if enabled else "Skipped",
        })
    items.append({"label": "Start research", "value": "__start__", "subtitle": "Begin with these sources"})
    register_list("sources", items, activate)

    enabled_count = sum(1 for v in toggles.values() if v)
    return page(
        panel(
            "New research (3/3)",
            Text(f"Query: {short_query(state['query'])}  ·  Iterations: {state['max_iter']}"),
            Text(f"Sources: {enabled_count}/{len(toggles)} enabled", fg=SECONDARY),
            gap(),
            *list_rows("sources"),
        ),
        gap(),
        footer=hint("↑↓ move · Enter toggle · Esc back"),
    )


def screen_run() -> Box:
    """Live run screen: context, the activity log, and the braille bar.

    The log block is padded to a fixed number of rows so the screen never
    grows or shrinks while events stream in.
    """
    return page(
        panel(
            "Research running",
            Text(f"Query: {short_query(state['query'])}", bold=True),
            Text(lambda: phase_sig(), fg=accent()),
            gap(),
            *log_block(activity_rows()),
            gap(),
            Row(
                Text(lambda: progress_parts()[0], fg=accent()),
                Text(lambda: progress_parts()[1], fg=TERTIARY),
                Text(lambda: progress_parts()[2], fg=SECONDARY),
            ),
            Text(lambda: clock_line(), fg=SECONDARY),
            Text(lambda: timing_line(), fg=TERTIARY),
        ),
        gap(),
        footer=hint("Ctrl+C quits — whatever is finished is saved"),
    )


def screen_logs() -> Box:
    """Scrollable history of the last run's activity log."""

    def activate(_item) -> None:
        return None

    entries = log_sig()
    items = []
    for index, (elapsed, level, text) in enumerate(entries):
        if level == "phase":
            label = f"▎{text}"
        else:
            label = f"{theme.format_duration(elapsed):>7}  {LOG_GLYPH.get(level, '·')} {text}"
        items.append({"label": label, "value": index, "fg": log_color(level)})
    register_list("logs", items, activate, visible=max(4, term_height() - 13 - MARGIN_TOP))

    return page(
        panel(
            "Activity log",
            Text(f"Query: {short_query(state['query'])}"),
            Text(f"{len(entries)} lines · from the most recent run", fg=TERTIARY),
            gap(),
            *list_rows("logs"),
        ),
        gap(),
        footer=hint("↑↓ move · PgUp/PgDn page · Home/End jump · Esc menu"),
    )


def save_lines(result: dict) -> list:
    """Blank line plus where the report went — or why it did not."""
    saved = result.get("saved")
    if saved:
        label = "Saved to" if result.get("ok") else "Partial report saved to"
        return [gap(), Text(f"{label}: {fit(saved, inner_width() - len(label) - 3)}", fg=SECONDARY)]
    error = result.get("save_error")
    if error:
        return [gap(), Text(f"Not saved: {error}", fg=WARNING)]
    return []


def screen_done() -> Box:
    result = state["result"] or {"ok": False, "error": "No result recorded."}
    if result.get("ok"):
        rows = kv_rows([
            ("Duration", theme.format_duration(result.get("duration", 0))),
            ("Iterations", str(result.get("iterations", 0))),
            ("Questions", str(result.get("questions", 0))),
            ("Searches", str(result.get("searches", 0))),
            ("Findings", str(result.get("findings", 0))),
            ("Gaps", str(result.get("gaps", 0))),
            ("Report", f"{result.get('report_len', 0):,} chars"),
        ])
        return page(
            panel("Research complete", *rows, *save_lines(result)),
            gap(),
            footer=hint("Enter menu · l activity log"),
        )
    return page(
        panel(
            "Research failed",
            Text(str(result.get("error", "Unknown error")), fg=DANGER),
            *save_lines(result),
        ),
        gap(),
        footer=hint("Enter menu · l activity log"),
    )


def screen_settings() -> Box:
    def on_pick(item) -> None:
        value = item["value"]
        clear_notice()
        if value == "back":
            go("menu")
        else:
            go(value)

    s = settings_sig()
    toggles = s.get("category_toggles", {})
    on = sum(1 for v in toggles.values() if v)
    tavily = mask_key(s.get("tavily_api_key", ""))

    items = [
        {"label": "Color theme", "value": "set-theme", "subtitle": f"Current: {str(s.get('theme', '')).title()}"},
        {"label": "Reports folder", "value": "set-reports", "subtitle": reports_dir(s)},
        {"label": "AI provider", "value": "set-provider", "subtitle": provider_blurb(s)},
        {"label": "Search", "value": "set-search", "subtitle": f"Tavily {tavily}"},
        {"label": "Research defaults", "value": "set-defaults", "subtitle": f"Iterations + results + sources {on}/{len(toggles)}"},
        {"label": "Setup wizard", "value": "setup-welcome", "subtitle": "Keys, folder, theme from scratch"},
        {"label": "Back", "value": "back", "subtitle": "Main menu"},
    ]
    register_list("settings", items, on_pick)
    return page(
        panel("Settings", *list_rows("settings")),
        gap(),
        footer=list_hint("edit", "menu"),
    )


def theme_options() -> list:
    current = str(settings_sig().get("theme", ""))
    items = []
    for name in theme.THEMES:
        blurb = THEME_BLURBS.get(name, "")
        if name == current:
            blurb = f"{blurb} · current" if blurb else "current"
        items.append({"label": name.title(), "value": name, "subtitle": blurb})
    return items


def screen_set_theme() -> Box:
    def on_pick(item) -> None:
        update_settings(theme=item["value"])
        settings_sig.set(load_settings())
        go("settings")

    register_list("set-theme", theme_options(), on_pick)
    return page(
        panel("Color theme", *list_rows("set-theme")),
        gap(),
        footer=list_hint("apply", "back"),
    )


def screen_set_provider() -> Box:
    def on_pick(item) -> None:
        value = item["value"]
        if value == "back":
            go("settings")
        else:
            clear_notice()
            go(value)

    llm = resolve_llm(settings_sig())
    fallback = llm.get("fallback_model") or "not set"
    items = [
        {"label": "Endpoint", "value": "set-endpoint", "subtitle": llm["base_url"]},
        {"label": "API key", "value": "set-llm-key", "subtitle": f"Key {mask_key(llm['api_key'])}"},
        {"label": "Model", "value": "set-model", "subtitle": llm["model"]},
        {"label": "Fallback model", "value": "set-fallback", "subtitle": f"Used if the model keeps failing · {fallback}"},
        {"label": "Back", "value": "back", "subtitle": "Settings"},
    ]
    register_list("set-provider", items, on_pick)
    return page(
        panel("AI provider", *list_rows("set-provider")),
        gap(),
        footer=list_hint("edit", "back"),
    )


def screen_set_endpoint() -> Box:
    def submit(value: str) -> None:
        raw = value.strip().rstrip("/")
        if not raw:
            clear_notice()
            go("set-provider")
            return
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            set_notice("error", "Enter a full URL with scheme and host.")
            return
        clear_notice()
        update_settings(llm_base_url=raw)
        settings_sig.set(load_settings())
        go("set-provider")

    endpoint_input = text_input(
        "set-endpoint",
        value="",
        placeholder="Any OpenAI-compatible URL (empty keeps current)",
        on_submit=submit,
    )
    return input_frame(
        "Endpoint",
        [Text(f"Current: {resolve_llm(settings_sig())['base_url']}")],
        endpoint_input,
        enter="save",
        back="back",
    )


def screen_set_llm_key() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if raw:
            update_settings(llm_api_key=raw)
            settings_sig.set(load_settings())
        clear_notice()
        go("set-provider")

    llm_key_input = text_input(
        "set-llm-key",
        placeholder="Paste a new key, Enter to save",
        on_submit=submit,
    )
    return input_frame("API key", [], llm_key_input, enter="save", back="back", kind="Paste")


def screen_set_model() -> Box:
    llm = resolve_llm(settings_sig())

    def commit(model) -> None:
        if model:
            update_settings(llm_model=model)
            settings_sig.set(load_settings())
        clear_notice()
        go("set-provider")

    return model_screen("set-model", "AI model", llm["base_url"], llm["api_key"], llm["model"], commit)


def screen_set_fallback() -> Box:
    llm = resolve_llm(settings_sig())

    def commit(model) -> None:
        if model or model == "":
            update_settings(llm_fallback_model=model)
            settings_sig.set(load_settings())
        clear_notice()
        go("set-provider")

    return model_screen(
        "set-fallback", "Fallback model", llm["base_url"], llm["api_key"],
        llm.get("fallback_model") or "not set", commit,
        extra_items=[{"label": "No fallback", "value": "", "subtitle": "Use the primary model only"}],
    )


def screen_set_search() -> Box:
    def on_pick(item) -> None:
        value = item["value"]
        if value == "back":
            go("settings")
        else:
            clear_notice()
            go(value)

    s = settings_sig()
    items = [
        {"label": "Tavily API key", "value": "set-tavily", "subtitle": mask_key(s.get("tavily_api_key", ""))},
        {"label": "Back", "value": "back", "subtitle": "Settings"},
    ]
    register_list("set-search", items, on_pick)
    return page(
        panel("Search", *list_rows("set-search")),
        gap(),
        footer=list_hint("edit", "back"),
    )


def screen_set_tavily() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if raw:
            update_settings(tavily_api_key=raw)
            settings_sig.set(load_settings())
        clear_notice()
        go("set-search")

    tavily_input = text_input(
        "set-tavily",
        placeholder="Paste a new key, Enter to save",
        on_submit=submit,
    )
    return input_frame("Tavily API key", [], tavily_input, enter="save", back="back", kind="Paste")


def screen_set_sources() -> Box:
    toggles = settings_sig().get("category_toggles", {})

    def activate(item) -> None:
        value = item["value"]
        if value == "__back__":
            go("set-defaults")
            return
        if value in toggles:
            toggles[value] = not toggles[value]
            update_settings(category_toggles={value: toggles[value]})
            settings_sig.set(load_settings())

    items = []
    for name, enabled in toggles.items():
        items.append({
            "label": f"{'●' if enabled else '○'} {name.replace('_', ' ').title()}",
            "value": name,
            "subtitle": "Enabled" if enabled else "Disabled",
        })
    items.append({"label": "Back", "value": "__back__", "subtitle": "Research defaults"})
    register_list("set-sources", items, activate)

    on = sum(1 for v in toggles.values() if v)
    return page(
        panel(
            f"Default sources ({on}/{len(toggles)})",
            Text("Applied to every new research run.", fg=SECONDARY),
            gap(),
            *list_rows("set-sources"),
        ),
        gap(),
        footer=list_hint("toggle", "back"),
    )


def screen_set_defaults() -> Box:
    def on_pick(item) -> None:
        value = item["value"]
        if value == "back":
            go("settings")
        elif value == "set-sources":
            clear_notice()
            go("set-sources")
        else:
            state["default_field"] = value
            clear_notice()
            go("set-default")

    s = settings_sig()
    toggles = s.get("category_toggles", {})
    on = sum(1 for v in toggles.values() if v)
    items = [
        {"label": "Max iterations", "value": "max_iterations", "subtitle": f"Current: {s.get('max_iterations', '')}"},
        {"label": "Max search results", "value": "max_search_results", "subtitle": f"Current: {s.get('max_search_results', '')}"},
        {"label": "Default sources", "value": "set-sources", "subtitle": f"{on}/{len(toggles)} categories enabled"},
        {"label": "Back", "value": "back", "subtitle": "Settings"},
    ]
    register_list("set-defaults", items, on_pick)
    return page(
        panel("Research defaults", *list_rows("set-defaults")),
        gap(),
        footer=list_hint("edit", "back"),
    )


def screen_set_default() -> Box:
    field = state["default_field"]
    label = "Max iterations" if field == "max_iterations" else "Max search results"

    def submit(value: str) -> None:
        raw = value.strip()
        if not raw:
            clear_notice()
            go("set-defaults")
            return
        if not raw.isdigit() or not 1 <= int(raw) <= 20:
            set_notice("error", "Enter a number from 1 to 20.")
            return
        clear_notice()
        update_settings(**{field: int(raw)})
        settings_sig.set(load_settings())
        go("set-defaults")

    default_input = text_input(
        "set-default",
        value="",
        placeholder="1-20 (empty keeps current)",
        on_submit=submit,
    )
    return input_frame(
        label,
        [Text(f"Current: {settings_sig().get(field, '')}")],
        default_input,
        enter="save",
        back="back",
    )


def screen_set_reports() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if not raw:
            clear_notice()
            go("settings")
            return
        expanded = os.path.expanduser(raw)
        try:
            os.makedirs(expanded, exist_ok=True)
        except OSError as e:
            set_notice("error", f"Cannot use that folder: {e}")
            return
        clear_notice()
        update_settings(reports_dir=raw)
        settings_sig.set(load_settings())
        go("settings")

    reports_input = text_input(
        "set-reports",
        value="",
        placeholder="New path (empty keeps current)",
        on_submit=submit,
    )
    return input_frame(
        "Reports folder",
        [Text(f"Current: {reports_dir(settings_sig())}")],
        reports_input,
        enter="save",
        back="back",
    )


# -- Setup wizard ----------------------------------------------------------

def screen_setup_welcome() -> Box:
    def on_pick(item) -> None:
        if item["value"] == "start":
            state["setup"] = {}
            state["search"] = {}
            clear_notice()
            go("setup-endpoint")
        else:
            go("menu")

    register_list("setup-welcome", [
        {"label": "Continue setup", "value": "start", "subtitle": "Takes about a minute"},
        {"label": "Back", "value": "menu", "subtitle": "Main menu"},
    ], on_pick, visible=2)

    return page(
        Box(*logo_lines()),
        gap(1),
        panel(
            "Welcome to Veritas",
            Text("Three things to get going: an API key for the AI brain, a Tavily key for web search, and a folder for reports."),
            Text("Any OpenAI-compatible endpoint works. Keys are stored only on this machine.", fg=SECONDARY),
            gap(),
            *list_rows("setup-welcome"),
        ),
        gap(),
        footer=hint("↑↓ move · Enter choose · Esc menu"),
    )


def screen_setup_endpoint() -> Box:
    def submit(value: str) -> None:
        raw = value.strip().rstrip("/")
        if not raw:
            set_notice("error", "Enter your endpoint URL to continue.")
            return
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            set_notice("error", "Enter a full URL with scheme and host.")
            return
        clear_notice()
        state.setdefault("setup", {})["llm_base_url"] = raw
        go("setup-llm-key")

    setup_endpoint_input = text_input(
        "setup-endpoint",
        value="",
        placeholder="Any OpenAI-compatible URL, e.g. https://host/v1",
        on_submit=submit,
    )
    return input_frame(
        "Setup (1/6)",
        [Text("AI endpoint", bold=True)],
        setup_endpoint_input,
        enter="continue",
        back="stop",
    )


def screen_setup_llm_key() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if not raw:
            set_notice("error", "Paste a key to continue, or Esc to stop.")
            return
        clear_notice()
        state.setdefault("setup", {})["llm_api_key"] = raw
        go("setup-model")

    setup_key_input = text_input(
        "setup-llm-key",
        placeholder="Paste your key for the endpoint above",
        on_submit=submit,
    )
    return input_frame(
        "Setup (2/6)",
        [Text("API key", bold=True)],
        setup_key_input,
        enter="continue",
        back="stop",
        kind="Paste",
    )


def screen_setup_model() -> Box:
    setup = state.get("setup", {})
    base_url = setup.get("llm_base_url", "")
    api_key = setup.get("llm_api_key", "")

    def commit(model) -> None:
        if model:
            state.setdefault("setup", {})["llm_model"] = model
        clear_notice()
        go("setup-tavily")

    return model_screen(
        "setup-model", "Setup (3/6) · AI model", base_url, api_key,
        setup.get("llm_model") or resolve_llm(settings_sig())["model"] or "not set",
        commit,
    )


def screen_setup_tavily() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if not raw:
            set_notice("error", "Paste a key to continue, or Esc to stop.")
            return
        clear_notice()
        state.setdefault("setup", {})["tavily_api_key"] = raw
        go("setup-reports")

    tavily_input = text_input(
        "setup-tavily",
        placeholder="Paste your key from tavily.com",
        on_submit=submit,
    )
    return input_frame(
        "Setup (4/6)",
        [Text("Tavily API key", bold=True)],
        tavily_input,
        enter="continue",
        back="stop",
        kind="Paste",
    )


def screen_setup_reports() -> Box:
    def submit(value: str) -> None:
        raw = value.strip()
        if raw:
            expanded = os.path.expanduser(raw)
            try:
                os.makedirs(expanded, exist_ok=True)
            except OSError as e:
                set_notice("error", f"Cannot use that folder: {e}")
                return
            state.setdefault("setup", {})["reports_dir"] = raw
        clear_notice()
        go("setup-theme")

    reports_input = text_input(
        "setup-reports",
        value="",
        placeholder="Empty keeps the default",
        on_submit=submit,
    )
    return input_frame(
        "Setup (5/6)",
        [
            Text("Reports folder", bold=True),
            Text(f"Default: {reports_dir(settings_sig())}", fg=SECONDARY),
        ],
        reports_input,
        enter="continue",
        back="stop",
    )


def screen_setup_theme() -> Box:
    def on_pick(item) -> None:
        setup = state.get("setup", {})
        update_settings(
            llm_api_key=setup.get("llm_api_key", ""),
            llm_base_url=setup.get("llm_base_url", ""),
            **(({"llm_model": setup["llm_model"]} if setup.get("llm_model") else {})),
            tavily_api_key=setup.get("tavily_api_key", ""),
            **(({"reports_dir": setup["reports_dir"]} if "reports_dir" in setup else {})),
            theme=item["value"],
        )
        settings_sig.set(load_settings())
        go("setup-done")

    register_list("setup-theme", theme_options(), on_pick)
    return page(
        panel("Setup (6/6)", Text("Pick a color theme", bold=True), gap(), *list_rows("setup-theme")),
        gap(),
        footer=hint("↑↓ move · Enter finish · Esc back"),
    )


def screen_setup_done() -> Box:
    s = settings_sig()
    return page(
        panel(
            "All set",
            Text("Saved on this machine:", bold=True),
            *kv_rows([
                ("Endpoint", str(s.get("llm_base_url", ""))),
                ("Model", str(s.get("llm_model", ""))),
                ("LLM key", mask_key(s.get("llm_api_key", ""))),
                ("Tavily key", mask_key(s.get("tavily_api_key", ""))),
                ("Reports", reports_dir(s)),
                ("Theme", str(s.get("theme", "")).title()),
            ]),
        ),
        gap(),
        footer=hint("Enter for the menu"),
    )


SCREEN_BUILDERS = {
    "menu": screen_menu,
    "query": screen_query,
    "iterations": screen_iterations,
    "sources": screen_sources,
    "run": screen_run,
    "done": screen_done,
    "logs": screen_logs,
    "settings": screen_settings,
    "set-theme": screen_set_theme,
    "set-reports": screen_set_reports,
    "set-provider": screen_set_provider,
    "set-endpoint": screen_set_endpoint,
    "set-llm-key": screen_set_llm_key,
    "set-model": screen_set_model,
    "set-fallback": screen_set_fallback,
    "set-search": screen_set_search,
    "set-tavily": screen_set_tavily,
    "set-defaults": screen_set_defaults,
    "set-default": screen_set_default,
    "set-sources": screen_set_sources,
    "setup-welcome": screen_setup_welcome,
    "setup-endpoint": screen_setup_endpoint,
    "setup-llm-key": screen_setup_llm_key,
    "setup-model": screen_setup_model,
    "setup-tavily": screen_setup_tavily,
    "setup-reports": screen_setup_reports,
    "setup-theme": screen_setup_theme,
    "setup-done": screen_setup_done,
}


# --------------------------------------------------------------------------
# Progress bar + live activity
# --------------------------------------------------------------------------

def bar_width() -> int:
    """Bar width: the panel interior minus the percent suffix (" 100%")."""
    return max(12, inner_width() - 5)


def particles_for(width: int) -> int:
    """Scale the particle count with the bar so wide bars never look empty."""
    return max(3, min(14, width // 11))


def activity_rows() -> int:
    """How many activity lines the run screen shows at this terminal height.

    Everything else on the screen is fixed height (two panel paddings, the
    title, query and phase lines, the bar and its two stat lines, the gaps,
    and the hint), so the log takes what is left.
    """
    return max(4, term_height() - 14 - MARGIN_TOP)


def progress_parts() -> tuple:
    """The bar as (filled, track, percent) so each part can be colored."""
    completed, total, _ = prog_sig()
    ratio = min(1.0, completed / total) if total else 0.0
    width = bar_width()
    filled, track = theme.braille_bar(
        ratio, width, anim_sig() * ANIM_TRAVEL, particles_for(width)
    )
    return filled, track, f" {ratio * 100:.0f}%"


def clock_line() -> str:
    """Elapsed time plus a rough estimate of what is left."""
    line = f"Elapsed {elapsed_sig()}"
    completed, total, _ = prog_sig()
    elapsed = time.time() - state.get("run_start", 0.0)
    if 0 < completed < total and 0 < elapsed < 86400:
        remaining = (elapsed / completed) * (total - completed)
        line += f"  ·  eta {theme.format_eta(remaining)}"
    return line


def timing_line() -> str:
    """Where the run has spent its time: model inference vs web search."""
    snapshot = stats_sig()
    calls = sum(count for _, _, count in snapshot)
    if not calls:
        return "Timing appears after the first model call..."
    parts = [
        f"{timings.LABELS.get(kind, kind.title())} {theme.format_duration(seconds)} ({count})"
        for kind, seconds, count in snapshot
    ]
    parts.append(f"{calls} calls")
    return "  ·  ".join(parts)


def activity_line(elapsed: float, level: str, text: str) -> Text:
    """One log line: time gutter, level glyph, and a single-line message.

    Messages are trimmed to the panel so the log block keeps its height: a
    wrapped line would push the progress bar down mid-run. Phase entries are
    headings — no gutter, accented bar, bold — so a run reads as sections.
    """
    if level == "phase":
        return Text(f"{'':>7}  ▎{fit(text, max(8, inner_width() - 11))}", fg=accent(), bold=True)
    room = max(8, inner_width() - 11)   # gutter + glyph + spaces
    return Text(
        f"{theme.format_duration(elapsed):>7}  {LOG_GLYPH.get(level, '·')} {fit(text, room)}",
        fg=log_color(level),
    )


def log_block(rows: int) -> list:
    """The tail of the activity log as exactly *rows* lines, headings spaced.

    Newest lines sit at the bottom (just above the progress bar) and a blank
    line is inserted before every heading, so phases group their own steps.
    """
    out: list = []
    for elapsed, level, text in reversed(log_sig()):
        if level == "phase":
            if out:
                out.append(Text(" "))
            out.append(activity_line(elapsed, level, text))
        else:
            out.append(activity_line(elapsed, level, text))
        if len(out) >= rows:
            break
    out = out[:rows]
    out.reverse()
    if not out:
        out = [Text("Waiting for activity...", fg=SECONDARY)]
    # Distinct instances (not a repeated one), each exactly one line tall.
    return [Text(" ") for _ in range(rows - len(out))] + out


# --------------------------------------------------------------------------
# Key handling
# --------------------------------------------------------------------------

def _text_key(char: str, modifiers: bool) -> bool:
    return len(char) == 1 and char.isprintable() and not modifiers


def _handle_input_key(e, target) -> None:
    """Route one key to a text input.

    Printable characters bypass handle_key because this OpenTUI port
    lowercases them (which would corrupt API keys, paths, and queries).
    The cursor is pinned to the end afterwards because this port's
    delete_char double-decrements it; caret navigation is not supported
    until that is fixed upstream.
    """
    key = getattr(e, "key", "") or ""
    modifiers = bool(
        getattr(e, "ctrl", False) or getattr(e, "alt", False) or getattr(e, "meta", False)
    )
    char = getattr(e, "sequence", "") or key
    if _text_key(char, modifiers):
        if notice_sig():
            clear_notice()
        target.insert_text(char)
    else:
        target.handle_key(e)
    try:
        target._cursor_position = len(target._value)
    except Exception:
        pass
    # Full rebuild (not just the input node's native repaint: that path
    # accumulates a horizontal panel offset per keystroke in this OpenTUI
    # build — see _after_text_change).
    _after_text_change()


def _handle_search_key(e, name: str) -> None:
    """Type-to-search list: printable chars filter, arrows move, Enter picks."""
    key = getattr(e, "key", "") or ""
    modifiers = bool(
        getattr(e, "ctrl", False) or getattr(e, "alt", False) or getattr(e, "meta", False)
    )
    char = getattr(e, "sequence", "") or key
    query = state["search"].get(name, "")

    if key in ("up", "k"):
        move_cursor(name, -1)
        return
    if key in ("down", "j"):
        move_cursor(name, 1)
        return
    if key == "pageup":
        move_cursor(name, -visible_rows(name))
        return
    if key == "pagedown":
        move_cursor(name, visible_rows(name))
        return
    if key == "home":
        set_cursor(name, 0)
        return
    if key == "end":
        set_cursor(name, len(state["lists"].get(name, [])) - 1)
        return
    if key in ("backspace", "delete") and query:
        state["search"][name] = query[:-1]
        set_cursor(name, 0)
        return
    if key in ("return", "enter"):
        items = state["lists"].get(name, [])
        if items:
            activate_current(name)
        elif query.strip():
            commit_query = state["meta"].get(name, {}).get("commit_query")
            if commit_query:
                commit_query(query.strip())
        return
    if _text_key(char, modifiers):
        state["search"][name] = query + char
        set_cursor(name, 0)
    # Same full-rebuild reason as _handle_input_key (see above).
    input_sig.set(input_sig.peek() + 1)


def _handle_list_key(e, name: str) -> None:
    key = getattr(e, "key", "") or ""
    items = state["lists"].get(name, [])
    visible = visible_rows(name)
    if key in ("up", "k"):
        move_cursor(name, -1)
    elif key in ("down", "j"):
        move_cursor(name, 1)
    elif key == "pageup":
        move_cursor(name, -visible)
    elif key == "pagedown":
        move_cursor(name, visible)
    elif key == "home":
        set_cursor(name, 0)
    elif key == "end":
        set_cursor(name, len(items) - 1)
    elif key in ("return", "enter"):
        activate_current(name)


# --------------------------------------------------------------------------
# Root component
# --------------------------------------------------------------------------

@component
def App():
    current = screen()
    list_sig()   # subscribe so list navigation repaints
    input_sig()  # subscribe so typing repaints from a clean layout

    def on_key(e) -> None:
        key = getattr(e, "key", "") or ""
        s = screen()
        # Ctrl+C always quits. OpenTUI disables ISIG on the tty, so the
        # terminal delivers \x03 as an ordinary key event instead of
        # raising SIGINT (which is why the KeyboardInterrupt path in run()
        # never fires while the app is running).
        if getattr(e, "ctrl", False) and key.lower() == "c":
            request_quit()
            return

        # Esc always navigates back, even from inside a text input
        # (Input.handle_key would otherwise consume it).
        if key == "escape":
            target = BACK.get(s)
            if target:
                clear_notice()
                go(target)
            return

        # Model pickers: manual-entry input when the endpoint can't be listed,
        # otherwise a searchable list.
        if s in SEARCH_SCREENS:
            target = inputs.get(s)
            if target is not None:
                _handle_input_key(e, target)
            else:
                _handle_search_key(e, s)
            return

        if s in INPUT_SCREENS and s in inputs:
            _handle_input_key(e, inputs[s])
            return

        # Screen shortcuts win over list navigation (the menu list would
        # otherwise swallow 'q' as an unhandled key).
        if key == "q" and s == "menu":
            request_quit()
            return
        if key == "l" and s == "done" and log_sig():
            go("logs")
            return
        if key in ("return", "enter") and s in ("done", "setup-done"):
            go("menu")
            return

        if s in state["meta"]:
            _handle_list_key(e, s)


    # Register the global key handler exactly once per renderer. App
    # re-evaluates on every signal change, and this port stacks duplicate
    # use_keyboard handlers (stale ones then swallow keys).
    current_renderer = use_renderer()
    renderer_ref["r"] = current_renderer
    if _keyboard_bound.get("renderer") is not current_renderer:
        _keyboard_bound["renderer"] = current_renderer
        use_keyboard(on_key)
    if _frame_bound.get("renderer") is not current_renderer:
        _frame_bound["renderer"] = current_renderer
        try:
            current_renderer.set_frame_callback(_on_frame)
        except Exception:
            pass
    use_paste(_on_paste)

    builder = SCREEN_BUILDERS.get(current)
    if builder is None:
        return page(panel("Screen unavailable", Text(f"Unknown screen: {current}", fg=DANGER)))
    return builder()


# --------------------------------------------------------------------------
# Terminal lifecycle
# --------------------------------------------------------------------------

# Every reporting mode the app (or OpenTUI) turns on. Left behind, they make
# the shell inherit a terminal that reports mouse moves, pastes, and focus
# changes; left on the alternate screen, they hide the shell's output.
_RESET_SEQUENCES = (
    "\x1b[?2026l"    # synchronized output (otherwise the shell's output stays held)
    "\x1b[?1000l"    # basic mouse tracking
    "\x1b[?1002l"    # button-event tracking
    "\x1b[?1003l"    # any-event tracking
    "\x1b[?1005l"    # utf8 mouse
    "\x1b[?1006l"    # SGR mouse
    "\x1b[?1015l"    # urxvt mouse
    "\x1b[?1016l"    # pixel mouse
    "\x1b[?2004l"    # bracketed paste
    "\x1b[?1004l"    # focus reporting
    "\x1b[?2027l"    # grapheme clustering
    "\x1b[?2031l"    # color-scheme reporting
    "\x1b[>4;0m"     # modifyOtherKeys off
    "\x1b[?1l\x1b>"  # normal cursor keys, numeric keypad
    "\x1b[<u"        # pop the kitty keyboard protocol
    "\x1b[0m"        # default attributes
    "\x1b[?25h"      # show cursor
    "\x1b[?1049l"    # leave the alternate screen
)


def _drain_stdin(budget: float = 0.6, idle: float = 0.15) -> None:
    """Read and discard terminal replies until the line goes quiet.

    Whatever we leave here is read by the shell instead, which then prints it
    as literal escape text at the next prompt.
    """
    try:
        import select as _select

        fd = sys.stdin.fileno()
    except Exception:
        return
    deadline = time.monotonic() + budget
    quiet_since = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= deadline:
            return
        try:
            ready, _, _ = _select.select([fd], [], [], 0.05)
        except Exception:
            return
        if ready:
            try:
                if not os.read(fd, 4096):
                    return
            except OSError:
                return
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= idle:
            return


def _reset_terminal() -> None:
    """Hand the terminal back to the shell exactly as we found it.

    OpenTUI restores echo at startup, so replies to its own capability
    queries ("997;1n" and friends) are echoed on screen the moment they
    arrive — which is after the UI stopped drawing and before the shell
    starts reading. Quitting therefore leaves rows of escape text behind.

    So: turn echo off for the handover, disable everything we may have left
    on, then drain the line until it stays quiet, and only then give the
    tty back to the shell.
    """
    fd = -1
    saved_attrs = None
    try:
        import termios

        fd = sys.stdin.fileno()
        saved_attrs = termios.tcgetattr(fd)
        quiet = list(saved_attrs)
        quiet[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSANOW, quiet)
    except Exception:
        saved_attrs = None
    try:
        sys.stdout.write(_RESET_SEQUENCES)
        sys.stdout.flush()
    except Exception:
        pass
    _drain_stdin()
    if saved_attrs is not None:
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, saved_attrs)
        except Exception:
            pass


def run() -> None:
    """Launch the OpenTUI app (blocking until quit)."""
    # Fresh installs land in the setup wizard; everyone else gets the menu.
    try:
        current_settings = load_settings()
        llm = resolve_llm(current_settings)
        _, tavily_key, _ = resolve_api_keys(current_settings)
        has_keys = bool(llm["api_key"] and llm["base_url"] and llm["model"] and tavily_key)
    except Exception:
        has_keys = False

    if not has_keys:
        set_notice("warn", "Finish setup: endpoint, model, and API keys are needed before research.")

    screen.set("setup-welcome" if not has_keys else "menu")
    try:
        # use_mouse=False: nothing here clicks; mouse tracking left on is
        # exactly what leaks escape garbage into the shell after exit.
        asyncio.run(render(App, {"use_mouse": False}))
    except KeyboardInterrupt:
        # Fallback for terminals/environments that still raise SIGINT;
        # normally Ctrl+C arrives as a key event and is handled in on_key.
        request_quit()
    finally:
        _reset_terminal()
