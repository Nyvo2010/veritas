"""New Veritas live renderer: spinner + determinate bar + scrolling event log.

Design fixes vs the old UIManager (utils/ui_manager.py):
- No screen=True alternate-screen takeover, so no flicker and no corruption
  when the process writes errors. Everything flows through one Rich Live.
- Fixed-height layout (header / log / progress) so frames never shift size
  as messages arrive — the old layout grew/shrank and jittered.
- No background thread + Event. Live auto-refreshes; state mutation is
  lock-guarded and cheap. Blocking LLM calls run via asyncio.to_thread in
  main.py so the spinner keeps animating at 12fps.
- No direct print() while live. Callers use log_* methods; final summaries
  are printed only after stop().
- Progress is event-driven (completed/total updated by the orchestrator),
  not fake fractional steps (step+0.2/0.5/0.7).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ui import theme
from ui.timings import Timings
from utils.text import strip_markup


@dataclass
class LogEntry:
    level: str  # phase | search | insight | ok | warn | err | info
    text: str
    timestamp: float


def _level_style(level: str) -> str:
    """Rich style for a log level.

    Resolved per call, not at import: the theme is applied after this module
    is imported (cli.apply_saved_theme), so a cached dict would keep the
    default accent no matter what the user picked.
    """
    return {
        "phase": f"bold {theme.PRIMARY}",
        "search": theme.PRIMARY,
        "insight": theme.SECONDARY,
        "ok": theme.SUCCESS,
        "warn": theme.WARNING,
        "err": f"bold {theme.ERROR}",
        "info": theme.SECONDARY,
    }.get(level, theme.SECONDARY)

_LEVEL_GLYPH = {
    "phase": theme.GLYPH_PHASE,
    "search": theme.GLYPH_SEARCH,
    "insight": theme.GLYPH_INSIGHT,
    "ok": theme.GLYPH_OK,
    "warn": theme.GLYPH_WARN,
    "err": theme.GLYPH_ERR,
    "info": "·",
}


class LiveRenderer:
    """Single Live display for a research run."""

    def __init__(self, console: Optional[Console] = None, log_height: int = 14):
        self.console = console or Console()
        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._started = False

        self.title = "Research"
        self.phase = "Preparing..."
        self.status = "Starting..."
        self.total_steps = 1
        self.completed_steps = 0.0
        self.start_time = time.time()
        self.done = False
        self.failed = False
        self.log_height = log_height
        self._log: Deque[LogEntry] = deque(maxlen=300)
        self.timings = Timings()

        self._layout = Layout()
        self._layout.split_column(
            Layout(name="header", size=5),
            Layout(name="log", ratio=1),
            Layout(name="progress", size=7),
        )
        self._live: Optional[Live] = None

    # -- lifecycle -----------------------------------------------------

    def start(self, title: str = "Research", total_steps: int = 1) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self.done = False
            self.failed = False
            self.title = title
            self.total_steps = max(1, total_steps)
            self.completed_steps = 0.0
            self.start_time = time.time()
            self._render_locked()
            live = Live(
                self._layout,
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._live = live
        live.start()

    def stop(self) -> None:
        with self._lock:
            live, self._live = self._live, None
            self._started = False
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass

    def complete(self, ok: bool = True) -> None:
        with self._lock:
            self.done = True
            self.failed = not ok
            self.completed_steps = float(self.total_steps)
            self.status = "Complete" if ok else "Failed"
        self.stop()
        elapsed = theme.format_duration(time.time() - self.start_time)
        msg = f"Research {'completed' if ok else 'failed'} in {elapsed}"
        style = theme.SUCCESS if ok else theme.ERROR
        self.console.print(
            Panel(
                Text(msg, style=f"bold {style}"),
                title="Done",
                box=theme.PANEL_BOX,
                style=style,
                padding=theme.PANEL_PADDING,
            )
        )
        self.console.print()

    # -- progress API (event-driven, dynamic totals) -------------------

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_steps = max(1, int(total))
            self._render_locked()

    def set_progress(self, completed: float, status: str = "") -> None:
        with self._lock:
            self.completed_steps = max(0.0, min(float(completed), float(self.total_steps)))
            if status:
                self.status = status
            self._render_locked()

    def advance(self, by: float = 1.0, status: str = "") -> None:
        with self._lock:
            self.completed_steps = min(float(self.total_steps), self.completed_steps + by)
            if status:
                self.status = status
            self._render_locked()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self._render_locked()
        self.log("phase", phase)

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
            self._render_locked()

    # -- log API (only path for output during a run) -------------------

    def log(self, level: str, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        # Strip any stray leading/trailing blank-line markers from old call sites.
        cleaned = cleaned.strip("\n").strip()
        if not cleaned:
            return
        with self._lock:
            self._log.append(LogEntry(level=level, text=cleaned, timestamp=time.time()))
            self._render_locked()

    # -- rendering (fixed heights, no jitter) --------------------------

    def _render_locked(self) -> None:
        self._layout["header"].update(self._header_panel())
        self._layout["log"].update(self._log_panel())
        self._layout["progress"].update(self._progress_panel())

    def _header_panel(self) -> Panel:
        if self.done:
            spinner = theme.GLYPH_OK if not self.failed else theme.GLYPH_ERR
            spin_style = theme.SUCCESS if not self.failed else theme.ERROR
        else:
            frame = int(time.time() * 10) % len(theme.SPINNER_FRAMES)
            spinner = theme.SPINNER_FRAMES[frame]
            spin_style = theme.PRIMARY
        text = Text()
        text.append(f"{spinner} ", style=f"bold {spin_style}")
        text.append("VERITAS", style=f"bold {theme.PRIMARY}")
        text.append(f"  ·  {self.title}", style=theme.SECONDARY)
        text.append(f"  ·  {self.phase}", style=theme.MUTED)
        return Panel(
            text,
            box=theme.PANEL_BOX,
            style=theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )

    def _log_panel(self) -> Panel:
        text = Text()
        entries = list(self._log)[-self.log_height:]
        if not entries:
            text.append("Waiting for activity...", style=theme.MUTED)
        for entry in entries:
            glyph = _LEVEL_GLYPH.get(entry.level, "·")
            style = _level_style(entry.level)
            elapsed = theme.format_duration(entry.timestamp - self.start_time)
            text.append(f"{elapsed:>7} ", style=theme.MUTED)
            text.append(f"{glyph} ", style=style)
            # Keep markup from old call sites inert: render as plain text.
            text.append(strip_markup(entry.text), style=theme.SECONDARY)
            text.append("\n")
        # Drop the final newline so the panel height stays constant.
        plain = text.plain
        if plain.endswith("\n"):
            text = text[:-1]
        return Panel(
            text,
            title="Activity",
            title_align="left",
            box=theme.PANEL_BOX,
            style=theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )

    def _progress_panel(self) -> Panel:
        elapsed = time.time() - self.start_time
        ratio = min(1.0, self.completed_steps / self.total_steps) if self.total_steps else 0.0
        if self.completed_steps > 0 and not self.done:
            eta = (elapsed / self.completed_steps) * max(0.0, self.total_steps - self.completed_steps)
            eta_str = theme.format_eta(eta)
        else:
            eta_str = "Complete" if self.done else "—"
        # Same one-row bar as the OpenTUI frontend: solid fill, particle track.
        # The particle clock is seconds since the run started, so no animation
        # state is kept here.
        particles = 0 if self.done else 20
        fill, track = theme.braille_bar(ratio, 40, elapsed * 11.0, particles)
        text = Text()
        text.append(fill, style=theme.PRIMARY)
        text.append(track, style=theme.MUTED)
        text.append(f" {ratio * 100:.1f}%", style=theme.SECONDARY)
        text.append("\n")
        text.append("Step ", style=f"bold {theme.PRIMARY}")
        text.append(f"{self.completed_steps:.0f}/{self.total_steps}", style=theme.SECONDARY)
        text.append("  |  Elapsed ", style=theme.MUTED)
        text.append(theme.format_duration(elapsed), style=theme.SECONDARY)
        text.append("  |  ETA ", style=theme.MUTED)
        text.append(eta_str, style=theme.SECONDARY)
        timing = self.timings.summary()
        if timing:
            text.append("\n")
            text.append(timing, style=theme.MUTED)
        text.append("\n")
        text.append(self.status, style=theme.MUTED)
        return Panel(
            text,
            title="Progress",
            title_align="left",
            box=theme.PANEL_BOX,
            style=theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )

