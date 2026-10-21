"""UIManager — backward-compatible facade over the new LiveRenderer.

Old import path (utils.ui_manager.UIManager) is preserved so agents and
main.py keep working, but all rendering goes through ui.live_renderer,
which fixes: alternate-screen flicker, growing/shrinking layout, mixed
print() corruption, and fake fractional progress.
"""

from typing import Optional

from rich.console import Console

from ui.live_renderer import LiveRenderer
from utils.text import classify_message


def notify(ui, level: str, text: str) -> None:
    """Write one leveled line to any UI sink, old or new.

    The agents used to call ``add_message`` directly, which made the UI
    contract implicit: a sink without the legacy method crashed mid-run even
    though it implemented the current ``log(level, text)`` API. Prefer the
    leveled method, fall back to the legacy one, and do nothing when there is
    no sink.
    """
    if ui is None or not text:
        return
    log_method = getattr(ui, "log", None)
    if callable(log_method):
        log_method(level, text)
        return
    add_message = getattr(ui, "add_message", None)
    if callable(add_message):
        add_message(text, style="yellow" if level == "warn" else "white")


class UIManager:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._renderer = LiveRenderer(console=self.console)
        self.description = "Overall Research Progress"
        self._total = 1

    # -- new API (preferred) -------------------------------------------
    @property
    def renderer(self) -> LiveRenderer:
        return self._renderer

    def set_phase(self, phase: str) -> None:
        self._renderer.set_phase(phase)

    def set_status(self, status: str) -> None:
        self._renderer.set_status(status)

    def add_timing(self, kind: str, seconds: float) -> None:
        """Record how long one blocking call took ("llm" or "search")."""
        self._renderer.timings.add(kind, seconds)

    def log(self, level: str, text: str) -> None:
        """Write one leveled line (same vocabulary as the OpenTUI sink)."""
        self._renderer.log(level, text)

    def set_counts(self, answered: int, findings: int, sources: int) -> None:
        """Running totals for the progress panel."""
        self._renderer.set_counts(answered, findings, sources)

    # -- legacy API (kept for agents/main.py) --------------------------
    def initialize(self, total_steps: int, description: str = "Research Progress") -> None:
        self.description = description
        self._total = max(1, int(total_steps))

    def start(self) -> None:
        self._renderer.start(title=self.description, total_steps=self._total)

    def stop(self) -> None:
        self._renderer.stop()

    def complete(self) -> None:
        self._renderer.complete(ok=True)

    def set_progress(self, completed: float, status: str = "") -> None:
        """Set absolute progress; the name main.py and QueueUI both use."""
        try:
            value = float(completed)
        except (TypeError, ValueError):
            value = 0.0
        # Clamp into dynamic total; totals can grow as follow-ups arrive.
        if value > self._renderer.total_steps:
            self._renderer.set_total(int(value) + 1)
        self._renderer.set_progress(value, status=status)

    # Kept because older callers (and the docs) use the update_* spelling.
    def update_progress(self, progress: float, status: str = "") -> None:
        self.set_progress(progress, status=status)

    def set_total(self, total: int) -> None:
        self._total = max(1, int(total))
        self._renderer.set_total(self._total)

    def advance(self, by: float = 1.0, status: str = "") -> None:
        self._renderer.advance(by=by, status=status)

    def add_message(self, message: str, style: str = "white") -> None:
        if message is None:
            return
        text = str(message).strip()
        if not text:
            return
        lowered = text.lower()
        if lowered in {"done", "done!"}:
            return
        level = classify_message(text)
        if style in ("red bold", "red", "bold red"):
            level = "err"
        elif style in ("yellow", "dim yellow"):
            level = "warn"
        self._renderer.log(level, text)
