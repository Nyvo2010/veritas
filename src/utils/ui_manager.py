"""UIManager — backward-compatible facade over the new LiveRenderer.

Old import path (utils.ui_manager.UIManager) is preserved so agents and
main.py keep working, but all rendering goes through ui.live_renderer,
which fixes: alternate-screen flicker, growing/shrinking layout, mixed
print() corruption, and fake fractional progress.
"""

from typing import Optional

from rich.console import Console

from ui.live_renderer import LiveRenderer


def _classify_legacy_message(message: str) -> str:
    lowered = message.lower()
    if "phase" in lowered and ("1" in lowered or "2" in lowered or "3" in lowered or "4" in lowered):
        return "phase"
    if "searching" in lowered:
        return "search"
    if "key insight" in lowered:
        return "insight"
    if "completed:" in lowered or "saved" in lowered or "found" in lowered:
        return "ok"
    if "error" in lowered or "failed" in lowered:
        return "err"
    if "warn" in lowered or "gap" in lowered:
        return "warn"
    if "generat" in lowered or "writ" in lowered or "analyz" in lowered:
        return "info"
    return "info"


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

    def update_progress(self, progress: float, status: str = "") -> None:
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        # Clamp into dynamic total; totals can grow as follow-ups arrive.
        if value > self._renderer.total_steps:
            self._renderer.set_total(int(value) + 1)
        self._renderer.set_progress(value, status=status)

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
        level = _classify_legacy_message(text)
        if style in ("red bold", "red", "bold red"):
            level = "err"
        elif style in ("yellow", "dim yellow"):
            level = "warn"
        self._renderer.log(level, text)
