"""Where a research run spends its wall-clock time.

The orchestrator records every blocking call here (model inference versus web
search) from its worker thread; the frontends poll :meth:`Timings.snapshot` on
their render tick to show the split. That split is the quickest way to answer
"why is this run slow, and is it the model or the search API?".
"""

import threading

# Fixed order and labels so both frontends read the same.
KINDS = (("llm", "LLM"), ("search", "Search"))
LABELS = dict(KINDS)


class Timings:
    """Thread-safe running totals and call counts per kind."""

    __slots__ = ("_lock", "_seconds", "_calls")

    def __init__(self):
        self._lock = threading.Lock()
        self._seconds = {kind: 0.0 for kind, _ in KINDS}
        self._calls = {kind: 0 for kind, _ in KINDS}

    def add(self, kind: str, seconds: float) -> None:
        """Record one finished call of *kind* (unknown kinds are kept too)."""
        try:
            seconds = max(0.0, float(seconds))
        except (TypeError, ValueError):
            return
        with self._lock:
            if kind not in self._seconds:
                self._seconds[kind] = 0.0
                self._calls[kind] = 0
            self._seconds[kind] += seconds
            self._calls[kind] += 1

    def snapshot(self) -> tuple:
        """Immutable ``((kind, seconds, calls), ...)``, safe to compare per frame."""
        with self._lock:
            return tuple(
                (kind, self._seconds[kind], self._calls[kind]) for kind in self._seconds
            )

    def summary(self) -> str:
        """One line, e.g. ``LLM 48s (12) · Search 18s (9)``."""
        from ui import theme

        parts = [
            f"{LABELS.get(kind, kind.title())} {theme.format_duration(seconds)} ({calls})"
            for kind, seconds, calls in self.snapshot()
        ]
        return " · ".join(parts)
