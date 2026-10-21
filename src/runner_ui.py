"""Thread-safe event-queue UI sink for research runs.

Implements the same method surface as utils.ui_manager.UIManager so
DeepResearchAgent can drive it without changes, but instead of rendering
anything it records progress counters and pushes (kind, payload) events
onto a queue. The OpenTUI frontend drains the queue on a timer and feeds
its reactive signals.

Kinds emitted:
  ("log", level, text)        level in phase|search|insight|ok|warn|err|info
  ("status", text)
  ("phase", text)
  ("progress", completed, total, status)
  ("counts", answered, findings, sources)
  ("done", ok)

Timing is not an event: the orchestrator calls add_timing() directly and the
frontend polls :attr:`QueueUI.timings` on its own render tick.
"""

import queue
import threading
import time

from ui.timings import Timings
from utils.text import level_for_style, strip_markup


class QueueUI:
    """Drop-in UI object for DeepResearchAgent.research(..., ui=...)."""

    def __init__(self):
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self._lock = threading.Lock()
        self.title = "Research"
        self.phase = "Preparing..."
        self.status = "Starting..."
        self.total_steps = 1
        self.completed_steps = 0.0
        self.start_time = time.time()
        self.counts = (0, 0, 0)     # (questions answered, findings, sources)
        self.timings = Timings()

    @property
    def renderer(self) -> "QueueUI":
        # Lets orchestrator code written against UIManager keep working
        # (it reads ui.renderer.completed_steps).
        return self

    # -- lifecycle ------------------------------------------------------
    def initialize(self, total_steps: int, description: str = "Research Progress") -> None:
        with self._lock:
            self.title = description
            self.total_steps = max(1, int(total_steps))
            self.completed_steps = 0.0
            self.start_time = time.time()
            self.counts = (0, 0, 0)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def complete(self) -> None:
        with self._lock:
            self.completed_steps = float(self.total_steps)
        self.events.put(("done", True))

    # -- progress --------------------------------------------------------
    def set_total(self, total: int) -> None:
        # Publish the new total with the current position: the frame callback
        # only redraws on events, so a silent total change would leave the bar
        # scaled against the old estimate until the next step finished.
        with self._lock:
            self.total_steps = max(1, int(total))
            snapshot = (self.completed_steps, self.total_steps, self.status)
        self.events.put(("progress",) + snapshot)

    def set_counts(self, answered: int, findings: int, sources: int) -> None:
        """Running totals for the run screen; cheap and idempotent."""
        counts = (int(answered), int(findings), int(sources))
        with self._lock:
            self.counts = counts
        self.events.put(("counts",) + counts)

    def set_progress(self, completed: float, status: str = "") -> None:
        with self._lock:
            self.completed_steps = max(0.0, min(float(completed), float(self.total_steps)))
            if status:
                self.status = status
            snapshot = (self.completed_steps, self.total_steps, self.status)
        self.events.put(("progress",) + snapshot)

    def advance(self, by: float = 1.0, status: str = "") -> None:
        with self._lock:
            self.completed_steps = min(float(self.total_steps), self.completed_steps + by)
            if status:
                self.status = status
            snapshot = (self.completed_steps, self.total_steps, self.status)
        self.events.put(("progress",) + snapshot)

    def update_progress(self, progress: float, status: str = "") -> None:
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        with self._lock:
            if value > self.total_steps:
                self.total_steps = int(value) + 1
        self.set_progress(value, status=status)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
        self.events.put(("phase", phase))
        self.log("phase", phase)

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
        self.events.put(("status", status))

    def add_timing(self, kind: str, seconds: float) -> None:
        """Record how long one blocking call took ("llm" or "search")."""
        self.timings.add(kind, seconds)

    # -- log --------------------------------------------------------------
    def log(self, level: str, text: str) -> None:
        cleaned = strip_markup(text)
        if not cleaned:
            return
        self.events.put(("log", level, cleaned))

    def add_message(self, message: str, style: str = "white") -> None:
        if message is None:
            return
        cleaned = strip_markup(str(message))
        if not cleaned or cleaned.lower() in {"done", "done!"}:
            return
        self.log(level_for_style(cleaned, style), cleaned)
