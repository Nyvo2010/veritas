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
  ("done", ok)
"""

import queue
import re
import threading
import time

_MARKUP_RE = re.compile(r"\[(?:/[a-z ]+|[a-z][a-z ]*)\]", re.IGNORECASE)


def strip_markup(text: str) -> str:
    cleaned = _MARKUP_RE.sub("", text or "")
    return " ".join(cleaned.split())


def classify_message(message: str) -> str:
    lowered = message.lower()
    if "phase" in lowered:
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
    return "info"


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
        self.done_ok = None

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
            self.done_ok = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def complete(self) -> None:
        with self._lock:
            self.completed_steps = float(self.total_steps)
            self.done_ok = True
        self.events.put(("done", True))

    # -- progress --------------------------------------------------------
    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_steps = max(1, int(total))

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
        level = classify_message(cleaned)
        if style in ("red bold", "red", "bold red"):
            level = "err"
        elif style in ("yellow", "dim yellow"):
            level = "warn"
        self.log(level, cleaned)

    def add_section(self, title: str, content: str, style: str = "white") -> None:
        self.log("info", f"{title}: {content}")
