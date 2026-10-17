"""Text helpers shared by the agents, the sinks, and the two frontends.

They live here — the bottom layer, importing nothing from the project — so
there is exactly one implementation of each, instead of one per module.
"""

import re

# Rich markup left in messages written for the old console UI.
_MARKUP_RE = re.compile(r"\[(?:/[a-z ]+|[a-z][a-z ]*)\]", re.IGNORECASE)


def clip(value: object, limit: int = 90) -> str:
    """Flatten *value* to one line and cap it at *limit* characters."""
    flat = " ".join(str(value if value is not None else "").split())
    return flat if len(flat) <= limit else flat[: max(1, limit - 1)] + "…"


def strip_markup(text: str) -> str:
    """Remove Rich markup tags and collapse whitespace runs."""
    return " ".join(_MARKUP_RE.sub("", text or "").split())


def classify_message(message: str) -> str:
    """Guess a log level for a message written for the old console UI.

    Legacy ``add_message(text, style)`` call sites predate leveled logging;
    this maps their free text back onto the levels the log renders with.
    """
    lowered = (message or "").lower()
    if "phase" in lowered and any(digit in lowered for digit in "1234"):
        return "phase"
    if "searching" in lowered:
        return "search"
    if "insight" in lowered:
        return "insight"
    # Problems before successes: "3 knowledge gaps found" is a warning, not a
    # completion, even though it contains "found".
    if "error" in lowered or "failed" in lowered:
        return "err"
    if "warn" in lowered or "gap" in lowered:
        return "warn"
    if "completed:" in lowered or "saved" in lowered or "found" in lowered:
        return "ok"
    return "info"
