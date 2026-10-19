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


def as_int(value: object, default: int = 0, low: int = None, high: int = None) -> int:
    """Coerce a model-supplied number, clamping it to a sane range."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def as_bool(value: object, default: bool = False) -> bool:
    """Read a model-supplied boolean, including the string forms models emit.

    ``"false"`` is truthy as a Python string, which would otherwise start an
    unnecessary follow-up research round.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0", ""}:
            return False
        return default
    return bool(value)


def string_list(value: object, limit: int = 0) -> list:
    """Coerce a model-supplied field to a list of non-empty strings.

    A bare string becomes one item, not one item per character: iterating a
    string where a list was requested would otherwise turn ``"false"`` into
    five follow-up questions.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    cleaned = [str(item).strip() for item in items if item is not None]
    cleaned = [item for item in cleaned if item]
    return cleaned[:limit] if limit else cleaned


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
