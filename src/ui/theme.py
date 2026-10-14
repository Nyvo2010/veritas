"""Single source of truth for Veritas TUI styling.

Rules enforced here so spacing never drifts again:
- One blank line between blocks (callers use console.print() once, never "\\n" in strings).
- One panel padding: (0, 2) everywhere.
- One box style: ROUNDED everywhere.
- One color palette, imported — never hardcoded elsewhere.
"""

from rich import box

PRIMARY = "cyan"
SECONDARY = "white"
SUCCESS = "green"
ERROR = "red"
WARNING = "yellow"
MUTED = "dim white"

PANEL_BOX = box.ROUNDED
PANEL_PADDING = (0, 2)

# Compact static wordmark that replaces the old animated noise banner.
# Small, cheap to render, never resizes or animates.
COMPACT_LOGO = [
    " __   __  _______  ______    ___   _______  _______  _______ ",
    "|  | |  ||       ||    _ |  |   | |       ||   _   ||       |",
    "|  |_|  ||    ___||   | ||  |   | |_     _||  |_|  ||  _____|",
    "|       ||   |___ |   |_| ||   |   | |   |  |       || |_____ ",
    "|       ||    ___||    __ ||   |___| |   |  |       ||_____  |",
    " |     | |   |___ |   |  ||       | |   |  |   _   | _____| |",
    "  |___|  |_______||___|  ||_______| |___|  |__| |__||_______|",
]

LOGO_SUBTITLE = "Deep Research Agent"

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

GLYPH_PHASE = "*"
GLYPH_SEARCH = ">"
GLYPH_OK = "+"
GLYPH_WARN = "!"
GLYPH_ERR = "x"
GLYPH_INSIGHT = "-"

# Selectable UI color schemes (primary accent only; rest of palette is fixed).
# Settings store the theme name; apply_theme() activates it process-wide.
THEMES = {
    "cyan": "cyan",
    "blue": "blue",
    "magenta": "magenta",
    "green": "green",
    "amber": "dark_orange",
}

DEFAULT_THEME = "cyan"


def apply_theme(name: str) -> str:
    """Activate a color theme by name. Returns the active theme name."""
    global PRIMARY, ACCENT_COLOR
    key = (name or "").strip().lower()
    if key not in THEMES:
        key = DEFAULT_THEME
    PRIMARY = THEMES[key]
    ACCENT_COLOR = THEMES[key]
    return key


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        mins, secs = divmod(total, 60)
        return f"{mins}m {secs}s" if secs else f"{mins}m"
    hours, rem = divmod(total, 3600)
    mins = rem // 60
    return f"{hours}h {mins}m" if mins else f"{hours}h"


def format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds < 60:
        return "< 1 min"
    minutes = seconds / 60
    if minutes < 1.5:
        return "≈ 1 min"
    if minutes < 60:
        return f"≈ {round(minutes)} min"
    hours = minutes / 60
    if hours < 1.5:
        return "≈ 1 hr"
    return f"≈ {round(hours)} hrs"


def progress_bar(ratio: float, width: int = 40) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return "█" * filled + "░" * (width - filled)
