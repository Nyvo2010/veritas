"""Single source of truth for Veritas styling: palette, glyphs, and the bar.

Rules enforced here so spacing never drifts again:
- One blank line between blocks (callers use console.print() once, never "\\n" in strings).
- One panel padding: (0, 2) everywhere.
- One color palette, imported — never hardcoded elsewhere.

The OpenTUI frontend (tui_app.py) does not use the Rich styles below; it
imports the glyphs, the themes, and :func:`braille_bar` so both frontends
share one vocabulary and one progress bar.
"""

from rich import box

# --------------------------------------------------------------------------
# One palette: imported by BOTH frontends so they always render identically.
# --------------------------------------------------------------------------
ACCENTS = {
    "blue": "#1e3fd8",       # vivid blue (default)
    "green": "#759e6b",      # sage
    "cyan": "#54a9ab",       # cadet teal
    "magenta": "#8f65a4",    # mauve
    "amber": "#c09549",      # brass
    "red": "#c14949",        # brick
    "orange": "#c17c3e",     # copper
    "purple": "#7665a4",     # lavender
    "white": "#d4d4d4",      # soft white
}

THEMES = ACCENTS             # selectable theme names == accent names
DEFAULT_THEME = "blue"

# Semantics shared by both frontends (never switch with the theme).
PANEL_BG = "#232326"         # neutral zinc, no color cast
TITLE_FG = "#1c1c1e"         # near-black for selected rows / title bars
SECONDARY = "#8e8e93"        # descriptions, values
TERTIARY = "#636366"         # hints, footnotes
SUBTITLE = "#707074"         # dimmer than body, softer than hints
SUCCESS = ACCENTS["green"]
DANGER = ACCENTS["red"]
ERROR = DANGER               # alias: Rich-side call sites say ERROR
WARNING = ACCENTS["amber"]
INFO = "#5d7fe8"             # blue for neutral notices

# Rich-side color roles. PRIMARY is set by apply_theme() below.
PRIMARY = ACCENTS[DEFAULT_THEME]
MUTED = TERTIARY

PANEL_BOX = box.ROUNDED
PANEL_PADDING = (0, 2)

# Static VERITAS wordmark. Never animates, never resizes mid-run.
COMPACT_LOGO = [
    "",
    "",
    ".sSSS s.",
    "SSSSS SSSs. .sSSSSs.    .sSSSSSSSs. SSSSS .sSSSSSSSSSSSSSs. .sSSSSs.    .sSSSSSSSs.",
    "S SSS SSSSS S SSSSSSSs. S SSS SSSSS S SSS SSSSS S SSS SSSSS S SSSSSSSs. S SSS SSSS'",
    "S  SS SSSSS S  SS SSSS' S  SS SSSS' S  SS SSSSS S  SS SSSSS S  SS SSSSS S  SS",
    "S..SS SSSSS S..SS       S..SSsSSSa. S..SS `:S:' S..SS `:S:' S..SSsSSSSS `SSSSsSSSa.",
    " S::S SSSS  S:::SSSS    S:::S SSSSS S:::S       S:::S       S:::S SSSSS .sSSS SSSSS",
    "  S;S SSS   S;;;S       S;;;S SSSSS S;;;S       S;;;S       S;;;S SSSSS S;;;S SSSSS",
    "   SS SS    S%%%S SSSSS S%%%S SSSSS S%%%S       S%%%S       S%%%S SSSSS S%%%S SSSSS",
    "    SsS     SSSSSsSS;:' SSSSS SSSSS SSSSS       SSSSS       SSSSS SSSSS SSSSSsSSSSS",
    "",
    " Made by Nyvø w/ love                                            Powered by Tavily "
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
# Blue is the default and sits at the top of the theme menu. Hex values match
# the OpenTUI accents so both frontends render the same scheme.

def apply_theme(name: str) -> str:
    """Activate a color theme by name. Returns the active theme name."""
    primary = (name or "").strip().lower()
    if primary not in THEMES:
        primary = DEFAULT_THEME
    globals()["PRIMARY"] = THEMES[primary]
    return primary


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


# Braille plotting: one character is a 2x4 dot grid at U+2800 + bit mask.
#   bit: 1 8     dot: 1 4
#        2 16         2 5
#        4 32         3 6
#       64 128        7 8
_BRAILLE_BIT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
_BRAILLE_FULL = 0xFF                       # all eight dots

def braille_bar(ratio: float, width: int, phase: float = 0.0, particles: int = 16) -> tuple:
    """A one-row braille progress bar: ``(filled, track)``.

    The completed part is solid, gap-free and static: it never ripples, so the
    filled width is exactly what has been done. Only the remaining track
    animates — *particles* drift left across four dot lanes at different
    speeds, each leaving a short streak. Both strings are exactly *width*
    cells wide, and *phase* is the animation clock in cells travelled, so the
    callers keep no animation state of their own.
    """
    try:
        ratio = max(0.0, min(1.0, float(ratio)))
    except (TypeError, ValueError):
        ratio = 0.0
    width = max(4, int(width))
    particles = max(0, int(particles))

    filled = min(width, int(round(ratio * width)))
    track_len = width - filled
    fill = chr(0x2800 + _BRAILLE_FULL) * filled

    track = bytearray(track_len)
    if track_len and particles:
        for index in range(particles):
            # Deterministic per-particle lane, speed, and offset: the field
            # spreads out and never marches in lockstep.
            seed = (index * 2654435761) & 0xFFFFFFFF
            lane = seed & 3
            speed = 0.7 + ((seed >> 9) & 0xFF) / 255.0 * 1.6
            start = ((seed >> 17) % 997) / 997.0 * (track_len + 8)
            gap = 4 + ((seed >> 5) % 9)
            position = (start - phase * speed) % (track_len + gap)
            if position >= track_len:
                continue  # still off-screen right
            cell = int(position)
            column = 0 if position - cell < 0.5 else 1
            track[cell] |= _BRAILLE_BIT[lane][column]
            # One dot further right reads as a streak, not a blip.
            if cell + 1 < track_len:
                track[cell + 1] |= _BRAILLE_BIT[lane][column]

    return fill, "".join(chr(0x2800 + mask) for mask in track)
