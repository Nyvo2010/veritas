"""Single source of truth for Veritas styling: palette, glyphs, and the bar.

Rules enforced here so spacing never drifts again:
- One blank line between blocks (callers use console.print() once, never "\\n" in strings).
- One panel padding: (0, 2) everywhere.
- One color palette, imported — never hardcoded elsewhere.

The OpenTUI frontend (tui_app.py) does not use the Rich styles below; it
imports the glyphs, the themes, and :func:`braille_wave` so both frontends
share one vocabulary and one progress bar.
"""

import math

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

# Dot rows of the leading cell, filled bottom to top across both text rows, so
# progress creeps forward as a rising meniscus rather than a flat cut.
_EDGE_ROWS = (
    (1, 3), (1, 2), (1, 1), (1, 0),
    (0, 3), (0, 2), (0, 1), (0, 0),
)


def braille_wave(ratio: float, width: int, phase: float = 0.0, particles: int = 16) -> tuple:
    """A two-row braille progress band: ``((fill_top, fill_bottom),
    (track_top, track_bottom))``.

    Eight dot rows tall, so it reads as a liquid band instead of a hairline.
    The filled cells carry a travelling ripple on their surface, the leading
    cell rises as a meniscus, and the remaining track is a particle field:
    *particles* drift left across eight lanes at different speeds, each leaving
    a short streak. Every row string is exactly *width* cells wide.

    *phase* is the animation clock in cells travelled, so the callers keep no
    animation state of their own.
    """
    try:
        ratio = max(0.0, min(1.0, float(ratio)))
    except (TypeError, ValueError):
        ratio = 0.0
    width = max(4, int(width))
    particles = max(0, int(particles))

    exact = ratio * width
    full = min(int(exact), width)
    edge_steps = int(round((exact - full) * 8)) if full < width else 0
    edge_steps = max(0, min(8, edge_steps))
    fill_cells = min(width, full + (1 if edge_steps else 0))
    track_len = width - fill_cells

    fill = [bytearray(width), bytearray(width)]
    track = [bytearray(width), bytearray(width)]

    for x in range(fill_cells):
        if x < full:
            surface = 0xFF
            crest = math.sin(x * 0.55 - phase * 1.15)
            if crest > 0.45:
                surface &= ~0x09   # one dot row shallower: the crest
            elif crest < -0.7:
                surface &= ~0x12   # two rows: the trough
            fill[0][x] = surface
            fill[1][x] = _BRAILLE_FULL
        else:
            for step in range(edge_steps):
                text_row, dot_row = _EDGE_ROWS[step]
                fill[text_row][x] |= (
                    _BRAILLE_BIT[dot_row][0] | _BRAILLE_BIT[dot_row][1]
                )

    if track_len and particles:
        for index in range(particles):
            # Deterministic per-particle lane, speed, and offset: the field
            # spreads out and never marches in lockstep.
            seed = (index * 2654435761) & 0xFFFFFFFF
            lane = seed & 7
            text_row = 0 if lane < 4 else 1
            dot_row = lane & 3
            speed = 0.7 + ((seed >> 9) & 0xFF) / 255.0 * 1.6
            start = ((seed >> 17) % 997) / 997.0 * (track_len + 8)
            gap = 4 + ((seed >> 5) % 9)
            position = (start - phase * speed) % (track_len + gap)
            if position >= track_len:
                continue  # still off-screen right
            cell = fill_cells + int(position)
            column = 0 if position - int(position) < 0.5 else 1
            track[text_row][cell] |= _BRAILLE_BIT[dot_row][column]
            # One dot further right reads as a streak, not a blip.
            if cell + 1 < width:
                track[text_row][cell + 1] |= _BRAILLE_BIT[dot_row][column]

    def render(masks) -> str:
        return "".join(chr(0x2800 + mask) for mask in masks)

    return ((render(fill[0][:fill_cells]), render(fill[1][:fill_cells])),
            (render(track[0][fill_cells:]), render(track[1][fill_cells:])))
