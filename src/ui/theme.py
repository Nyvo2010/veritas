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
_BRAILLE_TRACK = 0x04 | 0x20               # bottom row only: a dashed baseline
# Dot order used to fill one cell as progress crosses it: up the left column,
# then up the right column, so the fill level reads as a rising line.
_BRAILLE_FILL_ORDER = (0x40, 0x04, 0x02, 0x01, 0x80, 0x20, 0x10, 0x08)


def braille_bar(ratio: float, width: int, phase: float = 0.0, particles: int = 4) -> tuple:
    """Split a braille progress bar into (filled, track) strings.

    *filled* is the solid left part of the bar plus one partially filled edge
    cell, so progress creeps forward sub-cell by sub-cell. *track* is the
    remaining cells: a dim dashed baseline with *particles* drifting right to
    left through it. Both strings are returned separately so callers can color
    the fill and the track differently.

    *phase* is the animation clock in cells travelled, so the same function
    serves any renderer without holding animation state of its own.
    """
    try:
        ratio = max(0.0, min(1.0, float(ratio)))
    except (TypeError, ValueError):
        ratio = 0.0
    width = max(4, int(width))
    particles = max(0, int(particles))

    exact = ratio * width
    full = min(int(exact), width)
    edge = int(round((exact - full) * 8)) if full < width else 0

    fill_masks = [_BRAILLE_FULL] * full
    if edge:
        fill_masks.append(sum(_BRAILLE_FILL_ORDER[:edge]) | _BRAILLE_TRACK)
        full += 1
    filled = "".join(chr(0x2800 + mask) for mask in fill_masks)

    track_masks = [_BRAILLE_TRACK] * max(0, width - full)
    if track_masks and phase:
        for index in range(particles):
            # Deterministic per-particle lane, speed, and starting offset:
            # particles spread out and never all march in lockstep.
            seed = (index * 2654435761) & 0xFFFFFFFF
            lane = (seed >> 3) & 3
            speed = 0.55 + ((seed >> 8) & 0xFF) / 255.0 * 0.95
            start = ((seed >> 16) % 1013) / 1013.0 * width
            gap = 5 + ((seed >> 5) % 13)
            position = (start - phase * speed) % (width + gap)
            if position >= len(track_masks):
                continue  # still off-screen right: it has not entered the bar
            cell = int(position)
            column = 0 if position - cell < 0.5 else 1
            track_masks[cell] |= _BRAILLE_BIT[lane][column]
            # Trailing dot so each particle reads as a streak, not a blip.
            if cell + 1 < len(track_masks) and position - cell < 0.5:
                track_masks[cell + 1] |= _BRAILLE_BIT[lane][1]
    track = "".join(chr(0x2800 + mask) for mask in track_masks)

    return filled, track
