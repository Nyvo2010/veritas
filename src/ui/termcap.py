"""Terminal capability detection for consistent TUI colors.

The OpenTUI frontend speaks 24-bit hex colors. Terminals without truecolor
support (notably pre-Tahoe macOS Terminal.app, which advertises
``xterm-256color`` without ``COLORTERM=truecolor``) quantize those escapes
on their own terms: muted accents shift hue, dim grays wash out (worse
over a transparent background), and the logo shimmer bands into visible
steps. Terminal.app on macOS 26 Tahoe and later handles 24-bit color and
takes the full-quality path.

To look the same everywhere, detect truecolor once at startup. When it is
absent, snap every palette color to the nearest xterm-256 palette entry
before it reaches the renderer, so the output is deterministic instead of
emulator-dependent. Truecolor terminals are unaffected.

Override for testing: ``VERITAS_TRUECOLOR=1`` forces truecolor,
``VERITAS_TRUECOLOR=0`` forces the 256-color path.
"""

import os

# 6x6x6 color cube levels used by the xterm 256-color palette.
_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)

# First 16 canonical xterm colors.
_BASE_16 = (
    "#000000", "#800000", "#008000", "#808000",
    "#000080", "#800080", "#008080", "#c0c0c0",
    "#808080", "#ff0000", "#00ff00", "#ffff00",
    "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
)


def _build_palette() -> list:
    palette = list(_BASE_16)
    for r in _CUBE_LEVELS:
        for g in _CUBE_LEVELS:
            for b in _CUBE_LEVELS:
                palette.append(f"#{r:02x}{g:02x}{b:02x}")
    for i in range(24):
        v = 8 + 10 * i
        palette.append(f"#{v:02x}{v:02x}{v:02x}")
    return palette


PALETTE_256 = _build_palette()


def _hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def snap_hex(value: str) -> str:
    """Snap a hex color to the nearest xterm-256 palette entry (generic)."""
    try:
        r, g, b = _hex_to_rgb(value)
    except (ValueError, IndexError):
        return value
    best = None
    best_dist = None
    for entry in PALETTE_256:
        er = int(entry[1:3], 16)
        eg = int(entry[3:5], 16)
        eb = int(entry[5:7], 16)
        dist = (r - er) ** 2 + (g - eg) ** 2 + (b - eb) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = entry
    return best if best is not None else value


# Curated 256-color fallbacks for the Veritas palette. Plain nearest-RGB
# snapping collapses desaturated tones (sage -> pure gray) and merges
# neighboring themes (amber/orange, magenta/purple) into identical colors.
# These hand-picked entries preserve each hue's identity and stay distinct.
FALLBACK_256 = {
    "#6f8fb8": "#5f87af",  # dusty blue
    "#7d9b76": "#87af87",  # sage (nearest-RGB would give gray #878787)
    "#5f9ea0": "#5fafaf",  # cadet teal
    "#9a7fa8": "#af87af",  # muted mauve
    "#b3945e": "#af875f",  # muted brass
    "#b86767": "#af5f5f",  # muted brick
    "#b07d4f": "#d7875f",  # muted copper (kept distinct from brass)
    "#8a7fa8": "#8787af",  # muted lavender
    "#c9c9c9": "#c6c6c6",  # soft white
    "#232326": "#262626",  # panel zinc
    "#1c1c1e": "#1c1c1c",  # selected-row ink
    "#8e8e93": "#878787",  # secondary text
    "#636366": "#5f5f5f",  # hints
    "#707074": "#808080",  # subtitles (kept distinct from hints)
}


def to_256(value: str) -> str:
    """Map a hex color to its 256-color terminal equivalent."""
    mapped = FALLBACK_256.get(value.lower())
    if mapped is not None:
        return mapped
    return snap_hex(value)


_TRUECOLOR_PROGRAMS = {
    "iTerm.app",
    "Ghostty",
    "WezTerm",
    "vscode",
    "Hyper",
    "rio",
    "contour",
    "kitty",
    "Alacritty",
    "Tabby",
    "Termius",
}

_TRUECOLOR_TERMS = (
    "truecolor",
    "24bit",
    "direct",
    "kitty",
    "alacritty",
    "ghostty",
    "wezterm",
    "foot",
    "contour",
    "rio",
    "vte",
    "konsole",
)


def _apple_terminal_truecolor() -> bool:
    """Truecolor support in macOS Terminal.app arrived with macOS 26 Tahoe.

    (Darwin kernel major 25). Pre-Tahoe Terminal.app mangles 24-bit
    escapes, so only Tahoe and later get the truecolor path.
    """
    try:
        import platform

        ver = platform.mac_ver()[0]  # e.g. '26.1'
        if ver:
            return int(ver.split(".")[0]) >= 26
    except (ValueError, IndexError):
        pass
    try:
        release = os.uname().release  # Darwin major: 25 == Tahoe
        return int(release.split(".")[0]) >= 25
    except (ValueError, IndexError, AttributeError):
        return False


def supports_truecolor() -> bool:
    """Detect 24-bit color support from the environment.

    Unknown terminals conservatively report False: the 256-color fallback
    is clean and deterministic everywhere, while truecolor escapes on a
    terminal that cannot handle them (or a multiplexer without Tc
    passthrough) produce garbage. Use VERITAS_TRUECOLOR=1 to override.
    """
    force = os.environ.get("VERITAS_TRUECOLOR", "").strip().lower()
    if force in ("1", "true", "yes", "on"):
        return True
    if force in ("0", "false", "no", "off"):
        return False
    # tmux/screen only pass truecolor through with explicit Tc support
    # configured; without it 24-bit escapes garble, so play it safe.
    if os.environ.get("TMUX", "") or os.environ.get("STY", ""):
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    colorterm = os.environ.get("COLORTERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return True
    term = os.environ.get("TERM", "").lower()
    if any(key in term for key in _TRUECOLOR_TERMS):
        return True
    if os.environ.get("WT_SESSION", ""):
        # Windows Terminal (truecolor capable).
        return True
    if os.environ.get("VTE_VERSION", "") or os.environ.get("KONSOLE_VERSION", ""):
        # VTE-based terminals (GNOME Terminal, Terminator, Tilix, ...) and
        # Konsole all handle 24-bit color.
        return True
    program = os.environ.get("TERM_PROGRAM", "")
    if program in _TRUECOLOR_PROGRAMS:
        return True
    if program == "Apple_Terminal":
        # Terminal.app speaks 24-bit color since macOS 26 Tahoe; older
        # releases mangle those escapes and stay on the 256-color path.
        return _apple_terminal_truecolor()
    return False
