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


def _rgb_to_hex(rgb: tuple) -> str:
    """Format an (r, g, b) 0-255 tuple as lowercase #rrggbb."""
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in rgb])


def mix_hex(a: str, b: str, t: float) -> str:
    """Mix hex color *a* toward hex color *b* by fraction *t* (0-1).

    ``mix_hex(bg, fg, 0.10)`` is 90% bg + 10% fg. Used to derive panel
    (10%) and muted (50%, documented as ~45%) tints from the live
    terminal palette.
    """
    try:
        t = max(0.0, min(1.0, float(t)))
        ar, ag, ab = _hex_to_rgb(a)
        br, bg_, bb = _hex_to_rgb(b)
    except (ValueError, IndexError, TypeError, AttributeError):
        return a
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg_ - ag) * t)
    bl = round(ab + (bb - ab) * t)
    return _rgb_to_hex((r, g, bl))


def _srgb_channel_to_linear(c: int) -> float:
    """Convert an 8-bit sRGB channel to linear light (WCAG 2.x)."""
    v = max(0, min(255, int(c))) / 255.0
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def relative_luminance(value: str) -> float:
    """Relative luminance of a hex color (0 = black, 1 = white)."""
    r, g, b = _hex_to_rgb(value)
    rl = _srgb_channel_to_linear(r)
    gl = _srgb_channel_to_linear(g)
    bl = _srgb_channel_to_linear(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colors (1-21)."""
    la = relative_luminance(a)
    lb = relative_luminance(b)
    lighter = max(la, lb)
    darker = min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def is_dark(value: str) -> bool:
    """Whether a hex background counts as dark (luminance < 0.5)."""
    try:
        return relative_luminance(value) < 0.5
    except (ValueError, IndexError):
        return True


def parse_osc_color_spec(spec: str) -> str | None:
    """Parse one OSC 10/11 color payload to lowercase #rrggbb.

    Supports ``rgb:RR/GG/BB`` with 1-4 hex digits per component (X11-style,
    scaled to 8 bits) and ``#RRGGBB`` (plus ``#RGB`` for robustness).
    Returns None on parse failure.
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("rgb:"):
        body = s[4:]
        parts = body.split("/")
        if len(parts) != 3:
            return None
        rgb = []
        for part in parts:
            part = part.strip()
            if len(part) < 1 or len(part) > 4:
                return None
            try:
                raw = int(part, 16)
            except ValueError:
                return None
            bits = 4 * len(part)
            max_v = (1 << bits) - 1
            # Scale N-bit (4/8/12/16) to 8-bit: v * 255 / max.
            v8 = round(raw * 255 / max_v) if max_v else 0
            rgb.append(max(0, min(255, v8)))
        return _rgb_to_hex(tuple(rgb))
    if s.startswith("#"):
        h = s[1:].strip()
        if len(h) == 3:
            try:
                int(h, 16)
            except ValueError:
                return None
            h = h[0] * 2 + h[1] * 2 + h[2] * 2
        if len(h) != 6:
            return None
        try:
            int(h, 16)
        except ValueError:
            return None
        return "#" + h.lower()
    return None


def parse_osc_responses(data) -> dict:
    """Extract OSC 10 (fg) / 11 (bg) colors from a raw terminal reply.

    Accepts bytes or str. ST terminators BEL (``\\x07``) and ESC-backslash
    (``\\x1b\\\\``) are both supported. Returns ``{10: hex|None,
    11: hex|None}``; missing entries are None. Takes the last occurrence
    of each OSC number when several are present.
    """
    import re as _re

    if data is None:
        return {10: None, 11: None}
    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("ascii", errors="ignore")
        except Exception:
            return {10: None, 11: None}
    else:
        text = str(data)
    out: dict = {10: None, 11: None}
    # Payload runs to ST (BEL or ESC-backslash); it never contains BEL/ESC.
    for num in (10, 11):
        pat = _re.compile(
            r"\x1b\]%d;([^\x07\x1b]*)(?:\x07|\x1b\\)" % num
        )
        matches = pat.findall(text)
        for raw in reversed(matches):
            parsed = parse_osc_color_spec(raw)
            if parsed is not None:
                out[num] = parsed
                break
    return out


def query_terminal_bg_fg(timeout: float = 0.2) -> tuple:
    """Query the live terminal bg (OSC 11) / fg (OSC 10) via escape queries.

    Only runs when both stdin and stdout are TTYs; otherwise returns
    ``(None, None)`` immediately. Stdin is put in raw mode and restored
    afterwards even on error. Total blocking time is bounded by *timeout*
    (default 0.2s, always well under ~300ms). ANY failure (timeout, no
    response, parse error, exception) returns ``(None, None)`` silently.
    Returns ``(bg_hex, fg_hex)`` with lowercase #rrggbb or Nones.
    """
    import sys as _sys
    import time as _time

    try:
        if not _sys.stdin.isatty() or not _sys.stdout.isatty():
            return (None, None)
    except Exception:
        return (None, None)
    try:
        timeout = max(0.0, min(0.25, float(timeout)))
    except (ValueError, TypeError):
        timeout = 0.2
    try:
        import select as _select
        import termios as _termios
        import tty as _tty
    except ImportError:
        return (None, None)
    try:
        stdin_fd = _sys.stdin.fileno()
    except Exception:
        return (None, None)
    try:
        saved = _termios.tcgetattr(stdin_fd)
    except Exception:
        return (None, None)
    restore_blocking = None
    try:
        try:
            import os as _os

            try:
                restore_blocking = _os.get_blocking(stdin_fd)
            except Exception:
                restore_blocking = None
            _tty.setraw(stdin_fd)
            try:
                _os.set_blocking(stdin_fd, False)
            except Exception:
                pass
        except Exception:
            pass
        try:
            _sys.stdout.write("\x1b]10;?\x1b\\\x1b]11;?\x1b\\")
            _sys.stdout.flush()
        except Exception:
            return (None, None)
        import os as _os2

        buf = b""
        deadline = _time.monotonic() + timeout
        parsed: dict = {10: None, 11: None}
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break
            try:
                ready, _, _ = _select.select([stdin_fd], [], [], min(0.05, remaining))
            except Exception:
                break
            if not ready:
                # No input yet; if we already have both colors, stop early.
                if parsed.get(10) and parsed.get(11):
                    break
                continue
            try:
                chunk = _os2.read(stdin_fd, 4096)
            except OSError:
                break
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            try:
                parsed = parse_osc_responses(buf)
            except Exception:
                pass
            if parsed.get(10) and parsed.get(11):
                break
            if len(buf) > 8192:
                break
        try:
            final = parse_osc_responses(buf)
        except Exception:
            return (None, None)
        return (final.get(11), final.get(10))
    except Exception:
        return (None, None)
    finally:
        try:
            _termios.tcsetattr(stdin_fd, _termios.TCSADRAIN, saved)
        except Exception:
            pass
        if restore_blocking is not None:
            try:
                import os as _os3

                _os3.set_blocking(stdin_fd, restore_blocking)
            except Exception:
                pass


def derive_system_theme(bg_hex: str, fg_hex: str,
                        panel_mix: float = 0.10,
                        muted_mix: float = 0.50) -> dict:
    """Derive opencode-style system colors from the live terminal palette.

    - panel bg = mix of terminal bg toward terminal fg (~10%, always a
      visible tint of the real bg, opaque/transparency-safe). The mix is
      reduced (10% -> 7% -> 5% -> 3% -> 0%) only as far as needed to keep
      body-vs-panel contrast >= 4.5:1, so low-contrast palettes (e.g.
      solarized) still pass while staying a tint of the real bg.
    - body text = terminal fg EXACTLY (the terminal's own color, always
      readable on its bg).
    - muted/hints = mix(bg, fg, ~45-50%; starting at *muted_mix* and moving
      toward fg in 10% steps until muted-vs-panel >= 3:1, so both dark and
      light terminals pass).
    - dark/light is computed from bg luminance (returned as ``is_dark``).

    Accent roles are intentionally NOT derived here: the caller keeps the
    user's chosen theme accent on the truecolor path. Raises ValueError
    on invalid input so the caller can fall back to defaults byte-identical.
    """
    if bg_hex is None or fg_hex is None:
        raise ValueError("missing terminal bg/fg")
    bg = str(bg_hex).strip().lower()
    fg = str(fg_hex).strip().lower()
    # Normalize (#RGB -> #RRGGBB); reject anything else.
    for label, val in (("bg", bg), ("fg", fg)):
        h = val[1:] if val.startswith("#") else None
        if h is None or len(h) not in (3, 6):
            raise ValueError(f"invalid {label} color: {bg_hex!r}/{fg_hex!r}")
        try:
            int(h, 16)
        except ValueError:
            raise ValueError(f"invalid {label} color: {bg_hex!r}/{fg_hex!r}") from None
        if len(h) == 3:
            h = h[0] * 2 + h[1] * 2 + h[2] * 2
        if label == "bg":
            bg = "#" + h
        else:
            fg = "#" + h
    # Validate via luminance (raises on bad hex).
    dark = is_dark(bg)
    try:
        start_panel = max(0.0, min(1.0, float(panel_mix)))
    except (ValueError, TypeError):
        start_panel = 0.10
    try:
        start_muted = max(0.0, min(1.0, float(muted_mix)))
    except (ValueError, TypeError):
        start_muted = 0.50
    body = fg
    # Panel: prefer ~10% tint, shrink only to rescue 4.5:1 body contrast.
    panel_bg = mix_hex(bg, fg, start_panel)
    for candidate in (start_panel, 0.07, 0.05, 0.03, 0.0):
        if candidate > start_panel:
            continue
        trial = mix_hex(bg, fg, candidate)
        try:
            if contrast_ratio(body, trial) >= 4.5:
                panel_bg = trial
                break
        except (ValueError, IndexError):
            continue
    else:
        panel_bg = mix_hex(bg, fg, 0.0)
    # Muted: start at ~45-50%, move toward fg until 3:1 vs panel.
    muted = mix_hex(bg, fg, start_muted)
    candidates = []
    for step_up in (0.0, 0.10, 0.20, 0.30, 0.40):
        cand = round(min(0.90, start_muted + step_up), 10)
        if cand not in candidates:
            candidates.append(cand)
    if 0.90 not in candidates:
        candidates.append(0.90)
    for candidate in candidates:
        trial = mix_hex(bg, fg, candidate)
        try:
            if contrast_ratio(trial, panel_bg) >= 3.0:
                muted = trial
                break
        except (ValueError, IndexError):
            continue
    else:
        muted = body
    return {
        "bg": bg,
        "fg": fg,
        "panel_bg": panel_bg,
        "body": body,
        "muted": muted,
        "is_dark": dark,
    }


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
