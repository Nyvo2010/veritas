"""Terminal color compatibility for OpenTUI (pty proxy with SGR downconverter).

Problem: opentui-python 0.1.2's native engine detects non-truecolor terminals
(``rgb=False`` — no ``COLORTERM``, e.g. Apple Terminal) but still emits raw
24-bit SGR escapes (``38;2;r;g;b``/``48;2;...``). The Python-side ``write_out``
is bypassed because the native render loop writes straight to file
descriptor 1. Terminals without truecolor support ignore or misparse those
escapes, so panel backgrounds disappear and accents render as the wrong
color. VS Code's terminal supports truecolor, which is why it looked correct.

Fix: re-exec the TUI as a child behind a pty proxy. The child keeps the exact
ty structure it would normally have (fds 0/1/2 all the same tty — verified to
render correctly in a standalone capture), while the proxy:

- forwards tty → pty bytes (keys, paste, terminal query replies) untouched;
- forwards pty → tty bytes, rewriting 24-bit SGR runs to nearest xterm-256
  equivalents (6x6x6 cube + 24-step grayscale, the standard xterm projection);
- mirrors window-size changes and restores the terminal on child exit.
"""

from __future__ import annotations

import fcntl
import os
import re
import select
import struct
import sys
import termios
import tty

_TRUECOLOR_ENV = ("truecolor", "24bit")

# ESC[ 38;2;r;g;b m  and  ESC[ 48;2;r;g;b m  (foreground / background runs)
_SGR_TRUECOLOR = re.compile(rb"\x1b\[(38|48);2;(\d{1,3});(\d{1,3});(\d{1,3})m")

_PROXY_ENV = "VERITAS_COLOR_PROXY"
_READ_CHUNK = 65536


def supports_truecolor() -> bool:
    """Env-based truecolor probe; false exactly where opentui misrenders."""
    if os.environ.get("COLORTERM", "").lower() in _TRUECOLOR_ENV:
        return True
    if os.environ.get("WT_SESSION") or os.environ.get("ConEmuANSI", "").lower() == "on":
        return True
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if term_program in ("iterm.app", "ghostty", "vscode"):
        return True
    if os.environ.get("KITTY_WINDOW_ID") or os.environ.get("KITTY_PID"):
        return True
    if os.environ.get("WEZTERM_EXECUTABLE"):
        return True
    return False


def _rgb_to_256(r: int, g: int, b: int) -> int:
    """Closest xterm-256 index to an (r, g, b) triple.

    Candidates: the 6x6x6 cube (levels at 0/95/135/175/215/255) and, for
    near-equal channels, the 24-step grayscale ramp (8/18/.../238). The
    candidate with the smallest squared channel distance wins."""
    cube = (0, 95, 135, 175, 215, 255)
    cube_levels = tuple(min(range(6), key=lambda i: abs(cube[i] - v)) for v in (r, g, b))
    cube_sq = sum((cube[cube_levels[i]] - (r, g, b)[i]) ** 2 for i in range(3))
    cube_idx = 16 + 36 * cube_levels[0] + 6 * cube_levels[1] + cube_levels[2]

    if abs(r - g) < 5 and abs(g - b) < 5:  # near-gray: try the ramp
        level = min(range(24), key=lambda i: abs(8 + i * 10 - (r + g + b) / 3))
        v = 8 + level * 10
        ramp_sq = (r - v) ** 2 + (g - v) ** 2 + (b - v) ** 2
        if ramp_sq < cube_sq:
            return 232 + level
    return cube_idx


def _downconvert(match: re.Match[bytes]) -> bytes:
    slot, r, g, b = match.group(1), match.group(2), match.group(3), match.group(4)
    code = _rgb_to_256(int(r), int(g), int(b))
    return b"\x1b[" + slot + b";5;" + str(code).encode() + b"m"


def rewrite_truecolor(data: bytes) -> tuple[bytes, bytes]:
    """Rewrite complete 24-bit SGR runs. Returns (out, tail) where tail is a
    possibly-incomplete trailing escape sequence to resubmit with the next
    chunk; pass tail back in with the following data."""
    buf = data
    out = _SGR_TRUECOLOR.sub(_downconvert, buf) if b";2;" in buf else buf
    tail = b""
    idx = out.rfind(b"\x1b[")
    if idx != -1:
        rest = out[idx:]
        if not re.search(rb"\x1b\[[0-9;]*m", rest):
            tail = rest
            out = out[:idx]
    return out, tail


def should_proxy() -> bool:
    """One-time check, run only from the outermost interactive process."""
    if os.environ.get(_PROXY_ENV) not in (None, "", "0"):
        return False
    if not supports_truecolor():
        try:
            return os.isatty(0) and os.isatty(1) and os.fstat(0).st_rdev == os.fstat(1).st_rdev
        except OSError:
            return False
    return False


def run_proxy() -> None:
    """Replace this process with the TUI running behind a pty proxy."""
    import pty

    import cli

    src_dir = os.path.dirname(os.path.abspath(cli.__file__))

    real_tty = 0  # fds 0/1/2 are the same terminal (checked in should_proxy)
    attrs_before = termios.tcgetattr(real_tty)
    tty.setraw(real_tty, termios.TCSANOW)

    pid, master = pty.fork()
    if pid == 0:
        # Child: pristine TUI with the pty as its only terminal.
        os.environ[_PROXY_ENV] = "1"
        code = (
            "import sys; sys.path.insert(0, %r); import cli; cli.main()" % (src_dir,)
        )
        os.execv(sys.executable, [sys.executable, "-c", code])
        os._exit(127)

    # Parent (proxy): pump both directions until the child exits.
    def mirror_winsize() -> None:
        try:
            wins = fcntl.ioctl(real_tty, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
            fcntl.ioctl(master, termios.TIOCSWINSZ, wins)
        except OSError:
            pass

    mirror_winsize()
    status = 0
    tail = b""
    done = False
    try:
        while not done:
            try:
                r, _, _ = select.select([real_tty, master], [], [], 0.25)
            except InterruptedError:
                continue
            except OSError:
                break
            if master in r:
                try:
                    chunk = os.read(master, _READ_CHUNK)
                except OSError:
                    chunk = b""  # EIO: pty closed, child is gone
                if chunk:
                    data, tail = rewrite_truecolor(tail + chunk)
                    if data:
                        try:
                            os.write(1, data)
                        except OSError:
                            break
                else:
                    done = True
                mirror_winsize()
            if real_tty in r:
                try:
                    chunk = os.read(real_tty, _READ_CHUNK)
                except OSError:
                    chunk = b""
                if chunk:
                    try:
                        os.write(master, chunk)
                    except OSError:
                        break
                else:
                    done = True
            if not r:
                try:
                    wpid, wstatus = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if wpid:
                    if tail:
                        try:
                            os.write(1, tail)
                        except OSError:
                            pass
                        tail = b""
                    try:
                        status = os.waitstatus_to_exitcode(wstatus)
                    except ValueError:
                        status = 1
                    done = True
    finally:
        # Final drain: give the pty a beat to flush, then reap the child.
        if tail:
            try:
                os.write(1, tail)
            except OSError:
                pass
        try:
            _, wstatus = os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass
        termios.tcsetattr(real_tty, termios.TCSANOW, attrs_before)
        raise SystemExit(status)
