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

# Any SGR sequence: ESC [ params m
_SGR = re.compile(rb"\x1b\[([0-9;]*)m")

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


def _convert_sgr_params(params: bytes) -> bytes:
    """Rewrite 38;2;r;g;b / 48;2;r;g;b groups inside one SGR parameter list.

    A positional regex misses truecolor sub-parameters that share an SGR with
    other attributes ("1;38;2;r;g;b") or with each other
    ("38;2;r;g;b;48;2;r;g;b") — those leaks are what makes the colour bug
    intermittent, so we walk the parameters instead.
    """
    tokens = params.split(b";")
    out: list = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token in (b"38", b"48") and i + 4 < n and tokens[i + 1] == b"2":
            try:
                r, g, b = int(tokens[i + 2]), int(tokens[i + 3]), int(tokens[i + 4])
            except ValueError:
                out.append(token)
                i += 1
                continue
            out.append(token + b";5;" + str(_rgb_to_256(r, g, b)).encode())
            i += 5
        else:
            out.append(token)
            i += 1
    return b";".join(out)


def _rewrite_sgr(match: re.Match[bytes]) -> bytes:
    return b"\x1b[" + _convert_sgr_params(match.group(1)) + b"m"


def _safe_split(data: bytes) -> int:
    """Index where a possibly-incomplete trailing escape begins.

    A read can end in the middle of an escape sequence. Emitting that partial
    tail makes the terminal either print it as literal text or misparse the
    colour run, so the caller buffers from here and prepends it to the next
    read. Everything before the returned index is a complete sequence (or
    plain text) and safe to rewrite and emit.
    """
    i = 0
    n = len(data)
    safe = 0
    while i < n:
        if data[i] != 0x1B:
            i += 1
            safe = i
            continue
        if i + 1 >= n:
            return i                                   # lone trailing ESC
        kind = data[i + 1]
        if kind == ord("["):                           # CSI: ends at 0x40-0x7E
            j = i + 2
            while j < n and not (0x40 <= data[j] <= 0x7E):
                j += 1
            if j >= n:
                return i
            i = safe = j + 1
        elif kind == ord("]"):                         # OSC: ends at BEL or ST
            j = i + 2
            end = -1
            while j < n:
                if data[j] == 0x07:
                    end = j + 1
                    break
                if data[j] == 0x1B and j + 1 < n and data[j + 1] == ord("\\"):
                    end = j + 2
                    break
                j += 1
            if end == -1:
                return i
            i = safe = end
        elif kind in b"()*+#%":                        # designator escapes: 3 bytes
            if i + 2 >= n:
                return i
            i = safe = i + 3
        else:                                          # other two-byte escape
            i = safe = i + 2
    return safe


def rewrite_truecolor(data: bytes) -> tuple[bytes, bytes]:
    """Rewrite 24-bit SGR runs. Returns (out, tail) where tail is a
    possibly-incomplete trailing escape sequence to resubmit with the next
    chunk; pass tail back in with the following data."""
    split = _safe_split(data)
    out = data[:split]
    tail = data[split:]
    if b";2;" in out:
        out = _SGR.sub(_rewrite_sgr, out)
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
