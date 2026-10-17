"""Veritas terminal capability probe.

Run this in EVERY terminal you care about (VS Code, Terminal.app, Warp, ...)
and save the output. It prints the terminal's environment, the OpenTUI native
renderer's detected capabilities, and the detected palette, then exits.

Usage (from the Veritas project root, with the venv active):

    .venv/bin/python tools/term_caps.py
"""

from __future__ import annotations

import asyncio
import sys


def main() -> None:
    from opentui.renderer.core import create_cli_renderer
    from opentui.palette.detector import TerminalPalette

    envs = ("TERM", "COLORTERM", "TERM_PROGRAM", "TERM_PROGRAM_VERSION",
            "LC_TERMINAL", "WT_SESSION", "KITTY_WINDOW_ID", "WEZTERM_EXECUTABLE")
    print("== environment ==")
    for k in envs:
        v = __import__("os").environ.get(k)
        print(f"{k}={v!r}")

    async def probe() -> None:
        renderer = await create_cli_renderer()
        try:
            caps = renderer.capabilities
            print("== native-detected capabilities ==")
            for f in (
                "rgb", "kitty_graphics", "kitty_keyboard", "unicode",
                "explicit_width", "sgr_pixels", "sync", "sixel",
                "term_name", "term_version",
            ):
                print(f"{f}={getattr(caps, f)!r}")
            print("== palette detection (OSC 4/10/11, 1s timeout) ==")
            stdin, stdout = sys.stdin, sys.stdout
            try:
                detector = TerminalPalette(stdin, stdout)
                colors = await asyncio.wait_for(detector.detect(1000), timeout=5.0)
                print(f"palette={colors!r}")
            except Exception as e:  # noqa: BLE001
                print(f"palette detection failed: {type(e).__name__}: {e}")
        finally:
            renderer.destroy()

    asyncio.run(probe())


if __name__ == "__main__":
    main()
