"""Minimal OpenTUI color rendering repro (no Veritas code involved).

Run in the SAME terminals as tools/term_caps.py and compare what you see:

    .venv/bin/python tools/color_repro.py

It draws one static frame for ~2.5 seconds, then prints raw ANSI reference
swatches. If the OpenTUI frame and the raw-escape swatches look different in
a terminal, that terminal cannot display truecolor and the renderer is not
downgrading properly — the bug is below Veritas's UI code.
"""

from __future__ import annotations

import asyncio

from opentui import Box, Text, render

PANEL_BG = "#232326"
ACCENT = "#6f8fb8"
TITLE_FG = "#1c1c1e"


async def main() -> None:
    def scene() -> Box:
        return Box(
            Text("  truecolor repro frame  ", fg=TITLE_FG, bg=ACCENT, bold=True),
            Text("accent text #6f8fb8", fg=ACCENT),
            Text("panel bg #232326 ->", fg=ACCENT, bg=PANEL_BG),
            background_color=PANEL_BG,
            padding=1,
            border=True,
            border_color=ACCENT,
        )

    task = asyncio.ensure_future(render(scene, {"use_mouse": False, "exit_on_ctrl_c": False}))
    await asyncio.sleep(2.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"render ended: {type(e).__name__}: {e}")

    print()
    print("Raw-escape reference (what the terminal SHOULD show):")
    print("\x1b[48;2;35;35;38m  truecolor bg #232326  \x1b[0m")
    print("\x1b[48;5;235m  256-color bg 235      \x1b[0m")
    print("\x1b[38;2;111;143;184m\x1b[48;2;35;35;38m  truecolor fg #6f8fb8 on bg  \x1b[0m")
    print("\x1b[38;5;103m\x1b[48;5;235m  256 fg 103 on bg 235  \x1b[0m")


if __name__ == "__main__":
    asyncio.run(main())
