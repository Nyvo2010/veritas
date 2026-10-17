"""Static CLI screens with one consistent spacing system.

Every screen uses:
- compact logo header (no animation, fixed 7-line wordmark)
- single console.print() blank line between blocks
- Tinted blocks (accent title over a dark fill, no ASCII borders), matching
  the OpenTUI panels. Rich downgrades the hex fill to indexed escapes on
  256-color terminals, so the style holds everywhere.
"""

from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich import box as rich_box

from ui import theme


PANEL_PAD = (1, 2)


def block_renderable(title: Optional[str], body, title_style: Optional[str] = None):
    """Tinted block: bold title label over a dark fill, no ASCII borders.

    Mirrors the OpenTUI ``panel()`` (title, breathing room, tinted body).
    Returns a renderable for embedding in layouts; use the print_*
    helpers for direct screen output.
    """
    inner = [body] if not title else [
        Text(title, style=f"bold {title_style or theme.PRIMARY}"),
        Text(""),
        body,
    ]
    return Padding(Group(*inner), PANEL_PAD, style=f"on {theme.PANEL_BG}")


def print_header(console: Console, subtitle: Optional[str] = None) -> None:
    # One blank line between the terminal's top edge and the first block,
    # matching the OpenTUI frontend's MARGIN_TOP.
    console.print()
    title = theme.LOGO_SUBTITLE if subtitle is None else f"{theme.LOGO_SUBTITLE} — {subtitle}"
    try:
        width = console.width
    except Exception:
        width = 80
    if width is not None and width < 96:
        # Narrow terminals (e.g. default 80-col macOS Terminal): the wide
        # ASCII wordmark would wrap and look broken, so use a compact title
        # instead — same threshold as the OpenTUI banner fallback.
        console.print(block_renderable(title, Text("VERITAS", style=f"bold {theme.PRIMARY}")))
    else:
        # no_wrap keeps wide ASCII art intact instead of reflowing it.        logo = Text("\n".join(theme.COMPACT_LOGO), style=f"bold {theme.PRIMARY}", no_wrap=True)
        console.print(block_renderable(title, logo))
        console.print()
        console.print()
    console.print()


def print_panel(
    console: Console,
    body: str,
    title: str,
    style: Optional[str] = None,
    body_style: Optional[str] = None,
) -> None:
    style = style or theme.PRIMARY
    console.print(
        block_renderable(title, Text(body, style=body_style or theme.SECONDARY), title_style=style)
    )
    console.print()


def kv_panel(console: Console, title: str, rows: List[tuple], style: Optional[str] = None) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style=f"bold {theme.PRIMARY}")
    table.add_column("Value", style=theme.SECONDARY)
    for label, value in rows:
        table.add_row(label, value)
    console.print(block_renderable(title, table, title_style=style))
    console.print()


def menu_panel(console: Console, options: List[tuple]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style=f"bold {theme.PRIMARY}")
    table.add_column("Description", style=theme.SECONDARY)
    for key, label in options:
        table.add_row(f"[{key}]", label)
    console.print(block_renderable("Main Menu", table))
    console.print()


def filters_panel(console: Console, toggles: Dict[str, bool], title: str = "Sources") -> None:
    table = Table(show_header=True, box=rich_box.SIMPLE, padding=(0, 2))
    table.add_column("No.", style=f"bold {theme.PRIMARY}", justify="right")
    table.add_column("Category", style=theme.SECONDARY)
    table.add_column("Status", justify="center")
    for i, (name, enabled) in enumerate(toggles.items(), 1):
        status = f"[{theme.SUCCESS}]ON[/{theme.SUCCESS}]" if enabled else f"[{theme.ERROR}]OFF[/{theme.ERROR}]"
        table.add_row(str(i), name.replace("_", " ").title(), status)
    console.print(block_renderable(title, table))
    console.print()
