"""Static CLI screens with one consistent spacing system.

Every screen uses:
- compact logo header (no animation, fixed 7-line wordmark)
- single console.print() blank line between blocks
- Panel(padding=(0, 2), box=ROUNDED) everywhere
"""

from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box as rich_box

from ui import theme


def print_header(console: Console, subtitle: Optional[str] = None) -> None:
    # no_wrap keeps wide ASCII art intact instead of reflowing it.
    logo = Text("\n".join(theme.COMPACT_LOGO), style=f"bold {theme.PRIMARY}", no_wrap=True)
    title = theme.LOGO_SUBTITLE if subtitle is None else f"{theme.LOGO_SUBTITLE} — {subtitle}"
    console.print(
        Panel(
            logo,
            title=title,
            title_align="left",
            box=theme.PANEL_BOX,
            style=theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )
    )
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
        Panel(
            Text(body, style=body_style or theme.SECONDARY),
            title=title,
            title_align="left",
            box=theme.PANEL_BOX,
            style=style,
            padding=theme.PANEL_PADDING,
        )
    )
    console.print()


def kv_panel(console: Console, title: str, rows: List[tuple], style: Optional[str] = None) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style=f"bold {theme.PRIMARY}")
    table.add_column("Value", style=theme.SECONDARY)
    for label, value in rows:
        table.add_row(label, value)
    console.print(
        Panel(
            table,
            title=title,
            title_align="left",
            box=theme.PANEL_BOX,
            style=style or theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )
    )
    console.print()


def menu_panel(console: Console, options: List[tuple]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style=f"bold {theme.PRIMARY}")
    table.add_column("Description", style=theme.SECONDARY)
    for key, label in options:
        table.add_row(f"[{key}]", label)
    console.print(
        Panel(
            table,
            title="Main Menu",
            title_align="left",
            box=theme.PANEL_BOX,
            style=theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )
    )
    console.print()


def filters_panel(console: Console, toggles: Dict[str, bool], title: str = "Sources") -> None:
    table = Table(show_header=True, box=rich_box.SIMPLE, padding=(0, 2))
    table.add_column("No.", style=f"bold {theme.PRIMARY}", justify="right")
    table.add_column("Category", style=theme.SECONDARY)
    table.add_column("Status", justify="center")
    for i, (name, enabled) in enumerate(toggles.items(), 1):
        status = f"[{theme.SUCCESS}]ON[/{theme.SUCCESS}]" if enabled else f"[{theme.ERROR}]OFF[/{theme.ERROR}]"
        table.add_row(str(i), name.replace("_", " ").title(), status)
    console.print(
        Panel(
            table,
            title=title,
            title_align="left",
            box=theme.PANEL_BOX,
            style=theme.PRIMARY,
            padding=theme.PANEL_PADDING,
        )
    )
    console.print()
