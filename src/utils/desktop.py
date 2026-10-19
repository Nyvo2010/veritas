"""Open a report or reveal it in the platform file manager.

Popen is deliberate: opening a document must not block the TUI while the
user's editor or browser is alive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List


def _spawn(command: List[str]) -> None:
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _existing(path: str) -> str:
    target = os.path.abspath(os.path.expanduser(str(path or "")))
    if not path or not os.path.exists(target):
        raise FileNotFoundError(f"Not found: {target}")
    return target


def open_in_default_app(path: str) -> str:
    """Open *path* with the OS default application. Returns the resolved path."""
    target = _existing(path)
    if sys.platform == "darwin":
        _spawn(["open", target])
    elif os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
    else:
        _spawn(["xdg-open", target])
    return target


def reveal_in_file_manager(path: str) -> str:
    """Show *path* selected in Finder/Explorer, or its folder on Linux."""
    target = _existing(path)
    if sys.platform == "darwin":
        _spawn(["open", "-R", target])
    elif os.name == "nt":
        _spawn(["explorer", "/select,", target])
    else:
        # No portable "select this file" verb on Linux desktops; opening the
        # containing folder is the closest reliable equivalent.
        _spawn(["xdg-open", os.path.dirname(target) or target])
    return target
