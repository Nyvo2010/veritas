"""Detailed backend logging, written to a file and never to the screen.

The TUI owns the terminal, so anything the backend wants to say goes to
``<config>/logs/veritas.log`` instead: every model call with its model id,
prompt size, duration, finish reason and retries; every search with its query,
depth, result count and duration; every phase and step. DEBUG is the default
because the point is to be able to answer "where did the time go?" after the
fact.

Set ``VERITAS_LOG_LEVEL`` to raise or lower the file level, and
``VERITAS_LOG_CONSOLE=1`` to also stream the log to stderr (handy when running
the classic CLI, useless under the TUI).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import settings

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
_MAX_BYTES = 2_000_000
_BACKUPS = 3


def log_dir() -> str:
    """Directory holding the log file (created on demand)."""
    return os.path.join(settings.settings_dir(), "logs")


def log_path() -> str:
    """Full path of the rolling log file."""
    return os.path.join(log_dir(), "veritas.log")


def configure(force: bool = False) -> str:
    """Attach the file handler once. Returns the log path."""
    global _CONFIGURED
    path = log_path()
    if _CONFIGURED and not force:
        return path

    level_name = (os.environ.get("VERITAS_LOG_LEVEL") or "DEBUG").strip().upper()
    level = getattr(logging, level_name, logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(min(level, logging.INFO))

    handler = None
    try:
        os.makedirs(log_dir(), exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS,
                                      encoding="utf-8")
        try:
            # Prompts and queries end up in here, so keep it owner-only.
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        # A read-only home should never stop the app from running.
        handler = logging.StreamHandler(open(os.devnull, "w"))

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)

    if os.environ.get("VERITAS_LOG_CONSOLE", "").strip() in ("1", "true", "yes"):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(console)

    # These libraries are chatty at DEBUG and drown out our own lines.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "requests", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("veritas").info("logging to %s (level %s)", path, level_name)
    return path
