"""Persistent user settings for Veritas.

Stored as JSON outside the repo so a global install keeps working:
  POSIX: ~/.config/veritas/settings.json
  Windows: %APPDATA%/veritas/settings.json

Precedence for API keys and the endpoint: explicit argument > environment
variable (.env supported) > settings file. No provider is bundled, so the
base URL and model are empty until the user sets them.

The file holds secrets (API keys), so it is created with mode 0o600
on POSIX systems.

Optional: VERITAS_REPORTS_DIR to override the reports directory.
"""

import json
import os
import sys
import tempfile
from copy import deepcopy
from typing import Any, Dict, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# No bundled endpoint/model: the user provides their own OpenAI-compatible
# base URL and model id. Empty values force the setup wizard on first run.
DEFAULTS: Dict[str, Any] = {
    "theme": "blue",
    "reports_dir": os.path.join(os.path.expanduser("~"), "Veritas", "reports"),
    "llm_api_key": "",
    "llm_base_url": "",
    "llm_model": "",
    "llm_fallback_model": "",
    "tavily_api_key": "",
    "max_iterations": 10,
    "max_search_results": 15,
    "category_toggles": {
        "social": True,
        "educational": True,
        "open_knowledge": True,
        "news": True,
        "general": True,
    },
}


def settings_dir() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "veritas")
    return os.path.join(os.path.expanduser("~"), ".config", "veritas")


def settings_path() -> str:
    return os.path.join(settings_dir(), "settings.json")


def load_settings() -> Dict[str, Any]:
    """Load settings from disk, merged over defaults. Never raises."""
    merged = deepcopy(DEFAULTS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            for key, value in stored.items():
                if key == "category_toggles" and isinstance(value, dict):
                    merged["category_toggles"].update(value)
                elif key in merged:
                    merged[key] = value
    except (OSError, ValueError):
        pass
    return merged


def save_settings(settings: Dict[str, Any]) -> str:
    """Persist settings to disk. Returns the path written."""
    directory = settings_dir()
    os.makedirs(directory, exist_ok=True)
    path = settings_path()
    # Only persist known keys so hand-edited files stay clean.
    payload = {key: settings.get(key, DEFAULTS[key]) for key in DEFAULTS}
    # The file holds API keys, so it must never exist with default permissions,
    # and a crash mid-write must not leave a truncated file. ``mkstemp`` creates
    # it owner-only; the rename makes the swap atomic.
    handle, temporary = tempfile.mkstemp(prefix=".settings-", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def update_settings(**kwargs: Any) -> Dict[str, Any]:
    """Update individual settings and persist. Returns the new settings."""
    current = load_settings()
    for key, value in kwargs.items():
        if key == "category_toggles" and isinstance(value, dict):
            current["category_toggles"].update(value)
        elif key in DEFAULTS:
            current[key] = value
    save_settings(current)
    return current


def resolve_llm(settings: Dict[str, Any] = None) -> Dict[str, str]:
    """Resolve the LLM endpoint config: env (LLM_*) wins over the settings file."""
    settings = settings if settings is not None else load_settings()
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or str(settings.get("llm_api_key", "")).strip()
    )
    base_url = (
        os.environ.get("LLM_BASE_URL", "").strip()
        or str(settings.get("llm_base_url", "")).strip()
    )
    model = (
        os.environ.get("LLM_MODEL", "").strip()
        or str(settings.get("llm_model", "")).strip()
    )
    fallback_model = (
        os.environ.get("LLM_FALLBACK_MODEL", "").strip()
        or str(settings.get("llm_fallback_model", "") or "").strip()
    )
    if fallback_model == model:
        fallback_model = ""
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "fallback_model": fallback_model,
    }


def resolve_api_keys(settings: Dict[str, Any] = None) -> Tuple[str, str, str]:
    """Resolve (llm_key, tavily_key, base_url) with env-over-file precedence."""
    settings = settings if settings is not None else load_settings()
    llm = resolve_llm(settings)
    tavily_key = (
        os.environ.get("TAVILY_API_KEY", "").strip()
        or str(settings.get("tavily_api_key", "")).strip()
    )
    return llm["api_key"], tavily_key, llm["base_url"]


def mask_key(value: str) -> str:
    """Mask a secret for display, showing the last 4 chars when long enough."""
    text = str(value or "")
    if len(text) <= 4:
        return "not set" if not text else "****"
    return f"****{text[-4:]}"


def reports_dir(settings: Dict[str, Any] = None) -> str:
    """Resolved reports directory, expanded and created on demand by callers."""
    settings = settings if settings is not None else load_settings()
    override = os.environ.get("VERITAS_REPORTS_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.expanduser(str(settings.get("reports_dir") or DEFAULTS["reports_dir"]))
