"""Persistent user settings for Veritas.

Stored as JSON outside the repo so a global install keeps working:
  POSIX: ~/.config/veritas/settings.json
  Windows: %APPDATA%/veritas/settings.json

Precedence for API keys and base URL: explicit argument > environment
variable (.env supported) > settings file > built-in default.

The file holds secrets (API keys), so it is created with mode 0o600
on POSIX systems.

Optional: VERITAS_REPORTS_DIR to override the reports directory.
"""

import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

DEFAULTS: Dict[str, Any] = {
    "theme": "cyan",
    "reports_dir": os.path.join(os.path.expanduser("~"), "Veritas", "reports"),
    "llm_api_key": "",
    "llm_base_url": DEFAULT_NVIDIA_BASE_URL,
    "llm_model": DEFAULT_MODEL,
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

# Pre-generic-provider keys. Still honored on load (migrated in memory),
# but no longer written back to disk.
LEGACY_KEYS = ("nvidia_api_key", "nvidia_api_base_url")


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
    stored: Dict[str, Any] = {}
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
    # One-time migration from the NVIDIA-only era.
    if isinstance(stored, dict):
        if not merged.get("llm_api_key") and stored.get("nvidia_api_key"):
            merged["llm_api_key"] = stored["nvidia_api_key"]
        if stored.get("nvidia_api_base_url") and merged.get("llm_base_url") in (DEFAULT_NVIDIA_BASE_URL, ""):
            merged["llm_base_url"] = stored["nvidia_api_base_url"]
    return merged


def save_settings(settings: Dict[str, Any]) -> str:
    """Persist settings to disk. Returns the path written."""
    directory = settings_dir()
    os.makedirs(directory, exist_ok=True)
    path = settings_path()
    # Only persist known keys so hand-edited files stay clean.
    payload = {key: settings.get(key, DEFAULTS[key]) for key in DEFAULTS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
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
    """Resolve the LLM endpoint config: env (LLM_* then NVIDIA_*) wins over file."""
    settings = settings if settings is not None else load_settings()
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("NVIDIA_API_KEY", "").strip()
        or str(settings.get("llm_api_key", "") or settings.get("nvidia_api_key", "")).strip()
    )
    base_url = (
        os.environ.get("LLM_BASE_URL", "").strip()
        or os.environ.get("NVIDIA_API_BASE_URL", "").strip()
        or str(settings.get("llm_base_url", "") or settings.get("nvidia_api_base_url", "") or DEFAULT_NVIDIA_BASE_URL).strip()
    )
    model = (
        os.environ.get("LLM_MODEL", "").strip()
        or str(settings.get("llm_model", "") or DEFAULT_MODEL).strip()
    )
    return {"api_key": api_key, "base_url": base_url, "model": model}


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
