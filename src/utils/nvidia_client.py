"""Backward-compatible shim: the NVIDIA-only client is now the generic LLM client."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.llm_client import LLMClient, NvidiaClient, fetch_models  # noqa: F401
