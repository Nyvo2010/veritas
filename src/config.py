"""Environment-based defaults and tuning constants for Veritas.

Anything a user is expected to change lives in the settings file or is
resolved from the environment (see settings.py). These are fallbacks.
"""

import os
from dotenv import load_dotenv

# Load .env (if present) so config reads environment variables
load_dotenv()

# API Keys
# Default provider is NVIDIA NIM (OpenAI-compatible endpoint), but any
# OpenAI-compatible base URL works. LLM_* env vars win over NVIDIA_* ones.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_API_BASE_URL = os.environ.get("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", NVIDIA_API_KEY)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", NVIDIA_API_BASE_URL)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Model Configuration
# Default model for all LLM inference (analysis, reasoning, reports).
# Override per install via settings, the setup wizard, or LLM_MODEL.
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
LLM_MODEL = os.environ.get("LLM_MODEL", NVIDIA_MODEL)
# Optional second model the client switches to when the primary keeps
# failing (see LLMClient). Empty means "no fallback".
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "")
TEMPERATURE_ANALYSIS = 0.3  # Lower for consistent reasoning
TEMPERATURE_CREATIVE = 0.7  # Higher for report writing
TOP_P = 0.95

# Report Generation Settings
REPORT_SECTION_MAX_TOKENS = 12000
REPORT_SUMMARY_MAX_TOKENS = 4000

# Rate Limiting
MAX_RETRIES = 5
RETRY_DELAY = 2
EXPONENTIAL_BACKOFF = True

# API Timeout Settings (in seconds)
API_REQUEST_TIMEOUT = 120  # 2 minutes for API requests
SEARCH_TIMEOUT = 90  # 1.5 minutes for search operations

# Research Parameters
MAX_RESEARCH_ITERATIONS = 10
MAX_SEARCH_RESULTS = 15

# Website domain mappings
WEBSITE_CATEGORIES = {
    "social": [
        "reddit.com",
        "quora.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "linkedin.com",
        "medium.com",
    ],
    "educational": [
        "arxiv.org",
        "scholar.google.com",
        "researchgate.net",
        "sciencedirect.com",
        "springer.com",
        "nature.com",
        "science.org",
        "ieee.org",
        "acm.org",
        "jstor.org",
        "pubmed.ncbi.nlm.nih.gov",
    ],
    "open_knowledge": [
        "github.com",
        "stackoverflow.com",
        "stackexchange.com",
        "wikipedia.org",
        "wikidata.org",
        "wikihow.com",
    ],
    "news": [
        "cnn.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
        "reuters.com",
        "apnews.com",
        "bloomberg.com",
        "wsj.com",
    ],
}

# Parallel Processing
MAX_PARALLEL_SECTIONS = 6
