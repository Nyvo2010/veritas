"""
Configuration file for Deep Research Agent - NVIDIA NIM Version
"""

import os
from dotenv import load_dotenv

# Load .env (if present) so config reads environment variables
load_dotenv()

# API Keys
# NVIDIA NIM API key (OpenAI-compatible endpoint at https://integrate.api.nvidia.com/v1)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_API_BASE_URL = os.environ.get("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Model Configuration
# NVIDIA NIM models - Nemotron Lightning for all LLM inference
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # For analysis and reasoning
NVIDIA_REPORT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # For report generation
TEMPERATURE_ANALYSIS = 0.3  # Lower for consistent reasoning
TEMPERATURE_CREATIVE = 0.7  # Higher for report writing
TOP_P = 0.95

# Report Generation Settings
REPORT_SECTION_MAX_TOKENS = 12000
REPORT_SUMMARY_MAX_TOKENS = 4000
MIN_SECTION_LENGTH = 4000
TARGET_REPORT_LENGTH = 30000
MAX_PARALLEL_SECTIONS = 6

# Rate Limiting
MAX_RETRIES = 5
RETRY_DELAY = 2
EXPONENTIAL_BACKOFF = True

# API Timeout Settings (in seconds)
API_REQUEST_TIMEOUT = 120  # 2 minutes for API requests
SEARCH_TIMEOUT = 90  # 1.5 minutes for search operations

# Research Parameters
MAX_RESEARCH_ITERATIONS = 10
MAX_SUBQUESTIONS_PER_ITERATION = 5
MAX_SEARCH_RESULTS = 15
SEARCH_DEPTH_INITIAL = "basic"
SEARCH_DEPTH_DEEP = "advanced"

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

# Default category toggles (True = include, False = exclude)
DEFAULT_CATEGORY_TOGGLES = {
    "social": True,
    "educational": True,
    "open_knowledge": True,
    "news": True,
    "general": True,
}

# Report Generation
REPORT_SECTIONS = [
    "Executive Summary",
    "Background & Context",
    "Key Findings",
    "Detailed Analysis",
    "Uncertainties & Gaps",
    "Conclusion",
    "Sources",
]

# Parallel Processing
MAX_PARALLEL_SECTIONS = 6
