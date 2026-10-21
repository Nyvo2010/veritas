"""Source hygiene and citation helpers.

Two jobs, both deterministic and offline:

1. Decide which search results carry enough information to be worth analysing,
   and merge results that point at the same page (or the same syndicated text)
   so one source is never counted twice.
2. Give every remaining source one stable number for the whole report, so the
   ``[Source N]`` markers in the body and the bibliography cannot drift apart.

No network, no model, no third-party dependencies: these are heuristics, not
another inference pass. See ``tests/test_sources.py`` for the contract.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Query parameters that only track a visit, never select different content.
_TRACKING_KEYS = {
    "fbclid", "gclid", "dclid", "yclid", "igshid", "mc_cid", "mc_eid",
    "mkt_tok", "ref", "ref_src", "source", "campaign", "amp", "output",
    "_hsenc", "_hsmi", "vero_conv", "vero_id", "oly_enc_id", "oly_anon_id",
}
_TRACKING_PREFIXES = ("utm_", "pk_", "mtm_", "hsa_")

# Pages that exist to log in, sell something, or restate a policy. They can be
# long, but they do not answer a research question.
_LOW_VALUE_PATHS = {
    "login", "signin", "sign-in", "signup", "sign-up", "register",
    "subscribe", "newsletter", "cart", "checkout", "account", "privacy",
    "privacy-policy", "terms", "terms-of-service", "cookies", "cookie-policy",
    "search", "404", "not-found",
}

# Boilerplate that means the fetch was blocked or the page was empty. These
# are unambiguous, so a page that opens with one is not evidence no matter how
# much text follows: paywalls and JavaScript shells can run for thousands of
# characters.
_BLOCKED_MARKERS = (
    "enable javascript",
    "javascript is disabled",
    "access denied",
    "you have been blocked",
    "verify you are human",
    "subscribe to continue",
    "sign in to continue",
)

# Phrases that also appear in legitimate articles about captchas, errors, and
# authentication. They reject a short page, but one mention inside a long
# article is not a reason to throw the page away.
_AMBIGUOUS_BLOCKED_MARKERS = (
    "captcha",
    "page not found",
    "404 not found",
)

# A blocked page shows its notice at the top; later mentions are usually the
# article discussing the block.
_BLOCKED_HEAD_CHARS = 400

# Not every category is equally informative for a factual report. These are
# priors, not verdicts: the model still sees the real content and quality.
_CATEGORY_QUALITY = {
    "educational": 1.0,
    "open_knowledge": 0.95,
    "news": 0.90,
    "general": 0.80,
    "social": 0.65,
}

# What kind of page this is. A video, forum post, or vendor blog can be the
# only thing a search returns, but it is not the same evidence as a paper or a
# statistics agency, and the score should not pretend otherwise. The prior is
# a multiplier on the blended score, so a long marketing page cannot outrank a
# paper just by being long.
_SOURCE_TYPE_PRIOR = {
    "academic": 1.0,
    "institution": 0.95,
    "reference": 0.90,
    "news": 0.88,
    "docs": 0.85,
    "web": 0.80,
    "forum": 0.55,
    "video": 0.50,
}

_SOURCE_TYPE_LABEL = {
    "academic": "Academic paper",
    "institution": "Institution",
    "reference": "Reference work",
    "news": "News",
    "docs": "Documentation",
    "web": "Web page",
    "forum": "Forum post",
    "video": "Video",
}

_ACADEMIC_HOSTS = (
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "doi.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "link.springer.com", "ieee.org", "acm.org", "jstor.org", "mdpi.com",
    "plos.org", "frontiersin.org", "cell.com", "elifesciences.org",
    "tandfonline.com", "sagepub.com", "wiley.com", "onlinelibrary.wiley.com",
    "researchgate.net", "scholar.google.com", "semanticscholar.org",
)
_INSTITUTION_HOSTS = (
    "who.int", "un.org", "worldbank.org", "imf.org", "iea.org", "oecd.org",
    "europa.eu", "ec.europa.eu", "cdc.gov", "nih.gov", "nsf.gov",
)
_NEWS_HOSTS = (
    "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com", "theguardian.com",
    "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com",
    "economist.com", "npr.org", "aljazeera.com", "cnbc.com", "axios.com",
    "arstechnica.com", "theverge.com", "techcrunch.com", "wired.com",
)
_REFERENCE_HOSTS = ("wikipedia.org", "wikidata.org", "britannica.com")
_DOCS_HOSTS = (
    "readthedocs.io", "developer.mozilla.org", "docs.python.org",
    "github.com", "gitlab.com",
)
_VIDEO_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com")
_FORUM_HOSTS = (
    "reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com",
    "news.ycombinator.com", "x.com", "twitter.com", "discourse.org",
)


def _host_matches(host: str, domains: Iterable[str]) -> bool:
    """True when *host* is one of *domains* or a subdomain of one."""
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def source_type(url: object) -> str:
    """Classify a page by what kind of source it is (deterministic).

    One of ``academic``, ``institution``, ``reference``, ``news``, ``docs``,
    ``forum``, ``video``, or ``web``. This is what keeps a YouTube clip and a
    peer-reviewed paper from carrying the same evidence-quality score, and it
    is shown in the bibliography so the reader can judge the list themselves.
    """
    canonical = canonical_url(url)
    if not canonical:
        return "web"
    try:
        parsed = urlparse(canonical)
    except Exception:
        return "web"
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    if _host_matches(host, _VIDEO_HOSTS):
        return "video"
    # Facebook is a social network, but a /videos/ page is a video first.
    if _host_matches(host, ("facebook.com", "fb.watch")) and (
        "/videos/" in path or "watch" in path
    ):
        return "video"
    if _host_matches(host, _FORUM_HOSTS):
        return "forum"
    if _host_matches(host, _ACADEMIC_HOSTS):
        return "academic"
    # A university's newsroom is an institution, not a paper: med.stanford.edu
    # /news is a press article, and news.osu.edu is a news desk.
    if host.startswith("news."):
        return "news"
    if any(segment in path for segment in ("/news/", "/blog/", "/insights/")):
        if host.endswith((".edu", ".ac.uk", ".edu.au")):
            return "institution"
    if _host_matches(host, _INSTITUTION_HOSTS) or host.endswith((".gov", ".gov.uk", ".mil")):
        return "institution"
    if host.endswith((".edu", ".ac.uk", ".edu.au")):
        return "academic"
    if _host_matches(host, _REFERENCE_HOSTS):
        return "reference"
    if _host_matches(host, _DOCS_HOSTS) or host.startswith("docs."):
        return "docs"
    if _host_matches(host, _NEWS_HOSTS):
        return "news"
    return "web"


def source_type_label(kind: str) -> str:
    """Human label for a :func:`source_type` value."""
    return _SOURCE_TYPE_LABEL.get(str(kind or "").lower(), "Web page")


# Raw page text arrives as markdown or HTML soup: the analysis prompt used to
# start with `![logo](/img/s.png)` and a nav bar, spending its excerpt budget
# on things that are not evidence. These remove the cheap noise and nothing
# else — no sentence is rewritten, so a page cannot be made to say more.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]{1,300}>")
_BOILERPLATE_LINE_RE = re.compile(
    r"^\s*(?:skip to (?:main )?content|jump to (?:main )?content|"
    r"toggle navigation|accept (?:all )?cookies|advertisement|"
    r"menu|search|share|subscribe|sign in|log in)\s*$",
    re.IGNORECASE,
)


def clean_page_text(text: object) -> str:
    """Strip page chrome from fetched text before it reaches a prompt.

    Removes markdown images, keeps link labels, drops HTML tags, one-line
    cookie/navigation boilerplate, and lines with no alphanumerics. Paragraph
    breaks are preserved so the model can still see structure.
    """
    raw = str(text or "")
    if not raw:
        return ""
    cleaned = _MD_IMAGE_RE.sub(" ", raw)
    cleaned = _MD_LINK_RE.sub(lambda match: match.group(1) or " ", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _BOILERPLATE_LINE_RE.match(stripped):
            continue
        if not re.search(r"[A-Za-z0-9]", stripped):
            continue
        lines.append(re.sub(r"[ \t]+", " ", stripped))
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return result

# Tavily scores each result for relevance to the query. 0.02 is a page that
# matched one word; keeping it because it is long is how a report about heat
# pumps ended up citing the 1995 film Heat. A missing score arrives as 0.0 and
# is treated as unknown, not as irrelevant.
_MIN_RELEVANCE = 0.15

_MIN_TEXT_CHARS = 120
_FINGERPRINT_CHARS = 320
_FINGERPRINT_MIN_CHARS = 160

# Whole-text fingerprints. The prefix fingerprint above catches copies that
# are identical from the first word; syndicated copies usually differ in the
# site chrome around the article, so they are also compared by the words they
# contain. 400 word tokens is about two paragraphs, enough to tell two
# articles on the same subject apart.
_SIGNATURE_CHARS = 8000
_SIGNATURE_TOKENS = 400
_SIGNATURE_MIN_TOKENS = 80
_SIGNATURE_OVERLAP = 0.8


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canonical_url(url: object) -> str:
    """Normalise a URL for identity comparison.

    Scheme, ``www.``, trailing slashes, fragments, default ports, and tracking
    parameters are ignored, so the same article reached from two searches has
    one identity. The path is preserved (it is case-sensitive on many hosts).
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if not parsed.netloc:
        # A pasted bare host ("example.com/page") is still a usable URL.
        if "." in parsed.path and not parsed.path.startswith("/"):
            parsed = urlparse("https://" + raw)
        else:
            return ""

    try:
        host = (parsed.hostname or "").lower()
    except ValueError:
        return ""
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]

    try:
        port = parsed.port
    except ValueError:
        port = None
    if port and not ((parsed.scheme == "http" and port == 80)
                     or (parsed.scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = re.sub(r"/+", "/", parsed.path or "/")
    if path.rstrip("/").lower().endswith("/amp"):
        path = path.rstrip("/")[:-4]
    if not path:
        path = "/"
    elif path != "/":
        # One article, one identity: a trailing slash is formatting, not a
        # different page.
        path = path.rstrip("/") or "/"

    kept = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        lowered = key.lower()
        if lowered in _TRACKING_KEYS or lowered.startswith(_TRACKING_PREFIXES):
            continue
        if lowered == "output" and value.lower() == "amp":
            continue
        kept.append((key, value))
    query = urlencode(sorted(kept))

    return urlunparse(("https", host, path, "", query, ""))


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _best_text(result) -> str:
    """The fullest text we have for a result (full page preferred)."""
    full = getattr(result, "full_content", None) or ""
    snippet = getattr(result, "content", None) or ""
    return str(full) if len(str(full)) >= len(str(snippet)) else str(snippet)


def content_fingerprint(result) -> str:
    """A cheap identity for syndicated copies of the same article.

    Returns an empty string for thin content: two 80-character snippets are
    too easy to collide by accident.
    """
    text = _best_text(result)
    compact = re.sub(r"\W+", "", text.lower())
    if len(compact) < _FINGERPRINT_MIN_CHARS:
        return ""
    return compact[:_FINGERPRINT_CHARS]


def content_signature(result) -> frozenset:
    """The set of words a page contains, for near-duplicate detection.

    Reads only the opening of the page (8k characters, 400 words): enough to
    recognise an article behind different navigation, headers, and footers,
    without hashing a whole book for every comparison.
    """
    text = _best_text(result)[:_SIGNATURE_CHARS]
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())[:_SIGNATURE_TOKENS]
    if len(tokens) < _SIGNATURE_MIN_TOKENS:
        return frozenset()
    return frozenset(tokens)


def title_key(title: object) -> str:
    """Lowercase words of a page title, with decorations removed."""
    text = re.sub(r"^\s*\[[^\]]{1,12}\]\s*", "", str(title or "").lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


def same_title(first: str, second: str) -> bool:
    """True when two pages carry the same article title.

    The same paper is often served twice, as HTML and as PDF; the URLs and the
    text differ, the title does not. One title is frequently truncated ("..."),
    so a prefix of the other counts as the same title. Short titles are ignored:
    "Home" or "Blog" says nothing about which page it is.
    """
    if not first or not second:
        return False
    shorter, longer = sorted((first, second), key=len)
    if len(shorter) < 30:
        return False
    return shorter == longer or longer.startswith(shorter)


def same_article(first: frozenset, second: frozenset) -> bool:
    """True when two signatures are mostly the same words.

    Containment rather than Jaccard: a syndicated copy adds its own chrome,
    which inflates the union without making the article different. The shared
    words must cover nearly all of the smaller page.
    """
    if not first or not second:
        return False
    return len(first & second) / min(len(first), len(second)) >= _SIGNATURE_OVERLAP


def duplicate_of(result, existing: Iterable):
    """The earlier result *result* repeats, if any.

    Compares, in order of cost: the canonical URL, the opening text, the
    article title, then the words of the whole page.
    """
    canonical = canonical_url(getattr(result, "url", ""))
    fingerprint = content_fingerprint(result)
    title = title_key(getattr(result, "title", ""))
    signature = content_signature(result)
    for other in existing:
        if canonical and canonical == canonical_url(getattr(other, "url", "")):
            return other
        if fingerprint and fingerprint == content_fingerprint(other):
            return other
        if same_title(title, title_key(getattr(other, "title", ""))):
            return other
        if same_article(signature, content_signature(other)):
            return other
    return None


def _is_low_value_url(canonical: str) -> bool:
    try:
        parsed = urlparse(canonical)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    segments = [segment.lower() for segment in parsed.path.split("/") if segment]
    if not segments:
        return False
    last = segments[-1]
    if last in _LOW_VALUE_PATHS:
        return True
    # Search result pages (google.com/search, bing.com/search) are pointers,
    # not sources; the results they point at are already in the list.
    if last == "search" and any(host.endswith(site) for site in (
        "google.com", "bing.com", "duckduckgo.com", "yandex.com",
    )):
        return True
    return False


def looks_blocked(text: str) -> bool:
    """True when *text* reads as a blocked, paywalled, or empty page."""
    lowered = (text or "").lower()
    if any(marker in lowered[:_BLOCKED_HEAD_CHARS] for marker in _BLOCKED_MARKERS):
        return True
    # Ambiguous wording only condemns a page that is short enough to be
    # nothing but boilerplate.
    return len(lowered) < 1200 and any(
        marker in lowered for marker in _AMBIGUOUS_BLOCKED_MARKERS
    )


def is_usable_result(result) -> bool:
    """True when a result can support a claim, not just occupy a result row."""
    canonical = canonical_url(getattr(result, "url", ""))
    if not canonical or _is_low_value_url(canonical):
        return False

    title = str(getattr(result, "title", None) or "").strip()
    text = _best_text(result)
    score = _as_float(getattr(result, "score", 0.0))

    if 0 < score < _MIN_RELEVANCE:
        return False

    if len(text) < _MIN_TEXT_CHARS:
        # A short snippet is acceptable only when the search engine is very
        # confident and there is at least a real title to identify it.
        if not (score >= 0.6 and len(title) >= 8):
            return False

    if looks_blocked(text):
        return False

    return True


def quality_score(result) -> float:
    """A 0-1 estimate of how much useful evidence a result carries.

    Blends the search engine's relevance score, the amount of text actually
    retrieved, the category prior, and whether the title identifies the page,
    then multiplies by a source-type prior. The multiplier is what stops a
    long vendor page or a YouTube clip from scoring like a paper: length and
    relevance alone had every marketing page at 0.9+. Intentionally simple
    and explainable; no model call is needed.
    """
    text = _best_text(result)
    relevance = _clamp(_as_float(getattr(result, "score", 0.0)))
    length = _clamp(math.log10(len(text) + 1) / 4.0)
    category = _CATEGORY_QUALITY.get(
        str(getattr(result, "category", None) or "general").lower(), 0.8
    )
    title = 1.0 if len(str(getattr(result, "title", None) or "").strip()) >= 12 else 0.55
    kind = source_type(getattr(result, "url", ""))

    blend = 0.45 * relevance + 0.30 * length + 0.15 * category + 0.10 * title
    score = blend * _SOURCE_TYPE_PRIOR.get(kind, 0.80)
    canonical = canonical_url(getattr(result, "url", ""))
    if canonical and _is_low_value_url(canonical):
        score *= 0.5
    if looks_blocked(text):
        score *= 0.5
    return round(_clamp(score), 3)


def select_for_analysis(
    results: Iterable,
    limit: int = 10,
    per_domain: int = 3,
) -> List:
    """Choose the evidence the analysis model should see.

    Ranks by a blend of the evidence-quality score and the search engine's
    relevance, keeping first-appearance order as the tiebreaker. At most
    ``per_domain`` results come from one host, so agreement between sources
    means agreement between independent publishers rather than one site
    repeated. If the cap leaves room (for example, every result is from the
    same domain), the overflow fills the list rather than silently returning
    fewer results than asked for.
    """
    items = [
        result for result in results
        if canonical_url(getattr(result, "url", ""))
    ]
    if limit <= 0 or not items:
        return []

    def rank(result) -> float:
        quality = _as_float(getattr(result, "quality", 0.0)) or quality_score(result)
        relevance = _clamp(_as_float(getattr(result, "score", 0.0)))
        return 0.7 * quality + 0.3 * relevance

    # sorted() is stable, so equal scores retain the engine's ranking.
    ranked = sorted(items, key=rank, reverse=True)
    selected: List = []
    overflow: List = []
    counts: Dict[str, int] = {}

    for result in ranked:
        domain = _domain(canonical_url(getattr(result, "url", "")))
        if counts.get(domain, 0) >= per_domain:
            overflow.append(result)
            continue
        selected.append(result)
        counts[domain] = counts.get(domain, 0) + 1
        if len(selected) >= limit:
            return selected

    for result in overflow:
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected


def merge_result(target, other) -> None:
    """Fold *other* into *target*, keeping the richest version of each field."""
    target_text = _best_text(target)
    other_text = _best_text(other)
    if len(other_text) > len(target_text):
        if getattr(other, "full_content", None):
            target.full_content = other.full_content
        if len(str(getattr(other, "content", None) or "")) > len(
            str(getattr(target, "content", None) or "")
        ):
            target.content = other.content
    if _as_float(getattr(other, "score", 0.0)) > _as_float(getattr(target, "score", 0.0)):
        target.score = other.score
    target_category = str(getattr(target, "category", None) or "general")
    other_category = str(getattr(other, "category", None) or "general")
    if target_category == "general" and other_category != "general":
        target.category = other_category
    if not getattr(target, "title", None) and getattr(other, "title", None):
        target.title = other.title
    merged_quality = max(quality_score(target), _as_float(getattr(target, "quality", 0.0)))
    if hasattr(target, "quality"):
        target.quality = merged_quality


def prepare_results(results: Iterable) -> Tuple[List, List[Tuple[object, str]]]:
    """Filter unusable results and merge duplicate URLs or syndicated text.

    Returns ``(kept, rejected)`` where *rejected* is ``(result, reason)`` for
    logging. Order of first appearance is preserved, so the search engine's
    ranking still decides which sources the model sees first.
    """
    kept: List = []
    rejected: List[Tuple[object, str]] = []
    by_url: Dict[str, object] = {}
    by_fingerprint: Dict[str, object] = {}
    signatures: List[Tuple[frozenset, object]] = []

    for result in results:
        canonical = canonical_url(getattr(result, "url", ""))
        if not canonical:
            rejected.append((result, "missing or invalid URL"))
            continue
        if not is_usable_result(result):
            rejected.append((result, "no usable content (thin, blocked, or a non-source page)"))
            continue

        existing = by_url.get(canonical)
        if existing is not None:
            merge_result(existing, result)
            rejected.append((result, "duplicate URL"))
            continue

        fingerprint = content_fingerprint(result)
        if fingerprint and fingerprint in by_fingerprint:
            merge_result(by_fingerprint[fingerprint], result)
            rejected.append((result, "duplicate article text"))
            continue

        signature = content_signature(result)
        original = next(
            (other for other_signature, other in signatures
             if same_article(signature, other_signature)),
            None,
        )
        if original is None:
            original = next(
                (other for other in kept
                 if same_title(title_key(result.title),
                               title_key(getattr(other, "title", "")))),
                None,
            )
        if original is not None:
            merge_result(original, result)
            rejected.append((result, "near-duplicate article text"))
            continue

        if hasattr(result, "quality"):
            result.quality = quality_score(result)
        kept.append(result)
        by_url[canonical] = result
        if fingerprint:
            by_fingerprint[fingerprint] = result
        if signature:
            signatures.append((signature, result))

    return kept, rejected


@dataclass(frozen=True)
class SourceEntry:
    """One unique source as it appears in the report bibliography."""

    number: int
    url: str
    title: str
    category: str
    quality: float
    score: float
    published_date: str = ""
    analyzed: bool = False
    source_type: str = "web"


def build_source_registry(results: Iterable) -> Tuple[Dict[str, int], List[SourceEntry]]:
    """Number every unique source once, in first-appearance order.

    Returns ``(canonical_url_to_number, entries)``. The same mapping is used
    for section prompts, citation validation, and the bibliography, which is
    what keeps ``[Source N]`` meaningful across the whole document.
    """
    numbers: Dict[str, int] = {}
    entries: List[SourceEntry] = []
    accepted: List = []

    for result in results:
        canonical = canonical_url(getattr(result, "url", ""))
        if not canonical or canonical in numbers:
            continue
        if not is_usable_result(result):
            continue
        # The same page can be reached twice, republished, or served both as
        # HTML and as PDF. One bibliography entry is enough.
        if duplicate_of(result, accepted) is not None:
            continue
        accepted.append(result)

        number = len(entries) + 1
        numbers[canonical] = number

        title = str(getattr(result, "title", None) or "").strip() or _domain(canonical)
        entries.append(SourceEntry(
            number=number,
            url=str(getattr(result, "url", None) or "").strip(),
            title=title,
            category=str(getattr(result, "category", None) or "general"),
            quality=round(_as_float(getattr(result, "quality", 0.0)) or quality_score(result), 3),
            score=round(_as_float(getattr(result, "score", 0.0)), 3),
            published_date=str(getattr(result, "published_date", None) or "").strip(),
            analyzed=bool(getattr(result, "analyzed", False)),
            source_type=source_type(str(getattr(result, "url", None) or "")),
        ))

    return numbers, entries


def number_for(url: object, numbers: Dict[str, int]) -> Optional[int]:
    """Look up a source's stable number by its (possibly unnormalised) URL."""
    return numbers.get(canonical_url(url))


# ``[Source 4]``, ``[Sources 2, 5]``, ``[Source 1, Source 3]``, ``(Source 2)``.
_CITATION_RE = re.compile(
    r"[\[(]\s*sources?\s+"
    r"((?:(?:source\s+)?\d+)(?:\s*(?:,|&|\+|and)\s*(?:source\s+)?\d+)*)"
    r"\s*[\])]",
    re.IGNORECASE,
)


def normalize_citations(
    text: str,
    valid_numbers: Optional[Iterable[int]] = None,
    allowed_numbers: Optional[Iterable[int]] = None,
) -> str:
    """Rewrite every citation marker to the one canonical form.

    ``valid_numbers`` is the report-wide registry; ``allowed_numbers`` is the
    subset actually supplied as evidence for this text (omit it to allow the
    whole registry). Numbers outside the allowed set are removed rather than
    guessed at, so the report never cites evidence the model was not given.
    Markers are emitted as ``[Source 3]``, one bracket per source.
    """
    if not text:
        return text

    valid: Optional[Set[int]] = (
        {int(number) for number in valid_numbers} if valid_numbers is not None else None
    )
    allowed: Optional[Set[int]] = (
        valid if allowed_numbers is None else {int(number) for number in allowed_numbers}
    )

    def replace(match: re.Match) -> str:
        numbers: List[int] = []
        for raw in re.findall(r"\d+", match.group(1)):
            number = int(raw)
            if number <= 0 or number in numbers:
                continue
            if valid is not None and number not in valid:
                continue
            if allowed is not None and number not in allowed:
                continue
            numbers.append(number)
        return "".join(f"[Source {number}]" for number in numbers)

    cleaned = _CITATION_RE.sub(replace, text)
    # "[Source 1], [Source 2]" is two markers with a comma in between; the
    # canonical form puts them next to each other.
    cleaned = re.sub(r"\]\s*,\s*(?=\[Source )", "]", cleaned)
    # Removing an invalid marker can leave a space before punctuation.
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def renumber_citations(text: str, entries: Iterable[SourceEntry]) -> Tuple[str, List[SourceEntry]]:
    """Compact a citation set into a reference list of only what was cited.

    Numbers are reassigned 1..k in order of first appearance and every marker
    in *text* is rewritten to match. Entries that the body never cites are
    dropped: a bibliography should list the works it refers to, not every page
    the search engine returned.

    Returns ``(rewritten_text, cited_entries)``.
    """
    by_number = {entry.number: entry for entry in entries}
    mapping: Dict[int, int] = {}
    ordered: List[SourceEntry] = []

    for match in _CITATION_RE.finditer(text or ""):
        for raw in re.findall(r"\d+", match.group(1)):
            number = int(raw)
            if number in mapping or number not in by_number:
                continue
            mapping[number] = len(ordered) + 1
            ordered.append(replace(by_number[number], number=mapping[number]))

    def replace_marker(match: re.Match) -> str:
        numbers: List[int] = []
        for raw in re.findall(r"\d+", match.group(1)):
            new = mapping.get(int(raw))
            if new and new not in numbers:
                numbers.append(new)
        return "".join(f"[Source {number}]" for number in numbers)

    return _CITATION_RE.sub(replace_marker, text or ""), ordered


def cited_numbers(text: str) -> Set[int]:
    """All source numbers cited in *text*."""
    numbers: Set[int] = set()
    for match in _CITATION_RE.finditer(text or ""):
        numbers.update(int(raw) for raw in re.findall(r"\d+", match.group(1)))
    return numbers


def citation_count(text: str) -> int:
    """How many inline citation markers *text* contains (not unique sources)."""
    return sum(1 for _ in _CITATION_RE.finditer(text or ""))
