"""Shared site-restricted search + extract, for the sources with no stable
search API (ucipm_aps, plantwise, bugwood): search restricted to a domain,
fetch each candidate page fresh, extract clean text, keep only pages that
actually mention the class (a search engine ranking a page well doesn't mean
it's about this specific class)."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter

import requests
import trafilatura
from ddgs import DDGS

from agml_agent.common.cache import cached
from agml_agent.common.http import HEADERS, TIMEOUT

log = logging.getLogger(__name__)

FETCH_SLEEP = 1.0
BACKENDS = ["auto", "yahoo", "google", "brave"]  # ddgs 9.x engines; "auto" picks among them itself
# The engines rate-limit by request rate: the cache fill (2 concurrent, ~0.25 searches/s) got 0.5% empty, the first
# all-sources pilot (4 concurrent, unpaced) got most searches empty. So: at most 2 searches at a time, and engine
# requests at least SEARCH_INTERVAL apart across all threads.
SEARCHES = threading.Semaphore(2)
SEARCH_INTERVAL = 1.0
_pace_lock, _last_request = threading.Lock(), [0.0]
STATS = Counter()  # live searches and how many found nothing; read by bench_prefill to tell if search works here
# False: a search missing from the cache finds nothing instead of asking the engines (pages still fetch live).
# For runs where web search is rate-limited; the API sources are unaffected.
LIVE_SEARCH = True


def fetch_and_extract(url: str) -> str | None:
    """requests (with our own User-Agent) + trafilatura.extract — trafilatura
    2.x's fetch_url() no longer takes a `headers` kwarg, so the HTTP fetch is
    done ourselves and only the HTML is handed to trafilatura."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as e:
        log.debug("fetch failed for %s: %s", url, e)
        return None
    if resp.status_code != 200:
        return None
    return trafilatura.extract(resp.text)


def find_match(text: str, terms: list[str]) -> str | None:
    text_lower = text.lower()
    for term in terms:
        if term.lower() in text_lower:
            return term
    return None


def excerpt_around_match(text: str, term: str, max_len: int = 1200) -> str:
    idx = text.lower().find(term.lower())
    if idx == -1:
        return text[:max_len]
    start = max(0, idx - max_len // 4)
    return text[start:start + max_len]


def search_and_extract(
    query: str,
    sites: list[str],
    match_terms: list[str],
    max_results: int = 6,
    keep: int = 2,
    avoid_url_substrings: list[str] | None = None,
    skip_url_patterns: list[str] | None = None,
) -> list[dict]:
    """Returns up to `keep` dicts: {title, url, text, matched_term}. Empty
    `match_terms` keeps every extracted page (matched_term "") — for free-text
    searches with no class name to require.

    avoid_url_substrings deprioritizes matching hits (e.g. "/pdf/" links that
    trafilatura can't extract text from) rather than dropping them outright —
    they're tried last, so a source whose only hits are PDFs still gets
    something instead of nothing. skip_url_patterns (regexes) drops hits
    outright, before fetching, e.g. index pages that only list other pages."""
    # Search engines differ in which site-restricted queries they answer, and long multi-site queries often get
    # nothing anywhere: try the combined query, then one site at a time, each across several engines.
    queries = [f"({' OR '.join(f'site:{s}' for s in sites)}) {query}"] if len(sites) > 1 else []
    queries += [f"site:{s} {query}" for s in sites]
    def usable(hits):
        return [h for h in hits if (u := h.get("href")) and any(s in u for s in sites)
                and not any(re.search(p, u) for p in skip_url_patterns or [])]

    hits = usable(cached(["search", queries, max_results], lambda: _search(queries, max_results)))
    # The first query to answer can return only skipped pages (e.g. blocked wiki pages): then ask each site alone.
    for q in queries[1:] if not hits and len(queries) > 1 else []:
        if hits := usable(cached(["search", [q], max_results], lambda: _search([q], max_results))):
            break
    if not hits:
        log.warning("no search results for %r on any site/engine — DDGS backends can be flaky/rate-limited at scale, "
                    "retry this class individually if it matters", query)
        return []

    avoid = avoid_url_substrings or []
    if avoid:
        hits.sort(key=lambda h: any(a in (h.get("href") or "") for a in avoid))

    found: list[dict] = []
    for hit in hits:
        if len(found) >= keep:
            break
        url = hit["href"]
        try:
            text = cached(["page", url], lambda: _fetch(url))
            if not text:
                continue
            matched = find_match(text, match_terms) if match_terms else ""
            if matched is None:
                continue
            found.append({
                "title": hit.get("title") or url,
                "url": url,
                "text": text,
                "matched_term": matched,
            })
        except Exception as e:
            log.debug("fetch/extract failed for %s: %s", url, e)
    return found


def _pace() -> None:
    with _pace_lock:
        wait = _last_request[0] + SEARCH_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def _search(queries: list[str], max_results: int) -> list[dict]:
    if not LIVE_SEARCH:
        STATS["not_cached"] += 1
        return []
    with SEARCHES:
        return _search_live(queries, max_results)


def _search_live(queries: list[str], max_results: int) -> list[dict]:
    STATS["searches"] += 1
    for q in queries:
        for backend in BACKENDS:
            _pace()
            try:
                hits = list(DDGS().text(q, max_results=max_results, backend=backend))
            except Exception as e:
                log.debug("search failed for %r on %s: %s", q, backend, e)
                hits = []
            if hits:
                return hits
            time.sleep(1)
    STATS["empty"] += 1
    return []


def _fetch(url: str) -> str | None:
    text = fetch_and_extract(url)
    time.sleep(FETCH_SLEEP)  # only after live fetches: cached pages need no politeness delay
    return text
