"""Wikipedia — MediaWiki action API, free, no key:
    GET https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=<text>
    GET https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=<title>

Broad coverage, including tropical crops, pests and diseases the extension sites lack. Full-text search also takes a
description (search()), so it doubles as a symptom search. Articles about datasets or machine learning are dropped
(leakage: benchmark datasets have Wikipedia pages), and description hits must read like a plant/pest article.
"""

from __future__ import annotations

import re
from datetime import date

from agml_agent.common.cache import cached
from agml_agent.common.http import get
from agml_agent.common.schema import RetrievedChunk
from agml_agent.common.tagging import normalize_tags
from agml_agent.sources.base import ClassQuery

API = "https://en.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0 (Wikipedia)"
LEAK = re.compile(r"\b(data ?sets?|machine learning|deep learning|neural networks?|computer vision)\b", re.I)
PLANT_WORDS = re.compile(r"\b(plants?|crops?|leaf|leaves|fung\w*|insects?|pests?|pathogens?|disease|weeds?|cultivars?|fruit)\b", re.I)
MAX_CHARS = 4000


def _search(text: str, n: int) -> list[str]:
    r = get(API, params={"action": "query", "list": "search", "srsearch": text, "srlimit": n, "format": "json"}, sleep=0.2)
    return [h["title"] for h in (r.json().get("query", {}).get("search", []) if r is not None and r.ok else [])]


def _extract(title: str) -> str:
    r = get(API, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title,
                         "format": "json"}, sleep=0.2)
    pages = r.json().get("query", {}).get("pages", {}) if r is not None and r.ok else {}
    return next(iter(pages.values()), {}).get("extract") or ""


def page(title: str) -> dict | None:
    """{title, url, text} or None when the article is missing or about datasets/ML."""
    text = cached(["wikipedia", title], lambda: _extract(title))
    if not text or LEAK.search(title) or LEAK.search(text[:MAX_CHARS]):
        return None
    return {"title": title, "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", "text": text[:MAX_CHARS]}


def search(text: str, n: int = 4) -> list[dict]:
    """Articles matching free text (a description), keeping only plant/pest/disease-like ones."""
    titles = cached(["wikipedia-search", text, n], lambda: _search(text, n))
    return [p for t in titles if (p := page(t)) and PLANT_WORDS.search(p["text"][:1500])]


def fetch(query: ClassQuery) -> list[RetrievedChunk]:
    for term in query.search_terms():
        for title in cached(["wikipedia-search", term, 3], lambda: _search(term, 3)):
            p = page(title)
            # the article must be about the term: its title or opening names it
            if p and (term.lower() in title.lower() or term.lower() in p["text"][:600].lower()):
                return [RetrievedChunk(
                    class_name=query.class_name, tags=normalize_tags(["identification", "symptom", "host"]),
                    excerpt_text=p["text"], raw_response=RetrievedChunk.raw_json({"title": title, "term": term}),
                    source_url=p["url"], license=LICENSE, accessed_date=str(date.today()), crop=query.crop,
                    query_task=query.task)]
    return []
