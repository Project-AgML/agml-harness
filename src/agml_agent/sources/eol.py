"""Encyclopedia of Life — classic API, read without a key (heavy use asks for one):
    GET https://eol.org/api/search/1.0.json?q=<name>
    GET https://eol.org/api/pages/1.0/<id>.json?details=true&texts_per_page=5&images_per_page=0

English text descriptions (brief summary, comprehensive description) and English common names. Text only.
"""

from __future__ import annotations

import re
from datetime import date

from agml_agent.common.http import get
from agml_agent.common.schema import RetrievedChunk
from agml_agent.common.tagging import normalize_tags
from agml_agent.sources.base import ClassQuery

BASE = "https://eol.org/api"
LICENSE = "EOL page content, per-object licenses (mostly CC BY / CC BY-SA)"
MAX_CHARS = 3000


def _plain(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def fetch(query: ClassQuery) -> list[RetrievedChunk]:
    terms = ([query.scientific_name] if query.scientific_name else []) + query.search_terms()
    for term in dict.fromkeys(terms):
        r = get(f"{BASE}/search/1.0.json", params={"q": term})
        hits = r.json().get("results", []) if r is not None and r.ok else []
        hits = [h for h in hits if term.lower() in (h.get("title", "") + " " + h.get("content", "")).lower()]
        if not hits:
            continue
        pid = hits[0]["id"]
        r = get(f"{BASE}/pages/1.0/{pid}.json", params={"details": "true", "texts_per_page": 5, "images_per_page": 0,
                                                       "videos_per_page": 0, "common_names": "true"})
        tc = r.json().get("taxonConcept", {}) if r is not None and r.ok else {}
        texts = [_plain(d.get("description") or "") for d in tc.get("dataObjects", [])
                 if "Text" in (d.get("dataType") or "") and (d.get("language") or "en") == "en"]
        names = sorted({v["vernacularName"] for v in tc.get("vernacularNames", []) if v.get("language") == "en"})
        text = "\n".join(x for x in [f"Scientific name: {tc.get('scientificName') or hits[0].get('title')}",
                                     f"Common names: {', '.join(names[:8])}" if names else "", *texts] if x)[:MAX_CHARS]
        if not texts and not names:
            continue
        return [RetrievedChunk(
            class_name=query.class_name, tags=normalize_tags(["identification", "taxonomy"]), excerpt_text=text,
            raw_response=RetrievedChunk.raw_json({"page_id": pid, "term": term}), source_url=f"https://eol.org/pages/{pid}",
            license=LICENSE, accessed_date=str(date.today()), crop=query.crop, query_task=query.task)]
    return []
