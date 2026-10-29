"""iNaturalist — public API v1, read without a key:
    GET https://api.inaturalist.org/v1/taxa?q=<name>
    GET https://api.inaturalist.org/v1/taxa/<id>

Taxonomy, common name and iNaturalist's Wikipedia summary for plants, fungi and insects. Text only: observation
photos are never passed on. Scientific names match best; common names often find nothing.
"""

from __future__ import annotations

import re
from datetime import date

from agml_agent.common.http import get
from agml_agent.common.schema import RetrievedChunk
from agml_agent.common.tagging import normalize_tags
from agml_agent.sources.base import ClassQuery

API = "https://api.inaturalist.org/v1/taxa"
LICENSE = "iNaturalist taxon data; summary from Wikipedia (CC BY-SA)"


def fetch(query: ClassQuery) -> list[RetrievedChunk]:
    terms = ([query.scientific_name] if query.scientific_name else []) + query.search_terms()
    for term in dict.fromkeys(terms):
        r = get(API, params={"q": term, "per_page": 1}, sleep=1.0)  # iNaturalist asks for <= ~1 request/s
        hits = r.json().get("results", []) if r is not None and r.ok else []
        # the top hit must be the term itself (or have it as a name), not a fuzzy neighbour
        if not hits or term.lower() not in " ".join([hits[0].get("matched_term") or "", hits[0]["name"],
                                                     hits[0].get("preferred_common_name") or ""]).lower():
            continue
        r = get(f"{API}/{hits[0]['id']}", sleep=1.0)
        t = (r.json().get("results") or [None])[0] if r is not None and r.ok else None
        if not t:
            continue
        lineage = " > ".join(a["name"] for a in t.get("ancestors", []) if a.get("rank") not in ("stateofmatter",))
        summary = re.sub(r"<[^>]+>", "", t.get("wikipedia_summary") or "").strip()
        text = "\n".join(x for x in [f"Scientific name: {t['name']} ({t.get('rank', '')})",
                                     f"Common name: {t['preferred_common_name']}" if t.get("preferred_common_name") else "",
                                     f"Taxonomy: {lineage}" if lineage else "", summary] if x)
        return [RetrievedChunk(
            class_name=query.class_name, tags=normalize_tags(["taxonomy", "identification"]), excerpt_text=text,
            raw_response=RetrievedChunk.raw_json({"id": t["id"], "name": t["name"], "term": term}),
            source_url=f"https://www.inaturalist.org/taxa/{t['id']}", license=LICENSE, accessed_date=str(date.today()),
            crop=query.crop, query_task=query.task)]
    return []
