"""Catalogue of Life — ChecklistBank API, free, no key:
    GET https://api.checklistbank.org/dataset/3LR/nameusage/search?q=<name>
    GET https://api.checklistbank.org/dataset/3LR/taxon/<id>/info

Accepted name, status, classification and synonyms of plants, fungi, insects and other organisms.
"""

from __future__ import annotations

from datetime import date

from agml_agent.common.http import get
from agml_agent.common.schema import RetrievedChunk
from agml_agent.common.tagging import normalize_tags
from agml_agent.sources.base import ClassQuery

BASE = "https://api.checklistbank.org/dataset/3LR"  # 3LR: the latest Catalogue of Life release
LICENSE = "Catalogue of Life, CC BY 4.0"


def fetch(query: ClassQuery) -> list[RetrievedChunk]:
    terms = ([query.scientific_name] if query.scientific_name else []) + query.search_terms()
    for term in dict.fromkeys(terms):
        r = get(f"{BASE}/nameusage/search", params={"q": term, "limit": 1})
        hits = r.json().get("result", []) if r is not None and r.ok else []
        if not hits or term.lower() not in (hits[0]["usage"].get("label") or "").lower():
            continue
        u = hits[0]["usage"]
        r = get(f"{BASE}/taxon/{u['id']}/info")
        info = r.json() if r is not None and r.ok else {}
        syn = info.get("synonyms") or {}
        synonyms = [s.get("label") or s.get("name", {}).get("scientificName", "") for k in ("homotypic", "heterotypic")
                    for s in syn.get(k, [])][:8]
        text = "\n".join(x for x in [
            f"Scientific name: {u.get('label')} ({u.get('status', '')})",
            "Classification: " + " > ".join(c["name"] for c in hits[0].get("classification", [])),
            f"Synonyms: {'; '.join(s for s in synonyms if s)}" if synonyms else ""] if x)
        return [RetrievedChunk(
            class_name=query.class_name, tags=normalize_tags(["taxonomy", "synonym"]), excerpt_text=text,
            raw_response=RetrievedChunk.raw_json({"id": u["id"], "term": term}),
            source_url=f"https://www.catalogueoflife.org/data/taxon/{u['id']}", license=LICENSE,
            accessed_date=str(date.today()), crop=query.crop, query_task=query.task)]
    return []
