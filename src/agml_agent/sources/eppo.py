"""EPPO Global Database REST v2 — free account + API key, sent as an
X-Api-Key header:
    GET https://api.eppo.int/gd/v2/tools/name2codes?name=<term>
    GET https://api.eppo.int/gd/v2/taxons/taxon/<eppocode>/{overview,names,kingdom,hosts,photos}

Replaces the retired v1 API (data.eppo.int/api/rest/1.0, now 403). Ported
from AGgentic-context's agalign/eppo.py. Returns one names/taxonomy/hosts
chunk and, when EPPO has symptom/diagnostic photos, one chunk of their
captions (short visual descriptions). Only this source needs a key — see
README "API keys".
"""

from __future__ import annotations

import os
from datetime import date

from agml_agent.common.http import get
from agml_agent.common.schema import RetrievedChunk
from agml_agent.common.tagging import normalize_tags
from agml_agent.sources.base import ClassQuery

BASE = "https://api.eppo.int/gd/v2"
LICENSE = "EPPO Global Database — attribute source_url"


def _get(path: str, key: str, **params):
    r = get(BASE + path, params=params or None, headers={"X-Api-Key": key})
    return r.json() if r is not None and r.status_code == 200 else None


def _code(terms: list[str], key: str) -> str | None:
    for t in terms:
        hits = _get("/tools/name2codes", key, name=t)
        if hits:
            return next((h["eppocode"] for h in hits if h.get("preferred")), hits[0]["eppocode"])
    return None


def fetch(query: ClassQuery) -> list[RetrievedChunk]:
    key = os.environ.get("EPPO_API_KEY")
    if not key:
        raise RuntimeError(
            "EPPO_API_KEY not set. Register free at https://data.eppo.int/ui/#/user/register, "
            "log in, and your account page has your API token — put it in .env. See README 'API keys'."
        )

    # Scientific name first: EPPO is keyed on taxa.
    terms = query.search_terms()
    if query.scientific_name:
        terms = [query.scientific_name] + [t for t in terms if t != query.scientific_name]
    code = _code(terms, key)
    if not code:
        return []

    overview = _get(f"/taxons/taxon/{code}/overview", key) or {}
    names = _get(f"/taxons/taxon/{code}/names", key) or []
    kingdom = ((_get(f"/taxons/taxon/{code}/kingdom", key) or {}).get("eppocode") or {}).get("prefname")
    hosts = _get(f"/taxons/taxon/{code}/hosts", key) or []
    photos = _get(f"/taxons/taxon/{code}/photos", key) or []

    def chunk(text: str, tags: list[str], raw) -> RetrievedChunk:
        return RetrievedChunk(
            class_name=query.class_name,
            tags=normalize_tags(tags),
            excerpt_text=text,
            raw_response=RetrievedChunk.raw_json(raw),
            source_url=f"https://gd.eppo.int/taxon/{code}",
            license=LICENSE,
            accessed_date=str(date.today()),
            crop=query.crop,
            query_task=query.task,
            extra={"eppocode": code},
        )

    common = sorted({n["fullname"] for n in names if n.get("lang_iso") == "en"})
    major = [h["prefname"] for h in hosts if h.get("class_label") == "Major host"]
    parts = [f"Preferred name: {overview.get('prefname', code)} (EPPO code {code})",
             f"Kingdom: {kingdom}" if kingdom else "",
             f"English common names: {'; '.join(common)}" if common else "",
             f"Major hosts: {', '.join(major[:20])}" if major else ""]
    out = [chunk(". ".join(p for p in parts if p), ["taxonomy"] + (["host"] if major else []),
                 {"overview": overview, "names": names, "kingdom": kingdom, "hosts": hosts})]

    symptom_photos = [p for p in photos if {"Symptoms", "Diagnostic"} & set(p.get("tags") or [])]
    captions = [c for c in dict.fromkeys(" ".join((p.get("descinfo") or "").split()) for p in symptom_photos) if c]
    if captions:
        out.append(chunk("Symptoms shown in EPPO photos: " + " | ".join(captions), ["symptom"], symptom_photos))
    return out
