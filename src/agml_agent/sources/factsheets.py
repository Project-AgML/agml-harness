"""Open extension and fact-sheet sites without a search API (TNAU Agritech, APS Plant Disease Lessons, PNW and Texas
handbooks, NC State Extension, ProMusa, PlantVillage, Feedipedia): one fetcher, configured per site in
config/sources.yaml (`args: {sites: [...]}`), so each site is its own source in results. Same site-restricted search +
text extraction as ucipm_aps/bugwood (common/scrape.py); a page is kept only if it mentions the class.

Sources with `description: true` in sources.yaml are also searched by description (symptom_search.py).
"""

from __future__ import annotations

from datetime import date

from agml_agent.common.schema import RetrievedChunk
from agml_agent.common.scrape import excerpt_around_match, search_and_extract
from agml_agent.common.tagging import normalize_tags
from agml_agent.sources.base import ClassQuery

LICENSE = "public extension/fact-sheet page; attribute source_url"


def fetch(query: ClassQuery, sites: list[str], skip: list[str] | None = None) -> list[RetrievedChunk]:
    hits = search_and_extract(f"{query.class_name} {query.crop}".strip(), sites, query.search_terms(), max_results=6,
                              keep=2, skip_url_patterns=skip)
    return [RetrievedChunk(
        class_name=query.class_name,
        tags=normalize_tags(["identification", "symptom"]),
        excerpt_text=excerpt_around_match(h["text"], h["matched_term"], 2000),
        raw_response=RetrievedChunk.raw_json({"title": h["title"], "url": h["url"], "matched_term": h["matched_term"]}),
        source_url=h["url"],
        license=LICENSE,
        accessed_date=str(date.today()),
        crop=query.crop,
        query_task=query.task,
    ) for h in hits]
