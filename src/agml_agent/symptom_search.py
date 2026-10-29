"""
symptom_search — find reference pages from a description of what is seen
(symptoms, damage, appearance) instead of a class name.

Sources reached by site-restricted text search can take a description (the ones with `description_sites` in
config/sources.yaml: UC IPM, Bugwood, TNAU, APS lessons, PNW and Texas handbooks, NC State, ProMusa, PlantVillage),
plus Wikipedia through its own search API. EPPO, GBIF, USDA, iNaturalist, Catalogue of Life and EOL are name-keyed
and are not used here. Page titles usually name the disease/pest/species, so the result is a candidate list to check
with a name lookup (retrieve()). Pages about other crops are kept: a
similar problem on a related host is still worth comparing (the wide pilot's empty results all came from requiring the
crop name on the page: UC IPM/Bugwood barely cover tropical crops).

    from agml_agent.symptom_search import search_description, focus
    pages = search_description("puckered leaf, raised tissue between veins", crop="bean")
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from agml_agent.common.scrape import search_and_extract

# Navigation pages that matched symptom searches and took result slots without describing any problem: crop index
# pages (ipm.ucanr.edu/agriculture/<crop>/, pilot 2; pest pages are one level deeper), site sections and menus, the
# diagnostic tool, and Bugwood image collections (photo captions only).
INDEX_PAGES = [r"ipm\.ucanr\.edu/agriculture/[^/]+/?$", r"ipm\.ucanr\.edu/[^/]*/?$", r"ipm\.ucanr\.edu/PMG/menu\.",
               r"/plant-problem-diagnostic-tool", r"images\.bugwood\.org/(gallery|series)\.cfm"]
BLOCKED = [r"wiki\.bugwood\.org"]  # answers 403 (bot protection) to every request
# Other names a page may use for the crop: used to put pages about the crop first, never to drop pages.
CROP_SYNONYMS = [{"groundnut", "peanut"}, {"brinjal", "eggplant", "aubergine"}, {"maize", "corn"},
                 {"black gram", "urd bean", "urad bean", "vigna mungo"}, {"green gram", "mung bean", "mungbean"},
                 {"chili", "chilli", "chile", "pepper"}, {"okra", "lady finger", "bhindi"}, {"luffa", "sponge gourd"},
                 {"bottle gourd", "calabash"}, {"soybean", "soya", "soybeans"}, {"malabar spinach", "basella"}]
QUERY_WORDS = 8  # search engines answer short site-restricted queries; long descriptions get nothing
PER_SOURCE = 2  # pages kept per source before merging

_STOP = set("""a an and are as at be but by for from has have in into is it its leaf leaves of on or plant plants
shows showing the their there these this those to with without very some small large""".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def focus(text: str, description: str, max_len: int) -> str:
    """The sentences of `text` sharing the most words with `description`, in their original order, up to
    `max_len` chars; `text` itself (up to `max_len`) when it already fits or nothing overlaps. Selects the relevant
    part; it does not score how well the class fits."""
    text = re.sub(r"\s*\n\s*", "\n", re.sub(r"[ \t]+", " ", text)).strip()  # scraped pages carry layout whitespace
    want = _words(description)
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = [(len(want & _words(s)), i) for i, s in enumerate(sents)]
    if len(text) <= max_len or not want or not any(n for n, _ in scored):
        return text[:max_len]
    keep, size = set(), 0
    for n, i in sorted(scored, key=lambda x: (-x[0], x[1])):
        if not n:
            break
        if size + len(sents[i]) > max_len:
            continue
        keep.add(i)
        size += len(sents[i]) + 1
    return " ".join(sents[i] for i in sorted(keep)) or text[:max_len]


def crop_names(crop: str) -> list[str]:
    crop = crop.strip().lower()
    return [crop] + sorted(next((g - {crop} for g in CROP_SYNONYMS if crop in g), set())) if crop else []


def description_sources() -> dict[str, list[str]]:
    """{source name: sites} for the active sources searched by description (config/sources.yaml description_sites)."""
    from agml_agent.retrieve import _load_sources

    return {n: c["description_sites"] for n, c in _load_sources().items()
            if c.get("status") == "active" and c.get("description_sites")}


def _site_pages(sites: list[str], query: str) -> list[dict]:
    return search_and_extract(query, sites, [], max_results=6, keep=PER_SOURCE, avoid_url_substrings=["imgrecruit", "listxtax"],
                              skip_url_patterns=INDEX_PAGES + BLOCKED)


def search_description(description: str, crop: str = "", task: str = "", keep: int = 5) -> list[dict]:
    """Up to `keep` pages matching the description's keywords, from every description source and Wikipedia:
    {source, title, url, text, mentions_crop}. Pages mentioning the crop (or a synonym) come first; then sources take
    turns, so one site can't fill the result."""
    from agml_agent.sources import wikipedia

    words = list(dict.fromkeys(w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", description.lower())
                               if len(w) > 2 and w not in _STOP))[:QUERY_WORDS]
    query = " ".join([x for x in (crop, task) if x] + words)
    jobs = {n: (lambda s=s: _site_pages(s, query)) for n, s in description_sources().items()}
    jobs["wikipedia"] = lambda: wikipedia.search(query, PER_SOURCE)
    with ThreadPoolExecutor(len(jobs)) as ex:
        found = dict(zip(jobs, ex.map(lambda f: f(), jobs.values())))
    names = crop_names(crop)
    per_source = [[{"source": n, "title": h["title"], "url": h["url"], "text": h["text"],
                    "mentions_crop": any(re.search(rf"\b{re.escape(c)}\b", h["text"].lower()) for c in names)}
                   for h in hits] for n, hits in found.items()]
    turns = [p for rank in range(PER_SOURCE) for pages in per_source for p in pages[rank:rank + 1]]  # round robin
    return sorted(turns, key=lambda p: not p["mentions_crop"])[:keep]
