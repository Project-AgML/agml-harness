"""
symptom_search — find reference pages from a description of what is seen
(symptoms, damage, appearance) instead of a class name.

Only the sources that are reached by site-restricted text search can take a
description: UC IPM and Bugwood. EPPO, GBIF and USDA are name-keyed
APIs and are not used here. Page titles usually name the disease/pest/species,
so the result is a candidate list to check with a name lookup (retrieve()). Pages about other crops are kept: a
similar problem on a related host is still worth comparing (the wide pilot's empty results all came from requiring the
crop name on the page: UC IPM/Bugwood barely cover tropical crops).

    from agml_agent.symptom_search import search_description, focus
    pages = search_description("puckered leaf, raised tissue between veins", crop="bean")
"""

from __future__ import annotations

import re

from agml_agent.common.scrape import search_and_extract

SEARCH_SITES = ["ipm.ucanr.edu", "bugwood.org"]  # fact sheets; apsnet.org hits are mostly paywalled journal pages
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


def search_description(description: str, crop: str = "", task: str = "", keep: int = 4) -> list[dict]:
    """Up to `keep` pages from UC IPM/Bugwood matching the description's keywords:
    {site, title, url, text, mentions_crop}. Pages mentioning the crop (or a synonym) come first; the others are kept
    after them."""
    words = list(dict.fromkeys(w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", description.lower())
                               if len(w) > 2 and w not in _STOP))[:QUERY_WORDS]
    query = " ".join([x for x in (crop, task) if x] + words)
    hits = search_and_extract(query, SEARCH_SITES, [], max_results=8, keep=8,
                              avoid_url_substrings=["imgrecruit", "listxtax"], skip_url_patterns=INDEX_PAGES + BLOCKED)
    names = crop_names(crop)
    pages = [{"site": next(s for s in SEARCH_SITES if s in h["url"]), "title": h["title"], "url": h["url"], "text": h["text"],
              "mentions_crop": any(re.search(rf"\b{re.escape(n)}\b", h["text"].lower()) for n in names)} for h in hits]
    return sorted(pages, key=lambda p: not p["mentions_crop"])[:keep]
