"""
bench.py — benchmark the harness on AGgentic-benchmarks (agbench) WITHOUT
changing agbench: items, prompts, sampling/effort settings, grading and the
judge all come from the pinned agbench package, so a harness run is scored
exactly like the plain-model baselines. The only difference is that the
model may call tools.

    agml-agent-bench --model google/gemma-4-12B-it --data-dir /path/with/base-data --judge
    agml-agent-bench --model google/gemma-4-12B-it --endpoint http://127.0.0.1:8000 ...   # already-running server

The run dir is a standard agbench run dir (compare scores with
`agbench eval view --runs <baseline> <this run>`), plus `traces.jsonl`: one
line per item with every model turn, tool call and tool result (view with
`agml-agent-view`).

Leakage policy: only tools that query external databases are allowed here
(BENCH_TOOLS). search_agml is never offered — it returns AgML class lists
and benchmark results, and agbench items come from AgML. retrieve's result
is reduced to excerpt text + source URL: no images or image URLs reach the
model. Its description search is restricted to UC IPM/APSnet and Bugwood
fact-sheet sites (symptom_search.py), never the open web, where dataset pages
could be found.

"Exploit" policy (the first harness baseline): one pass, tools available,
at most `max_tool_calls` calls, no planning/search/self-consistency. The
prompt is agbench's, unchanged; the tools are described only through their
schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agbench.eval import Benchmark, EvalItem, Prediction, VLMMethod
from agbench.eval.client import encode_image

from agml_agent.common import cache, scrape
from agml_agent.common.cache import RetrievalCache

MAX_CANDIDATES = 4
DATABASES = ("EPPO, UC IPM/APS, Bugwood, TNAU Agritech, APS lessons, PNW and Texas handbooks, NC State Extension, "
             "ProMusa, PlantVillage, Feedipedia, Wikipedia, GBIF, iNaturalist, Catalogue of Life, EOL, USDA")
# Leakage guard on every source's text: benchmark datasets are described on the web (dataset pages, papers, Wikipedia
# articles); text about datasets or machine learning never reaches the model.
LEAK = re.compile(r"\b(data ?sets?|machine learning|deep learning|neural networks?|computer vision)\b", re.I)

RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Look up crop diseases, pests, weed/plant species and produce-quality classes in external agricultural "
            f"databases ({DATABASES}). Two ways to ask, usable together:\n"
            "- description: what you see in the image (symptoms, damage, colours, shapes). Searches extension fact "
            "sheets (UC IPM, Bugwood, TNAU Agritech, APS, PNW and Texas handbooks, NC State, ProMusa, PlantVillage) "
            "and Wikipedia, and returns pages with matching features (diseases, pests, species, quality problems), on "
            "this crop first, then on other hosts; their titles usually name what they describe.\n"
            "- candidates: up to 4 names to check. Returns reference text for each (taxonomy, hosts, symptoms, "
            "identification notes) to compare against the image. With a description too, each excerpt is narrowed to "
            "the parts matching it.\n"
            "Typical use: describe what you see, then check the most likely causes from the results with candidates. "
            "Names are looked up by name only: put symptoms in description, not in candidate names. Every excerpt "
            "comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What you see, e.g. 'bean leaf with small "
                                "reddish-brown pustules surrounded by yellow halos'."},
                "candidates": {
                    "type": "array", "maxItems": MAX_CANDIDATES,
                    "description": "Diseases, pests, species or classes to check, e.g. "
                                   "[{'name': 'bean rust', 'scientific_name': 'Uromyces appendiculatus'}].",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Common name, e.g. 'coffee leaf rust'."},
                            "scientific_name": {"type": "string", "description": "Scientific name of the pathogen, "
                                                "pest or species if known; tried first, usually the most reliable."},
                            "aliases": {"type": "array", "items": {"type": "string"},
                                        "description": "Other names, tried if the main names find nothing."},
                        },
                        "required": ["name"],
                    },
                },
                "crop": {"type": "string", "description": "Host crop, e.g. 'coffee'. Optional: added to the search, and "
                         "pages mentioning it come first; pages about other hosts are still returned. To look up the "
                         "crop itself, give it as a candidate with task 'species'."},
                "task": {"type": "string", "enum": ["disease", "pest", "species", "quality"],
                         "description": "Kind of class. Every database is searched; the ones covering this "
                                        "kind come first."},
            },
        },
    },
}

# --retrieve-mode candidates: the model proposes names and the tool checks them; no search by description. To compare
# with the default (pilots 2-3: description search anchored the model on whatever pages it returned).
CANDIDATES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Look up crop diseases, pests, weed/plant species and produce-quality classes by name in external "
            f"agricultural databases ({DATABASES}). Give up to 4 candidate names, e.g. the "
            "causes you are considering; returns reference text for each (taxonomy, hosts, symptoms, identification "
            "notes) to compare against the image. Lookup is by name only: symptom descriptions find nothing. Every "
            "excerpt comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {k: v for k, v in RETRIEVE_SCHEMA["function"]["parameters"]["properties"].items() if k != "description"},
            "required": ["candidates"],
        },
    },
}
# --retrieve-mode crosscheck: the grower's flow. Describe the photo, make an initial diagnosis (up to 4), cross-check it
# against the databases, decide. Both are required; the description narrows each excerpt to what was seen. API pilot
# (2026-10-30): name checks with a description fixed 9 / broke 2 of 60 items; description search alone did not help.
CROSSCHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Cross-check your initial diagnosis against external agricultural databases "
            f"({DATABASES}). Give a description of what you see and up to 4 candidates: the diseases, pests, species "
            "or quality classes you are considering (the answer options, if you are given options). Returns reference "
            "text for each candidate (appearance, symptoms, hosts, taxonomy), narrowed to the parts matching your "
            "description, to compare against the image before you decide. Every excerpt comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **RETRIEVE_SCHEMA["function"]["parameters"]["properties"],
                "description": {"type": "string", "description": "What you see in 2-4 sentences: plant part, colours, "
                                "shapes, texture, pattern and spread of any symptoms or damage, e.g. 'Bean leaf with many "
                                "small reddish-brown powdery pustules on the underside, each surrounded by a yellow halo; "
                                "older leaves are most affected.'"},
            },
            "required": ["description", "candidates"],
        },
    },
}
# --retrieve-mode crosscheck-healthy: crosscheck, plus the reminder that nothing may be wrong. Repeats (2026-10-31): on
# open-ended items with a healthy plant crosscheck fell from 69% to 36%; candidates almost never included "healthy", so
# the model compared problems only and picked one. Tool description and result only: agbench's prompt is unchanged.
NO_PROBLEM_NOTE = ("The plant may be healthy with no problem, or none of the candidates may fit: reference text is returned "
                   "whether or not the plant has the candidate.")
CROSSCHECK_HEALTHY_SCHEMA = {
    "type": "function",
    "function": {
        **CROSSCHECK_SCHEMA["function"],
        "description": f"{CROSSCHECK_SCHEMA['function']['description']} {NO_PROBLEM_NOTE}",
    },
}
RESULT_NOTE = (f"\n\nThis text describes each candidate, not this plant. {NO_PROBLEM_NOTE} Compare it with the image, and "
               "keep only what the image shows.")
# --retrieve-mode crosscheck-placebo: the same tool with no reference text and no lookups, to separate what asking for a
# diagnosis does from what the databases add.
PLACEBO_RESULT = f"No reference text was found for these candidates. {NO_PROBLEM_NOTE} Decide from the image."
# --retrieve-mode crosscheck-assess: first say whether anything is wrong. Traces of crosscheck-healthy (2026-11-01): on
# healthy plants the description already reported symptoms before any result, and candidates were all problems.
NO_PROBLEM_SEEN = "no problem seen"
ASSESSMENTS = [NO_PROBLEM_SEEN, "problem seen", "not a health question"]
NOTHING_TO_CHECK = "No candidates to check. Decide from the image."
CROSSCHECK_ASSESS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Cross-check your view of the image against external agricultural databases "
            f"({DATABASES}). First decide whether anything is actually wrong: veins, surface texture and a few small "
            "marks can be normal. Then describe what you see and, unless you see no problem, give up to 4 candidates: "
            "the diseases, pests, species or quality classes you are considering (the answer options, if you are given "
            "options). Returns reference text for each candidate (appearance, symptoms, hosts, taxonomy), narrowed to "
            f"the parts matching your description, to compare against the image before you decide. {NO_PROBLEM_NOTE} "
            "Every excerpt comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "assessment": {"type": "string", "enum": ASSESSMENTS,
                               "description": "Whether the plant looks healthy or something is wrong."},
                **CROSSCHECK_SCHEMA["function"]["parameters"]["properties"],
                "description": {"type": "string", "description": "What you see in 2-4 sentences: plant part, colours, "
                                "shapes, texture and any marks or damage, e.g. 'Bean leaf, evenly green with prominent "
                                "veins and a slightly uneven surface; no spots or lesions.'"},
            },
            "required": ["assessment", "description"],
        },
    },
}
RETRIEVE_MODES = {"description+candidates": RETRIEVE_SCHEMA, "candidates": CANDIDATES_SCHEMA, "crosscheck": CROSSCHECK_SCHEMA,
                  "crosscheck-healthy": CROSSCHECK_HEALTHY_SCHEMA, "crosscheck-placebo": CROSSCHECK_HEALTHY_SCHEMA,
                  "crosscheck-assess": CROSSCHECK_ASSESS_SCHEMA, "crosscheck-look-first": CROSSCHECK_HEALTHY_SCHEMA,
                  "crosscheck-look-first-placebo": CROSSCHECK_HEALTHY_SCHEMA}
PLACEBO_MODES = {"crosscheck-placebo", "crosscheck-look-first-placebo"}
# --retrieve-mode crosscheck-look-first: answer with no tool offered, then see that answer and may cross-check it. Placebo
# (2026-11-01): healthy plants lost as much with no reference text, and detection fell with no calls, so the damage starts
# when the tool is offered. Detection and segmentation stop at the first answer. If the check ends without a complete
# answer (token budget, server error), the first answer stands.
LOOK_ONLY_TASKS = {"detection", "segmentation"}
CHECK_PROMPT = "That is your first answer. You may cross-check it, then give your final answer in the format asked."
# --code-tool: every task gets run_python next to retrieve, and the model chooses: no tool, either, or both in any
# order, within the same call budget. Detection and segmentation then get the look-first check too (without the code
# tool they still stop at the first answer). Detection baseline (fresh full baseline, 2026-11-09): the model misses
# objects rather than drawing bad boxes (matched boxes overlap 0.76, but it answers [] on half the images that
# contain objects and finds 1 in 20 objects). The tool descriptions say only what is available; how (or whether) to
# use them is the model's choice. Runs in sandbox.py: no network, no files but the item's images, no libraries
# beyond plain image processing (so no pretrained models).
CODE_TOOL = False
MAX_PROMPT_IMAGES = 8  # vLLM allows 8 images per prompt (agbench models.yaml max_images), the item's own included
CODE_SLOTS = 8         # sandboxes running at once (the farm job has 8 CPUs)
RUN_PYTHON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Look at the image more closely with code: crop and enlarge a part of it, measure colours or "
                        "sizes, or count regions, then see the result before you decide. `img` is a PIL image of the "
                        "picture you were given (`W` x `H` pixels; `imgs` lists all images when there are several; the "
                        "files are input_file_0.png, input_file_1.png, ...), with `np` (numpy) and `Image` (PIL) ready; "
                        "numpy, PIL, cv2, skimage and scipy can be imported. Text you print() and images you pass to "
                        "show(image) or img.show() (at most 2 per call) are returned to you. Coordinates in code are "
                        "pixels; give your answer in the scale the question asks for. Example: "
                        "show(img.crop((0, 0, W // 2, H // 2)).resize((W, H))) shows the top-left quarter enlarged. No "
                        "network, other files or other libraries."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run."}},
                       "required": ["code"]},
    },
}
# --keep-healthy: a first answer that finds no problem gets a second look without the tool. Knowledge-base runs
# (2026-11-04): vivid symptom text turned 12 of 103 healthy first answers into a problem (4 with no cleanup, 8 with
# --clean-text 1). The model can still change its answer when the image clearly shows a problem.
KEEP_HEALTHY = False
HEALTHY_ANSWER = re.compile(r"(?<!un)(?<!not )\bhealthy\b|\bno (visible |obvious |apparent |significant |clear |notable )?"
                            r"(signs?|disease|pests?|damage|stress|symptoms?|problems?|issues?)\b|\bnormal\b|\bfresh\b", re.I)
HEALTHY_CHECK_PROMPT = ("That is your first answer: you found no problem. Change it only if the image shows clear visible "
                        "evidence of a specific problem; otherwise keep it. Give your final answer in the format asked.")

PER_CHUNK_CHARS = 1200
# After a description search. Pilot 2: the model answered with the first page's cause and never checked it, and a
# search always returns pages, even for a healthy plant.
NEXT_STEP = ("\n\nThis is a search by symptoms, not a diagnosis: it returns pages even when nothing is wrong. Before "
             "answering, check the most likely causes (from these pages or your own) with candidates and compare their "
             "symptoms with the image.")
RESULT_CHARS = 6000
# Concurrent live lookups across all items. Was 4 while web search ran live (rate-limited); with --no-live-search only
# the APIs are asked, and look-first runs (2026-11-02) spent most of each call's 2-6 min queueing for these slots.
LIVE_LOOKUPS = 12
# A lookup returns the sources that answered within this many seconds. Timing 24 real candidates (2026-11-03): one
# lookup took ~15 s, set by Catalogue of Life (median 14 s, max 68 s) and GBIF (median 9 s), both taxonomy only.
SOURCE_DEADLINE = 15
_live = threading.Semaphore(LIVE_LOOKUPS)
# Sites whose robots.txt blocks AI crawlers (checked 2026-11-03): their pages are never shown, even from the cache.
BLOCKED = re.compile(r"plantvillage\.psu\.edu|pnwhandbooks\.org|apsnet\.org|promusa\.org", re.I)
# --kb: the local knowledge base (kb.py), searched by candidate name next to the API sources; None when not used.
KB_INDEX = None
# --kb-only: the local knowledge base and names table are the only sources; no API, web search, Wikipedia or cache,
# and any connection off this machine raises (so a run cannot quietly use the internet)
KB_ONLY = False
NAMES = None
NAME_LABEL = {"eppo": "EPPO", "usda_plants": "USDA PLANTS", "gbif": "GBIF"}
NAME_URL = {"eppo": "https://gd.eppo.int/taxon/{}", "usda_plants": "https://plants.usda.gov/plant-profile/{}",
            "gbif": "https://www.gbif.org/species/{}"}
KB_TAGS = {"symptom": ["symptom", "identification"], "life_cycle": ["life_cycle"], "management": ["management"],
           "other": ["identification"]}
# --check-options: on multiple choice, every answer option is looked up, then the model's own candidates while there is
# room. Look-first runs (2026-11-02): the model checked every option in 21% of calls and the correct one in 58%.
CHECK_OPTIONS = False
# --tidy-names: candidate names cleaned before lookup (shown to the model as given). Full run (2026-11-06): 14% of
# 97,059 lookups found nothing, many for names in dataset-label form: the crop in front ("Tomato Tomato mosaic virus"),
# a plant part ("Leaf spot on Leaves"), a code or second name in brackets ("Echinochloa crus-galli (ECHCG)"),
# underscores and hyphens ("kalanchoe-pinnata"). The name as given stays as the last alias.
TIDY_NAMES = False
# --crop-from-prompt: a call without `crop` gets the item's crop when the prompt names it (21% of calls had none);
# nothing the model was not already shown.
CROP_FROM_PROMPT = False
MAX_WITH_OPTIONS = 6
NOT_OPTIONS = {"none of the above", "i don't know"}
# --clean-text: what each candidate's reference text shows, and in what order (see _clean_parts). 0: as looked up;
# 1: ordered and filtered; 2: also Wikipedia's appearance sections and taxonomy shrunk to name lines.
CLEAN_TEXT = 0
APPEARANCE_TAGS, IDENTIFICATION_TAGS = {"symptom", "damage"}, {"identification", "grade_defect"}
DISAMBIGUATION = re.compile(r"\bmay (also )?refer to\b|\bmay mean\b|disambiguation", re.I)
TAXONOMY_TAGS = {"taxonomy", "synonym", "distribution"}
NAME_SOURCES = {"eppo", "gbif", "catalogue_of_life"}
DISEASE_NAME = re.compile(r"wilt|spot|blight|rot\b|rust|mildew|mosaic|canker|scab|curl|virus|disease|damage|mold|mould|"
                          r"blast|smut|anthracnose|streak|stripe|yellows|dieback|die back|deficien", re.I)
NAME_FIELD = re.compile(r"(Scientific name|Preferred name|Common names?|English common names|Synonyms)\s*:", re.I)
NAME_CHARS = 300
BOILERPLATE = re.compile(r"An official website of the United States government|^N/A \(or sometimes", re.I)
# Wikipedia pages about the wrong thing ("rust" -> iron corrosion, carambola -> carbon nanotubes) name no living thing.
BIOLOGY = re.compile(r"plant|leaf|leaves|crop|fung|bacteri|virus|insect|pest|disease|species|genus|famil|weed|fruit|"
                     r"seed|grain|root|stem|flower|cultivar|variet|host|larva|mite|nematode|tree|herb|grass", re.I)


def _cached(key: list, fetch):
    def live():
        with _live:
            return fetch()
    return cache.cached(key, live)


def _lookup(c: dict, crop: str, task: str) -> dict:
    """Every database is searched whatever the task, so a wrong guess about the kind of class misses nothing; the
    databases covering `task` come first (and get the result space first)."""
    from agml_agent.retrieve import _load_sources, retrieve

    name, sci, aliases = c["name"], c.get("scientific_name") or None, c.get("aliases") or []
    if KB_ONLY:
        return _local_lookup(name, sci, aliases, crop)
    found = _cached(["name-all", name.strip().lower(), sci, aliases, crop.lower()],
                    lambda: retrieve(name, crop=crop, scientific_name=sci, aliases=aliases, deadline=SOURCE_DEADLINE))
    # cached lookups made before these sites were left out can still hold their pages
    found = {src: kept for src, chs in found.items() if (kept := [ch for ch in chs if not BLOCKED.search(ch.get("source_url") or "")])}
    if KB_INDEX is not None:
        hits = KB_INDEX.search([name, sci, *aliases], crop)
        if hits:
            found = {"local_kb": [{"excerpt_text": f"{h['title']} > {h['heading']}: {h['text']}", "source_url": h["url"],
                                   "tags": KB_TAGS[h["kind"]], "raw_response": json.dumps({"site": h["site"]})} for h in hits],
                     **found}
    covers = {src: cfg.get("task") or [] for src, cfg in _load_sources().items()}
    covers["local_kb"] = ["disease", "pest", "species", "quality"]
    return dict(sorted(found.items(), key=lambda kv: task not in covers.get(kv[0], [])))


def _local_lookup(name: str, sci: str | None, aliases: list, crop: str) -> dict:
    """--kb-only: the names table resolves the candidate (and its scientific name and aliases) to a taxon; the
    knowledge base is searched for the name as given, then the scientific names, synonyms and common names found."""
    info = None
    if NAMES is not None:
        for n in [name, sci, *aliases]:
            if n and (info := NAMES.lookup(n, crop=crop)):
                break
        if info and info.get("rank") in ("kingdom", "phylum", "class", "order"):
            info = None  # "caterpillar" -> Lepidoptera, "fungal spores" -> Fungi: too broad to help
    terms = [name, sci, *aliases]
    if info:
        terms += [info["scientific_name"], *info["synonyms"], *info["common_names"]]
    if info and info["source"] == "wikipedia":
        info = None  # a Wikipedia redirect gives the article to search, not a scientific name for the names line
    found = {}
    hits = KB_INDEX.search(list(dict.fromkeys(t for t in terms if t)), crop) if KB_INDEX is not None else []
    if hits:
        found["local_kb"] = [{"excerpt_text": f"{h['title']} > {h['heading']}: {h['text']}", "source_url": h["url"],
                              "tags": KB_TAGS[h["kind"]], "raw_response": json.dumps({"site": h["site"]})} for h in hits]
    if info:
        group = ", ".join(x for x in (info["kingdom"], info["family"]) if x)
        syn = [x for x in info["synonyms"] if re.fullmatch(r"[A-Z][a-z]+( [a-z-]+){1,2}", x)][:5]
        text = "\n".join(x for x in (
            f"Scientific name: {info['scientific_name']}" + (f" ({group})" if group else ""),
            info["common_names"] and "English common names: " + ", ".join(info["common_names"][:5]),
            syn and "Synonyms: " + ", ".join(syn)) if x)
        found[NAME_LABEL[info["source"]]] = [{"excerpt_text": text, "tags": ["taxonomy"],
                                 "source_url": NAME_URL[info["source"]].format(info["id"]),
                                 "raw_response": json.dumps({"source": info["source"], "id": info["id"]})}]
    return found


def _label(src: str, ch: dict) -> str:
    """The source tag the model sees: a knowledge-base section names its site ("local_kb: UC IPM"), so the source of
    every piece of text is visible in the result, not only in the URL."""
    if src == "local_kb":
        from agml_agent.kb import site_label

        site = (_loads(ch.get("raw_response") or "{}") or {}).get("site")
        return f"local_kb: {site_label(site)}" if site else src
    return src


def _local_sources_text(kb_root: Path) -> str:
    """The sources a --kb-only lookup searches, by display name, from the knowledge base's manifest."""
    from agml_agent.kb import site_label

    m = _loads((kb_root / "manifest.json").read_text()) if (kb_root / "manifest.json").exists() else {}
    names = [n for n, c in (m.get("sites") or {}).items() if c.get("sections")] + list((m.get("names") or {}).get("rows") or {})
    return ", ".join(dict.fromkeys(site_label(n) for n in names))


def _describe_local_sources(kb_root: Path) -> None:
    """--kb-only: retrieve's description names the databases actually searched (all on disk), not v1's live list."""
    local = _local_sources_text(kb_root)
    for schema in [RETRIEVE_SCHEMA, *RETRIEVE_MODES.values()]:
        f = schema["function"]
        f["description"] = f["description"].replace("external agricultural databases", "agricultural reference databases") \
            .replace(f"({DATABASES})", f"({local}; all downloaded locally)")


def _block_network() -> None:
    """--kb-only: connections to anything but this machine (the model and judge servers) raise."""
    import ipaddress
    import socket

    connect = socket.socket.connect

    def local_only(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None:
            try:
                ok = ipaddress.ip_address(host).is_loopback
            except ValueError:
                ok = host == "localhost"
            if not ok:
                raise RuntimeError(f"--kb-only: network access to {address} is not allowed")
        return connect(self, address)

    socket.socket.connect = local_only


def tidy_name(c: dict, crop: str = "") -> dict:
    """The candidate with its name cleaned for lookup: in dataset-label form ("Pepper,_bell___Bacterial_spot") only the
    part after the last "___"; underscores, commas and hyphens between words become spaces; text in brackets becomes
    an alias; a trailing plant part ("on leaves") and a repeated first word ("Tomato Tomato mosaic virus") are dropped.
    A crop word that is part of the name ("Banana bunchy top", "Coffee leaf rust") stays. The name as given is kept as
    the last alias."""
    orig = c["name"]
    name = re.sub(r"[_,]+", " ", orig.rsplit("___", 1)[-1] if "___" in orig else orig)
    brackets = [b.strip() for b in re.findall(r"\(([^)]*)\)", name) if b.strip()]
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z]{3})", " ", name)        # kalanchoe-pinnata, not "K-1" codes
    name = re.sub(r"(?<=[a-z]{3})-[A-Za-z]{1,2}$", "", name.strip())  # label suffixes: "aphids-p", "potasium-K"
    name = re.sub(r"\s+on (the )?(leaves|leaf|fruits?|stems?|plants?|pods?)\s*$", "", name.strip(), flags=re.I)
    words = name.split()
    while len(words) > 2 and words[0].lower() == words[1].lower():
        words = words[1:]
    name = " ".join(words).strip() or orig
    aliases = list(dict.fromkeys([*(c.get("aliases") or []), *brackets, *([orig] if name != orig else [])]))
    return {**c, "name": name, "aliases": aliases}


def with_options(candidates, options: list[str]) -> list:
    """The answer options first (in their order, without 'None of the above' / 'I don't know'), then the model's
    candidates that are not options, up to MAX_WITH_OPTIONS."""
    opts = [o for o in options if o.strip().lower() not in NOT_OPTIONS]
    own = [{"name": c} if isinstance(c, str) else c for c in candidates or []]
    names = {o.strip().lower() for o in opts}
    extra = [c for c in own if isinstance(c, dict) and c.get("name") and c["name"].strip().lower() not in names]
    return ([next((c for c in own if isinstance(c, dict) and (c.get("name") or "").strip().lower() == o.strip().lower()), {"name": o})
             for o in opts] + extra)[:MAX_WITH_OPTIONS]


def _name_line(text: str) -> str:
    """'Scientific name: ...; English common names: ...' from a taxonomy excerpt (lines or EPPO's '. '-joined fields)."""
    fields = [f.strip() for f in re.split(r"\n|\.\s+(?=[A-Z][A-Za-z ]+:)", text)]
    return "; ".join(f for f in fields if NAME_FIELD.match(f))


def _clean_parts(res: dict, crop: str, task: str, description: str, cand: dict | None = None) -> list[str]:
    """--clean-text. Repeats (2026-10-31): EPPO's taxonomy came first for 45% of candidates, 39% of results mentioned the
    crop, and there were disambiguation pages, wrong entities and USDA grades on non-grade names. So: appearance text
    (symptoms, damage) first, then identification notes, then taxonomy; text mentioning the crop first within each;
    disambiguation pages, site boilerplate, Wikipedia pages naming no living thing, and USDA grades unless the task is
    quality (or not given) are dropped. Nothing is dropped for being about another host."""
    from agml_agent.sources import wikipedia
    from agml_agent.symptom_search import crop_names, focus

    crops = crop_names(crop)
    parts, names, name_srcs = [], {}, {}
    for turn, (src, chs) in enumerate(res.items()):
        for i, ch in enumerate(chs):
            text, tags = ch["excerpt_text"], set(ch.get("tags") or [])
            if (DISAMBIGUATION.search(text[:300]) or BOILERPLATE.search(text[:200])
                    or (src == "wikipedia" and not BIOLOGY.search(text))):
                continue
            if src == "usda_ams_grades" and task and task != "quality":
                continue
            if CLEAN_TEXT >= 2 and src == "wikipedia":  # the article's appearance sections, not only its opening
                raw = _loads(ch.get("raw_response") or "{}")
                title = raw.get("title") if isinstance(raw, dict) else None
                text = (title and wikipedia.appearance_text(title, crops)) or text
            if CLEAN_TEXT >= 2 and src in NAME_SOURCES and cand and DISEASE_NAME.search(cand["name"]) and not any(
                    n and n.lower() in text.lower() for n in (cand["name"], cand.get("scientific_name"))):
                # a disease's common name looked up as a name finds some organism, often on another host ("Fusarium
                # wilt" -> EPPO's onion basal rot): its names and hosts are shown only if the text names the candidate
                continue
            if CLEAN_TEXT >= 2 and tags <= TAXONOMY_TAGS:
                # taxonomy-only text shrinks to name fields (scientific name, common names, synonyms), merged across
                # sources into one line: they map local and common names to the options; the lineage takes space from
                # appearance text
                for field in _name_line(text).split("; "):
                    if field and field.lower() not in names:
                        names[field.lower()] = field
                        name_srcs.setdefault(src, ch["source_url"])
                continue
            kind = 0 if tags & APPEARANCE_TAGS else 1 if tags & IDENTIFICATION_TAGS else 2
            on_crop = any(c in text.lower() for c in crops)
            parts.append(((kind, not on_crop, i, turn),
                          f"[{_label(src, ch)}] {focus(text, description, PER_CHUNK_CHARS)} (source: {ch['source_url']})"))
    if names:
        line = "; ".join(names.values())[:NAME_CHARS]
        parts.append(((-1, False, 0, 0), f"[names: {', '.join(name_srcs)}] {line} (source: {next(iter(name_srcs.values()))})"))
    return [p for _, p in sorted(parts, key=lambda kp: kp[0])]


def _fit(parts: list[str], limit: int) -> str:
    """Like _join, but the first part that doesn't fit is cut at a sentence end to the space left (keeping its source
    and URL) rather than dropped, when at least 300 characters are left."""
    out, size = [], 0
    for part in parts:
        room = limit - size
        if len(part) <= room:
            out.append(part)
            size += len(part) + 2
            continue
        m = re.match(r"(\[[^\]]*\] )(.*)( \(source: [^)]*\))$", part, re.S)
        if m and room >= 300:
            body = m.group(2)[:room - len(m.group(1)) - len(m.group(3)) - 4]
            body = body[:body.rfind(". ") + 1] or body
            out.append(f"{m.group(1)}{body} ...{m.group(3)}")
        break
    return "\n\n".join(out)


def _join(parts: list[str], limit: int) -> str:
    """Whole parts while they fit (a cut part would lose its source URL); the first one is cut if it alone is too long."""
    out, size = [], 0
    for part in parts:
        if size + len(part) > limit:
            if not out:
                out.append(part[:limit])
            break
        out.append(part)
        size += len(part) + 2
    return "\n\n".join(out)


def run_retrieve(description: str = "", candidates: list | None = None, crop: str = "", task: str = "",
                 max_candidates: int = MAX_CANDIDATES) -> str:
    """The retrieve tool, reduced to what the model may see: excerpt text and source URL per chunk."""
    from agml_agent.symptom_search import focus, search_description

    cands = [{"name": c} if isinstance(c, str) else c for c in candidates or []]
    cands = [c for c in cands if isinstance(c, dict) and c.get("name")][:max_candidates]
    looked_up = [tidy_name(c, crop) for c in cands] if TIDY_NAMES else cands
    if not description and not cands:
        return "error: give a description of what you see and/or candidates (names) to check."
    if cands:
        with ThreadPoolExecutor(len(cands)) as ex:
            found = list(ex.map(lambda c: _lookup(c, crop, task), looked_up))
        sections = []
        for c, res in zip(cands, found):
            res = {src: [ch for ch in chs if not LEAK.search(ch["excerpt_text"])] for src, chs in res.items()}
            if CLEAN_TEXT:
                chunks = _clean_parts(res, crop, task, description, c)
            else:
                # sources take turns (first chunk of each, then second...), task-matching sources first
                chunks = [f"[{_label(src, chs[i])}] {focus(chs[i]['excerpt_text'], description, PER_CHUNK_CHARS)} (source: {chs[i]['source_url']})"
                          for i in range(max(map(len, res.values()), default=0)) for src, chs in res.items() if i < len(chs)]
            body = (_fit(chunks, RESULT_CHARS // len(cands)) if CLEAN_TEXT >= 2 else _join(chunks, RESULT_CHARS // len(cands))) or (
                f"No results found for {c['name']!r}. Try its scientific name or aliases, or another candidate.")
            sections.append(f"## {c['name']}\n{body}")
        return "\n\n".join(sections)
    # "v4": pages cached before every description source and Wikipedia were searched are not reused
    pages = _cached(["description-v4", description.strip().lower(), crop.lower(), task],
                    lambda: search_description(description, crop, task))
    pages = [p for p in pages if not LEAK.search(p["text"])]
    if not pages:
        return ("No fact-sheet or Wikipedia pages matched this description. Check candidate names instead, or answer "
                "from the image.")
    other = f" (does not mention {crop})" if crop else ""
    head = (f"Pages matching the description, those mentioning {crop} first; pages about other hosts can still show "
            "a similar problem, compare them with the image" if crop else "Pages matching the description")
    return (f"{head} (titles usually name what they describe):\n\n"
            + _join([f"[{p['source']}] {p['title']}{'' if p.get('mentions_crop') else other}: "
                     f"{focus(p['text'], description, PER_CHUNK_CHARS)} (source: {p['url']})"
                     for p in pages], RESULT_CHARS - len(NEXT_STEP)) + NEXT_STEP)


# name -> (schema, function). Only external-database tools: see the leakage policy above. (run_python, for detection and
# segmentation with --code-tool, is not listed here: it reaches no data at all, only the item's own image, in sandbox.py.)
BENCH_TOOLS = {"retrieve": (RETRIEVE_SCHEMA, run_retrieve)}


def tool_args(model_id: str) -> list[str]:
    """vLLM flags that turn on tool calling for an agbench registry model (llama.cpp: none, --jinja does it)."""
    from agbench.eval.methods import model_spec

    if model_spec(model_id)["backend"] != "vllm":
        return []
    parsers = yaml.safe_load(files("agml_agent.config").joinpath("tool_parsers.yaml").read_text())
    if model_id not in parsers:
        raise ValueError(f"{model_id}: no entry in config/tool_parsers.yaml (vLLM --tool-call-parser)")
    return ["--enable-auto-tool-choice", "--tool-call-parser", parsers[model_id]]


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=path,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _agbench_commit() -> str | None:
    url = json.loads(importlib.metadata.distribution("agbench").read_text("direct_url.json") or "{}")
    return (url.get("vcs_info") or {}).get("commit_id")


class HarnessMethod(VLMMethod):
    """agbench's VLMMethod (same prompt, box order, sampling, effort, max_tokens) plus a tool loop.

    `max_tokens` stays the per-item generation budget: completion tokens summed over turns. Tool results
    count only toward the context window (max_model_len), not toward it. Every item's turns go to
    `trace_path` (traces.jsonl) when set."""

    def __init__(self, endpoint: str, model_id: str, thinking: bool = True, effort: str | None = "low",
                 max_tokens: int | None = None, tools: list[str] | tuple[str, ...] = ("retrieve",), max_tool_calls: int = 4,
                 trace_path: str | Path | None = None, served_name: str | None = None, retries: int = 3,
                 retrieve_mode: str = "description+candidates"):
        super().__init__(endpoint, model_id, thinking, effort, max_tokens, served_name=served_name)
        if bad := [t for t in tools if t not in BENCH_TOOLS]:
            raise ValueError(f"tools {bad} not allowed in benchmark runs; allowed: {sorted(BENCH_TOOLS)} "
                             "(search_agml is excluded: AgML class lists/results leak benchmark labels)")
        self.tools, self.max_tool_calls, self.retries = list(tools), max_tool_calls, retries
        self.trace_path = Path(trace_path) if trace_path else None
        if retrieve_mode not in RETRIEVE_MODES:
            raise ValueError(f"retrieve_mode {retrieve_mode!r}; one of {sorted(RETRIEVE_MODES)}")
        self.retrieve_mode = retrieve_mode
        self.name = f"{model_id} + tools" + ("" if retrieve_mode == "description+candidates" else f" ({retrieve_mode})") + (
            " + code" if CODE_TOOL else "")
        self._lock = asyncio.Lock()
        self._code_slots = asyncio.Semaphore(CODE_SLOTS)

    def config(self) -> dict:
        return {**super().config(), "name": self.name, "harness": "exploit", "tools": self.tools,
                "max_tool_calls": self.max_tool_calls, "retrieve_mode": self.retrieve_mode, "agml_agent": importlib.metadata.version("agml-agent"),
                "agml_agent_git": _git_sha(Path(__file__).parent), "agbench_commit": _agbench_commit(),
                "retrieve_cache": str(cache.CACHE.path) if cache.CACHE else None, "live_search": scrape.LIVE_SEARCH,
                "clean_text": CLEAN_TEXT, "check_options": CHECK_OPTIONS, "keep_healthy": KEEP_HEALTHY,
                "tidy_names": TIDY_NAMES, "crop_from_prompt": CROP_FROM_PROMPT, "code_tool": CODE_TOOL,
                "kb": str(KB_INDEX.path) if KB_INDEX is not None else None, "kb_only": KB_ONLY,
                "kb_manifest": _loads((KB_INDEX.path.parent / "manifest.json").read_text())
                if KB_INDEX is not None and (KB_INDEX.path.parent / "manifest.json").exists() else None}

    async def _create(self, **kw):
        err = None
        for attempt in range(self.retries + 1):
            try:
                return await self.client.api.chat.completions.create(**kw), None
            except Exception as e:  # connection drops, 5xx, timeouts
                err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt * 5)
        return None, err

    async def _call_tool(self, name: str, raw_args: str, options: list[str] = (), item: EvalItem | None = None) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            args = json.loads(raw_args or "{}")
            if (name == "retrieve" and CROP_FROM_PROMPT and isinstance(args, dict) and not args.get("crop") and item
                    and item.crop and item.crop.lower() in (item.prompt or "").lower()):
                args["crop"] = item.crop
            if name == "retrieve" and self.retrieve_mode == "candidates":
                args.pop("description", None)  # not offered in this mode
            if name == "retrieve" and self.retrieve_mode == "crosscheck-assess":
                assessment = args.pop("assessment", None)
                if assessment not in ASSESSMENTS or not args.get("description"):
                    return f"error: give an assessment (one of {ASSESSMENTS}) and a description of what you see.", time.perf_counter() - t0
                if not args.get("candidates"):
                    result = NOTHING_TO_CHECK if assessment == NO_PROBLEM_SEEN else "error: give candidates (up to 4 names) to cross-check."
                    return result, time.perf_counter() - t0
            elif name == "retrieve" and self.retrieve_mode.startswith("crosscheck") and not (args.get("description") and args.get("candidates")):
                result = ("error: give both a description of what you see and candidates (your initial diagnosis, up "
                          "to 4 names) to cross-check.")
                return result, time.perf_counter() - t0
            if name == "retrieve" and self.retrieve_mode in PLACEBO_MODES:
                return PLACEBO_RESULT, time.perf_counter() - t0
            if name == "retrieve" and CHECK_OPTIONS and options:
                args["candidates"], args["max_candidates"] = with_options(args.get("candidates"), options), MAX_WITH_OPTIONS
            result = await asyncio.to_thread(BENCH_TOOLS[name][1], **args) if name in self.tools else f"error: unknown tool {name!r}"
            if name in self.tools and self.retrieve_mode in ("crosscheck-healthy", "crosscheck-assess", "crosscheck-look-first"):
                result += RESULT_NOTE
        except Exception as e:  # bad JSON or arguments, source failures
            result = f"error: {type(e).__name__}: {e}"
        return result, time.perf_counter() - t0

    async def _run_code(self, raw_args: str, images: list) -> tuple[str, float, list[str]]:
        """run_python: the model's code on the images it saw, in the sandbox (sandbox.py)."""
        from agml_agent import sandbox

        t0 = time.perf_counter()
        try:
            args = json.loads(raw_args or "{}")
            code = args.get("code") if isinstance(args, dict) else None
            if not code or not isinstance(code, str):
                return "error: give the Python code to run as `code`.", time.perf_counter() - t0, []
            out_dir = (self.trace_path.parent if self.trace_path else Path("runs")) / "tool_images"
            async with self._code_slots:
                text, images = await asyncio.to_thread(sandbox.run, code, images, out_dir)
        except Exception as e:  # bad JSON, sandbox failure
            text, images = f"error: {type(e).__name__}: {e}", []
        return text, time.perf_counter() - t0, images

    async def predict(self, item: EvalItem) -> Prediction:
        model = await self.client.resolve_model()
        encoded = [encode_image(p, self.client.max_pixels) for p in item.images]
        sent_size = encoded[0][1] if encoded else None
        content = [{"type": "image_url", "image_url": {"url": u}} for u, _ in encoded] + [{"type": "text", "text": self.prompt(item)}]
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": content}]
        body = dict(self.body)
        temperature = body.pop("temperature", None)
        schemas = [RETRIEVE_MODES[self.retrieve_mode] if t == "retrieve" else BENCH_TOOLS[t][0] for t in self.tools]
        seen_images, n_shown = [], 0
        max_shown = max(0, MAX_PROMPT_IMAGES - len(item.images))
        if CODE_TOOL:
            from PIL import Image

            schemas = schemas + [RUN_PYTHON_SCHEMA]
            for p, (_, size) in zip(item.images, encoded):  # the code sees what the model saw, at the size it saw
                im = Image.open(p).convert("RGB")
                seen_images.append(im if list(im.size) == list(size) else im.resize(tuple(size), Image.BICUBIC))

        trace, n_calls, tool_s, used = [], 0, 0.0, []
        prompt_tokens = completion_tokens = 0
        text, finish, err, t0 = "", "stop", None, time.perf_counter()
        look_first = self.retrieve_mode.startswith("crosscheck-look-first") or CODE_TOOL
        looking, first_answer, tools_off = look_first, None, False
        for _ in range(self.max_tool_calls + 2 + look_first):
            remaining = self.max_tokens - completion_tokens
            if remaining <= 0:
                finish = "length"
                break
            offer = {} if looking or tools_off else {"tools": schemas, "tool_choice": "auto" if n_calls < self.max_tool_calls else "none"}
            r, err = await self._create(model=model, messages=messages, max_tokens=remaining, temperature=temperature,
                                        extra_body=body or None, **offer)
            if r is None:
                finish = "error"
                break
            msg, finish = r.choices[0].message, r.choices[0].finish_reason or ""
            prompt_tokens += getattr(r.usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(r.usage, "completion_tokens", 0) or 0
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            calls = list(msg.tool_calls or [])
            text = msg.content or ""
            trace.append({"role": "assistant", "reasoning": reasoning, "text": text,
                          "tool_calls": [{"name": c.function.name, "arguments": _loads(c.function.arguments)} for c in calls]})
            if looking:
                looking = False
                if finish != "stop" or not text or (item.task in LOOK_ONLY_TASKS and not CODE_TOOL):
                    break
                first_answer = text
                tools_off = KEEP_HEALTHY and bool(HEALTHY_ANSWER.search(text))  # a healthy first answer: second look, no tool
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": HEALTHY_CHECK_PROMPT if tools_off else CHECK_PROMPT}]
                continue
            if not calls:
                break
            messages.append({"role": "assistant", "content": text,
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            shown = []  # images from run_python this turn: sent after the tool results, as one user message
            for c in calls:
                images = []
                if n_calls < self.max_tool_calls:
                    n_calls += 1
                    if CODE_TOOL and c.function.name == "run_python":
                        result, dt, images = await self._run_code(c.function.arguments, seen_images)
                        if n_shown + len(images) > max_shown:
                            images = images[:max(0, max_shown - n_shown)]
                            result += f"\n[only {max_shown} images per item can be returned; the rest were dropped]"
                        n_shown += len(images)
                    else:
                        result, dt = await self._call_tool(c.function.name, c.function.arguments, item.choices or [], item)
                    used.append(c.function.name)
                else:
                    result, dt = "error: tool-call budget used up; answer now", 0.0
                tool_s += dt
                trace.append({"role": "tool", "name": c.function.name, "result": result, "latency_s": round(dt, 3),
                              **({"images": images} if images else {})})
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": result})
                shown += images
            if shown:
                messages.append({"role": "user", "content": [
                    *({"type": "image_url", "image_url": {"url": encode_image(p, self.client.max_pixels)[0]}} for p in shown),
                    {"type": "text", "text": "Images from show(), in order."}]})

        latency = time.perf_counter() - t0
        checked_finish, kept_first = finish, first_answer is not None and (finish != "stop" or not text)
        if kept_first:
            text, finish = first_answer, "stop"
        await self._write_trace({"item_id": item.item_id, "dataset": item.dataset, "n_tool_calls": n_calls,
                                 "tools_used": used, "tool_s": round(tool_s, 3), "turns": sum(s["role"] == "assistant" for s in trace),
                                 "finish_reason": checked_finish, "error": err, "first_answer": first_answer, "kept_first": kept_first,
                                 "healthy_first": tools_off,
                                 "trace": trace})
        reasoning_all = "\n\n".join(
            s["reasoning"] + "".join(f"\n[called {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})]" for c in s["tool_calls"])
            for s in trace if s["role"] == "assistant").strip()
        if finish == "error":
            return Prediction(text="", finish_reason="error", latency_s=latency, sent_size=sent_size, error=err)
        return Prediction(text=text, reasoning=reasoning_all, finish_reason=finish, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens, latency_s=latency, sent_size=sent_size)

    async def _write_trace(self, rec: dict) -> None:
        if not self.trace_path:
            return
        async with self._lock:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _loads(s: str):
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError:
        return s


def _judge(run_dir: Path, endpoint: str | None) -> None:
    from agbench.eval.gpu import sampler
    from agbench.eval.judge import judge_run
    from agbench.eval.resources import models
    from agbench.eval.runner import log_compute
    from agbench.eval.servers import Server

    jm = models()["judge"]
    if endpoint:
        judge_run(run_dir, endpoint, jm)
        return
    with sampler() as gpu:
        t0 = time.time()
        with Server(jm, run_dir / "judge_server.log") as srv:
            log_compute(run_dir, phase="judge_load", model=jm, wall_s=round(srv.load_seconds, 1), **gpu.summary(t0, time.time()))
            judge_run(run_dir, srv.endpoint, jm, gpu=gpu)


def run(args: argparse.Namespace) -> Path:
    from agbench.eval.gpu import sampler
    from agbench.eval.methods import key_formats, model_spec
    from agbench.eval.report import leaderboard_md, load_runs
    from agbench.eval.runner import log_compute
    from agbench.eval.servers import Server

    load_dotenv()
    cache.CACHE = None if args.no_retrieve_cache or args.kb_only else RetrievalCache(
        args.retrieve_cache or args.out / "retrieve_cache.jsonl")
    scrape.LIVE_SEARCH = not args.no_live_search
    global CLEAN_TEXT, CHECK_OPTIONS, KB_INDEX, KEEP_HEALTHY, TIDY_NAMES, CROP_FROM_PROMPT, KB_ONLY, NAMES, CODE_TOOL
    CLEAN_TEXT, CHECK_OPTIONS, KEEP_HEALTHY = args.clean_text, args.check_options, args.keep_healthy
    CODE_TOOL = args.code_tool
    TIDY_NAMES, CROP_FROM_PROMPT = args.tidy_names, args.crop_from_prompt
    if args.kb:
        from agml_agent.kb import KB
        KB_INDEX = KB(args.kb)
    if args.kb_only:
        if not args.kb:
            raise SystemExit("--kb-only needs --kb")
        from agml_agent.kb import Names
        KB_ONLY = True
        NAMES = Names(args.kb) if (args.kb / "names.sqlite").exists() else None
        _describe_local_sources(args.kb)
        _block_network()
    bench = Benchmark(args.data_dir, args.task or None, args.format or None, args.only, args.limit_per_dataset, args.seed)
    print(f"{len(bench.items)} items; {len(bench.skipped)} datasets skipped")
    spec = model_spec(args.model)
    conc = args.concurrency or (spec.get("parallel", 2) if spec["backend"] == "llamacpp" else spec["concurrency"])
    tag = ("tools" if args.retrieve_mode == "description+candidates" else f"tools-{args.retrieve_mode}") + (
        {0: "", 1: "-clean", 2: "-clean2"}[args.clean_text] + ("-options" if args.check_options else "") + ("-kb" if args.kb else "") + (
        "-keephealthy" if args.keep_healthy else "") + ("-tidy" if args.tidy_names else "") + (
        "-crop" if args.crop_from_prompt else "") + ("-kbonly" if args.kb_only else "") + ("-code" if args.code_tool else ""))
    rd = args.run_dir or args.out / f"{args.model.replace('/', '__')}__{tag}_{datetime.now():%Y%m%d-%H%M%S}"
    rd.mkdir(parents=True, exist_ok=True)

    def method(endpoint: str) -> HarnessMethod:
        return HarnessMethod(endpoint, args.model, args.thinking, args.reasoning_effort, args.max_tokens, args.tools,
                             args.max_tool_calls, trace_path=rd / "traces.jsonl", retrieve_mode=args.retrieve_mode)

    if args.endpoint:
        res = bench.run(method(args.endpoint), rd, conc, key_formats=key_formats(spec))
    else:
        with sampler() as gpu:
            t0 = time.time()
            with Server(args.model, rd / "server.log", max_model_len=args.max_model_len, extra_args=tool_args(args.model)) as srv:
                log_compute(rd, phase="load", model=args.model, wall_s=round(srv.load_seconds, 1), **gpu.summary(t0, time.time()))
                res = bench.run(method(srv.endpoint), rd, conc, key_formats=key_formats(spec), gpu=gpu)
    if args.judge and res.metrics["pending_judge"]:
        _judge(rd, args.judge_endpoint)
    print(leaderboard_md(load_runs([rd])))
    print(f"results: {rd}")
    return rd


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the harness (model + tools) on AGgentic-benchmarks via the pinned agbench.")
    p.add_argument("--data-dir", type=Path, required=True, help="Folder with base-data/.")
    p.add_argument("--model", required=True, help="agbench registry model id.")
    p.add_argument("--endpoint", help="Use a running OpenAI-compatible server (started with tool calling on) instead of launching one.")
    p.add_argument("--tools", nargs="+", default=["retrieve"], help=f"Tools offered to the model; allowed: {sorted(BENCH_TOOLS)}.")
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--retrieve-mode", choices=sorted(RETRIEVE_MODES), default="description+candidates",
                   help="description+candidates: search by what the model sees and check names; candidates: check names only.")
    p.add_argument("--task", action="append", help="classification | detection | segmentation | image_text_to_text (repeatable).")
    p.add_argument("--format", action="append", help="mcqa_within | mcqa_across | oeq | grounding | native (repeatable).")
    p.add_argument("--only", help="Regex on dataset name.")
    p.add_argument("--limit-per-dataset", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.add_argument("--reasoning-effort", default="low", help="low | medium | high (agbench default: low).")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-dir", type=Path, help="Resume into this run dir.")
    p.add_argument("--judge", action="store_true", help="Judge open-ended answers after the run (agbench judge).")
    p.add_argument("--judge-endpoint")
    p.add_argument("--retrieve-cache", type=Path, help="Lookup cache shared across runs (default: <out>/retrieve_cache.jsonl).")
    p.add_argument("--no-retrieve-cache", action="store_true", help="Always look up live.")
    p.add_argument("--clean-text", type=int, nargs="?", const=1, default=0, choices=[0, 1, 2],
                   help="Candidate reference text. 1: appearance first, then identification, then taxonomy; text on the "
                        "crop first; disambiguation pages, off-topic Wikipedia pages and off-task USDA grades dropped. "
                        "2: also Wikipedia's appearance sections (Symptoms, Description...) and taxonomy as name lines."),
    p.add_argument("--check-options", action="store_true",
                   help=f"On multiple choice, look up every answer option, then the model's candidates (up to {MAX_WITH_OPTIONS}).")
    p.add_argument("--tidy-names", action="store_true",
                   help="Clean candidate names before lookup (crop prefix, brackets, underscores, plant part); the name "
                        "as given stays an alias.")
    p.add_argument("--crop-from-prompt", action="store_true",
                   help="A call without crop gets the item's crop when the prompt names it.")
    p.add_argument("--keep-healthy", action="store_true",
                   help="crosscheck-look-first: a first answer that finds no problem gets a second look without the tool, "
                        "kept unless the image clearly shows a problem.")
    p.add_argument("--kb", type=Path, help="Local knowledge base folder (agml-agent-kb), searched next to the API sources.")
    p.add_argument("--code-tool", action="store_true",
                   help="Offer run_python (sandboxed Python on the item's images: numpy, PIL, cv2, skimage, scipy; print() "
                        "and show() come back) next to retrieve on every task, after the first answer; the model "
                        "chooses which to use. Detection and segmentation then get the check step too.")
    p.add_argument("--kb-only", action="store_true",
                   help="Only the local knowledge base (--kb) and its names table: no API sources, web search, Wikipedia "
                        "or lookup cache, and connections off this machine raise.")
    p.add_argument("--no-live-search", action="store_true",
                   help="Web searches (fact-sheet sites) come only from the cache; API sources still look up live. "
                        "Lookups made this way lack fact-sheet pages, so give the run its own --retrieve-cache copy.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
