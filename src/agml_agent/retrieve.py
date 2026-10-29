"""
retrieve() — live, demand-driven external grounding lookup.

    from agml_agent.retrieve import retrieve
    result = retrieve("Coffee leaf rust", crop="coffee", scientific_name="Hemileia vastatrix")

External sources only — for AgML's own dataset catalog, see the separate
search_agml() tool (agml_agent/search_agml.py), a deliberately different
tool for a deliberately different task (see README "Why two tools, not
one, not three").

No corpus is built or stored anywhere. Every call hits each active source's
fetcher module (agml_agent/sources/<name>.py) live, over the network, right
now. Results are kept in a per-process in-memory cache only — nothing is
written to disk, nothing is published anywhere — so repeated lookups for the
same class within one run/process don't re-hit the network, but nothing
persists across runs.

`sources`:
  - unset (None)            -> every active source in config/sources.yaml
                                (i.e. everything except paused/parked ones).
  - a list of source names  -> only those (must be `status: active`)

CLI, for manually testing one class against every active source (from a
source checkout: `uv run python -m agml_agent.retrieve ...`; installed:
`agml-agent-retrieve ...`):
    agml-agent-retrieve "Coffee leaf rust" --crop coffee --sci "Hemileia vastatrix" --aliases roya
    agml-agent-retrieve "Coffea arabica" --crop coffee --task species
    agml-agent-retrieve "Coffee leaf rust" --sources gbif eppo
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import yaml
from dotenv import load_dotenv

from agml_agent.sources.base import ClassQuery

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ddgs's internal HTTP client (primp) logs every search-backend request it
# tries while resolving one site-restricted query (Wikipedia, Brave, Yahoo,
# etc.) at INFO — noisy, and easy to mistake for "we're pulling content from
# these sites," which we never do (only site-restricted result URLs on the
# domains a fetcher actually lists get fetched for content). Quieted here;
# raise back to INFO/DEBUG if you need to debug the search step itself.
for _noisy in ("primp", "httpx", "httpcore", "urllib3", "h2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# process-lifetime cache: {(source_name, class_name_lower): [chunk_dict, ...]}
# Never persisted, never shared across processes — see module docstring.
_CACHE: dict[tuple[str, str], list[dict]] = {}


def _load_sources() -> dict[str, dict]:
    # importlib.resources, not a filesystem-relative path — config/ ships
    # inside the installed package, and this works whether running from a
    # source checkout or a pip/uvx install, unlike a Path(__file__)-relative
    # read (which only reliably works when there's a real repo checkout).
    text = files("agml_agent").joinpath("config", "sources.yaml").read_text()
    cfg = yaml.safe_load(text)
    return {s["name"]: s for s in cfg["sources"]}


def _active_source_names() -> list[str]:
    return [name for name, cfg in _load_sources().items() if cfg.get("status") == "active"]


def retrieve(
    class_name: str,
    crop: str = "",
    task: str = "",
    scientific_name: str | None = None,
    aliases: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Returns {source_name: [chunk_dict, ...]} — only sources that returned
    at least one chunk are present as keys. A source erroring (bad key,
    network failure, page layout changed) is logged and simply omitted, so
    one broken source never breaks the whole call."""
    registry = _load_sources()

    requested = _active_source_names() if sources is None else sources

    query = ClassQuery(
        class_name=class_name, crop=crop, task=task,
        scientific_name=scientific_name, aliases=aliases or [],
    )

    def one(name: str) -> list[dict]:
        cfg = registry.get(name)
        if cfg is None:
            log.warning("unknown source %r — skipping (not in config/sources.yaml)", name)
            return []
        if cfg.get("status") != "active":
            log.warning("source %r is %s, not active — skipping", name, cfg.get("status"))
            return []
        if query.task and cfg["task"] and query.task not in cfg["task"]:
            return []  # this source doesn't cover this task, not an error

        cache_key = (name, class_name.strip().lower())
        if cache_key in _CACHE:
            return _CACHE[cache_key]
        try:
            module = importlib.import_module(cfg["module"])
            chunks = module.fetch(query, **(cfg.get("args") or {}))
        except Exception as e:
            log.error("%s: %s", name, e)
            return []
        chunk_dicts = [c.to_dict() for c in chunks]
        if chunk_dicts:  # empties aren't cached: search backends fail transiently, so a later call may find it
            _CACHE[cache_key] = chunk_dicts
        return chunk_dicts

    # Sources are independent (different sites and APIs): asked in parallel, results kept in config order.
    with ThreadPoolExecutor(max(1, min(len(requested), 8))) as ex:
        found = list(ex.map(one, requested))
    results = {name: chunks for name, chunks in zip(requested, found) if chunks}
    return results


def _run(args: argparse.Namespace) -> None:
    result = retrieve(
        class_name=args.class_name,
        crop=args.crop or "",
        task=args.task or "",
        scientific_name=args.sci,
        aliases=args.aliases,
        sources=args.sources,
    )

    if not result:
        print(f"No results for {args.class_name!r} from any active source.")
        return

    for source, chunks in result.items():
        print(f"\n=== {source} — {len(chunks)} chunk(s) ===")
        for c in chunks:
            print(f"  tags: {', '.join(c['tags'])}")
            print(f"  url:  {c['source_url']}")
            print(f"  {c['excerpt_text'][:300]}{'...' if len(c['excerpt_text']) > 300 else ''}")

    if args.json:
        print("\n" + json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test agml.retrieve() against one class, live, over every active source "
                     "(see config/sources.yaml) — or a subset via --sources.",
        epilog='Example: agml-agent-retrieve "Coffee leaf rust" --crop coffee --sci "Hemileia vastatrix" '
               '--aliases roya --task disease',
    )
    parser.add_argument(
        "class_name",
        help='the class/concept to look up, exactly as you want it searched for, e.g. '
             '"Coffee leaf rust" or "Coffea arabica". This is the only required argument — '
             'everything else narrows or helps the search.',
    )
    parser.add_argument(
        "--crop", default=None,
        help='the crop/host this class belongs to, e.g. "coffee". Used both as extra search '
             'context (e.g. USDA AMS Grade Standards uses it to find the right commodity page) '
             'and carried through into each result chunk\'s `crop` field.',
    )
    parser.add_argument(
        "--task", default=None, choices=["disease", "pest", "species", "quality"],
        help="restricts which sources are even tried, since each source in config/sources.yaml "
             "only covers certain tasks (e.g. GBIF/USDA PLANTS/USDA NALT are species-only, USDA "
             "AMS is quality-only). Omit to try every active source regardless of task.",
    )
    parser.add_argument(
        "--sci", default=None,
        help='scientific name, e.g. "Hemileia vastatrix". Tried as a search term if the plain '
             "class_name doesn't get a match — most useful for species-ID sources like GBIF/"
             "USDA PLANTS, which key off scientific names more reliably than common names.",
    )
    parser.add_argument(
        "--aliases", nargs="*", default=None,
        help='extra alternate names to also try, space-separated, e.g. --aliases roya "coffee rust". '
             "Tried in order after class_name and --sci, so put the most likely match first.",
    )
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="restrict the call to specific source names instead of every active source, "
             "space-separated, e.g. --sources gbif eppo. Names must match config/sources.yaml "
             "and be status: active, or they're skipped with a warning.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="also print the full result as indented JSON after the human-readable summary "
             "(every field per chunk — tags, raw_response, license, etc., not just the excerpt).",
    )
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
