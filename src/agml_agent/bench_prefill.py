"""
bench_prefill.py — fill the retrieve cache before a benchmark run: look up every classification class name of the
agbench items (the names the model sees as options), so the run mostly reads cached text instead of waiting on slow,
rate-limited site searches.

    agml-agent-bench-prefill --data-dir /path/with/base-data --cache runs/retrieve_cache.jsonl

Point the run at the same file (`agml-agent-bench --retrieve-cache ...`). The model still gets only what it asks for:
the cache changes how fast a lookup returns, not what it returns. Each name is looked up with its dataset's crop (when
the dataset has one crop) and with no crop, as the model often gives none. Rerunning resumes: found lookups are
cached, and lookups that found nothing are tried again.

Also tells whether site search works from where it runs: the progress lines count live searches that found nothing
(farm compute nodes: 40% in the wide pilot).
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from agml_agent.common import cache, scrape

TASKS = {"disease": "disease", "pest_damage": "pest", "species_id": "species", "quality_maturity": "quality"}


def lookups(items) -> list[tuple[str, str, str]]:
    """(name, crop, task) for every class name of the classification items: with the dataset's crop first (when it
    names one crop), then without. Names with fewer than 3 letters (codes like '4', 'R') are skipped."""
    out = {}
    for it in items:
        if it.task != "classification":
            continue
        task, crop = TASKS.get(it.ag_task, ""), (it.crop or "").strip().lower()
        crop = "" if crop == "plant" or "," in crop or " or " in crop else crop
        for name in it.class_names or []:
            if len(re.findall(r"[A-Za-z]", name)) >= 3:
                out.setdefault((name.strip(), crop, task), None)
    keys = list(out)
    return [k for k in keys if k[1]] + list(dict.fromkeys((n, "", t) for n, _, t in keys))


def run(args: argparse.Namespace) -> None:
    from agbench.eval import Benchmark

    from agml_agent.bench import _lookup

    load_dotenv()
    cache.CACHE = cache.RetrievalCache(args.cache)
    bench = Benchmark(args.data_dir, ["classification"], None, args.only, None, 0)
    todo = lookups(bench.items)
    print(f"{len(todo)} lookups ({len(cache.CACHE.data)} cache entries already)", flush=True)
    t0, found = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_lookup, {"name": n}, crop, task) for n, crop, task in todo]
        for i, f in enumerate(as_completed(futs), 1):
            found += bool(f.result())
            if i % args.every == 0 or i == len(todo):
                s = scrape.STATS
                print(f"{i}/{len(todo)} done, {found} found something; live searches {s['searches']}, "
                      f"{s['empty']} empty ({s['empty'] / max(s['searches'], 1):.0%}); {(time.time() - t0) / 60:.1f} min",
                      flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Fill the retrieve cache with every classification class name of the benchmark.")
    p.add_argument("--data-dir", type=Path, required=True, help="Folder with base-data/.")
    p.add_argument("--cache", type=Path, default=Path("runs/retrieve_cache.jsonl"))
    p.add_argument("--only", help="Regex on dataset name.")
    p.add_argument("--workers", type=int, default=2, help="Concurrent lookups (the search engines rate-limit).")
    p.add_argument("--every", type=int, default=25, help="Print progress every N lookups.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
