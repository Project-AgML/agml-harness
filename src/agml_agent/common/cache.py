"""Persistent lookup cache (JSON lines: {key, value}), shared by the benchmark harness and the cache-fill script.

Off unless a caller sets CACHE (bench.run, bench_prefill). Two levels use it: whole retrieve() lookups (bench.py) and,
below them, site-restricted search results and extracted page text (common/scrape.py). The search is the slow and
unreliable step, and its query depends only on the class name and crop (scientific names and aliases only filter the
pages), so the lower level is reused across calls that name the same class differently."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class RetrievalCache:
    """Lookup results kept across runs, so a full run doesn't repeat the same slow, rate-limited lookups and every
    model sees the same retrieved text. Empty results are not stored: they are often transient search failures."""

    def __init__(self, path: str | Path):
        self.path, self._lock = Path(path), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.data[r["key"]] = r["value"]

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        if not value:
            return
        with self._lock:
            self.data[key] = value
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


CACHE: RetrievalCache | None = None


def cached(key: list, fetch):
    """fetch() unless CACHE holds a value for key; a non-empty fetched value is stored."""
    k = json.dumps(key)
    if CACHE and (hit := CACHE.get(k)) is not None:
        return hit
    value = fetch()
    if CACHE:
        CACHE.put(k, value)
    return value
