"""Local knowledge base: extension fact-sheet sites downloaded once, split into sections and searched locally, the way
AGgentic-benchmarks keeps its data on disk. Lookups then take milliseconds, are not rate-limited, and every run sees
the same text.

Sites are listed in config/kb_sites.yaml. Only sites whose robots.txt allows us are crawled; sites that block AI
crawlers (PlantVillage, PNW Handbooks, APS, ProMusa, 2026-11-03) are left out, and the crawler obeys robots.txt and
its Crawl-delay (at least 1 s between requests per site). Text only: images and PDFs are not downloaded.

Layout of the knowledge-base folder (default ./kb, or $AGML_KB):
    raw/<site>/pages.jsonl        one line per fetched URL: url, status, content type, sha1, fetched date
    raw/<site>/html/<sha1>.html.gz  the page as downloaded (kept so extraction can be redone without re-crawling)
    docs.jsonl.gz                 one line per section: site, url, title, heading, kind, text
    kb.sqlite                     the same sections with a full-text index (SQLite FTS5, BM25 ranking)
    manifest.json                 sites, page and section counts, build date

    agml-agent-kb crawl [--site NAME] [--max-pages N]   download (resumable: fetched URLs are skipped)
    agml-agent-kb build                                  extract sections and index them
    agml-agent-kb search "coffee leaf rust" [--crop coffee]
    agml-agent-kb stats
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.robotparser
from datetime import date
from importlib.resources import files
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import yaml

from agml_agent.common.http import HEADERS, TIMEOUT

LEAK = re.compile(r"\b(data ?sets?|machine learning|deep learning|neural networks?|computer vision)\b", re.I)
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
LINK = re.compile(r"""href=["']([^"'#]+)""", re.I)
LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
# "Diagnostic procedures" sections describe lab tests, not what the plant looks like: not symptom sections.
KINDS = [("symptom", re.compile(r"symptom|signs?\b|identif|damage|descri|appearance|diagnosis|characteristic|injur", re.I)),
         ("life_cycle", re.compile(r"life ?cycle|biology|disease cycle|development|epidemiolog|spread", re.I)),
         ("management", re.compile(r"manag|control|treatment|prevent|monitor", re.I))]
# Names that describe no organism: looking them up finds sections that only mention the word ("Healthy flowers...").
NOT_NAMES = re.compile(r"^(healthy|normal|fresh|good|bad|ripe|unripe|rotten|damaged|other|none|unknown)\b", re.I)
SECTION_CHARS = 1500  # long sections are split at paragraph breaks into parts of about this size
MIN_SECTION_CHARS = 80


def kb_dir(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("AGML_KB") or "kb")


def sites() -> dict[str, dict]:
    cfg = yaml.safe_load(files("agml_agent").joinpath("config", "kb_sites.yaml").read_text())
    return {s["name"]: s for s in cfg["sites"]}


# ---------------------------------------------------------------- crawl

class Crawler:
    def __init__(self, name: str, cfg: dict, root: Path, max_pages: int | None = None):
        self.name, self.cfg = name, cfg
        self.dir = root / "raw" / name
        (self.dir / "html").mkdir(parents=True, exist_ok=True)
        self.max_pages = max_pages or cfg.get("max_pages", 5000)
        self.allow = re.compile(cfg["allow"])
        self.deny = re.compile(cfg["deny"]) if cfg.get("deny") else None
        # robots.txt fetched with our User-Agent: urllib's reader uses its own, which some sites refuse (it then
        # treats the whole site as disallowed)
        self.robots = urllib.robotparser.RobotFileParser(cfg["robots"])
        r = requests.get(cfg["robots"], headers=HEADERS, timeout=TIMEOUT)
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"{name}: robots.txt answered {r.status_code}; not crawling")
        self.robots.parse(r.text.splitlines() if r.ok else [])
        self.delay = max(1.0, float(self.robots.crawl_delay(HEADERS["User-Agent"]) or 0), float(cfg.get("delay", 0)))
        self.done = {}
        if (self.dir / "pages.jsonl").exists():
            for line in (self.dir / "pages.jsonl").read_text().splitlines():
                r = json.loads(line)
                self.done[r["url"]] = r
        self._last = 0.0
        # links found but not fetched, by reason: reported in crawl.json so nothing is lost silently
        self.skipped = {"robots": set(), "documents": set()}

    def ok(self, url: str) -> bool:
        if not self.allow.search(url) or (self.deny and self.deny.search(url)):
            return False
        if not self.robots.can_fetch(HEADERS["User-Agent"], url):
            self.skipped["robots"].add(url)
            return False
        return True

    def get(self, url: str) -> requests.Response | None:
        time.sleep(max(0.0, self._last + self.delay - time.monotonic()))
        self._last = time.monotonic()
        try:
            return requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        except requests.RequestException:
            return None

    def seeds(self) -> list[str]:
        """Page URLs from the site's sitemaps (sitemap indexes followed), then its start pages."""
        urls, maps = [], list(self.cfg.get("sitemaps") or [])
        while maps:
            r = self.get(maps.pop(0))
            for loc in LOC.findall(r.text if r is not None and r.ok else ""):
                (maps if loc.endswith(".xml") else urls).append(loc)
        return urls + list(self.cfg.get("start") or [])

    def run(self) -> dict:
        queue, seen = [], set(self.done)
        for u in self.seeds():
            if u not in seen and self.ok(u):
                queue.append(u)
                seen.add(u)
        fetched = sum(r["status"] == 200 for r in self.done.values())
        # pages already fetched are re-read from disk for their links, so an interrupted crawl continues where it was
        for r in list(self.done.values()):
            if r.get("sha1") and self.cfg.get("follow", True):
                html = gzip.decompress((self.dir / "html" / f"{r['sha1']}.html.gz").read_bytes()).decode("utf-8", "ignore")
                queue += [u for u in self._links(r["url"], html) if u not in seen and not seen.add(u)]
        with open(self.dir / "pages.jsonl", "a") as log:
            while queue and fetched < self.max_pages:
                url = queue.pop(0)
                r = self.get(url)
                rec = {"url": url, "status": r.status_code if r is not None else None, "fetched": str(date.today()),
                       "type": (r.headers.get("content-type", "") if r is not None else "").split(";")[0]}
                if r is not None and r.ok and rec["type"] == "text/html":
                    rec["final_url"], rec["sha1"] = r.url, hashlib.sha1(r.url.encode()).hexdigest()
                    (self.dir / "html" / f"{rec['sha1']}.html.gz").write_bytes(gzip.compress(r.content))
                    fetched += 1
                    if self.cfg.get("follow", True):
                        queue += [u for u in self._links(r.url, r.text) if u not in seen and not seen.add(u)]
                log.write(json.dumps(rec) + "\n")
                log.flush()
                self.done[url] = rec
        report = {"site": self.name, "pages": fetched, "delay_s": self.delay, "max_pages": self.max_pages,
                  "cap_reached": bool(queue), "not_fetched_queue": len(queue),
                  "errors": sum(r["status"] != 200 for r in self.done.values()),
                  "skipped_robots": len(self.skipped["robots"]), "skipped_documents": len(self.skipped["documents"]),
                  "finished": str(date.today())}
        (self.dir / "crawl.json").write_text(json.dumps(
            {**report, "skipped_documents_urls": sorted(self.skipped["documents"]),
             "skipped_robots_urls": sorted(self.skipped["robots"]), "queue": queue}, indent=1))
        return report

    def _links(self, base: str, html: str) -> list[str]:
        out = []
        for href in LINK.findall(html):
            u = urldefrag(urljoin(base, href.strip()))[0]
            if urlparse(u).scheme not in ("http", "https"):
                continue
            if re.search(r"\.(pdf|docx?|xlsx?|pptx?)$", u, re.I):  # documents: not downloaded (yet), but listed
                if self.allow.search(u):
                    self.skipped["documents"].add(u)
            elif not re.search(r"\.(jpe?g|png|gif|svg|webp|ico|css|js|zip|mp4|mp3)$", u, re.I) and self.ok(u):
                out.append(u)
        return out


def crawl(root: Path, only: list[str] | None = None, max_pages: int | None = None) -> list[dict]:
    """Sites are crawled in parallel (one thread each); requests to one site are spaced by its delay."""
    todo = {n: c for n, c in sites().items() if c.get("status", "active") == "active" and (not only or n in only)}
    results = []
    def one(n, c):
        results.append(Crawler(n, c, root, max_pages).run())
        print(json.dumps(results[-1]), flush=True)
    threads = [threading.Thread(target=one, args=(n, c)) for n, c in todo.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------- build

def kind_of(heading: str) -> str:
    return next((k for k, rx in KINDS if rx.search(heading)), "other")


def sections(md: str, title: str) -> list[tuple[str, str]]:
    """(heading, text) parts of an extracted page (markdown headings), long sections split at paragraph breaks."""
    out, head, buf = [], title, []
    def flush():
        text = "\n".join(buf).strip()
        while text:
            if len(text) <= SECTION_CHARS:
                part, text = text, ""
            else:
                cut = text.rfind("\n", 0, SECTION_CHARS)
                cut = cut if cut > SECTION_CHARS // 3 else text.rfind(". ", 0, SECTION_CHARS) + 1 or SECTION_CHARS
                part, text = text[:cut].strip(), text[cut:].strip()
            if len(part) >= MIN_SECTION_CHARS:
                out.append((head, part))
    for line in md.splitlines():
        m = HEADING.match(line.strip())
        if m:
            flush()
            head, buf = m.group(1).strip(" *"), []
        else:
            buf.append(line)
    flush()
    return out


def build(root: Path) -> dict:
    import trafilatura

    rows, counts = [], {}
    for name in sites():
        d = root / "raw" / name
        if not (d / "pages.jsonl").exists():
            continue
        seen_text = set()
        n_pages = 0
        for line in (d / "pages.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line still being written by a running crawl
                continue
            if not r.get("sha1"):
                continue
            html = gzip.decompress((d / "html" / f"{r['sha1']}.html.gz").read_bytes()).decode("utf-8", "ignore")
            meta = trafilatura.extract_metadata(html)
            title = (meta.title if meta and meta.title else "").strip()
            md = trafilatura.extract(html, output_format="markdown", include_formatting=True, include_tables=True,
                                     include_links=False, include_images=False, url=r["final_url"]) or ""
            if not md or LEAK.search(title) or LEAK.search(md):  # leakage: no pages about datasets or ML
                continue
            n_pages += 1
            for head, text in sections(md, title):
                key = hashlib.sha1(text.encode()).hexdigest()
                if key in seen_text:  # navigation blocks and boilerplate repeated across pages
                    continue
                seen_text.add(key)
                rows.append({"site": name, "url": r["final_url"], "title": title, "heading": head,
                             "kind": kind_of(head), "text": text})
        counts[name] = {"pages": n_pages, "sections": sum(x["site"] == name for x in rows)}
        if (d / "crawl.json").exists():  # what the crawl did not fetch, and whether it stopped at its page cap
            c = json.loads((d / "crawl.json").read_text())
            counts[name]["crawl"] = {k: c[k] for k in ("pages", "cap_reached", "not_fetched_queue", "errors",
                                                       "skipped_robots", "skipped_documents", "finished") if k in c}
    with gzip.open(root / "docs.jsonl.gz", "wt") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    db = root / "kb.sqlite"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(title, heading, text, site UNINDEXED, url UNINDEXED, "
                "kind UNINDEXED, tokenize='porter unicode61')")
    con.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?)",
                    [(x["title"], x["heading"], x["text"], x["site"], x["url"], x["kind"]) for x in rows])
    con.commit()
    con.close()
    manifest = {"built": str(date.today()), "sites": counts, "sections": len(rows),
                "kinds": {k: sum(x["kind"] == k for x in rows) for k in ["symptom", "life_cycle", "management", "other"]}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------- search

class KB:
    """Read-only, thread-safe search over kb.sqlite."""

    def __init__(self, root: str | Path | None = None):
        self.path = kb_dir(root) / "kb.sqlite"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path}: build the knowledge base first (agml-agent-kb build)")
        self._local = threading.local()

    def _con(self) -> sqlite3.Connection:
        if not hasattr(self._local, "con"):
            self._local.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        return self._local.con

    def search(self, terms: list[str], crop: str = "", limit: int = 4, per_page: int = 2) -> list[dict]:
        """Sections about one of `terms` (tried in order), best first. A section counts as about the term if the
        term is in its page title or heading; sections that only mention it in their text are used only when no
        title or heading names it (quality check 2026-11-03: passing mentions were mostly tables and other topics).
        Then appearance sections, then the crop in the title or text (ordering only, nothing is filtered by crop),
        then BM25. Tables and names that describe no organism ("healthy", "Mg") are skipped."""
        out, seen_pages = [], {}
        crop = crop.lower().strip()
        for term in dict.fromkeys(t.strip() for t in terms if t and len(t.strip()) >= 4 and not NOT_NAMES.match(t.strip())):
            phrase = '"' + term.replace('"', " ").strip() + '"'
            try:
                rows = self._con().execute(
                    "SELECT title, heading, text, site, url, kind, bm25(docs, 8.0, 4.0, 1.0) FROM docs "
                    "WHERE docs MATCH ? ORDER BY bm25(docs, 8.0, 4.0, 1.0) LIMIT 60", (phrase,)).fetchall()
            except sqlite3.OperationalError:
                continue
            low = term.lower()
            rows = [r for r in rows if r[2].count("|") < 8]  # tables: lists of names, not descriptions
            named = [r for r in rows if low in (r[0] + " " + r[1]).lower()]
            rows = named or rows
            rows.sort(key=lambda r: (low not in r[0].lower(), r[5] != "symptom",
                                     bool(crop) and crop not in (r[0] + " " + r[2]).lower(), r[6]))
            for title, heading, text, site, url, kind, score in rows:
                if seen_pages.get(url, 0) >= per_page or any(o["text"] == text for o in out):
                    continue
                seen_pages[url] = seen_pages.get(url, 0) + 1
                out.append({"title": title, "heading": heading, "text": text, "site": site, "url": url, "kind": kind,
                            "term": term})
                if len(out) >= limit:
                    return out
            if out:  # the first term that finds anything wins (class name, then scientific name, then aliases)
                return out
        return out


def main() -> None:
    p = argparse.ArgumentParser(description="Local knowledge base of extension fact sheets (download, build, search).")
    p.add_argument("--kb", help="Knowledge-base folder (default $AGML_KB or ./kb).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="Download the sites in config/kb_sites.yaml (resumable).")
    c.add_argument("--site", action="append", help="Only this site (repeatable).")
    c.add_argument("--max-pages", type=int, help="Stop each site after this many pages (default: per-site max_pages).")
    sub.add_parser("build", help="Extract sections from the downloaded pages and index them.")
    s = sub.add_parser("search", help="Search the built index.")
    s.add_argument("query", nargs="+")
    s.add_argument("--crop", default="")
    s.add_argument("--limit", type=int, default=4)
    sub.add_parser("stats", help="Print the manifest.")
    a = p.parse_args()
    root = kb_dir(a.kb)
    if a.cmd == "crawl":
        crawl(root, a.site, a.max_pages)
    elif a.cmd == "build":
        print(json.dumps(build(root), indent=2))
    elif a.cmd == "search":
        for r in KB(root).search([" ".join(a.query)], a.crop, a.limit):
            print(f"[{r['site']}] {r['title']} > {r['heading']} ({r['kind']})\n  {r['text'][:400]}\n  {r['url']}\n")
    else:
        print((root / "manifest.json").read_text())


if __name__ == "__main__":
    main()
