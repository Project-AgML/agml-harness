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
import html as html_lib
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
# Site navigation that text extraction keeps on small pages (UC IPM photo pages): such sections are dropped.
NAVIGATION = re.compile(r"Notifications\s+Loading|Manage Pests\s+Home|Skip to (main )?content", re.I)
# Headings of sections that are not about the subject: reference lists, credits, publication details.
NOT_CONTENT = re.compile(r"^(references?|literature cited|bibliography|citations?|sources?|further reading|"
                         r"acknowledg|authors?|publication|resources|for more information|contacts?)\b", re.I)
SECTION_CHARS = 1500  # long sections are split at paragraph breaks into parts of about this size
MIN_SECTION_CHARS = 80


def site_label(name: str) -> str:
    """The site's display name (`label` in kb_sites.yaml), e.g. ucipm -> UC IPM."""
    return sites().get(name, {}).get("label", name)


def kb_dir(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("AGML_KB") or "kb")


_SITES = None


def sites() -> dict[str, dict]:
    global _SITES
    if _SITES is None:
        cfg = yaml.safe_load(files("agml_agent").joinpath("config", "kb_sites.yaml").read_text())
        _SITES = {s["name"]: s for s in cfg["sites"]}
    return _SITES


# ---------------------------------------------------------------- crawl

class Crawler:
    def __init__(self, name: str, cfg: dict, root: Path, max_pages: int | None = None):
        self.name, self.cfg = name, cfg
        self.dir = root / "raw" / name
        (self.dir / "html").mkdir(parents=True, exist_ok=True)
        self.max_pages = max_pages or cfg.get("max_pages", 5000)
        self.allow = re.compile(cfg.get("allow", "."))
        self.deny = re.compile(cfg["deny"]) if cfg.get("deny") else None
        self.follow_from = re.compile(cfg["follow_from"]) if cfg.get("follow_from") else None
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
            body = r.content if r is not None and r.ok else b""
            if body[:2] == b"\x1f\x8b":  # sitemap.xml.gz served as a gzip file
                body = gzip.decompress(body)
            for loc in LOC.findall(body.decode("utf-8", "ignore")):
                (maps if re.search(r"\.xml(\.gz)?$", loc) else urls).append(loc)
        return urls + list(self.cfg.get("start") or [])

    def files(self) -> dict:
        """Fixed downloads (`files:` in the config: data dumps such as a taxonomy or a regulation volume), streamed to
        raw/<site>/files/ as they are. Already downloaded files are kept."""
        (self.dir / "files").mkdir(exist_ok=True)
        with open(self.dir / "pages.jsonl", "a") as log:
            for url in self.cfg["files"]:
                if url in self.done and self.done[url].get("file"):
                    continue
                if not self.robots.can_fetch(HEADERS["User-Agent"], url):
                    self.skipped["robots"].add(url)
                    continue
                name = urlparse(url).path.rsplit("/", 1)[-1]
                rec = {"url": url, "status": None, "fetched": str(date.today()), "file": None}
                try:
                    with requests.get(url, headers=HEADERS, timeout=120, stream=True) as r:
                        rec["status"], rec["type"] = r.status_code, r.headers.get("content-type", "").split(";")[0]
                        if r.ok:
                            tmp = self.dir / "files" / (name + ".part")
                            with open(tmp, "wb") as f:
                                for chunk in r.iter_content(1 << 20):
                                    f.write(chunk)
                            tmp.rename(self.dir / "files" / name)
                            rec["file"], rec["bytes"] = name, (self.dir / "files" / name).stat().st_size
                except requests.RequestException as e:
                    rec["error"] = str(e)
                log.write(json.dumps(rec) + "\n")
                self.done[url] = rec
        report = {"site": self.name, "files": sum(bool(r.get("file")) for r in self.done.values()),
                  "bytes": sum(r.get("bytes", 0) for r in self.done.values()),
                  "errors": sum(not r.get("file") for r in self.done.values()),
                  "skipped_robots": len(self.skipped["robots"]), "finished": str(date.today())}
        (self.dir / "crawl.json").write_text(json.dumps(report, indent=1))
        return report

    def run(self) -> dict:
        if self.cfg.get("files"):
            return self.files()
        queue, seen = [], set(self.done)
        for u in self.seeds():
            if u not in seen and self.ok(u):
                queue.append(u)
                seen.add(u)
        fetched = sum(r["status"] == 200 for r in self.done.values())
        # pages already fetched are re-read from disk for their links, so an interrupted crawl continues where it was
        for r in list(self.done.values()):
            if r.get("sha1") and r.get("type", "text/html") == "text/html" and self._follows(r["url"]):
                html = gzip.decompress((self.dir / "html" / f"{r['sha1']}.html.gz").read_bytes()).decode("utf-8", "ignore")
                queue += [u for u in self._links(r["url"], html) if u not in seen and not seen.add(u)]
        with open(self.dir / "pages.jsonl", "a") as log:
            while queue and fetched < self.max_pages:
                url = queue.pop(0)
                r = self.get(url)
                rec = {"url": url, "status": r.status_code if r is not None else None, "fetched": str(date.today()),
                       "type": (r.headers.get("content-type", "") if r is not None else "").split(";")[0]}
                # by content: some sites send PDFs as application/force-download or octet-stream
                is_pdf = r is not None and r.ok and r.content[:5] == b"%PDF-"
                if r is not None and r.ok and (rec["type"] == "text/html" or (is_pdf and self.cfg.get("documents"))):
                    rec["final_url"], rec["sha1"] = r.url, hashlib.sha1(r.url.encode()).hexdigest()
                    if is_pdf:
                        rec["type"] = "application/pdf"
                    ext = "pdf" if is_pdf else "html"
                    (self.dir / "html" / f"{rec['sha1']}.{ext}.gz").write_bytes(gzip.compress(r.content))
                    fetched += 1
                    if not is_pdf and self._follows(url):
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

    def _follows(self, url: str) -> bool:
        return self.cfg.get("follow", True) and (self.follow_from is None or bool(self.follow_from.search(url)))

    def _links(self, base: str, html: str) -> list[str]:
        out = []
        for href in LINK.findall(html):
            u = urldefrag(urljoin(base, html_lib.unescape(href.strip())))[0]  # "&amp;" in hrefs is "&"
            if self.cfg.get("https"):  # the site links both http:// and https:// copies of its pages
                u = re.sub(r"^http://", "https://", u)
            if urlparse(u).scheme not in ("http", "https"):
                continue
            if re.search(r"\.(pdf|docx?|xlsx?|pptx?)$", u, re.I):  # documents: PDFs where the site config says
                if self.cfg.get("documents") and re.search(r"\.pdf$", u, re.I) and self.ok(u):  # so, else listed
                    out.append(u)
                elif self.allow.search(u):
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


# title-page lines that name the publisher or series, not the subject
PDF_BOILERPLATE = re.compile(r"acknowledg|organi[sz]ation|data sheets?|fiches|prepared by|contract|bulletin|publishing|quarantine "
                             r"pest|^eppo\b|^oepp|standards?$|^pm \d|diagnostics?$|guide$|^tnau|university", re.I)
ENGLISH = re.compile(r"\b(the|of|and|in|to|is|are|for|with|on|by|from|leaves|plants?)\b", re.I)


def english(text: str) -> bool:
    """At least 1 in 12 words is a common English word (Tamil pages, also those in legacy fonts that extract as Latin
    gibberish, fall below)."""
    words = text.split()
    return bool(words) and len(ENGLISH.findall(text)) / len(words) >= 1 / 12


PDF_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z ,&/()'-]{2,70})$")


def pdf_markdown(data: bytes) -> tuple[str, str]:
    """(title, text with '# ' headings) of a PDF, via poppler's pdftotext. Headings: short lines in capitals or
    numbered ("3.2 Symptoms"); running headers and footers (lines on a third of the pages or more) are dropped and
    wrapped lines are joined into paragraphs."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", f.name, "-"], capture_output=True, timeout=120)
    pages = out.stdout.decode("utf-8", "ignore").split("\f")
    count = {}
    for pg in pages:
        for line in {x.strip() for x in pg.splitlines() if x.strip()}:
            count[line] = count.get(line, 0) + 1
    repeated = {x for x, n in count.items() if len(pages) >= 3 and n >= max(2, len(pages) / 3)}
    lines, before = [], []  # before: the lines ahead of the first heading (publisher, series, then the subject)
    for pg in pages:
        for raw in pg.splitlines():
            x = raw.strip()
            if not x or x in repeated or re.fullmatch(r"[\d\s/|-]+|page \d+.*", x, re.I):
                lines.append("")
                continue
            m = PDF_HEADING.match(x)
            if m and (x.isupper() or re.match(r"^\d", x)) and not x.endswith(".") and len(x.split()) <= 10 \
                    and not PDF_BOILERPLATE.search(x):
                if before is not None:  # the first heading: what came before it is the title page
                    lines, before = [], [y for y in before if 3 <= len(y) <= 120 and not PDF_BOILERPLATE.search(y)
                                         and re.match(r"[A-Z0-9*]", y)
                                         and english(y + " the") and not re.search(r"[А-я]", y)]
                    # the subject, else a running header naming it (classic EPPO datasheets), else the heading
                    heads = sorted((y for y in repeated if not PDF_BOILERPLATE.search(y) and 3 <= len(y) <= 120
                                    and not re.search(r"\d{2,}|[А-я]", y)), key=lambda y: -count[y])
                    title = before[-1] if before else heads[0] if heads else m.group(1).strip()
                    before = None
                lines += ["", "# " + m.group(1).strip().capitalize(), ""]
            else:
                if before is not None:
                    before.append(x)
                lines.append(x)
    if before is not None:  # no headings: the whole text, titled by its first line
        title = next((y for y in before if len(y) >= 3), "")[:120]
    paras, buf = [], []
    for x in lines + [""]:
        if x.startswith("# ") or not x:
            if buf:
                paras.append(re.sub(r"(\w)- (\w)", r"\1\2", " ".join(buf)))
                buf = []
            if x:
                paras.append(x)
        else:
            buf.append(x)
    return title, "\n".join(paras)


def cfr_sections(zip_path: Path, parts=("51", "52", "810")) -> list[dict]:
    """One row per CFR section of the given parts of a GovInfo CFR title zip: title = the subpart heading (for
    Part 51, "United States Standards for Grades of Apples"), heading = section number and subject."""
    import zipfile

    from lxml import etree

    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for member in sorted(n for n in z.namelist() if n.endswith(".xml")):
            tree = etree.fromstring(z.read(member), etree.XMLParser(recover=True, huge_tree=True))
            for part in tree.iter("PART"):
                hd = " ".join((part.findtext("HD") or "").split())
                num = re.search(r"PART\s+(\d+)", hd)
                if not num or num.group(1) not in parts:
                    continue
                for sec in part.iter("SECTION"):
                    sub = next((a for a in sec.iterancestors() if a.tag == "SUBPART"), None)
                    sub_hd = " ".join((sub.findtext("HD") if sub is not None else hd).split()) if sub is not None else hd
                    sub_hd = re.sub(r"^Subpart\s*[A-Z]*\s*[—-]+\s*", "", sub_hd)
                    no = " ".join((sec.findtext("SECTNO") or "").split())
                    subj = " ".join((sec.findtext("SUBJECT") or "").split())
                    text = "\n".join(" ".join("".join(p.itertext()).split()) for p in sec.iter("P", "FP"))
                    if len(text) >= MIN_SECTION_CHARS:
                        rows.append({"title": sub_hd, "heading": f"{no} {subj}".strip(), "text": text,
                                     "url": f"https://www.ecfr.gov/current/title-7/section-{no.strip('§ ').strip()}"})
    return rows


# GBIF Backbone descriptions kept (English): what the organism looks like and does; not specimens examined, etymology,
# distribution, conservation, or type material
GBIF_TYPES = {"description": "symptom", "diagnosis": "symptom", "biology_ecology": "life_cycle",
              "food_feeding": "life_cycle", "breeding": "life_cycle", "activity": "life_cycle", "notes": "other",
              "discussion": "other", "general": "other", "morphology": "symptom", "biology": "life_cycle"}


def gbif_descriptions(zip_path: Path, names_db: Path, site: str) -> list[dict]:
    """English descriptions from the GBIF Backbone archive (Description.tsv; from treatments in its source checklists),
    titled by the taxon's scientific name from names.sqlite; one section per description, long ones split."""
    import csv
    import io
    import zipfile

    if not names_db.exists():
        return []
    con = sqlite3.connect(f"file:{names_db}?mode=ro", uri=True)
    csv.field_size_limit(1 << 30)
    out, seen = [], set()
    with zipfile.ZipFile(zip_path) as z, z.open("Description.tsv") as f:
        for t in csv.DictReader(io.TextIOWrapper(f, "utf-8", errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE):
            kind = GBIF_TYPES.get((t.get("type") or "").strip().lower())
            text = (t.get("description") or "").strip()
            if not kind or (t.get("language") or "").lower() not in ("en", "eng") or len(text) < MIN_SECTION_CHARS:
                continue
            key = hashlib.sha1(text.encode()).hexdigest()
            if key in seen or LEAK.search(text):
                continue
            seen.add(key)
            row = con.execute("SELECT name, kingdom, family FROM names WHERE source = 'gbif' AND id = ? LIMIT 1",
                              (t["taxonID"],)).fetchone()
            if not row:
                continue
            heading = t["type"].strip().replace("_", " ").capitalize()
            for head, part in sections(text, heading):
                out.append({"site": site, "url": f"https://www.gbif.org/species/{t['taxonID']}", "title": row[0],
                            "heading": head, "kind": kind, "text": part})
    con.close()
    return out


def build(root: Path) -> dict:
    import trafilatura

    rows, counts = [], {}
    for name, cfg in sites().items():
        d = root / "raw" / name
        if not (d / "pages.jsonl").exists() or cfg.get("parse", "html") in ("names_plants", "names_eppo"):
            continue
        if cfg.get("parse") == "names_gbif":  # names go to names.sqlite; the backbone's descriptions are text
            n0 = len(rows)
            for f in sorted((d / "files").glob("*.zip")):
                rows += gbif_descriptions(f, root / "names.sqlite", name)
            counts[name] = {"pages": len({r["url"] for r in rows[n0:]}), "sections": len(rows) - n0}
            continue
        if cfg.get("parse") == "cfr":
            n0 = len(rows)
            for f in sorted((d / "files").glob("*.zip")):
                for x in cfr_sections(f):
                    for i, (head, text) in enumerate(sections(x["text"], x["heading"])):
                        rows.append({"site": name, "url": x["url"], "title": x["title"], "heading": head,
                                     "kind": kind_of(head), "text": text})
            counts[name] = {"pages": len({r["url"] for r in rows[n0:]}), "sections": len(rows) - n0}
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
            if r.get("type") == "application/pdf":
                try:
                    title, md = pdf_markdown(gzip.decompress((d / "html" / f"{r['sha1']}.pdf.gz").read_bytes()))
                except Exception:  # damaged or scanned PDF
                    continue
            else:
                html = gzip.decompress((d / "html" / f"{r['sha1']}.html.gz").read_bytes()).decode("utf-8", "ignore")
                meta = trafilatura.extract_metadata(html)
                title = (meta.title if meta and meta.title else "").strip()
                md = trafilatura.extract(html, output_format="markdown", include_formatting=True, include_tables=True,
                                         include_links=False, include_images=False, url=r["final_url"]) or ""
            if not md or LEAK.search(title) or LEAK.search(md):  # leakage: no pages about datasets or ML
                continue
            if not english(md):  # Tamil and other non-English pages
                continue
            n_pages += 1
            for head, text in sections(md, title):
                if NAVIGATION.search(text[:300]):
                    continue
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


# GBIF treatments describe specimens under a microscope, cultures on plates and nomenclature: nothing a photo shows
# (offline check 2026-11-09: "Colonies on PDA ... cottony", typification notes, host lists under "Discussion")
GBIF_NOT_VISIBLE = re.compile(r"\b(PDA|SNA|MEA|OA)\b|colonies (on|reach)|in culture|culture characteristics|holotype|paratype|"
                              r"lectotype|neotype|typif|herbarium|specimens? examined|\bArt\. \d|µm.{0,40}µm.{0,40}µm", re.I)
# EPPO standards' laboratory and inspection procedures (the diagnostic protocols' symptom sections stay)
LAB_METHOD = re.compile(r"extraction|isolation|culture|\bpcr\b|elisa|molecular|sampling|reagent|primer|traps?\b|"
                        r"inspection|reference material|procedure|test(s|ing)?\b|equipment|consignment|treatment|"
                        r"acknowledg|references?\b|appendix|annex|authors?\b", re.I)
# Feedipedia sections about feeding animals (the species description sections stay)
FEED_USE = re.compile(r"\b(goats?|cattle|sheep|pigs?|poultry|chickens?|rabbits?|fish|horses?|ruminants?|buffalo|camels?|"
                      r"ducks?|turkeys?|livestock|nutritional|nutrition|feeding|forage|silage|hay|potential constraints|"
                      r"processing|digestibility)\b", re.I)


# ---------------------------------------------------------------- names table

NAMES_SCHEMA = ("CREATE TABLE names (source TEXT, id TEXT, accepted_id TEXT, name TEXT, norm TEXT, kind TEXT, "
                "lang TEXT, rank TEXT, kingdom TEXT, family TEXT)")


def norm_name(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def _gbif_rows(zip_path: Path):
    """GBIF Backbone (Darwin Core archive): every name, accepted or synonym (canonical form, no authors), then every
    vernacular name. A synonym's accepted_id points to its accepted taxon."""
    import csv
    import io
    import zipfile

    csv.field_size_limit(1 << 30)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("Taxon.tsv") as f:
            r = csv.DictReader(io.TextIOWrapper(f, "utf-8", errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE)
            for t in r:
                name = t.get("canonicalName") or t.get("scientificName") or ""
                if not name:
                    continue
                syn = "synonym" in (t.get("taxonomicStatus") or "").lower()
                yield ("gbif", t["taxonID"], t.get("acceptedNameUsageID") or t["taxonID"] if syn else t["taxonID"], name,
                       norm_name(name), "synonym" if syn else "scientific", "", (t.get("taxonRank") or "").lower(),
                       t.get("kingdom") or "", t.get("family") or "")
        with z.open("VernacularName.tsv") as f:
            r = csv.DictReader(io.TextIOWrapper(f, "utf-8", errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE)
            for t in r:
                name = t.get("vernacularName") or ""
                if name:
                    yield ("gbif", t["taxonID"], t["taxonID"], name, norm_name(name), "common",
                           (t.get("language") or "").lower(), "", "", "")


def _plants_rows(path: Path):
    """USDA PLANTS checklist: accepted rows (no synonym symbol) carry the common name; synonym rows point to the
    accepted symbol. Scientific names lose their authors (the epithets are the lower-case words after the genus)."""
    import csv

    def canonical(sci: str) -> str:
        w = sci.split()
        out = w[:1]
        for x in w[1:]:
            if x in ("var.", "ssp.", "subsp.", "f.", "×", "x") or (x[:1].islower() and x.isalpha()):
                out.append(x)
            elif x.startswith("(") or x[:1].isupper():
                if len(out) > 1 and out[-1] in ("var.", "ssp.", "subsp.", "f."):
                    continue
                break
        return " ".join(out)

    with open(path, encoding="utf-8", errors="ignore") as f:
        for t in csv.DictReader(f):
            sym, syn = t.get("Symbol", ""), t.get("Synonym Symbol", "")
            name = canonical(t.get("Scientific Name with Author", ""))
            if not name:
                continue
            fam = t.get("Family", "")
            if syn:
                yield ("usda_plants", syn, sym, name, norm_name(name), "synonym", "", "", "Plantae", fam)
            else:
                yield ("usda_plants", sym, sym, name, norm_name(name), "scientific", "", "", "Plantae", fam)
                if t.get("Common Name"):
                    yield ("usda_plants", sym, sym, t["Common Name"], norm_name(t["Common Name"]), "common", "en", "",
                           "Plantae", fam)


LANGS = {"English": "en", "French": "fr", "German": "de", "Spanish": "es", "Portuguese": "pt", "Italian": "it",
         "Dutch": "nl", "Russian": "ru", "Chinese": "zh", "Japanese": "ja"}


def _eppo_rows(d: Path):
    """EPPO taxon pages: preferred name, other scientific names and common names (with language), kingdom, family."""
    td = re.compile(r"<tr>\s*<td>(.*?)</td>\s*<td[^>]*>(.*?)</td>", re.S)
    for line in (d / "pages.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("sha1"):
            continue
        h = gzip.decompress((d / "html" / f"{r['sha1']}.html.gz").read_bytes()).decode("utf-8", "ignore")
        code = r["url"].rstrip("/").rsplit("/", 1)[-1]
        txt = lambda x: " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", x)).split())
        pref = re.search(r"<strong>Preferred name:</strong>(.*?)</li>", h, re.S)
        if not pref:
            continue
        rank = lambda k: (re.search(rf"{k}\s*<span[^>]*>\s*<a[^>]*>(.*?)</a>", h, re.S) or [None, ""])[1]
        kingdom, family = txt(rank("Kingdom")), txt(rank("Family"))
        name = txt(pref.group(1))
        yield ("eppo", code, code, name, norm_name(name), "scientific", "", "", kingdom, family)
        other = re.search(r"Other scientific names</span>.*?<tbody>(.*?)</tbody>", h, re.S)
        for n, _ in td.findall(other.group(1) if other else ""):
            yield ("eppo", code, code, txt(n), norm_name(txt(n)), "synonym", "", "", kingdom, family)
        common = re.search(r'id="tbcommon".*?<tbody>(.*?)</tbody>', h, re.S)
        for n, lang in td.findall(common.group(1) if common else ""):
            yield ("eppo", code, code, txt(n), norm_name(txt(n)), "common", LANGS.get(txt(lang), txt(lang).lower()),
                   "", kingdom, family)


def build_names(root: Path) -> dict:
    """names.sqlite: one table of names from the GBIF Backbone, USDA PLANTS and EPPO taxon pages, indexed by
    normalised name, so a class name can be resolved to its accepted scientific name, synonyms and common names."""
    db = root / "names.sqlite"
    tmp = root / "names.sqlite.part"
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute(NAMES_SCHEMA)
    counts = {}
    for name, cfg in sites().items():
        d = root / "raw" / name
        kind = cfg.get("parse", "")
        if not kind.startswith("names") or not (d / "pages.jsonl").exists():
            continue
        if kind == "names_gbif":
            rows = (x for f in sorted((d / "files").glob("*.zip")) for x in _gbif_rows(f))
        elif kind == "names_plants":
            rows = (x for f in sorted((d / "files").glob("*.txt")) for x in _plants_rows(f))
        else:
            rows = _eppo_rows(d)
        n, batch = 0, []
        for x in rows:
            batch.append(x)
            if len(batch) >= 100000:
                con.executemany("INSERT INTO names VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
                n += len(batch)
                batch = []
        con.executemany("INSERT INTO names VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
        counts[name] = n + len(batch)
        con.commit()
    con.execute("CREATE INDEX names_norm ON names(norm)")
    con.execute("CREATE INDEX names_id ON names(source, id)")
    con.execute("CREATE INDEX names_acc ON names(source, accepted_id)")
    con.commit()
    con.close()
    tmp.replace(db)
    m = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    m["names"] = {"built": str(date.today()), "rows": counts}
    (root / "manifest.json").write_text(json.dumps(m, indent=2))
    return m["names"]


class Names:
    """Resolve a class name to its taxon: accepted scientific name, synonyms, English common names, kingdom and
    family. EPPO first (pests and diseases, disease common names), then USDA PLANTS, then GBIF."""

    ORDER = ("eppo", "usda_plants", "gbif")

    def __init__(self, root: str | Path | None = None):
        self.path = kb_dir(root) / "names.sqlite"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path}: build the names table first (agml-agent-kb names)")
        self._local = threading.local()

    def _con(self) -> sqlite3.Connection:
        if not hasattr(self._local, "con"):
            self._local.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        return self._local.con

    def lookup(self, name: str, max_names: int = 8, crop: str = "") -> dict | None:
        """The taxon a class name stands for. Tried as given, then with its host ("<name> of <crop>", "<crop> <name>",
        and "leaf rust of coffee" for "coffee leaf rust": EPPO and GBIF name diseases that way). A common name shared
        by several taxa ("powdery mildew") resolves only if one of them names the crop; otherwise None, as a wrong
        scientific name would mislead more than none."""
        n, c = norm_name(name), norm_name(crop)
        if len(n) < 4 or NOT_NAMES.match(n):
            return None
        from agml_agent.symptom_search import crop_names

        tries = [n]
        for c2 in [norm_name(x) for x in crop_names(crop)] if c else []:  # grape, grapevine, vine ...
            rest = " ".join(w for w in n.split() if w not in c2.split())
            tries += [f"{n} of {c2}", f"{c2} {n}"] + ([f"{rest} of {c2}"] if rest and rest != n else [])
        rank = {s: i for i, s in enumerate(self.ORDER)}
        for q in dict.fromkeys(tries):
            hits = self._con().execute("SELECT source, accepted_id, kind FROM names WHERE norm = ?", (q,)).fetchall()
            if not hits:
                continue
            best = min(rank.get(h[0], 9) for h in hits)
            hits = [h for h in hits if rank.get(h[0], 9) == best]
            src = hits[0][0]
            sci_hits = [h for h in hits if h[2] in ("scientific", "synonym")]
            taxa = list(dict.fromkeys(self._accepted(src, h[1]) for h in (sci_hits or hits)))
            if len(taxa) > 1 and not sci_hits:
                on_crop = [t for t in taxa if c and self._names_crop(src, t, c)]
                if len(on_crop) != 1:
                    return None  # ambiguous common name
                taxa = on_crop
            return self._taxon(src, taxa[0], name, max_names)
        return None

    def _accepted(self, src: str, tid: str) -> str:
        """Follow a synonym (also one a vernacular name is attached to) to its accepted taxon."""
        for _ in range(3):
            row = self._con().execute("SELECT accepted_id FROM names WHERE source = ? AND id = ? AND kind = 'synonym' "
                                      "LIMIT 1", (src, tid)).fetchone()
            if not row or row[0] == tid:
                return tid
            tid = row[0]
        return tid

    def _names_crop(self, src: str, tid: str, crop: str) -> bool:
        return any(crop in norm_name(r[0]) for r in self._con().execute(
            "SELECT name FROM names WHERE source = ? AND id = ? AND kind = 'common' UNION ALL "
            "SELECT name FROM names WHERE source = ? AND accepted_id = ? AND kind = 'common'", (src, tid, src, tid)))

    def _taxon(self, src: str, acc: str, name: str, max_names: int) -> dict:
        ids = [acc] + [r[0] for r in self._con().execute(
            "SELECT id FROM names WHERE source = ? AND accepted_id = ? AND kind = 'synonym'", (src, acc))]
        marks = ",".join("?" * len(ids))
        rows = self._con().execute(
            f"SELECT name, kind, lang, kingdom, family, id FROM names WHERE source = ? AND id IN ({marks}) UNION "
            f"SELECT name, kind, lang, kingdom, family, id FROM names WHERE source = ? AND accepted_id = ?",
            (src, *ids, src, acc)).fetchall()
        sci = next((r[0] for r in rows if r[1] == "scientific" and r[5] == acc), None) or \
            next((r[0] for r in rows if r[1] == "scientific"), name)
        kingdom = next((r[3] for r in rows if r[3]), "")
        family = next((r[4] for r in rows if r[4]), "")
        syn = list(dict.fromkeys(r[0] for r in rows if r[1] == "synonym" and r[0] != sci))[:max_names]
        en = [r[0] for r in rows if r[1] == "common" and r[2] in ("en", "eng")]
        common = list(dict.fromkeys(x.lower() for x in (en or [r[0] for r in rows if r[1] == "common" and not r[2]])))
        rank = self._con().execute("SELECT rank FROM names WHERE source = ? AND id = ? AND kind = 'scientific' LIMIT 1",
                                   (src, acc)).fetchone()
        return {"name": name, "scientific_name": sci, "synonyms": syn, "common_names": common[:max_names],
                "kingdom": kingdom, "family": family, "source": src, "id": acc, "rank": (rank or [""])[0]}


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
        for term in dict.fromkeys(t.strip() for t in terms if t and len(t.strip()) >= 4 and not NOT_NAMES.match(t.strip())):
            phrase = '"' + term.replace('"', " ").strip() + '"'
            try:
                rows = self._con().execute(
                    "SELECT title, heading, text, site, url, kind, bm25(docs, 8.0, 4.0, 1.0) FROM docs "
                    "WHERE docs MATCH ? ORDER BY bm25(docs, 8.0, 4.0, 1.0) LIMIT 60", (phrase,)).fetchall()
            except sqlite3.OperationalError:
                continue
            low = term.lower()
            rows = [r for r in rows if r[2].count("|") < 8 and not NOT_CONTENT.match(r[1].strip())  # tables, references
                    and not (r[3] == "gbif_backbone" and (GBIF_NOT_VISIBLE.search(r[2]) or r[1] in ("Discussion", "Notes")))
                    and not (r[3] == "feedipedia" and FEED_USE.search(r[1]))
                    and not (r[3] == "eppo_docs" and LAB_METHOD.search(r[1]))]
            # the whole term, not inside another word ("longan" is not "Aulacaspis longanae")
            whole = re.compile(rf"(?<![a-z]){re.escape(low)}(?![a-z])")
            named = [r for r in rows if whole.search((r[0] + " " + r[1]).lower())]
            # without a title or heading naming it, the exact term must be in the text (stemming matches "Thumbe" to
            # "thumb") and so must the crop: passing mentions elsewhere were mostly off topic (2026-11-04: "bruised" ->
            # Jimsonweed, "Leaf Rot" -> spinach watering, "pink disease" -> a forage grass)
            # (GBIF treatments of insects often name a crop as a host in passing: title matches only)
            rows = named or [r for r in rows if r[3] != "gbif_backbone" and whole.search(r[2].lower()) and crop
                             and crop.lower() in r[2].lower()]
            on_crop = self._pages_mentioning({r[4] for r in rows}, crop)
            rows.sort(key=lambda r: (low not in r[0].lower(), r[4] not in on_crop, r[5] != "symptom", r[6]))
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


    def _pages_mentioning(self, urls: set[str], crop: str) -> set[str]:
        """The pages (any of their sections) that mention the crop or one of its other names."""
        if not crop or not urls:
            return set()
        from agml_agent.symptom_search import crop_names
        q = " OR ".join('"' + c.replace('"', " ") + '"' for c in crop_names(crop))
        marks = ",".join("?" * len(urls))
        try:
            return {u for (u,) in self._con().execute(
                f"SELECT DISTINCT url FROM docs WHERE docs MATCH ? AND url IN ({marks})", (q, *urls))}
        except sqlite3.OperationalError:
            return set()


def main() -> None:
    p = argparse.ArgumentParser(description="Local knowledge base of extension fact sheets (download, build, search).")
    p.add_argument("--kb", help="Knowledge-base folder (default $AGML_KB or ./kb).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="Download the sites in config/kb_sites.yaml (resumable).")
    c.add_argument("--site", action="append", help="Only this site (repeatable).")
    c.add_argument("--max-pages", type=int, help="Stop each site after this many pages (default: per-site max_pages).")
    sub.add_parser("build", help="Extract sections from the downloaded pages and index them.")
    sub.add_parser("names", help="Build the names table (names.sqlite) from the downloaded name sources.")
    nl = sub.add_parser("name", help="Look a name up in the names table.")
    nl.add_argument("query", nargs="+")
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
    elif a.cmd == "names":
        print(json.dumps(build_names(root), indent=2))
    elif a.cmd == "name":
        print(json.dumps(Names(root).lookup(" ".join(a.query)), indent=2, ensure_ascii=False))
    elif a.cmd == "search":
        for r in KB(root).search([" ".join(a.query)], a.crop, a.limit):
            print(f"[{r['site']}] {r['title']} > {r['heading']} ({r['kind']})\n  {r['text'][:400]}\n  {r['url']}\n")
    else:
        print((root / "manifest.json").read_text())


if __name__ == "__main__":
    main()
