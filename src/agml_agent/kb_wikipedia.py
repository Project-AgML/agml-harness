"""Wikipedia for the local knowledge base: a subset of the official English dump (pages-articles-multistream).

Which articles: chosen by a rule that does not look at benchmark class names (that would itself be a mild leak):
  - every article with a taxonomy infobox (Taxobox, Automatic taxobox, Speciesbox, Virusbox, ...): all organisms;
  - articles in plant-disease, plant-pathogen, pest, weed, crop, fruit, vegetable or cultivar categories.
Redirects to the kept articles become names ("Late blight" -> "Phytophthora infestans").

The dump is read in parallel by its multistream index (each bz2 stream holds about 100 pages). Output, under
raw/wikipedia/extract/:
    articles.jsonl.gz   one line per kept article: title, why kept, sections [[heading, text], ...]
    redirects.tsv.gz    redirect title <tab> target title, for every redirect in the dump (filtered at names build)
    extract.json        dump file, counts per rule, date
"""

from __future__ import annotations

import bz2
import gzip
import html
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

TAXOBOX = re.compile(r"\{\{\s*(taxobox|automatic taxobox|speciesbox|subspeciesbox|infraspeciesbox|virusbox|hybridbox|"
                     r"ichnobox|oobox|bacteriabox)\b", re.I)
AGRI_CATEGORY = re.compile(r"\[\[\s*category\s*:[^\]|]*(plant pathogens|plant diseases|plant disease|fungal plant|"
                           r"bacterial plant|plant viruses|viral plant|plant virus|oomycete|nematode|agricultural pest|"
                           r"insect pests|pests \(organism\)|pest insects|pest animals|weeds|noxious weeds|invasive plant|"
                           r"crops|fruits|vegetables|cultivars|cereals|legumes|spices|medicinal plants|"
                           r"physiological plant disorders|plant nutrition|nutrient deficiency)", re.I)
PAGE = re.compile(r"<page>(.*?)</page>", re.S)
SKIP_HEADINGS = re.compile(r"^(references?|external links|see also|further reading|notes|bibliography|sources|gallery|"
                           r"footnotes|citations|literature|works cited|in popular culture|etymology)$", re.I)
HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)


def _field(page: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", page, re.S)
    return html.unescape(m.group(1)) if m else ""


def _plain(wikitext: str) -> str:
    """Readable text of a section: templates, references, tables, files and markup removed."""
    import mwparserfromhell as mw

    t = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", wikitext, flags=re.S)
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)  # tables
    t = re.sub(r"\[\[(file|image|category):[^\[\]]*(\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", t, flags=re.I)
    code = mw.parse(t)
    for tpl in code.filter_templates(recursive=False):
        try:
            code.remove(tpl)
        except ValueError:
            pass
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _sections(wikitext: str, title: str) -> list[list[str]]:
    out, marks = [], [(m.start(), m.end(), m.group(2)) for m in HEADING.finditer(wikitext)]
    bounds = [(0, title)] + [(e, h) for _, e, h in marks]
    ends = [s for s, _, _ in marks] + [len(wikitext)]
    for (start, head), end in zip(bounds, ends):
        head = re.sub(r"'''?|\[\[|\]\]", "", head).strip()
        if SKIP_HEADINGS.match(head):
            continue
        text = _plain(wikitext[start:end])
        if len(text) >= 80:
            out.append([head or title, text])
    return out


def parse_pages(xml: str) -> tuple[list[dict], list[tuple[str, str]], dict]:
    """Kept articles, all redirects, and counts from a chunk of dump XML (whole <page> elements)."""
    arts, reds, n = [], [], {"pages": 0, "taxobox": 0, "category": 0}
    for m in PAGE.finditer(xml):
        page = m.group(1)
        if _field(page, "ns") != "0":
            continue
        n["pages"] += 1
        title = _field(page, "title")
        red = re.search(r'<redirect title="([^"]*)"', page)
        if red:
            reds.append((title, html.unescape(red.group(1))))
            continue
        text = _field(page, "text")
        why = "taxobox" if TAXOBOX.search(text) else "category" if AGRI_CATEGORY.search(text) else None
        if not why:
            continue
        n[why] += 1
        arts.append({"title": title, "why": why, "sections": _sections(text, title)})
    return arts, reds, n


def _read_streams(args) -> tuple[list[dict], list[tuple[str, str]], dict]:
    path, spans = args
    arts, reds, n = [], [], {"pages": 0, "taxobox": 0, "category": 0}
    with open(path, "rb") as f:
        for start, end in spans:
            f.seek(start)
            raw = f.read(end - start) if end else f.read()
            try:
                xml = bz2.decompress(raw).decode("utf-8", "ignore")
            except (OSError, ValueError):
                continue
            a, r, c = parse_pages(xml)
            arts += a
            reds += r
            for k in n:
                n[k] += c[k]
    return arts, reds, n


def extract(root: Path, workers: int = 8, batch: int = 200) -> dict:
    """Read the downloaded dump in parallel and write raw/wikipedia/extract/ (see the module docstring)."""
    d = root / "raw" / "wikipedia" / "files"
    dump = next(d.glob("*pages-articles-multistream.xml.bz2"))
    index = next(d.glob("*pages-articles-multistream-index.txt.bz2"))
    offsets = set()
    with bz2.open(index, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            offsets.add(int(line.split(":", 1)[0]))
    starts = sorted(offsets)
    spans = list(zip(starts, starts[1:] + [0]))
    jobs = [(str(dump), spans[i:i + batch]) for i in range(0, len(spans), batch)]
    out = root / "raw" / "wikipedia" / "extract"
    out.mkdir(parents=True, exist_ok=True)
    totals = {"pages": 0, "taxobox": 0, "category": 0, "redirects": 0, "sections": 0}
    with gzip.open(out / "articles.jsonl.gz.part", "wt") as fa, gzip.open(out / "redirects.tsv.gz.part", "wt") as fr, \
            ProcessPoolExecutor(workers) as ex:
        for arts, reds, n in ex.map(_read_streams, jobs, chunksize=1):
            for a in arts:
                fa.write(json.dumps(a, ensure_ascii=False) + "\n")
                totals["sections"] += len(a["sections"])
            for r in reds:
                fr.write(f"{r[0]}\t{r[1]}\n")
            totals["redirects"] += len(reds)
            for k in n:
                totals[k] += n[k]
    (out / "articles.jsonl.gz.part").replace(out / "articles.jsonl.gz")
    (out / "redirects.tsv.gz.part").replace(out / "redirects.tsv.gz")
    report = {"dump": dump.name, "extracted": str(date.today()), "rule": {"taxobox": TAXOBOX.pattern,
              "category": AGRI_CATEGORY.pattern}, **totals}
    (out / "extract.json").write_text(json.dumps(report, indent=1))
    return report


def articles(root: Path):
    with gzip.open(root / "raw" / "wikipedia" / "extract" / "articles.jsonl.gz", "rt") as f:
        for line in f:
            yield json.loads(line)


def url(title: str) -> str:
    from urllib.parse import quote

    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,'")
