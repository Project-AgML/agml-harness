import gzip
import hashlib
import json

import pytest

from agml_agent import bench, kb

PAGE = """<html><head><title>Coffee Leaf Rust</title></head><body><article>
<h1>Coffee Leaf Rust</h1><p>Coffee leaf rust is the most damaging disease of coffee worldwide, found wherever arabica
coffee is grown, and it has caused large losses in many regions over the past century.</p>
<h2>Symptoms</h2><p>Small pale yellow spots on the upper leaf surface grow into orange powdery pustules on the underside
of coffee leaves; heavily infected leaves drop early, and repeated defoliation weakens the tree over the seasons.</p>
<h2>Management</h2><p>Plant resistant varieties, keep shade moderate, and apply copper fungicides before the rainy
season when the disease starts to spread from older leaves.</p>
</article></body></html>"""


@pytest.fixture
def tiny_kb(tmp_path, monkeypatch):
    """One downloaded page under raw/, as `crawl` leaves it."""
    monkeypatch.setattr(kb, "sites", lambda: {"ucipm": {}})
    d = tmp_path / "raw" / "ucipm"
    (d / "html").mkdir(parents=True)
    url = "https://ipm.example/coffee-rust"
    sha = hashlib.sha1(url.encode()).hexdigest()
    (d / "html" / f"{sha}.html.gz").write_bytes(gzip.compress(PAGE.encode()))
    (d / "pages.jsonl").write_text(json.dumps({"url": url, "final_url": url, "status": 200, "sha1": sha}) + "\n")
    return tmp_path


def test_build_splits_pages_into_tagged_sections_and_search_finds_them(tiny_kb):
    manifest = kb.build(tiny_kb)
    assert manifest["sites"]["ucipm"]["pages"] == 1 and manifest["kinds"]["symptom"] >= 1
    hits = kb.KB(tiny_kb).search(["coffee leaf rust"], crop="coffee")
    assert hits[0]["heading"] == "Coffee Leaf Rust" or hits[0]["kind"] == "symptom"
    assert any(h["kind"] == "symptom" and "orange powdery pustules" in h["text"] for h in hits)
    assert kb.KB(tiny_kb).search(["Hemileia vastatrix", "coffee leaf rust"])  # later terms are tried when earlier find nothing


def test_sections_split_long_text_at_paragraphs():
    md = "# Title\n" + "\n".join(["word " * 60] * 10)
    parts = kb.sections(md, "T")
    assert len(parts) > 1 and all(len(t) <= kb.SECTION_CHARS for _, t in parts) and parts[0][0] == "Title"


def test_kb_results_come_first_and_blocked_sites_are_never_shown(tiny_kb, monkeypatch):
    kb.build(tiny_kb)
    monkeypatch.setattr(bench, "KB_INDEX", kb.KB(tiny_kb))
    monkeypatch.setattr(bench.cache, "CACHE", None)
    import agml_agent.retrieve as r
    monkeypatch.setattr(r, "retrieve", lambda *a, **k: {"plantvillage": [{"excerpt_text": "x", "source_url": "https://plantvillage.psu.edu/a"}],
                                                       "eol": [{"excerpt_text": "Hemileia vastatrix", "source_url": "https://eol.org/1"}]})
    found = bench._lookup({"name": "coffee leaf rust"}, "coffee", "disease")
    assert list(found) == ["local_kb", "eol"]


def test_options_are_looked_up_first_then_own_candidates():
    got = bench.with_options([{"name": "bean rust", "scientific_name": "Uromyces"}, "angular leaf spot"],
                             ["Bean Rust", "Anthracnose", "Mosaic", "None of the above", "I don't know"])
    assert [c["name"] for c in got] == ["bean rust", "Anthracnose", "Mosaic", "angular leaf spot"]
    assert got[0]["scientific_name"] == "Uromyces"  # the model's details for an option are kept


def test_viewer_checks_relevance_and_use_of_kb_sections():
    from agml_agent import bench_view
    trace = [{"role": "assistant", "reasoning": "dark spots", "text": "",
              "tool_calls": [{"name": "retrieve", "arguments": {"description": "dark spots", "crop": "coffee"}}]},
             {"role": "tool", "result": "## coffee leaf rust\n[local_kb] Coffee Leaf Rust > Symptoms: orange powdery "
                                        "pustules on the underside of coffee leaves (source: https://x)\n\n[names: eppo] "
                                        "Preferred name: Hemileia vastatrix (source: https://e)"},
             {"role": "assistant", "reasoning": "I see orange powdery pustules on the underside of the leaves", "text": "rust"}]
    rows = bench_view.kb_sections(trace, "Coffee leaf rust")
    assert len(rows) == 1 and rows[0]["page is the candidate"] and rows[0]["appearance section"] and rows[0]["names crop"]
    assert rows[0]["correct class"] and rows[0]["used"]
    assert bench_view.source_use(trace) == (["local_kb", "names"], ["local_kb"])


def _names_kb(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "sites", lambda: {"usda_plants": {"parse": "names_plants"}})
    d = tmp_path / "raw" / "usda_plants"
    (d / "files").mkdir(parents=True)
    (d / "pages.jsonl").write_text('{"url": "x", "file": "plantlst.txt"}\n')
    (d / "files" / "plantlst.txt").write_text(
        '"Symbol","Synonym Symbol","Scientific Name with Author","Common Name","Family"\n'
        '"SOLY2","","Solanum lycopersicum L.","garden tomato","Solanaceae"\n'
        '"SOLY2","LYES","Lycopersicon esculentum Mill.","","Solanaceae"\n')
    kb.build_names(tmp_path)
    return tmp_path


def test_names_table_resolves_synonyms_and_common_names(tmp_path, monkeypatch):
    root = _names_kb(tmp_path, monkeypatch)
    got = kb.Names(root).lookup("Lycopersicon esculentum")
    assert got["scientific_name"] == "Solanum lycopersicum" and got["common_names"] == ["garden tomato"]
    assert got["family"] == "Solanaceae" and "Lycopersicon esculentum" in got["synonyms"]
    assert kb.Names(root).lookup("healthy") is None and kb.Names(root).lookup("unknown thing") is None


def test_kb_only_lookup_uses_local_names_and_kb_only(tiny_kb, tmp_path, monkeypatch):
    kb.build(tiny_kb)
    monkeypatch.setattr(bench, "KB_INDEX", kb.KB(tiny_kb))
    monkeypatch.setattr(bench, "KB_ONLY", True)
    monkeypatch.setattr(bench, "NAMES", None)
    import agml_agent.retrieve as r
    monkeypatch.setattr(r, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no live lookups")))
    found = bench._lookup({"name": "coffee leaf rust"}, "coffee", "disease")
    assert list(found) == ["local_kb"]


def test_block_network_allows_only_this_machine(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    bench._block_network()
    s = socket.socket()
    with pytest.raises(RuntimeError, match="kb-only"):
        s.connect(("93.184.216.34", 80))
    s.close()


def test_cfr_sections_are_titled_by_subpart(tmp_path):
    import zipfile
    xml = """<CFRDOC><PART><HD>PART 51—FRESH FRUITS, VEGETABLES AND OTHER PRODUCTS</HD><SUBPART>
<HD SOURCE="HED">Subpart—United States Standards for Grades of Apples</HD>
<SECTION><SECTNO>§ 51.301</SECTNO><SUBJECT>U.S. Extra Fancy.</SUBJECT><P>U.S. Extra Fancy consists of apples of one
variety which are mature but not overripe, carefully hand picked, clean, fairly well formed, free from decay.</P></SECTION>
</SUBPART></PART><PART><HD>PART 1—ADMINISTRATIVE</HD><SECTION><SECTNO>§ 1.1</SECTNO><SUBJECT>Scope.</SUBJECT>
<P>Not a grade standard, so not indexed even though this paragraph is long enough to be a section here.</P></SECTION></PART></CFRDOC>"""
    with zipfile.ZipFile(tmp_path / "t.zip", "w") as z:
        z.writestr("CFR-2025-title7-vol2.xml", xml)
    rows = kb.cfr_sections(tmp_path / "t.zip")
    assert len(rows) == 1 and rows[0]["title"] == "United States Standards for Grades of Apples"
    assert rows[0]["heading"].startswith("§ 51.301") and "hand picked" in rows[0]["text"]


def test_pdf_markdown_finds_headings_and_drops_running_headers():
    import shutil, subprocess
    if not shutil.which("pdftotext"):
        pytest.skip("poppler not installed")
    fake = "EPPO Bulletin 2020\n\nSYMPTOMS\nLeaves show small brown spots that\nenlarge and merge.\n\fEPPO Bulletin 2020\n\nHOSTS\nPear.\n\fEPPO Bulletin 2020\nend\n"
    import agml_agent.kb as k
    real = subprocess.run
    monkey = lambda *a, **kw: type("R", (), {"stdout": fake.encode()})()
    try:
        subprocess.run = monkey
        title, md = k.pdf_markdown(b"%PDF-")
    finally:
        subprocess.run = real
    assert "# Symptoms" in md and "small brown spots that enlarge and merge." in md and "EPPO Bulletin" not in md


def test_kb_text_names_its_site_and_viewer_lists_sources(tiny_kb, monkeypatch):
    kb.build(tiny_kb)
    monkeypatch.setattr(kb, "_SITES", None)
    monkeypatch.setattr(kb, "sites", lambda: {"ucipm": {"label": "UC IPM"}})
    monkeypatch.setattr(bench, "KB_INDEX", kb.KB(tiny_kb))
    monkeypatch.setattr(bench, "KB_ONLY", True)
    monkeypatch.setattr(bench, "NAMES", None)
    monkeypatch.setattr(bench, "CLEAN_TEXT", 2)
    out = bench.run_retrieve("orange powder", [{"name": "coffee leaf rust"}], "coffee", "disease")
    assert "[local_kb: UC IPM] Coffee Leaf Rust" in out
    from agml_agent import bench_view
    trace = [{"role": "assistant", "reasoning": "", "text": "", "tool_calls": [{"name": "retrieve", "arguments": {}}]},
             {"role": "tool", "result": out},
             {"role": "assistant", "reasoning": "small pale yellow spots on the upper leaf surface grow into orange", "text": "rust"}]
    rows = bench_view.source_list(trace)
    assert rows and rows[0]["source"] == "local_kb: UC IPM" and rows[0]["url"] == "https://ipm.example/coffee-rust"
    assert any(r["used"] for r in rows)
    assert bench_view.source_use(trace)[0] == ["local_kb"]
