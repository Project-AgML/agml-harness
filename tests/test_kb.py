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
