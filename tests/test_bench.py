import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from agbench.eval import Benchmark, VLMMethod

from agml_agent import bench
from agml_agent.common import cache


def _cls(root: Path, name: str, labels: list[str], crops: list[str], per: int = 3):
    d = root / "base-data" / "classification" / name
    (d / "images").mkdir(parents=True)
    samples = [{"sample_id": f"{name}-{l}-{i}", "label": l, "images": [f"images/{l}-{i}.jpg"], "image_size": [100, 80]}
               for l in labels for i in range(per)]
    for s in samples:
        Image.new("RGB", (100, 80), (90, 140, 60)).save(d / s["images"][0])
    (d / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    (d / "meta.json").write_text(json.dumps({"name": name, "info": {}, "catalog": {
        "classes": labels, "crops": crops, "agricultural_task": "disease_classification"}}))


@pytest.fixture
def data_dir(tmp_path):
    _cls(tmp_path, "tomato_leaf_disease", ["Early_blight", "Late_blight", "healthy", "Leaf_Mold"], ["tomato"])
    _cls(tmp_path, "bean_cowpea_disease", ["bean_rust", "cowpea_mosaic", "angular_spot"], ["bean"])
    return tmp_path


class FakeVLLM(BaseHTTPRequestHandler):
    """Calls `retrieve` while tool_choice is auto and it hasn't seen a tool result (always, if `greedy`); then answers."""
    requests: list[dict] = []
    greedy = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.requests.append(body)
        seen_tool = any(m["role"] == "tool" for m in body["messages"])
        if body.get("tool_choice") == "auto" and (FakeVLLM.greedy or not seen_tool):
            n = sum(m["role"] == "tool" for m in body["messages"])
            msg = {"content": "", "reasoning_content": "check the database",
                   "tool_calls": [{"id": f"call_{n}", "type": "function",
                                   "function": {"name": "retrieve", "arguments": json.dumps(ARGS)}}]}
            finish = "tool_calls"
        else:
            msg, finish = {"content": "Late blight", "reasoning_content": "it matches"}, "stop"
        self._send({"choices": [{"message": {"role": "assistant", **msg}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ARGS = {"description": "dark lesions", "candidates": [{"name": "late blight"}], "task": "disease"}


@pytest.fixture
def fake_server(monkeypatch):
    FakeVLLM.requests, FakeVLLM.greedy = [], False
    calls = []
    monkeypatch.setitem(bench.BENCH_TOOLS, "retrieve",
                        (bench.RETRIEVE_SCHEMA, lambda **kw: calls.append(kw) or "[eppo] Late blight: dark lesions (source: x)"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", calls
    srv.shutdown()


def _args(data_dir, tmp_path, endpoint, **kw):
    a = dict(data_dir=data_dir, model="google/gemma-4-12B-it", endpoint=endpoint, tools=["retrieve"], max_tool_calls=4,
             task=None, format=["mcqa_within"], only="tomato", limit_per_dataset=None, concurrency=4, thinking=True,
             reasoning_effort="low", max_tokens=None, max_model_len=None, seed=0, out=tmp_path / "runs", run_dir=None,
             judge=False, judge_endpoint=None, retrieve_cache=None, no_retrieve_cache=False,
             retrieve_mode="description+candidates")
    return argparse.Namespace(**{**a, **kw})


def test_harness_run_matches_agbench_format_and_logs_traces(data_dir, tmp_path, fake_server):
    endpoint, calls = fake_server
    rd = bench.run(_args(data_dir, tmp_path, endpoint))
    recs = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().splitlines()]
    traces = [json.loads(l) for l in (rd / "traces.jsonl").read_text().splitlines()]
    assert recs and len(traces) == len(recs) and {t["item_id"] for t in traces} == {r["item_id"] for r in recs}

    # same record format as a plain agbench model run on the same items
    base = Benchmark(data_dir, formats=["mcqa_within"], only="tomato").run(VLMMethod(endpoint, "google/gemma-4-12B-it"), tmp_path / "base")
    assert set(recs[0]) == set(base.records[0])
    assert {r["item_id"] for r in recs} == {r["item_id"] for r in base.records}

    t = traces[0]
    assert t["n_tool_calls"] == 1 and t["tools_used"] == ["retrieve"] and t["turns"] == 2
    assert t["trace"][0]["tool_calls"] == [{"name": "retrieve", "arguments": ARGS}]
    assert t["trace"][1]["role"] == "tool" and "dark lesions" in t["trace"][1]["result"]
    assert calls[0] == ARGS
    assert all(r["completion_tokens"] == 10 and r["answer"] for r in recs)

    # the baseline's sampling and effort settings, plus tools
    first = FakeVLLM.requests[0]
    assert first["top_k"] == 64 and first["thinking_token_budget"] == 1024 and first["chat_template_kwargs"]["enable_thinking"]
    assert [x["function"]["name"] for x in first["tools"]] == ["retrieve"]
    second = next(b for b in FakeVLLM.requests if any(m["role"] == "tool" for m in b["messages"]))
    assert second["messages"][-1]["tool_call_id"] == "call_0" and second["messages"][-2]["tool_calls"][0]["id"] == "call_0"

    cfg = json.loads((rd / "run.json").read_text())["method"]
    assert cfg["name"] == "google/gemma-4-12B-it + tools" and cfg["tools"] == ["retrieve"] and cfg["effort"] == "low"
    assert len(cfg["agbench_commit"]) == 40 and cfg["retrieve_cache"] == str(tmp_path / "runs" / "retrieve_cache.jsonl")


def test_tool_budget_forces_an_answer(data_dir, tmp_path, fake_server):
    endpoint, calls = fake_server
    FakeVLLM.greedy = True
    rd = bench.run(_args(data_dir, tmp_path, endpoint, max_tool_calls=2, limit_per_dataset=1))
    t = [json.loads(l) for l in (rd / "traces.jsonl").read_text().splitlines()][0]
    assert t["n_tool_calls"] == 2 and t["turns"] == 3
    assert [b["tool_choice"] for b in FakeVLLM.requests[:3]] == ["auto", "auto", "none"]
    rec = json.loads((rd / "predictions.jsonl").read_text().splitlines()[0])
    assert rec["answer"] == "Late blight" and not rec["error"]


def test_candidates_mode_offers_no_description_search(data_dir, tmp_path, fake_server):
    endpoint, calls = fake_server
    rd = bench.run(_args(data_dir, tmp_path, endpoint, retrieve_mode="candidates", limit_per_dataset=1))
    assert rd.name.startswith("google__gemma-4-12B-it__tools-candidates_")
    params = FakeVLLM.requests[0]["tools"][0]["function"]["parameters"]
    assert "description" not in params["properties"] and params["required"] == ["candidates"]
    assert calls[0] == {"candidates": [{"name": "late blight"}], "task": "disease"}  # description dropped
    cfg = json.loads((rd / "run.json").read_text())["method"]
    assert cfg["retrieve_mode"] == "candidates" and cfg["name"] == "google/gemma-4-12B-it + tools (candidates)"


def test_agml_tools_are_not_allowed():
    with pytest.raises(ValueError, match="search_agml"):
        bench.HarnessMethod("http://127.0.0.1:1", "google/gemma-4-12B-it", tools=["retrieve", "search_agml"])


GBIF = {"gbif": [{"excerpt_text": "Scientific name: Phytophthora infestans", "source_url": "https://gbif.org/species/1",
                  "raw_response": '{"media": "https://img/1.jpg"}', "extra": {"images": [{"url": "https://img/1.jpg"}]}}]}


@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(cache, "CACHE", None)


def test_retrieve_result_is_text_only(monkeypatch, no_cache):
    import agml_agent.retrieve as r
    monkeypatch.setattr(r, "retrieve", lambda *a, **k: GBIF)
    out = bench.run_retrieve(candidates=[{"name": "late blight"}], task="disease")
    assert out == "## late blight\n[gbif] Scientific name: Phytophthora infestans (source: https://gbif.org/species/1)"
    monkeypatch.setattr(r, "retrieve", lambda *a, **k: {})
    assert "No results found for 'x'." in bench.run_retrieve(candidates=["x"])
    assert bench.run_retrieve().startswith("error:")


def test_candidates_are_looked_up_by_name(monkeypatch, no_cache):
    import agml_agent.retrieve as r
    seen = []
    text = "Late blight is a disease of potato and tomato. Lesions are dark and water-soaked. Spread by wind."
    monkeypatch.setattr(r, "retrieve", lambda name, **k: seen.append((name, k)) or {"ucipm_aps": [
        {"excerpt_text": text, "source_url": f"https://ipm/{name}"}]})
    out = bench.run_retrieve("dark water-soaked lesions", [{"name": "late blight", "scientific_name": "Phytophthora infestans"},
                                                         {"name": "early blight", "aliases": ["Alternaria"]}], "tomato", "disease")
    assert out.index("## late blight") < out.index("## early blight")
    assert f"[ucipm_aps] {text} (source: https://ipm/late blight)" in out  # short excerpts are kept whole
    assert ("late blight", {"crop": "tomato", "scientific_name": "Phytophthora infestans", "aliases": []}) in seen
    assert ("early blight", {"crop": "tomato", "scientific_name": None, "aliases": ["Alternaria"]}) in seen


def test_every_database_is_searched_task_only_orders(monkeypatch, no_cache):
    import agml_agent.retrieve as r
    seen = []
    monkeypatch.setattr(r, "retrieve", lambda name, **k: seen.append(k) or {
        "gbif": [{"excerpt_text": "Puccinia, a rust fungus.", "source_url": "https://gbif.org/1"}],
        "eppo": [{"excerpt_text": "Puccinia graminis, stem rust of wheat.", "source_url": "https://eppo.int/1"}]})
    out = bench.run_retrieve(candidates=["stem rust"], crop="wheat", task="disease")
    assert "task" not in seen[0] and out.index("[eppo]") < out.index("[gbif]")
    out = bench.run_retrieve(candidates=["stem rust"], task="species")
    assert out.index("[gbif]") < out.index("[eppo]")


def test_description_search(monkeypatch, no_cache):
    import agml_agent.symptom_search as ss
    got = []
    monkeypatch.setattr(ss, "search_description", lambda d, c, t: got.append((d, c, t)) or [
        {"site": "ipm.ucanr.edu", "title": "Bean Common Mosaic", "url": "https://ipm.ucanr.edu/bcm", "text": "Leaves pucker.",
         "mentions_crop": True},
        {"site": "bugwood.org", "title": "Soybean mosaic", "url": "https://bugwood.org/smv", "text": "Leaves pucker.",
         "mentions_crop": False}])
    out = bench.run_retrieve("puckered leaf", crop="bean", task="disease")
    assert got == [("puckered leaf", "bean", "disease")]
    assert "[ipm.ucanr.edu] Bean Common Mosaic: Leaves pucker. (source: https://ipm.ucanr.edu/bcm)" in out
    assert "[bugwood.org] Soybean mosaic (does not mention bean): Leaves pucker." in out
    assert out.endswith(bench.NEXT_STEP) and "check the most likely causes" in out
    monkeypatch.setattr(ss, "search_description", lambda *a: [])
    assert bench.run_retrieve("nothing").startswith("UC IPM and Bugwood returned no pages")


def test_description_search_skips_crop_index_pages(monkeypatch):
    import agml_agent.common.scrape as sc
    import agml_agent.symptom_search as ss
    urls = ["https://ipm.ucanr.edu/agriculture/dry-beans/", "https://ipm.ucanr.edu/agriculture/dry-beans/bean-rust/",
            "https://ipm.ucanr.edu/home-and-landscape/bean-rust"]

    class FakeDDGS:
        def text(self, q, **kw):
            return [{"href": u, "title": u} for u in urls]

    fetched = []
    monkeypatch.setattr(sc, "DDGS", FakeDDGS)
    monkeypatch.setattr(sc, "FETCH_SLEEP", 0)
    monkeypatch.setattr(sc, "fetch_and_extract", lambda u: fetched.append(u) or "bean rust pustules")
    pages = ss.search_description("pustules", crop="bean")
    assert fetched == urls[1:] and [p["url"] for p in pages] == urls[1:]


def test_description_search_keeps_other_hosts_after_the_crop(monkeypatch, no_cache):
    import agml_agent.common.scrape as sc
    import agml_agent.symptom_search as ss
    pages = {"https://ipm.ucanr.edu/dry-beans/halo-blight/": "Halo blight of dry beans: water-soaked spots, yellow halos.",
             "https://wiki.bugwood.org/Peanut_Diseases": "blocked",
             "https://bugwood.org/peanut-leaf-spot": "Early leaf spot of peanut: brown spots with yellow halos."}

    class FakeDDGS:
        def text(self, q, **kw):
            return [{"href": u, "title": u} for u in pages]

    monkeypatch.setattr(sc, "DDGS", FakeDDGS)
    monkeypatch.setattr(sc, "FETCH_SLEEP", 0)
    monkeypatch.setattr(sc, "fetch_and_extract", lambda u: pages[u])
    got = ss.search_description("spots with yellow halos", crop="groundnut")
    assert [(p["url"], p["mentions_crop"]) for p in got] == [("https://bugwood.org/peanut-leaf-spot", True),
                                                             ("https://ipm.ucanr.edu/dry-beans/halo-blight/", False)]


def test_cache_persists_results_but_not_empties(monkeypatch, tmp_path):
    import agml_agent.retrieve as r
    n = []
    monkeypatch.setattr(r, "retrieve", lambda *a, **k: n.append(1) or ({} if len(n) == 1 else GBIF))
    monkeypatch.setattr(cache, "CACHE", cache.RetrievalCache(tmp_path / "c.jsonl"))
    for _ in range(3):
        bench.run_retrieve(candidates=["late blight"])
    assert len(n) == 2  # empty -> retried; found -> cached
    monkeypatch.setattr(cache, "CACHE", cache.RetrievalCache(tmp_path / "c.jsonl"))  # a later run
    assert "Phytophthora" in bench.run_retrieve(candidates=["Late Blight "]) and len(n) == 2


def test_focus_keeps_matching_sentences():
    from agml_agent.symptom_search import focus
    text = "Rust is common. Pustules are reddish-brown with yellow halos. Manage with fungicides."
    assert focus(text, "reddish-brown pustules", 60) == "Pustules are reddish-brown with yellow halos."
    assert focus(text, "reddish-brown pustules", 200) == text  # fits: kept whole
    assert focus(text, "", 10) == text[:10] and focus(text, "zebra", 10) == text[:10]


def test_tool_server_flags():
    assert bench.tool_args("google/gemma-4-12B-it") == ["--enable-auto-tool-choice", "--tool-call-parser", "gemma4"]
    assert bench.tool_args("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF") == []
    with pytest.raises(ValueError, match="tool_parsers.yaml"):
        bench.tool_args("some/unlisted-model")


def test_search_and_pages_are_cached_whatever_the_names(monkeypatch, tmp_path):
    import agml_agent.common.scrape as sc
    from agml_agent.sources import ucipm_aps
    from agml_agent.sources.base import ClassQuery
    searched, fetched = [], []

    class FakeDDGS:
        def text(self, q, **kw):
            searched.append(q)
            return [{"href": "https://ipm.ucanr.edu/bean-rust", "title": "Bean rust"}]

    monkeypatch.setattr(sc, "DDGS", FakeDDGS)
    monkeypatch.setattr(sc, "FETCH_SLEEP", 0)
    monkeypatch.setattr(sc, "fetch_and_extract", lambda u: fetched.append(u) or "Bean rust, Uromyces appendiculatus.")
    monkeypatch.setattr(cache, "CACHE", cache.RetrievalCache(tmp_path / "c.jsonl"))
    ucipm_aps.fetch(ClassQuery("bean rust", "bean"))
    monkeypatch.setattr(cache, "CACHE", cache.RetrievalCache(tmp_path / "c.jsonl"))  # a later run
    chunks = ucipm_aps.fetch(ClassQuery("bean rust", "bean", scientific_name="Uromyces appendiculatus"))
    assert len(searched) == 1 and len(fetched) == 1 and chunks[0].source_url == "https://ipm.ucanr.edu/bean-rust"


def test_prefill_lookups():
    from types import SimpleNamespace as NS
    from agml_agent.bench_prefill import lookups
    items = [NS(task="classification", ag_task="disease", crop="Bean", class_names=["bean rust", "Healthy", "4"]),
             NS(task="classification", ag_task="species_id", crop="plant", class_names=["Swinecress"]),
             NS(task="classification", ag_task="disease", crop="tomato or grape", class_names=["bean rust"]),
             NS(task="detection", ag_task="disease", crop="bean", class_names=["blight"])]
    assert lookups(items) == [("bean rust", "bean", "disease"), ("Healthy", "bean", "disease"),
                              ("bean rust", "", "disease"), ("Healthy", "", "disease"), ("Swinecress", "", "species")]


def test_search_asks_each_site_when_first_answer_is_all_skipped(monkeypatch, no_cache):
    import agml_agent.common.scrape as sc
    import agml_agent.symptom_search as ss
    asked = []

    class FakeDDGS:
        def text(self, q, **kw):
            asked.append(q)
            return ([{"href": "https://wiki.bugwood.org/Rust", "title": "Rust"}] if " OR " in q
                    else [{"href": "https://ipm.ucanr.edu/home-and-landscape/rust/", "title": "Rust"}] if "ucanr" in q else [])

    monkeypatch.setattr(sc, "DDGS", FakeDDGS)
    monkeypatch.setattr(sc, "FETCH_SLEEP", 0)
    monkeypatch.setattr(sc, "fetch_and_extract", lambda u: "Rust: orange pustules.")
    assert [p["url"] for p in ss.search_description("orange pustules")] == ["https://ipm.ucanr.edu/home-and-landscape/rust/"]
    assert len(asked) == 2
