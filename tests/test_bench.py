import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from agbench.eval import Benchmark, VLMMethod

from agml_agent import bench


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
             judge=False, judge_endpoint=None, retrieve_cache=None, no_retrieve_cache=False)
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


def test_agml_tools_are_not_allowed():
    with pytest.raises(ValueError, match="search_agml"):
        bench.HarnessMethod("http://127.0.0.1:1", "google/gemma-4-12B-it", tools=["retrieve", "search_agml"])


GBIF = {"gbif": [{"excerpt_text": "Scientific name: Phytophthora infestans", "source_url": "https://gbif.org/species/1",
                  "raw_response": '{"media": "https://img/1.jpg"}', "extra": {"images": [{"url": "https://img/1.jpg"}]}}]}


@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(bench, "CACHE", None)


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
    assert ("late blight", {"crop": "tomato", "task": "disease", "scientific_name": "Phytophthora infestans", "aliases": []}) in seen
    assert ("early blight", {"crop": "tomato", "task": "disease", "scientific_name": None, "aliases": ["Alternaria"]}) in seen


def test_description_search(monkeypatch, no_cache):
    import agml_agent.symptom_search as ss
    got = []
    monkeypatch.setattr(ss, "search_description", lambda d, c, t: got.append((d, c, t)) or [
        {"site": "ipm.ucanr.edu", "title": "Bean Common Mosaic", "url": "https://ipm.ucanr.edu/bcm", "text": "Leaves pucker."}])
    out = bench.run_retrieve("puckered leaf", crop="bean", task="disease")
    assert got == [("puckered leaf", "bean", "disease")]
    assert "[ipm.ucanr.edu] Bean Common Mosaic: Leaves pucker. (source: https://ipm.ucanr.edu/bcm)" in out
    assert out.endswith(bench.NEXT_STEP) and "check the most likely causes" in out
    monkeypatch.setattr(ss, "search_description", lambda *a: [])
    assert bench.run_retrieve("nothing").startswith("No pages matched")


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


def test_cache_persists_results_but_not_empties(monkeypatch, tmp_path):
    import agml_agent.retrieve as r
    n = []
    monkeypatch.setattr(r, "retrieve", lambda *a, **k: n.append(1) or ({} if len(n) == 1 else GBIF))
    monkeypatch.setattr(bench, "CACHE", bench.RetrievalCache(tmp_path / "c.jsonl"))
    for _ in range(3):
        bench.run_retrieve(candidates=["late blight"])
    assert len(n) == 2  # empty -> retried; found -> cached
    monkeypatch.setattr(bench, "CACHE", bench.RetrievalCache(tmp_path / "c.jsonl"))  # a later run
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
