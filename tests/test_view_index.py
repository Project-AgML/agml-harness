"""Trace viewer index: summaries and offsets per item, last trace wins, traces read by offset, cache rebuilt when
traces.jsonl changes."""
import json
import os
import time

from agml_agent import bench_view


def _write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _rec(item, answer, calls=0):
    trace = [{"role": "assistant", "text": answer, "tool_calls": []}]
    trace += [{"role": "tool", "name": "retrieve", "result": "x"}] * calls
    return {"item_id": item, "n_tool_calls": calls, "tools_used": ["retrieve"] * bool(calls), "tool_s": 1.5 * calls,
            "trace": trace}


def test_index_reads_last_trace_by_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_view, "CACHE_DIR", tmp_path / "cache")
    f = tmp_path / "traces.jsonl"
    _write(f, [_rec("a", "first try", 1), _rec("b", "bee", 0), _rec("a", "retried", 2)])
    with open(f, "a") as fh:
        fh.write("\n")  # blank lines are skipped
    idx = bench_view.trace_index(f)
    assert set(idx) == {"a", "b"}
    assert idx["a"]["n_tool_calls"] == 2 and idx["a"]["n_tool_results"] == 2 and idx["a"]["tool_s"] == 3.0
    assert idx["b"]["tools_used"] == []
    ta, tb = bench_view.read_traces(f, [idx["a"], idx["b"]])
    assert ta[0]["text"] == "retried" and tb[0]["text"] == "bee"
    assert [i for i, _ in bench_view._all_traces(f)] == ["b", "a"]  # file order of each item's last trace


def test_cache_used_and_rebuilt_when_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_view, "CACHE_DIR", tmp_path / "cache")
    f = tmp_path / "traces.jsonl"
    _write(f, [_rec("a", "one")])
    assert set(bench_view.trace_index(f)) == {"a"}
    assert len(list((tmp_path / "cache").glob("index-*.pkl"))) == 1
    calls = []
    real = bench_view.json.loads
    monkeypatch.setattr(bench_view.json, "loads", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert set(bench_view.trace_index(f)) == {"a"} and not calls  # second call: from the cache, nothing parsed
    monkeypatch.setattr(bench_view.json, "loads", real)
    _write(f, [_rec("a", "one"), _rec("c", "new")])  # e.g. a resumed run appended items
    os.utime(f, ns=(time.time_ns(), time.time_ns() + 10**9))
    assert set(bench_view.trace_index(f)) == {"a", "c"}


def test_corrupt_cache_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_view, "CACHE_DIR", tmp_path / "cache")
    f = tmp_path / "traces.jsonl"
    _write(f, [_rec("a", "one")])
    bench_view.trace_index(f)
    for c in (tmp_path / "cache").glob("*.pkl"):
        c.write_bytes(b"not a pickle")
    assert set(bench_view.trace_index(f)) == {"a"}


def test_sources_and_kb_for_whole_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_view, "CACHE_DIR", tmp_path / "cache")
    f = tmp_path / "traces.jsonl"
    _write(f, [_rec("a", "x", 1), _rec("b", "y")])
    su = bench_view.trace_sources(f)
    assert set(su) == {"a", "b"} and su["b"] == ([], [])
    assert bench_view.trace_kb(f, {"a": "rust"}) == []  # no knowledge-base text in these traces
