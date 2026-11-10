import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from agml_agent import bench, sandbox

IMG = Image.new("RGB", (64, 48), (90, 140, 60))


def test_print_and_show_come_back(tmp_path):
    text, images = sandbox.run("print(W, H, img.size)\nshow(img.crop((0, 0, 32, 24)))\nshow(np.zeros((10, 10), bool))",
                               IMG, tmp_path)
    assert "64 48 (64, 48)" in text and "2 image(s) from show()" in text
    assert len(images) == 2 and all(Path(p).exists() for p in images)
    assert Image.open(images[0]).size == (32, 24)


def test_all_item_images_are_available(tmp_path):
    text, _ = sandbox.run("print(len(imgs), imgs[1].size, img.size)", [IMG, Image.new("RGB", (20, 10))], tmp_path)
    assert "2 (20, 10) (64, 48)" in text


def test_at_most_two_images_per_call(tmp_path):
    text, images = sandbox.run("for _ in range(4): show(img)", IMG, tmp_path)
    assert len(images) == sandbox.MAX_IMAGES and "only 2 images" in text


def test_errors_are_returned_not_raised(tmp_path):
    text, images = sandbox.run("x = 1 / 0", IMG, tmp_path)
    assert "ZeroDivisionError" in text and images == []


def test_timeout_stops_the_code(tmp_path):
    text, _ = sandbox.run("while True:\n    pass", IMG, tmp_path, timeout=2)
    assert "longer than 2 s" in text


@pytest.mark.parametrize("code", ["import os\nprint(os.listdir('/'))", "import subprocess", "import socket",
                                  "__import__('os')", "Image.os.system('ls')", "eval('1+1')"])
def test_escape_routes_are_refused_before_running(code, tmp_path):
    text, _ = sandbox.run(code, IMG, tmp_path)
    assert text.startswith("error:") and "not allowed" in text


@pytest.mark.parametrize("code", ["import torch", "import urllib.request", "import transformers", "import pathlib"])
def test_only_listed_libraries_can_be_imported(code, tmp_path):
    text, _ = sandbox.run(code, IMG, tmp_path)
    assert "not allowed" in text


def test_files_outside_the_work_folder_cannot_be_read(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("label: late blight")
    for code in (f"print(open({str(secret)!r}).read())", f"Image.open({str(secret)!r})"):
        text, _ = sandbox.run(code, IMG, tmp_path / "out")
        assert "PermissionError" in text and "late blight" not in text


def test_image_libraries_work(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")
    code = ("import cv2\nfrom skimage import measure\nfrom scipy import ndimage\n"
            "a = np.array(img)\nm = cv2.inRange(a, (0, 100, 0), (255, 255, 255))\n"
            "print(int(m.sum() > 0), measure.label(m > 0).max(), ndimage.label(m)[1])")
    text, _ = sandbox.run(code, IMG, tmp_path)
    assert text.startswith("1 1 1"), text


# ---------------------------------------------------------------- bench flow with --code-tool

CODE = "print(W, H)\nshow(img.crop((0, 0, W // 2, H // 2)))"


class FakeCodeVLLM(BaseHTTPRequestHandler):
    """First answer [] (no tools offered); then calls run_python once; then answers with a box."""
    requests: list[dict] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeCodeVLLM.requests.append(body)
        seen_tool = any(m["role"] == "tool" for m in body["messages"])
        if body.get("tools") and not seen_tool:
            msg = {"content": "", "tool_calls": [{"id": "call_0", "type": "function",
                                                  "function": {"name": "run_python", "arguments": json.dumps({"code": CODE})}}]}
            finish = "tool_calls"
        elif seen_tool:
            msg, finish = {"content": '[{"box_2d": [0, 0, 500, 500], "label": "apple"}]'}, "stop"
        else:
            msg, finish = {"content": "[]"}, "stop"
        self._send({"choices": [{"message": {"role": "assistant", **msg}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def det_dir(tmp_path):
    d = tmp_path / "base-data" / "detection" / "apple_detection"
    (d / "images").mkdir(parents=True)
    samples = [{"sample_id": f"apple-{i}", "images": [f"images/{i}.jpg"], "image_size": [100, 80],
                "objects": [{"label": "apple", "bbox": [0, 0, 50, 40]}]} for i in range(2)]
    for s in samples:
        Image.new("RGB", (100, 80), (200, 40, 40)).save(d / s["images"][0])
    (d / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    (d / "meta.json").write_text(json.dumps({"name": "apple_detection", "info": {"class_names": ["apple"]},
                                             "catalog": {"classes": ["apple"], "crops": ["apple"],
                                                         "agricultural_task": "fruit_detection"}}))
    return tmp_path


@pytest.fixture
def code_server():
    FakeCodeVLLM.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeCodeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _det_args(data_dir, tmp_path, endpoint, **kw):
    from test_bench import _args

    return _args(data_dir, tmp_path, endpoint, format=None, only=None, task=["detection"],
                 retrieve_mode="crosscheck-look-first", **kw)


def test_code_tool_runs_code_and_sends_images_back(det_dir, tmp_path, code_server, monkeypatch):
    """Detection with --code-tool: both tools are offered after the first answer; the model here picks run_python."""
    monkeypatch.setattr(bench, "CODE_TOOL", False)
    rd = bench.run(_det_args(det_dir, tmp_path, code_server, code_tool=True))
    traces = [json.loads(l) for l in (rd / "traces.jsonl").read_text().splitlines()]
    preds = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().splitlines()]
    t = traces[0]
    assert t["n_tool_calls"] == 1 and t["tools_used"] == ["run_python"] and t["first_answer"] == "[]"
    tool = next(s for s in t["trace"] if s["role"] == "tool")
    assert "image(s) from show()" in tool["result"] and len(tool["images"]) == 1 and Path(tool["images"][0]).exists()
    # the code saw the image the model saw, and the next request carried the shown image
    after = next(r for r in FakeCodeVLLM.requests if any(m["role"] == "tool" for m in r["messages"]))
    last = after["messages"][-1]
    assert last["role"] == "user" and sum(c["type"] == "image_url" for c in last["content"]) == 1
    assert {t["function"]["name"] for t in after["tools"]} == {"retrieve", "run_python"}
    assert all(p["tp"] == 1 for p in preds)  # the final answer is scored, not the first
    assert json.loads((rd / "run.json").read_text())["method"]["code_tool"] is True


def test_without_code_tool_detection_stops_at_the_first_answer(det_dir, tmp_path, code_server, monkeypatch):
    monkeypatch.setattr(bench, "CODE_TOOL", False)
    rd = bench.run(_det_args(det_dir, tmp_path, code_server))
    traces = [json.loads(l) for l in (rd / "traces.jsonl").read_text().splitlines()]
    assert all(t["n_tool_calls"] == 0 and t["turns"] == 1 for t in traces)
    assert not any(r.get("tools") for r in FakeCodeVLLM.requests)


def test_code_tool_is_offered_with_retrieve_on_classification_too(tmp_path, code_server, monkeypatch):
    from test_bench import _cls

    monkeypatch.setattr(bench, "CODE_TOOL", False)
    monkeypatch.setitem(bench.BENCH_TOOLS, "retrieve", (bench.RETRIEVE_SCHEMA, lambda **kw: "[eppo] x (source: y)"))
    _cls(tmp_path, "tomato_leaf_disease", ["Early_blight", "Late_blight"], ["tomato"], per=1)
    from test_bench import _args

    bench.run(_args(tmp_path, tmp_path, code_server, format=["mcqa_within"], only="tomato", code_tool=True,
                    retrieve_mode="crosscheck-look-first"))
    offered = [r for r in FakeCodeVLLM.requests if r.get("tools")]
    assert offered and all({t["function"]["name"] for t in r["tools"]} == {"retrieve", "run_python"} for r in offered)


def test_images_are_files_with_the_names_the_model_expects_and_pil_show_works(tmp_path):
    text, images = sandbox.run("im = Image.open('input_file_0.png')\nprint(im.size)\nim.crop((0, 0, 8, 8)).show()", IMG, tmp_path)
    assert "(64, 48)" in text and len(images) == 1 and Image.open(images[0]).size == (8, 8)


def test_retrieve_description_names_the_local_sources_with_kb_only(tmp_path, monkeypatch):
    import copy

    from agml_agent import kb
    (tmp_path / "manifest.json").write_text(json.dumps({"sites": {"ucipm": {"sections": 5}, "texas_pdh": {"sections": 0}},
                                                        "names": {"rows": {"gbif_backbone": 9}}}))
    monkeypatch.setattr(kb, "sites", lambda: {"ucipm": {"label": "UC IPM"}, "gbif_backbone": {"label": "GBIF Backbone Taxonomy"}})
    monkeypatch.setattr(bench, "RETRIEVE_SCHEMA", copy.deepcopy(bench.RETRIEVE_SCHEMA))
    monkeypatch.setattr(bench, "RETRIEVE_MODES", copy.deepcopy(bench.RETRIEVE_MODES))
    bench._describe_local_sources(tmp_path)
    d = bench.RETRIEVE_MODES["crosscheck-look-first"]["function"]["description"]
    assert "(UC IPM, GBIF Backbone Taxonomy; all downloaded locally)" in d and "PlantVillage" not in d
