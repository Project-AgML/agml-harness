"""
bench.py — benchmark the harness on AGgentic-benchmarks (agbench) WITHOUT
changing agbench: items, prompts, sampling/effort settings, grading and the
judge all come from the pinned agbench package, so a harness run is scored
exactly like the plain-model baselines. The only difference is that the
model may call tools.

    agml-agent-bench --model google/gemma-4-12B-it --data-dir /path/with/base-data --judge
    agml-agent-bench --model google/gemma-4-12B-it --endpoint http://127.0.0.1:8000 ...   # already-running server

The run dir is a standard agbench run dir (compare scores with
`agbench eval view --runs <baseline> <this run>`), plus `traces.jsonl`: one
line per item with every model turn, tool call and tool result (view with
`agml-agent-view`).

Leakage policy: only tools that query external databases are allowed here
(BENCH_TOOLS). search_agml is never offered — it returns AgML class lists
and benchmark results, and agbench items come from AgML. retrieve's result
is reduced to excerpt text + source URL: no images or image URLs reach the
model.

"Exploit" policy (the first harness baseline): one pass, tools available,
at most `max_tool_calls` calls, no planning/search/self-consistency. The
prompt is agbench's, unchanged; the tools are described only through their
schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agbench.eval import Benchmark, EvalItem, Prediction, VLMMethod
from agbench.eval.client import encode_image

RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Look up reference text about a crop disease, pest, weed/plant species or produce-quality class in external "
            "agricultural databases (EPPO, UC IPM/APS, Bugwood, GBIF, USDA). Returns short excerpts such as taxonomy, "
            "hosts, symptoms and identification notes, each with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "Common name of the disease, pest, species or class, e.g. 'coffee leaf rust'."},
                "crop": {"type": "string", "description": "Host crop, e.g. 'coffee'."},
                "task": {"type": "string", "enum": ["disease", "pest", "species", "quality"],
                         "description": "Kind of class; picks which databases are searched."},
                "scientific_name": {"type": "string", "description": "Scientific name if known, e.g. 'Hemileia vastatrix'."},
            },
            "required": ["class_name"],
        },
    },
}

PER_CHUNK_CHARS = 1200
RESULT_CHARS = 6000


def run_retrieve(class_name: str, crop: str = "", task: str = "", scientific_name: str | None = None) -> str:
    """retrieve(), reduced to what the model may see: excerpt text and source URL per chunk."""
    from agml_agent.retrieve import retrieve

    res = retrieve(class_name, crop=crop, task=task, scientific_name=scientific_name)
    parts = [f"[{src}] {c['excerpt_text'][:PER_CHUNK_CHARS]} (source: {c['source_url']})" for src, chunks in res.items() for c in chunks]
    return "\n\n".join(parts)[:RESULT_CHARS] or "No results found."


# name -> (schema, function). Only external-database tools: see the leakage policy above.
BENCH_TOOLS = {"retrieve": (RETRIEVE_SCHEMA, run_retrieve)}


def tool_args(model_id: str) -> list[str]:
    """vLLM flags that turn on tool calling for an agbench registry model (llama.cpp: none, --jinja does it)."""
    from agbench.eval.methods import model_spec

    if model_spec(model_id)["backend"] != "vllm":
        return []
    parsers = yaml.safe_load(files("agml_agent.config").joinpath("tool_parsers.yaml").read_text())
    if model_id not in parsers:
        raise ValueError(f"{model_id}: no entry in config/tool_parsers.yaml (vLLM --tool-call-parser)")
    return ["--enable-auto-tool-choice", "--tool-call-parser", parsers[model_id]]


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=path,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _agbench_commit() -> str | None:
    url = json.loads(importlib.metadata.distribution("agbench").read_text("direct_url.json") or "{}")
    return (url.get("vcs_info") or {}).get("commit_id")


class HarnessMethod(VLMMethod):
    """agbench's VLMMethod (same prompt, box order, sampling, effort, max_tokens) plus a tool loop.

    `max_tokens` stays the per-item generation budget: completion tokens summed over turns. Tool results
    count only toward the context window (max_model_len), not toward it. Every item's turns go to
    `trace_path` (traces.jsonl) when set."""

    def __init__(self, endpoint: str, model_id: str, thinking: bool = True, effort: str | None = "low",
                 max_tokens: int | None = None, tools: list[str] | tuple[str, ...] = ("retrieve",), max_tool_calls: int = 4,
                 trace_path: str | Path | None = None, served_name: str | None = None, retries: int = 3):
        super().__init__(endpoint, model_id, thinking, effort, max_tokens, served_name=served_name)
        if bad := [t for t in tools if t not in BENCH_TOOLS]:
            raise ValueError(f"tools {bad} not allowed in benchmark runs; allowed: {sorted(BENCH_TOOLS)} "
                             "(search_agml is excluded: AgML class lists/results leak benchmark labels)")
        self.tools, self.max_tool_calls, self.retries = list(tools), max_tool_calls, retries
        self.trace_path = Path(trace_path) if trace_path else None
        self.name = f"{model_id} + tools"
        self._lock = asyncio.Lock()

    def config(self) -> dict:
        return {**super().config(), "name": self.name, "harness": "exploit", "tools": self.tools,
                "max_tool_calls": self.max_tool_calls, "agml_agent": importlib.metadata.version("agml-agent"),
                "agml_agent_git": _git_sha(Path(__file__).parent), "agbench_commit": _agbench_commit()}

    async def _create(self, **kw):
        err = None
        for attempt in range(self.retries + 1):
            try:
                return await self.client.api.chat.completions.create(**kw), None
            except Exception as e:  # connection drops, 5xx, timeouts
                err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt * 5)
        return None, err

    async def _call_tool(self, name: str, raw_args: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            args = json.loads(raw_args or "{}")
            result = await asyncio.to_thread(BENCH_TOOLS[name][1], **args) if name in self.tools else f"error: unknown tool {name!r}"
        except Exception as e:  # bad JSON or arguments, source failures
            result = f"error: {type(e).__name__}: {e}"
        return result, time.perf_counter() - t0

    async def predict(self, item: EvalItem) -> Prediction:
        model = await self.client.resolve_model()
        encoded = [encode_image(p, self.client.max_pixels) for p in item.images]
        sent_size = encoded[0][1] if encoded else None
        content = [{"type": "image_url", "image_url": {"url": u}} for u, _ in encoded] + [{"type": "text", "text": self.prompt(item)}]
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": content}]
        body = dict(self.body)
        temperature = body.pop("temperature", None)
        schemas = [BENCH_TOOLS[t][0] for t in self.tools]

        trace, n_calls, tool_s, used = [], 0, 0.0, []
        prompt_tokens = completion_tokens = 0
        text, finish, err, t0 = "", "stop", None, time.perf_counter()
        for _ in range(self.max_tool_calls + 2):
            remaining = self.max_tokens - completion_tokens
            if remaining <= 0:
                finish = "length"
                break
            r, err = await self._create(model=model, messages=messages, max_tokens=remaining, temperature=temperature,
                                        tools=schemas, tool_choice="auto" if n_calls < self.max_tool_calls else "none",
                                        extra_body=body or None)
            if r is None:
                finish = "error"
                break
            msg, finish = r.choices[0].message, r.choices[0].finish_reason or ""
            prompt_tokens += getattr(r.usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(r.usage, "completion_tokens", 0) or 0
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            calls = list(msg.tool_calls or [])
            text = msg.content or ""
            trace.append({"role": "assistant", "reasoning": reasoning, "text": text,
                          "tool_calls": [{"name": c.function.name, "arguments": _loads(c.function.arguments)} for c in calls]})
            if not calls:
                break
            messages.append({"role": "assistant", "content": text,
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                if n_calls < self.max_tool_calls:
                    n_calls += 1
                    result, dt = await self._call_tool(c.function.name, c.function.arguments)
                    used.append(c.function.name)
                else:
                    result, dt = "error: tool-call budget used up; answer now", 0.0
                tool_s += dt
                trace.append({"role": "tool", "name": c.function.name, "result": result, "latency_s": round(dt, 3)})
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name, "content": result})

        latency = time.perf_counter() - t0
        await self._write_trace({"item_id": item.item_id, "dataset": item.dataset, "n_tool_calls": n_calls,
                                 "tools_used": used, "tool_s": round(tool_s, 3), "turns": sum(s["role"] == "assistant" for s in trace),
                                 "finish_reason": finish, "error": err, "trace": trace})
        reasoning_all = "\n\n".join(
            s["reasoning"] + "".join(f"\n[called {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})]" for c in s["tool_calls"])
            for s in trace if s["role"] == "assistant").strip()
        if finish == "error":
            return Prediction(text="", finish_reason="error", latency_s=latency, sent_size=sent_size, error=err)
        return Prediction(text=text, reasoning=reasoning_all, finish_reason=finish, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens, latency_s=latency, sent_size=sent_size)

    async def _write_trace(self, rec: dict) -> None:
        if not self.trace_path:
            return
        async with self._lock:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _loads(s: str):
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError:
        return s


def _judge(run_dir: Path, endpoint: str | None) -> None:
    from agbench.eval.gpu import sampler
    from agbench.eval.judge import judge_run
    from agbench.eval.resources import models
    from agbench.eval.runner import log_compute
    from agbench.eval.servers import Server

    jm = models()["judge"]
    if endpoint:
        judge_run(run_dir, endpoint, jm)
        return
    with sampler() as gpu:
        t0 = time.time()
        with Server(jm, run_dir / "judge_server.log") as srv:
            log_compute(run_dir, phase="judge_load", model=jm, wall_s=round(srv.load_seconds, 1), **gpu.summary(t0, time.time()))
            judge_run(run_dir, srv.endpoint, jm, gpu=gpu)


def run(args: argparse.Namespace) -> Path:
    from agbench.eval.gpu import sampler
    from agbench.eval.methods import key_formats, model_spec
    from agbench.eval.report import leaderboard_md, load_runs
    from agbench.eval.runner import log_compute
    from agbench.eval.servers import Server

    load_dotenv()
    bench = Benchmark(args.data_dir, args.task or None, args.format or None, args.only, args.limit_per_dataset, args.seed)
    print(f"{len(bench.items)} items; {len(bench.skipped)} datasets skipped")
    spec = model_spec(args.model)
    conc = args.concurrency or (spec.get("parallel", 2) if spec["backend"] == "llamacpp" else spec["concurrency"])
    rd = args.run_dir or args.out / f"{args.model.replace('/', '__')}__tools_{datetime.now():%Y%m%d-%H%M%S}"
    rd.mkdir(parents=True, exist_ok=True)

    def method(endpoint: str) -> HarnessMethod:
        return HarnessMethod(endpoint, args.model, args.thinking, args.reasoning_effort, args.max_tokens, args.tools,
                             args.max_tool_calls, trace_path=rd / "traces.jsonl")

    if args.endpoint:
        res = bench.run(method(args.endpoint), rd, conc, key_formats=key_formats(spec))
    else:
        with sampler() as gpu:
            t0 = time.time()
            with Server(args.model, rd / "server.log", max_model_len=args.max_model_len, extra_args=tool_args(args.model)) as srv:
                log_compute(rd, phase="load", model=args.model, wall_s=round(srv.load_seconds, 1), **gpu.summary(t0, time.time()))
                res = bench.run(method(srv.endpoint), rd, conc, key_formats=key_formats(spec), gpu=gpu)
    if args.judge and res.metrics["pending_judge"]:
        _judge(rd, args.judge_endpoint)
    print(leaderboard_md(load_runs([rd])))
    print(f"results: {rd}")
    return rd


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the harness (model + tools) on AGgentic-benchmarks via the pinned agbench.")
    p.add_argument("--data-dir", type=Path, required=True, help="Folder with base-data/.")
    p.add_argument("--model", required=True, help="agbench registry model id.")
    p.add_argument("--endpoint", help="Use a running OpenAI-compatible server (started with tool calling on) instead of launching one.")
    p.add_argument("--tools", nargs="+", default=["retrieve"], help=f"Tools offered to the model; allowed: {sorted(BENCH_TOOLS)}.")
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--task", action="append", help="classification | detection | segmentation | image_text_to_text (repeatable).")
    p.add_argument("--format", action="append", help="mcqa_within | mcqa_across | oeq | grounding | native (repeatable).")
    p.add_argument("--only", help="Regex on dataset name.")
    p.add_argument("--limit-per-dataset", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.add_argument("--reasoning-effort", default="low", help="low | medium | high (agbench default: low).")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-dir", type=Path, help="Resume into this run dir.")
    p.add_argument("--judge", action="store_true", help="Judge open-ended answers after the run (agbench judge).")
    p.add_argument("--judge-endpoint")
    run(p.parse_args())


if __name__ == "__main__":
    main()
