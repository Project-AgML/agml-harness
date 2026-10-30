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
model. Its description search is restricted to UC IPM/APSnet and Bugwood
fact-sheet sites (symptom_search.py), never the open web, where dataset pages
could be found.

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
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agbench.eval import Benchmark, EvalItem, Prediction, VLMMethod
from agbench.eval.client import encode_image

from agml_agent.common import cache, scrape
from agml_agent.common.cache import RetrievalCache

MAX_CANDIDATES = 4
DATABASES = ("EPPO, UC IPM/APS, Bugwood, TNAU Agritech, APS lessons, PNW and Texas handbooks, NC State Extension, "
             "ProMusa, PlantVillage, Feedipedia, Wikipedia, GBIF, iNaturalist, Catalogue of Life, EOL, USDA")
# Leakage guard on every source's text: benchmark datasets are described on the web (dataset pages, papers, Wikipedia
# articles); text about datasets or machine learning never reaches the model.
LEAK = re.compile(r"\b(data ?sets?|machine learning|deep learning|neural networks?|computer vision)\b", re.I)

RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Look up crop diseases, pests, weed/plant species and produce-quality classes in external agricultural "
            f"databases ({DATABASES}). Two ways to ask, usable together:\n"
            "- description: what you see in the image (symptoms, damage, colours, shapes). Searches extension fact "
            "sheets (UC IPM, Bugwood, TNAU Agritech, APS, PNW and Texas handbooks, NC State, ProMusa, PlantVillage) "
            "and Wikipedia, and returns pages with matching features (diseases, pests, species, quality problems), on "
            "this crop first, then on other hosts; their titles usually name what they describe.\n"
            "- candidates: up to 4 names to check. Returns reference text for each (taxonomy, hosts, symptoms, "
            "identification notes) to compare against the image. With a description too, each excerpt is narrowed to "
            "the parts matching it.\n"
            "Typical use: describe what you see, then check the most likely causes from the results with candidates. "
            "Names are looked up by name only: put symptoms in description, not in candidate names. Every excerpt "
            "comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What you see, e.g. 'bean leaf with small "
                                "reddish-brown pustules surrounded by yellow halos'."},
                "candidates": {
                    "type": "array", "maxItems": MAX_CANDIDATES,
                    "description": "Diseases, pests, species or classes to check, e.g. "
                                   "[{'name': 'bean rust', 'scientific_name': 'Uromyces appendiculatus'}].",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Common name, e.g. 'coffee leaf rust'."},
                            "scientific_name": {"type": "string", "description": "Scientific name of the pathogen, "
                                                "pest or species if known; tried first, usually the most reliable."},
                            "aliases": {"type": "array", "items": {"type": "string"},
                                        "description": "Other names, tried if the main names find nothing."},
                        },
                        "required": ["name"],
                    },
                },
                "crop": {"type": "string", "description": "Host crop, e.g. 'coffee'. Optional: added to the search, and "
                         "pages mentioning it come first; pages about other hosts are still returned. To look up the "
                         "crop itself, give it as a candidate with task 'species'."},
                "task": {"type": "string", "enum": ["disease", "pest", "species", "quality"],
                         "description": "Kind of class. Every database is searched; the ones covering this "
                                        "kind come first."},
            },
        },
    },
}

# --retrieve-mode candidates: the model proposes names and the tool checks them; no search by description. To compare
# with the default (pilots 2-3: description search anchored the model on whatever pages it returned).
CANDIDATES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Look up crop diseases, pests, weed/plant species and produce-quality classes by name in external "
            f"agricultural databases ({DATABASES}). Give up to 4 candidate names, e.g. the "
            "causes you are considering; returns reference text for each (taxonomy, hosts, symptoms, identification "
            "notes) to compare against the image. Lookup is by name only: symptom descriptions find nothing. Every "
            "excerpt comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {k: v for k, v in RETRIEVE_SCHEMA["function"]["parameters"]["properties"].items() if k != "description"},
            "required": ["candidates"],
        },
    },
}
# --retrieve-mode crosscheck: the grower's flow. Describe the photo, make an initial diagnosis (up to 4), cross-check it
# against the databases, decide. Both are required; the description narrows each excerpt to what was seen. API pilot
# (2026-10-30): name checks with a description fixed 9 / broke 2 of 60 items; description search alone did not help.
CROSSCHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Cross-check your initial diagnosis against external agricultural databases "
            f"({DATABASES}). Give a description of what you see and up to 4 candidates: the diseases, pests, species "
            "or quality classes you are considering (the answer options, if you are given options). Returns reference "
            "text for each candidate (appearance, symptoms, hosts, taxonomy), narrowed to the parts matching your "
            "description, to compare against the image before you decide. Every excerpt comes with its source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **RETRIEVE_SCHEMA["function"]["parameters"]["properties"],
                "description": {"type": "string", "description": "What you see in 2-4 sentences: plant part, colours, "
                                "shapes, texture, pattern and spread of any symptoms or damage, e.g. 'Bean leaf with many "
                                "small reddish-brown powdery pustules on the underside, each surrounded by a yellow halo; "
                                "older leaves are most affected.'"},
            },
            "required": ["description", "candidates"],
        },
    },
}
RETRIEVE_MODES = {"description+candidates": RETRIEVE_SCHEMA, "candidates": CANDIDATES_SCHEMA, "crosscheck": CROSSCHECK_SCHEMA}

PER_CHUNK_CHARS = 1200
# After a description search. Pilot 2: the model answered with the first page's cause and never checked it, and a
# search always returns pages, even for a healthy plant.
NEXT_STEP = ("\n\nThis is a search by symptoms, not a diagnosis: it returns pages even when nothing is wrong. Before "
             "answering, check the most likely causes (from these pages or your own) with candidates and compare their "
             "symptoms with the image.")
RESULT_CHARS = 6000
LIVE_LOOKUPS = 4  # concurrent live lookups across all items: the search backends rate-limit
_live = threading.Semaphore(LIVE_LOOKUPS)


def _cached(key: list, fetch):
    def live():
        with _live:
            return fetch()
    return cache.cached(key, live)


def _lookup(c: dict, crop: str, task: str) -> dict:
    """Every database is searched whatever the task, so a wrong guess about the kind of class misses nothing; the
    databases covering `task` come first (and get the result space first)."""
    from agml_agent.retrieve import _load_sources, retrieve

    name, sci, aliases = c["name"], c.get("scientific_name") or None, c.get("aliases") or []
    found = _cached(["name-all", name.strip().lower(), sci, aliases, crop.lower()],
                    lambda: retrieve(name, crop=crop, scientific_name=sci, aliases=aliases))
    covers = {src: cfg.get("task") or [] for src, cfg in _load_sources().items()}
    return dict(sorted(found.items(), key=lambda kv: task not in covers.get(kv[0], [])))


def _join(parts: list[str], limit: int) -> str:
    """Whole parts while they fit (a cut part would lose its source URL); the first one is cut if it alone is too long."""
    out, size = [], 0
    for part in parts:
        if size + len(part) > limit:
            if not out:
                out.append(part[:limit])
            break
        out.append(part)
        size += len(part) + 2
    return "\n\n".join(out)


def run_retrieve(description: str = "", candidates: list | None = None, crop: str = "", task: str = "") -> str:
    """The retrieve tool, reduced to what the model may see: excerpt text and source URL per chunk."""
    from agml_agent.symptom_search import focus, search_description

    cands = [{"name": c} if isinstance(c, str) else c for c in candidates or []]
    cands = [c for c in cands if isinstance(c, dict) and c.get("name")][:MAX_CANDIDATES]
    if not description and not cands:
        return "error: give a description of what you see and/or candidates (names) to check."
    if cands:
        with ThreadPoolExecutor(len(cands)) as ex:
            found = list(ex.map(lambda c: _lookup(c, crop, task), cands))
        sections = []
        for c, res in zip(cands, found):
            res = {src: [ch for ch in chs if not LEAK.search(ch["excerpt_text"])] for src, chs in res.items()}
            # sources take turns (first chunk of each, then second...), task-matching sources first
            chunks = [f"[{src}] {focus(chs[i]['excerpt_text'], description, PER_CHUNK_CHARS)} (source: {chs[i]['source_url']})"
                      for i in range(max(map(len, res.values()), default=0)) for src, chs in res.items() if i < len(chs)]
            body = _join(chunks, RESULT_CHARS // len(cands)) or (
                f"No results found for {c['name']!r}. Try its scientific name or aliases, or another candidate.")
            sections.append(f"## {c['name']}\n{body}")
        return "\n\n".join(sections)
    # "v4": pages cached before every description source and Wikipedia were searched are not reused
    pages = _cached(["description-v4", description.strip().lower(), crop.lower(), task],
                    lambda: search_description(description, crop, task))
    pages = [p for p in pages if not LEAK.search(p["text"])]
    if not pages:
        return ("No fact-sheet or Wikipedia pages matched this description. Check candidate names instead, or answer "
                "from the image.")
    other = f" (does not mention {crop})" if crop else ""
    head = (f"Pages matching the description, those mentioning {crop} first; pages about other hosts can still show "
            "a similar problem, compare them with the image" if crop else "Pages matching the description")
    return (f"{head} (titles usually name what they describe):\n\n"
            + _join([f"[{p['source']}] {p['title']}{'' if p.get('mentions_crop') else other}: "
                     f"{focus(p['text'], description, PER_CHUNK_CHARS)} (source: {p['url']})"
                     for p in pages], RESULT_CHARS - len(NEXT_STEP)) + NEXT_STEP)


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
                 trace_path: str | Path | None = None, served_name: str | None = None, retries: int = 3,
                 retrieve_mode: str = "description+candidates"):
        super().__init__(endpoint, model_id, thinking, effort, max_tokens, served_name=served_name)
        if bad := [t for t in tools if t not in BENCH_TOOLS]:
            raise ValueError(f"tools {bad} not allowed in benchmark runs; allowed: {sorted(BENCH_TOOLS)} "
                             "(search_agml is excluded: AgML class lists/results leak benchmark labels)")
        self.tools, self.max_tool_calls, self.retries = list(tools), max_tool_calls, retries
        self.trace_path = Path(trace_path) if trace_path else None
        if retrieve_mode not in RETRIEVE_MODES:
            raise ValueError(f"retrieve_mode {retrieve_mode!r}; one of {sorted(RETRIEVE_MODES)}")
        self.retrieve_mode = retrieve_mode
        self.name = f"{model_id} + tools" + ("" if retrieve_mode == "description+candidates" else f" ({retrieve_mode})")
        self._lock = asyncio.Lock()

    def config(self) -> dict:
        return {**super().config(), "name": self.name, "harness": "exploit", "tools": self.tools,
                "max_tool_calls": self.max_tool_calls, "retrieve_mode": self.retrieve_mode, "agml_agent": importlib.metadata.version("agml-agent"),
                "agml_agent_git": _git_sha(Path(__file__).parent), "agbench_commit": _agbench_commit(),
                "retrieve_cache": str(cache.CACHE.path) if cache.CACHE else None, "live_search": scrape.LIVE_SEARCH}

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
            if name == "retrieve" and self.retrieve_mode == "candidates":
                args.pop("description", None)  # not offered in this mode
            if name == "retrieve" and self.retrieve_mode == "crosscheck" and not (args.get("description") and args.get("candidates")):
                result = ("error: give both a description of what you see and candidates (your initial diagnosis, up "
                          "to 4 names) to cross-check.")
                return result, time.perf_counter() - t0
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
        schemas = [RETRIEVE_MODES[self.retrieve_mode] if t == "retrieve" else BENCH_TOOLS[t][0] for t in self.tools]

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
    cache.CACHE = None if args.no_retrieve_cache else RetrievalCache(args.retrieve_cache or args.out / "retrieve_cache.jsonl")
    scrape.LIVE_SEARCH = not args.no_live_search
    bench = Benchmark(args.data_dir, args.task or None, args.format or None, args.only, args.limit_per_dataset, args.seed)
    print(f"{len(bench.items)} items; {len(bench.skipped)} datasets skipped")
    spec = model_spec(args.model)
    conc = args.concurrency or (spec.get("parallel", 2) if spec["backend"] == "llamacpp" else spec["concurrency"])
    tag = "tools" if args.retrieve_mode == "description+candidates" else f"tools-{args.retrieve_mode}"
    rd = args.run_dir or args.out / f"{args.model.replace('/', '__')}__{tag}_{datetime.now():%Y%m%d-%H%M%S}"
    rd.mkdir(parents=True, exist_ok=True)

    def method(endpoint: str) -> HarnessMethod:
        return HarnessMethod(endpoint, args.model, args.thinking, args.reasoning_effort, args.max_tokens, args.tools,
                             args.max_tool_calls, trace_path=rd / "traces.jsonl", retrieve_mode=args.retrieve_mode)

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
    p.add_argument("--retrieve-mode", choices=sorted(RETRIEVE_MODES), default="description+candidates",
                   help="description+candidates: search by what the model sees and check names; candidates: check names only.")
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
    p.add_argument("--retrieve-cache", type=Path, help="Lookup cache shared across runs (default: <out>/retrieve_cache.jsonl).")
    p.add_argument("--no-retrieve-cache", action="store_true", help="Always look up live.")
    p.add_argument("--no-live-search", action="store_true",
                   help="Web searches (fact-sheet sites) come only from the cache; API sources still look up live. "
                        "Lookups made this way lack fact-sheet pages, so give the run its own --retrieve-cache copy.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
