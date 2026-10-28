"""
bench_view.py — Streamlit viewer for harness runs' tool use and traces
(traces.jsonl, written by bench.py). Scores themselves are compared in
agbench's own viewer (`agbench eval view --runs <baseline> <harness run>`);
this one adds what agbench doesn't know about: which tools were called,
with what arguments and results, and how that relates to the score.

    agml-agent-view --runs runs/<harness run> --baseline <agbench baseline run dir> [--data-dir /path]

Runs without traces.jsonl given in --runs/--baseline are baselines: their
answers are shown next to each trace for the same item.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="View harness tool use and traces.")
    p.add_argument("--runs", nargs="+", required=True, help="Harness run dir(s) (or folders of runs).")
    p.add_argument("--baseline", nargs="*", default=[], help="agbench run dir(s) to compare against (e.g. the no-tools run).")
    p.add_argument("--data-dir", help="Folder with base-data/, if images live elsewhere than when the runs were made.")
    p.add_argument("--port", type=int, default=8503)
    a = p.parse_args()
    cmd = [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", str(a.port), "--",
           "--runs", *[str(Path(r).resolve()) for r in a.runs + a.baseline]]
    if a.data_dir:
        cmd += ["--data-dir", str(Path(a.data_dir).resolve())]
    subprocess.run(cmd)


def app() -> None:
    import pandas as pd
    import streamlit as st
    from PIL import Image

    from agbench.eval.metrics import cv_key
    from agbench.eval.report import load_runs
    from agbench.eval.runner import read_jsonl

    st.set_page_config(page_title="Harness traces", layout="wide")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--data-dir", default=None)
    args, _ = parser.parse_known_args()

    @st.cache_data(ttl=60)
    def runs() -> list[dict]:
        return load_runs(args.runs)

    @st.cache_data(ttl=60)
    def frame(run_dir: str) -> pd.DataFrame:
        """Predictions joined with the last trace per item (a resumed run may have retried an item)."""
        p = pd.DataFrame(read_jsonl(Path(run_dir) / "predictions.jsonl"))
        if not len(p):
            return p
        p["cv"] = [cv_key(r) for r in p.to_dict("records")]
        p["score"] = [r["judge_correct"] if r.get("judge_correct") is not None and r["judge_correct"] == r["judge_correct"]
                      else float(r["correct"]) for r in p.to_dict("records")]
        tr = {t["item_id"]: t for t in read_jsonl(Path(run_dir) / "traces.jsonl")}
        if tr:
            p["n_tool_calls"] = [tr.get(i, {}).get("n_tool_calls", 0) for i in p.item_id]
            p["tools_used"] = [tr.get(i, {}).get("tools_used", []) for i in p.item_id]
            p["tool_s"] = [tr.get(i, {}).get("tool_s", 0.0) for i in p.item_id]
            p["trace"] = [tr.get(i, {}).get("trace") for i in p.item_id]
        return p

    def image_path(path: str) -> str | None:
        if Path(path).exists():
            return path
        if args.data_dir and "base-data/" in path:
            q = Path(args.data_dir) / path[path.index("base-data/"):]
            if q.exists():
                return str(q)
        return None

    harness = [r for r in runs() if (Path(r["dir"]) / "traces.jsonl").exists()]
    baselines = [r for r in runs() if r not in harness]

    def base_scores() -> pd.DataFrame:
        """item_id x baseline run -> score."""
        cols = {r["name"]: frame(r["dir"]).set_index("item_id")["score"] for r in baselines if len(frame(r["dir"]))}
        return pd.DataFrame(cols)

    def usage_row(name: str, p: pd.DataFrame) -> dict:
        n = p.n_tool_calls
        by_tool = pd.Series([t for ts in p.tools_used for t in ts]).value_counts()
        return {"run": name, "items": len(p), "items with tool calls (%)": round((n > 0).mean() * 100, 1),
                "calls / item": round(n.mean(), 2), "max calls": int(n.max()),
                "score with tools (%)": round(p.score[n > 0].mean() * 100, 1) if (n > 0).any() else None,
                "score without tools (%)": round(p.score[n == 0].mean() * 100, 1) if (n == 0).any() else None,
                "tool s / item": round(p.tool_s.mean(), 2),
                "calls by tool": ", ".join(f"{k} ({v})" for k, v in by_tool.items())}

    def page_usage():
        st.title("Tool use")
        if not harness:
            st.info(f"No harness runs (with traces.jsonl) under {args.runs}.")
            return
        st.caption("Pooled over items (micro). Score: judge verdict where judged, box score for detection/segmentation. "
                   "Compare macro scores with the baseline in agbench's viewer.")
        st.dataframe(pd.DataFrame([usage_row(r["name"], frame(r["dir"])) for r in harness]), hide_index=True, width="stretch")
        for r in harness:
            p = frame(r["dir"])
            st.subheader(r["name"])
            st.markdown("**By CV task**")
            st.dataframe(pd.DataFrame([usage_row(k, g) for k, g in p.groupby("cv")]).rename(columns={"run": "cv task"}),
                         hide_index=True, width="stretch")
            bs = base_scores()
            for b in bs.columns:
                j = p.set_index("item_id")[["score", "n_tool_calls", "cv"]].join(bs[b].rename("base"), how="inner")
                if not len(j):
                    continue
                j["fixed"], j["broken"] = (j.base < 0.5) & (j.score >= 0.5), (j.base >= 0.5) & (j.score < 0.5)
                st.markdown(f"**vs {b}** ({len(j)} shared items): fixed = baseline wrong, harness right; broken = the reverse")
                t = j.groupby("cv").agg(items=("score", "size"), baseline=("base", "mean"), harness=("score", "mean"),
                                        fixed=("fixed", "sum"), broken=("broken", "sum"),
                                        fixed_with_tools=("fixed", lambda s: int((s & (j.loc[s.index, "n_tool_calls"] > 0)).sum())))
                t["baseline"], t["harness"] = (t.baseline * 100).round(1), (t.harness * 100).round(1)
                st.dataframe(t, width="stretch")

    def show_trace(trace: list):
        step = 0
        for s in trace:
            if s.get("role") == "tool":
                st.markdown(f"↳ **{s.get('name')}** result ({s.get('latency_s', 0):.1f}s)")
                st.code(str(s.get("result", ""))[:4000], language=None, wrap_lines=True)
                continue
            step += 1
            st.markdown(f"**Model turn {step}**")
            if s.get("reasoning"):
                st.caption("Reasoning")
                st.text(s["reasoning"][:4000])
            for c in s.get("tool_calls") or []:
                st.code(f"{c.get('name')}({json.dumps(c.get('arguments'), ensure_ascii=False)})", language="python", wrap_lines=True)
            if s.get("text"):
                st.caption("Text")
                st.text(s["text"][:2000])

    def page_traces():
        st.title("Traces")
        if not harness:
            st.info(f"No harness runs (with traces.jsonl) under {args.runs}.")
            return
        names = {f"{r['name']} · {Path(r['dir']).name}": r["dir"] for r in harness}
        c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1, 1, 1, 1])
        run_dir = names[c1.selectbox("Run", list(names))]
        p = frame(run_dir)
        dataset = c2.selectbox("Dataset", ["all"] + sorted(p.dataset.unique(), key=str.lower))
        v = p if dataset == "all" else p[p.dataset == dataset]
        fmt = c3.selectbox("Format", ["all"] + sorted(v.format.unique()))
        v = v if fmt == "all" else v[v.format == fmt]
        which = c4.selectbox("Show", ["all", "wrong", "correct"])
        if which != "all":
            v = v[(v.score >= 0.5) == (which == "correct")]
        tools = c5.selectbox("Tools", ["all", "used", "not used"])
        if tools != "all":
            v = v[(v.n_tool_calls > 0) == (tools == "used")]
        bs = base_scores()
        if len(bs.columns):
            base = c6.selectbox("vs baseline", ["all", "fixed", "broken"], help=f"Against {bs.columns[0]}")
            if base != "all":
                b = v.item_id.map(bs.iloc[:, 0])
                v = v[(b < 0.5) & (v.score >= 0.5)] if base == "fixed" else v[(b >= 0.5) & (v.score < 0.5)]
        items = {it["item_id"]: it for it in read_jsonl(Path(run_dir) / "items.jsonl")}
        base_recs = {r["name"]: frame(r["dir"]).set_index("item_id") for r in baselines if len(frame(r["dir"]))}
        st.caption(f"{len(v)} items")
        page_size = 10
        page = st.number_input("Page", 1, max(1, (len(v) + page_size - 1) // page_size), 1)
        for _, r in v.iloc[(page - 1) * page_size: page * page_size].iterrows():
            it = items.get(r["item_id"], {})
            with st.container(border=True):
                a, b = st.columns([1, 2])
                with a:
                    for img in list(filter(None, map(image_path, (it.get("images") or [])[:2]))):
                        st.image(Image.open(img).convert("RGB"), width="stretch")
                with b:
                    st.markdown(f"**{'✅' if r['score'] >= 0.5 else '❌'}** `{r['item_id']}` · {r['dataset']} · {r['cv']} · "
                                f"ag task: {r['ag_task']} · {int(r['n_tool_calls'])} tool calls, {r['tool_s']:.1f}s")
                    if it.get("prompt"):
                        with st.expander("Prompt"):
                            st.text(it["prompt"][:3000])
                    st.markdown(f"**Answer:** {r['answer'] or '_(empty)_'}  \n**Ground truth:** {r['gt']}")
                    for n, br in base_recs.items():
                        if r["item_id"] in br.index:
                            x = br.loc[r["item_id"]]
                            st.caption(f"{n}: {'✅' if x['score'] >= 0.5 else '❌'} {str(x['answer'])[:300]}")
                    if isinstance(r.get("trace"), list):
                        with st.expander(f"Trace ({int(r['n_tool_calls'])} tool calls)", expanded=r["n_tool_calls"] > 0):
                            show_trace(r["trace"])

    st.navigation([st.Page(page_usage, title="Tool use", default=True), st.Page(page_traces, title="Traces")]).run()


if __name__ == "__main__":
    app()
