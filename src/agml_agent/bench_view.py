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
import re
import subprocess
import sys
from pathlib import Path


# Runs before results were labelled by source name labelled description pages by site.
OLD_LABELS = {"ipm.ucanr.edu": "ucipm_aps", "bugwood.org": "bugwood"}
# "[names: eppo, gbif]" (--clean-text 2) is one block labelled "names".
BLOCK = re.compile(r"^\[([a-z_.]+)(?::[^\]\n]*)?\] ?(.*?)(?=^\[[a-z_.]+(?::[^\]\n]*)?\] |^## |\Z)", re.M | re.S)
SECTION = re.compile(r"^## (.+?)\n(.*?)(?=^## |\Z)", re.M | re.S)


def _shingles(text: str, n: int = 4) -> set[tuple]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _source_names() -> set[str]:
    from agml_agent.retrieve import _load_sources

    return set(_load_sources()) | set(OLD_LABELS) | {"local_kb", "names"}


def _turns(trace: list, i: int) -> tuple[set, set, str]:
    """Shingles of the model's text before and after step i, and the later text itself (lower case)."""
    text = lambda ts: " ".join((t.get("reasoning") or "") + " " + (t.get("text") or "") + " " + json.dumps(
        [c.get("arguments") for c in t.get("tool_calls") or []]) for t in ts if t.get("role") == "assistant")
    later = text(trace[i + 1:])
    return _shingles(text(trace[:i])), _shingles(later), later.lower()


def _uses(body: str, earlier: set, later: set) -> bool:
    """The later text repeats at least two 4-word phrases from `body` that the model had not written before the result
    (its description and first answer are not evidence of use, and reference text is narrowed to match them)."""
    return len((_shingles(body) & later) - earlier) >= 2


def source_use(trace: list | None) -> tuple[list[str], list[str]]:
    """(sources shown, sources used) in one item's trace. Shown: the source's text was in a tool result. Used: the
    model's later reasoning or answer repeats two 4-word phrases from that text that it had not written before, or a
    page title from it."""
    shown, used, known = [], [], _source_names()
    for i, s in enumerate(trace or []):
        if s.get("role") != "tool":
            continue
        earlier_sh, later_sh, later_low = _turns(trace, i)
        for m in BLOCK.finditer(str(s.get("result", ""))):
            if m.group(1) not in known:  # a bracketed word inside page text, not a source label
                continue
            src, body = OLD_LABELS.get(m.group(1), m.group(1)), m.group(2)
            body = re.sub(r"\(source: [^)]*\)", "", body)
            title = body.split(":", 1)[0].strip().lower() if ":" in body[:150] else ""
            shown.append(src)
            if _uses(body, earlier_sh, later_sh) or (len(title) > 12 and title in later_low):
                used.append(src)
    return sorted(set(shown)), sorted(set(used))


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower().replace("_", " ")).split())


def kb_sections(trace: list | None, answer: str = "") -> list[dict]:
    """One row per knowledge-base section shown to the model: the candidate it was looked up for, the crop given, the
    page title and heading, and checks of relevance (title or heading names the candidate; the crop is named; the
    heading is about appearance; the candidate is the correct class) and use (as in source_use)."""
    from agml_agent.kb import kind_of

    rows, gt = [], _norm(answer)
    for i, s in enumerate(trace or []):
        if s.get("role") != "tool" or "[local_kb]" not in str(s.get("result", "")):
            continue
        args = next((c.get("arguments") for c in reversed(trace[i - 1].get("tool_calls") or [])), {}) if i else {}
        crop = (args or {}).get("crop") or "" if isinstance(args, dict) else ""
        earlier_sh, later_sh, _ = _turns(trace, i)
        for sec in SECTION.finditer(str(s["result"])):
            cand = sec.group(1).strip()
            for m in BLOCK.finditer(sec.group(2)):
                if m.group(1) != "local_kb":
                    continue
                body = re.sub(r"\(source: ([^)]*)\)", "", m.group(2)).strip()
                url = (re.search(r"\(source: ([^)]*)\)", m.group(2)) or [None, ""])[1]
                head, _, text = body.partition(": ")
                title, _, heading = head.partition(" > ")
                c, low = _norm(cand), _norm(head)
                page = _norm(re.sub(r"\|.*$", "", title))  # title without "| NC State Extension Publications"
                rows.append({"candidate": cand, "crop": crop, "title": title, "heading": heading, "text": text, "url": url,
                             "names candidate": bool(c) and c in low,
                             "page is the candidate": bool(c) and (page == c or page.startswith(c + " ") and len(page) <= len(c) + 25
                                                                   and not re.search(r"\b(moth|beetle|weevil|borer|mite|aphid)\b", page[len(c):])),
                             "names crop": bool(crop) and _norm(crop) in _norm(body),
                             "appearance section": kind_of(heading) == "symptom",
                             "correct class": bool(c and gt) and (c == gt or (len(gt) > 3 and gt in c) or (len(c) > 3 and c in gt)),
                             "used": _uses(body, earlier_sh, later_sh)})
    return rows


def kb_summary(p) -> dict:
    """Knowledge-base relevance and use over one run's frame (rows of predictions with `trace` and `gt`)."""
    import pandas as pd

    rows = [dict(r, item_id=i, score=sc) for i, t, gt, sc in zip(p["item_id"], p["trace"], p["gt"], p["score"])
            for r in kb_sections(t, str(gt))]
    calls = sum(1 for t in p.trace for s in (t or []) if s.get("role") == "tool")
    k = pd.DataFrame(rows)
    out = {"tool results": calls, "with knowledge-base text": int(k.item_id.nunique()) if len(k) else 0,
           "sections shown": len(k)}
    if len(k):
        for col in ("names candidate", "page is the candidate", "appearance section", "names crop", "correct class", "used"):
            out[f"{col} (%)"] = round(k[col].mean() * 100, 1)
        cc = k[k["correct class"]]
        out["used, correct class (%)"] = round(cc.used.mean() * 100, 1) if len(cc) else None
    return out, k


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
            use = [source_use(t) for t in p.trace]
            p["sources_shown"], p["sources_used"] = [u[0] for u in use], [u[1] for u in use]
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

    def page_sources():
        st.title("Sources")
        if not harness:
            st.info(f"No harness runs (with traces.jsonl) under {args.runs}.")
            return
        bs = base_scores()
        if not len(bs.columns):
            st.info("Pass a baseline run (--baseline) to see which items each source fixed or broke.")
        st.caption("Shown: the source's text was in a tool result for the item. Used: the model's later reasoning or "
                   "answer repeats a phrase (2+ four-word phrases) or a page title from it. Fixed / broken: against the "
                   "baseline (baseline wrong, harness right / the reverse). An item counts for every source it showed, so "
                   "sources overlap. Compare with the noise line: items where no tool was called, which flip only from "
                   "sampling.")
        for r in harness:
            p = frame(r["dir"])
            if "sources_shown" not in p:
                continue
            st.subheader(r["name"])
            j = p.set_index("item_id")
            if len(bs.columns):
                j = j.join(bs.iloc[:, 0].rename("base"), how="inner")
                j["fixed"], j["broken"] = (j.base < 0.5) & (j.score >= 0.5), (j.base >= 0.5) & (j.score < 0.5)
            rows = []
            for src in sorted({x for xs in j.sources_shown for x in xs}):
                row = {"source": src}
                for kind in ("shown", "used"):
                    m = j[f"sources_{kind}"].map(lambda xs: src in xs)
                    row[f"items {kind}"] = int(m.sum())
                    if "base" in j:
                        row[f"fixed ({kind})"], row[f"broken ({kind})"] = int(j.fixed[m].sum()), int(j.broken[m].sum())
                        row[f"net ({kind})"] = row[f"fixed ({kind})"] - row[f"broken ({kind})"]
                    row[f"score ({kind}, %)"] = round(j.score[m].mean() * 100, 1) if m.any() else None
                rows.append(row)
            t = pd.DataFrame(rows)
            if "base" in j and len(t):
                quiet = j.n_tool_calls == 0
                t = pd.concat([t, pd.DataFrame([{"source": "(noise: no tool call)", "items shown": int(quiet.sum()),
                                                 "fixed (shown)": int(j.fixed[quiet].sum()),
                                                 "broken (shown)": int(j.broken[quiet].sum()),
                                                 "net (shown)": int(j.fixed[quiet].sum() - j.broken[quiet].sum()),
                                                 "score (shown, %)": round(j.score[quiet].mean() * 100, 1) if quiet.any() else None}])])
            st.dataframe(t, hide_index=True, width="stretch")
            if "base" in j and len(t) > 1:
                st.markdown("**By CV task** (items where the source was used)")
                by = [{"cv task": cv, "source": src, "items": int(m.sum()), "fixed": int(g.fixed[m].sum()),
                       "broken": int(g.broken[m].sum())}
                      for cv, g in j.groupby("cv") for src in sorted({x for xs in g.sources_used for x in xs})
                      if (m := g.sources_used.map(lambda xs, s=src: s in xs)).any()]
                st.dataframe(pd.DataFrame(by), hide_index=True, width="stretch")

    def page_kb():
        st.title("Knowledge base")
        st.caption("Each section the local knowledge base returned for a candidate. Names candidate: the page title or "
                   "heading contains the candidate's name. Page is the candidate: the page is about exactly it (not "
                   "'Banana Moth' for 'banana'). Names crop: the crop given in the call appears in the section. "
                   "Appearance section: the heading is about symptoms, damage, identification or description. Correct "
                   "class: the candidate is the item's answer. Used: the model's later reasoning or answer repeats two "
                   "4-word phrases from the section that it had not written before the result.")
        runs_kb = [r for r in harness if frame(r["dir"]).trace.map(lambda t: "[local_kb]" in json.dumps(t)).any()]
        if not runs_kb:
            st.info("No run here showed knowledge-base text (run with --kb).")
            return
        summ = []
        for r in runs_kb:
            o, _ = kb_summary(frame(r["dir"]))
            summ.append({"run": f"{r['name']} · {Path(r['dir']).name}", **o})
        st.dataframe(pd.DataFrame(summ), hide_index=True, width="stretch")
        st.markdown("**Use by source** (share of items where a source was shown and then used)")
        rows = []
        for r in harness:
            p = frame(r["dir"])
            if "sources_shown" not in p:
                continue
            for src in sorted({x for xs in p.sources_shown for x in xs}):
                shown = p.sources_shown.map(lambda xs: src in xs)
                used = p.sources_used.map(lambda xs: src in xs)
                rows.append({"run": Path(r["dir"]).name, "source": src, "items shown": int(shown.sum()),
                             "used (%)": round(used[shown].mean() * 100, 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        names = {f"{r['name']} · {Path(r['dir']).name}": r["dir"] for r in runs_kb}
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        _, k = kb_summary(frame(names[c1.selectbox("Run", list(names))]))
        filt = {c: c2.selectbox(c, ["all", "yes", "no"], key=c) if i == 0 else (c3 if i == 1 else c4).selectbox(c, ["all", "yes", "no"], key=c)
                for i, c in enumerate(["page is the candidate", "correct class", "used"])}
        for c, f in filt.items():
            if f != "all":
                k = k[k[c] == (f == "yes")]
        st.caption(f"{len(k)} sections")
        st.dataframe(k[["item_id", "candidate", "crop", "title", "heading", "names candidate", "page is the candidate",
                        "names crop", "appearance section", "correct class", "used", "text", "url"]],
                     hide_index=True, width="stretch", height=600)

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
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 2, 1, 1, 1, 1, 1])
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
        if "sources_shown" in v:
            src = c7.selectbox("Source", ["all"] + sorted({x for xs in p.sources_shown for x in xs}),
                               help="Items where this source was in a tool result")
            if src != "all":
                v = v[v.sources_shown.map(lambda xs: src in xs)]
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
                    if r.get("sources_shown"):
                        st.caption("Sources shown: " + ", ".join(f"**{x}**" if x in r["sources_used"] else x
                                                                 for x in r["sources_shown"]) + " (bold: used)")
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

    st.navigation([st.Page(page_usage, title="Tool use", default=True), st.Page(page_sources, title="Sources"),
                   st.Page(page_kb, title="Knowledge base"), st.Page(page_traces, title="Traces")]).run()


if __name__ == "__main__":
    app()
