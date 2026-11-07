#!/usr/bin/env python3
"""Aggregate results, paired bootstrap CIs, McNemar tests, Holm correction."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score

from graph_core import embed_texts, load_corpus, load_gold, stanza_sort_key
from llm import OllamaClient

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"

COND_ORDER = [
    ("c0_closed", "Closed-book"),
    ("c1_naive", "NaiveRAG"),
    ("c2_full", "LightRAG-Full"),
    ("c3_low", "LightRAG-LowOnly"),
    ("c4_high", "LightRAG-HighOnly"),
    ("c5_origin", "LightRAG-NoOriginalText"),
    ("c6_empty", "LightRAG-EmptyProfiles"),
    ("c7_shuffled", "LightRAG-ShuffledStructure"),
    ("c8_extractive", "LightRAG-ExtractiveProfiles"),
    ("c9_temporal", "LightRAG-TemporalGraph"),
    ("c10_naive_temporal", "NaiveRAG-TemporalOrder"),
]
COND_LABEL = dict(COND_ORDER)


def parse_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def parse_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_rows() -> pd.DataFrame:
    gold = {q["id"]: q for q in load_gold()}
    rows = []
    for cond, label in COND_ORDER:
        path = RESULTS / f"answers_{cond}.json"
        if not path.exists():
            print("missing", path)
            continue
        answers = json.loads(path.read_text(encoding="utf-8"))
        judged = {}
        jp = RESULTS / f"judged_judge1_{cond}.json"
        if jp.exists():
            judged = json.loads(jp.read_text(encoding="utf-8"))
        j2p = RESULTS / f"judged_judge2_{cond}.json"
        judged2 = json.loads(j2p.read_text(encoding="utf-8")) if j2p.exists() else {}
        for qid, r in answers.items():
            q = gold[qid]
            j = (judged.get(qid, {}) or {}).get("parsed", {}) or {}
            j2 = (judged2.get(qid, {}) or {}).get("parsed", {}) or {}
            corr = parse_int(j.get("correctness"), None)
            corr2 = parse_int(j2.get("correctness"), None)
            claims = j.get("claims") or []
            supported = [c for c in claims if isinstance(c, dict) and c.get("supported") in (True, False)]
            unsupported_rate = (
                sum(1 for c in supported if c.get("supported") is False) / len(supported)
                if supported else np.nan)
            order = j.get("order_results") or []
            order_correct = sum(1 for o in order if o.get("result") == "correct")
            order_wrong = sum(1 for o in order if o.get("result") == "wrong")
            order_total = order_correct + order_wrong
            order_acc = order_correct / order_total if order_total else np.nan
            ctx = r.get("context", "") or ""
            ev = q.get("evidence", [])
            ev_hits = [s for s in ev if re.search(rf"\[{s}\]", ctx)]
            cited = r.get("citations", []) or []
            cit_recall = (len(set(cited) & set(ev)) / len(set(ev))) if ev else np.nan
            rows.append({
                "condition": cond, "label": label, "qid": qid, "type": q["type"],
                "question": q["question"], "answer": r.get("answer", ""),
                "temporal_question": r.get("temporal_question", False),
                "temporal_activated": r.get("temporal_activated", False),
                "correctness": corr, "correctness2": corr2,
                "correct_norm": corr / 2 if corr is not None else np.nan,
                "strict_correct": 1.0 if corr == 2 else (0.0 if corr is not None else np.nan),
                "lenient_correct": 1.0 if corr is not None and corr >= 1 else (0.0 if corr is not None else np.nan),
                "key_point_coverage": parse_float(j.get("key_point_coverage"), np.nan),
                "citation_precision": parse_float(j.get("citation_precision"), np.nan),
                "citation_recall": cit_recall,
                "n_claims": len(supported),
                "unsupported_rate": unsupported_rate,
                "order_accuracy": order_acc,
                "sequence_correct": 1.0 if j.get("sequence_correct") is True else (0.0 if j.get("sequence_correct") is False else np.nan),
                "duration_correct": 1.0 if j.get("duration_correct") is True else (0.0 if j.get("duration_correct") is False else np.nan),
                "evidence_recall": len(ev_hits) / len(ev) if ev else np.nan,
                "n_citations": len(cited),
                "has_citation": 1.0 if cited else 0.0,
                "refusal": 1.0 if "provided context does not establish" in (r.get("answer", "").lower()) else 0.0,
                "context_tokens": r.get("context_meta", {}).get("context_tokens_est", np.nan),
                "graph_tokens": r.get("context_meta", {}).get("graph_tokens_est", np.nan),
                "chunk_tokens": r.get("context_meta", {}).get("chunk_tokens_est", np.nan),
                "n_chunks": len(r.get("context_meta", {}).get("used_chunks", [])),
                "answer_chars": len(r.get("answer", "")),
                "wall_seconds": r.get("wall_seconds", np.nan),
                "usage": r.get("usage", {}),
                "error": r.get("error"),
            })
    return pd.DataFrame(rows)


def attach_semantic(df: pd.DataFrame) -> pd.DataFrame:
    gold = {q["id"]: q for q in load_gold()}
    cache_path = RESULTS / "semantic_sim.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = []
    for i, r in df.iterrows():
        key = f"{r['condition']}|{r['qid']}"
        if key not in cache:
            todo.append((i, key))
    if todo:
        client = OllamaClient()
        texts = []
        for i, key in todo:
            r = df.loc[i]
            texts.extend([r["answer"], gold[r["qid"]]["gold_answer"]])
        emb = embed_texts(client, texts)
        for k, (i, key) in enumerate(todo):
            a, b = emb[2 * k], emb[2 * k + 1]
            cache[key] = float(np.dot(a, b))
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    df["semantic_cos"] = [cache.get(f"{r['condition']}|{r['qid']}", np.nan) for _, r in df.iterrows()]
    return df


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 42):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    if len(d) == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return dict(diff=float(d.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), p=float(min(1.0, p)), n=int(len(d)))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    out = [1.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        adj = max(adj, prev)
        out[i] = adj
        prev = adj
    return out


def mcnemar(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return dict(b=n01, c=n10, p=1.0)
    return dict(b=n01, c=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def main() -> None:
    df = load_rows()
    # The judge does not emit a separate duration label for every item; for the
    # three duration questions we use the correctness score as the duration score.
    for i, r in df.iterrows():
        if r["qid"] in {"Q18", "Q19", "Q20"}:
            df.at[i, "duration_correct"] = r["correct_norm"]
    print("rows", len(df), "conditions", df["condition"].nunique())
    if df.empty:
        return
    # Key-point coverage is only defined for thematic questions.
    df.loc[df["type"] != "thematic", "key_point_coverage"] = np.nan
    # No citations -> citation precision is undefined, not perfect.
    df.loc[df["has_citation"] == 0, "citation_precision"] = np.nan
    df.to_csv(RESULTS / "all_answer_rows.csv", index=False)
    df = attach_semantic(df)
    df.to_csv(RESULTS / "all_answer_rows_with_semantic.csv", index=False)

    # Main table
    metrics = ["correct_norm", "strict_correct", "lenient_correct", "key_point_coverage",
               "f1" if "f1" in df else None, "semantic_cos", "citation_precision",
               "citation_recall", "unsupported_rate", "evidence_recall",
               "context_tokens", "chunk_tokens", "answer_chars", "wall_seconds"]
    metrics = [m for m in metrics if m and m in df]
    # deterministic EM/F1 not yet attached; compute quickly
    from evaluate import answer_metrics
    gold = {q["id"]: q for q in load_gold()}
    df["em"] = [answer_metrics(r["answer"], gold[r["qid"]])["em"] for _, r in df.iterrows()]
    df["f1"] = [answer_metrics(r["answer"], gold[r["qid"]])["f1"] for _, r in df.iterrows()]
    df["contains"] = [answer_metrics(r["answer"], gold[r["qid"]])["contains"] for _, r in df.iterrows()]
    df.to_csv(RESULTS / "all_answer_rows_with_metrics.csv", index=False)
    metrics = ["correct_norm", "strict_correct", "lenient_correct", "em", "f1", "contains",
               "semantic_cos", "key_point_coverage", "citation_precision", "citation_recall",
               "unsupported_rate", "has_citation", "refusal", "evidence_recall", "context_tokens",
               "chunk_tokens", "answer_chars", "wall_seconds"]
    rows = []
    for cond, label in COND_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        row = {"condition": cond, "label": label, "n": len(sub)}
        for m in metrics:
            mean, lo, hi = boot_ci(sub[m].to_numpy())
            row[m] = mean
            row[m + "_lo"] = lo
            row[m + "_hi"] = hi
        rows.append(row)
    main_tab = pd.DataFrame(rows)
    main_tab.to_csv(RESULTS / "table_main.csv", index=False)
    print("\n=== main table ===")
    print(main_tab[["label", "n", "correct_norm", "strict_correct", "citation_precision",
                    "citation_recall", "has_citation", "refusal", "unsupported_rate", "evidence_recall",
                    "context_tokens", "chunk_tokens"]].to_string(index=False))

    # By type
    by_rows = []
    for (cond, typ), sub in df.groupby(["condition", "type"]):
        row = {"condition": cond, "label": COND_LABEL.get(cond, cond), "type": typ, "n": len(sub)}
        for m in ["correct_norm", "strict_correct", "key_point_coverage", "citation_precision",
                  "unsupported_rate", "evidence_recall", "order_accuracy", "sequence_correct",
                  "duration_correct", "context_tokens"]:
            mean, lo, hi = boot_ci(sub[m].to_numpy())
            row[m] = mean; row[m + "_lo"] = lo; row[m + "_hi"] = hi
        by_rows.append(row)
    by_tab = pd.DataFrame(by_rows)
    by_tab.to_csv(RESULTS / "table_by_type.csv", index=False)

    # Paired comparisons
    def arr(cond, qids, metric="correct_norm"):
        s = df[(df["condition"] == cond) & (df["qid"].isin(qids))].set_index("qid")[metric]
        return s

    all_qids = sorted(df["qid"].unique())
    temporal_qids = sorted(df[df["type"] == "temporal"]["qid"].unique())
    nontemporal_qids = [q for q in all_qids if q not in temporal_qids]
    comparisons = []
    h1 = [("c2_full", "c1_naive"), ("c2_full", "c6_empty"), ("c2_full", "c7_shuffled"),
          ("c2_full", "c5_origin"), ("c2_full", "c3_low"), ("c2_full", "c4_high"),
          ("c6_empty", "c1_naive"), ("c0_closed", "c1_naive")]
    for a, b in h1:
        sa, sb = arr(a, all_qids), arr(b, all_qids)
        common = sa.index.intersection(sb.index)
        res = paired_diff(sa.loc[common].to_numpy(), sb.loc[common].to_numpy())
        res.update({"family": "H1", "a": a, "b": b, "scope": "all"})
        comparisons.append(res)
    for a, b in [("c2_full", "c8_extractive")]:
        sa, sb = arr(a, all_qids), arr(b, all_qids)
        common = sa.index.intersection(sb.index)
        for metric in ["correct_norm", "citation_precision", "unsupported_rate"]:
            sa2 = arr(a, all_qids, metric); sb2 = arr(b, all_qids, metric)
            common = sa2.index.intersection(sb2.index)
            res = paired_diff(sa2.loc[common].to_numpy(), sb2.loc[common].to_numpy())
            res.update({"family": "H2", "a": a, "b": b, "scope": metric})
            comparisons.append(res)
    for a, b, scope in [("c9_temporal", "c2_full", "temporal"), ("c9_temporal", "c10_naive_temporal", "temporal"),
                        ("c10_naive_temporal", "c1_naive", "temporal"), ("c2_full", "c1_naive", "temporal")]:
        sa, sb = arr(a, temporal_qids), arr(b, temporal_qids)
        common = sa.index.intersection(sb.index)
        res = paired_diff(sa.loc[common].to_numpy(), sb.loc[common].to_numpy())
        res.update({"family": "H3", "a": a, "b": b, "scope": scope})
        comparisons.append(res)
    comp = pd.DataFrame(comparisons)
    # Holm within each family
    comp["p_holm"] = np.nan
    for fam, idxs in comp.groupby("family").groups.items():
        ps = comp.loc[idxs, "p"].fillna(1.0).tolist()
        comp.loc[idxs, "p_holm"] = holm(ps)
    comp.to_csv(RESULTS / "table_paired.csv", index=False)
    print("\n=== paired comparisons ===")
    print(comp[["family", "a", "b", "scope", "diff", "lo", "hi", "p", "p_holm", "n"]].to_string(index=False))

    # McNemar for strict binary
    mc_rows = []
    for fam, pairs in {
        "H1": h1,
        "H3": [("c9_temporal", "c2_full"), ("c9_temporal", "c10_naive_temporal"),
               ("c10_naive_temporal", "c1_naive"), ("c2_full", "c1_naive")],
    }.items():
        for a, b in pairs:
            qids = temporal_qids if fam == "H3" else all_qids
            sa = arr(a, qids, "strict_correct"); sb = arr(b, qids, "strict_correct")
            common = sa.index.intersection(sb.index)
            m = mcnemar(sa.loc[common].to_numpy(), sb.loc[common].to_numpy())
            m.update({"family": fam, "a": a, "b": b})
            mc_rows.append(m)
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(RESULTS / "table_mcnemar.csv", index=False)
    print("\n=== McNemar (strict) ===")
    print(mc.to_string(index=False))

    # H3 within-C2 temporal vs non-temporal
    def type_mean(cond, typ, metric="correct_norm"):
        sub = df[(df["condition"] == cond) & (df["type"] == typ)][metric].dropna().to_numpy()
        return sub
    h3_rows = []
    for cond in ["c0_closed", "c1_naive", "c2_full", "c3_low", "c4_high", "c5_origin",
                 "c6_empty", "c7_shuffled", "c8_extractive", "c9_temporal", "c10_naive_temporal"]:
        for typ in ["factual", "multihop", "thematic", "temporal"]:
            x = type_mean(cond, typ)
            if len(x):
                m, lo, hi = boot_ci(x)
                h3_rows.append({"condition": cond, "type": typ, "n": len(x),
                                "correct_norm": m, "lo": lo, "hi": hi,
                                "order_accuracy": type_mean(cond, typ, "order_accuracy").mean() if typ == "temporal" else np.nan,
                                "sequence_correct": type_mean(cond, typ, "sequence_correct").mean() if typ == "temporal" else np.nan,
                                "duration_correct": type_mean(cond, typ, "duration_correct").mean() if typ == "temporal" else np.nan})
    h3 = pd.DataFrame(h3_rows)
    h3.to_csv(RESULTS / "table_h3_temporal.csv", index=False)

    # Judge agreement
    judge_rows = []
    for cond, label in COND_ORDER:
        sub = df[(df["condition"] == cond) & df["correctness"].notna() & df["correctness2"].notna()]
        if len(sub) < 3:
            continue
        y1 = sub["correctness"].astype(int).to_numpy()
        y2 = sub["correctness2"].astype(int).to_numpy()
        try:
            kap = cohen_kappa_score(y1, y2, weights="quadratic")
        except Exception:
            kap = np.nan
        b1 = (sub["correctness"] == 2).astype(int).to_numpy()
        b2 = (sub["correctness2"] == 2).astype(int).to_numpy()
        try:
            kappab = cohen_kappa_score(b1, b2)
        except Exception:
            kappab = np.nan
        agree = float((y1 == y2).mean())
        rho = spearmanr(y1, y2).correlation if len(set(y1) | set(y2)) > 1 else np.nan
        judge_rows.append({"condition": cond, "label": label, "n": len(sub),
                           "agreement": agree, "kappa_3": kap, "kappa_binary": kappab,
                           "spearman": rho})
    judge_tab = pd.DataFrame(judge_rows)
    judge_tab.to_csv(RESULTS / "table_judge_agreement.csv", index=False)
    print("\n=== judge agreement ===")
    print(judge_tab.to_string(index=False))

    # Profile audit summary
    pa_path = RESULTS / "profile_audit.json"
    if pa_path.exists():
        pa = json.loads(pa_path.read_text(encoding="utf-8"))["samples"]
        pa_rows = []
        for mode in ("llm", "extractive"):
            for kind in ("entity", "relation"):
                vals = [s["parsed"].get("support_rate") for s in pa
                        if s["mode"] == mode and s["kind"] == kind and isinstance(s["parsed"].get("support_rate"), (int, float))]
                if vals:
                    pa_rows.append({"mode": mode, "kind": kind, "n": len(vals),
                                    "mean_support": float(np.mean(vals)),
                                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
        pd.DataFrame(pa_rows).to_csv(RESULTS / "table_profile_audit.csv", index=False)
        print("\n=== profile audit ===")
        print(pd.DataFrame(pa_rows).to_string(index=False))

    # Save summary JSON
    summary = {
        "n_rows": len(df),
        "conditions": [c for c, _ in COND_ORDER],
        "main": main_tab.replace({np.nan: None}).to_dict(orient="records"),
        "paired": comp.replace({np.nan: None}).to_dict(orient="records"),
        "mcnemar": mc.to_dict(orient="records"),
        "judge_agreement": judge_tab.replace({np.nan: None}).to_dict(orient="records") if not judge_tab.empty else [],
    }
    (RESULTS / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote tables and stats.json")


if __name__ == "__main__":
    main()
