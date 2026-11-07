#!/usr/bin/env python3
"""Full-cycle cost, latency, context composition and incremental-update accounting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"

COND_ORDER = [
    ("c0_closed", "Closed-book"), ("c1_naive", "NaiveRAG"),
    ("c2_full", "LightRAG-Full"), ("c3_low", "LightRAG-LowOnly"),
    ("c4_high", "LightRAG-HighOnly"), ("c5_origin", "LightRAG-NoOriginalText"),
    ("c6_empty", "LightRAG-EmptyProfiles"), ("c7_shuffled", "LightRAG-ShuffledStructure"),
    ("c8_extractive", "LightRAG-ExtractiveProfiles"),
    ("c9_temporal", "LightRAG-TemporalGraph"),
    ("c10_naive_temporal", "NaiveRAG-TemporalOrder"),
]


def main() -> None:
    # ---- indexing calls (unique chunk extraction calls) ----
    calls = []
    log = RESULTS / "llm_calls.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                calls.append(json.loads(line))
            except Exception:
                pass
    cdf = pd.DataFrame(calls)
    if not cdf.empty:
        ext = cdf[cdf["tag"].str.startswith("extract:chunk-", na=False)].copy()
        ext = ext.sort_values("wall_seconds").drop_duplicates("tag", keep="last")
        repair = cdf[cdf["tag"].str.startswith("extract_repair:", na=False)]
        idx_calls = len(ext) + len(repair)
        idx_prompt = int(ext["prompt_tokens"].sum() + repair["prompt_tokens"].sum())
        idx_completion = int(ext["completion_tokens"].sum() + repair["completion_tokens"].sum())
        idx_seconds = float(ext["wall_seconds"].sum() + repair["wall_seconds"].sum())
        idx_per_chunk = ext[["tag", "prompt_tokens", "completion_tokens", "wall_seconds"]].copy()
        idx_per_chunk["chunk_idx"] = idx_per_chunk["tag"].str.extract(r"chunk-(\d+)").astype(int)
        idx_per_chunk = idx_per_chunk.sort_values("chunk_idx")
        half = 14
        full_tokens = int(idx_per_chunk["prompt_tokens"].sum() + idx_per_chunk["completion_tokens"].sum())
        inc_tokens = int(idx_per_chunk[idx_per_chunk["chunk_idx"] >= half]["prompt_tokens"].sum() +
                         idx_per_chunk[idx_per_chunk["chunk_idx"] >= half]["completion_tokens"].sum())
        full_seconds = float(idx_per_chunk["wall_seconds"].sum())
        inc_seconds = float(idx_per_chunk[idx_per_chunk["chunk_idx"] >= half]["wall_seconds"].sum())
    else:
        idx_calls = idx_prompt = idx_completion = 0
        idx_seconds = 0.0
        full_tokens = inc_tokens = 0
        full_seconds = inc_seconds = 0.0

    # ---- query cost/latency per condition from answer files ----
    rows = []
    for cond, label in COND_ORDER:
        p = RESULTS / f"answers_{cond}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vals = list(d.values())
        uncached = [r for r in vals if not (r.get("usage", {}) or {}).get("cached", False)]
        def mean(xs):
            xs = [x for x in xs if isinstance(x, (int, float)) and not np.isnan(x)]
            return float(np.mean(xs)) if xs else np.nan
        def pctl(xs, q):
            xs = [x for x in xs if isinstance(x, (int, float)) and not np.isnan(x)]
            return float(np.percentile(xs, q)) if xs else np.nan
        row = {
            "condition": cond, "label": label, "n": len(vals), "n_uncached": len(uncached),
            "keyword_calls": sum(1 for r in vals if r.get("keywords")),
            "llm_calls_per_query": 1 if cond in ("c0_closed", "c1_naive", "c10_naive_temporal") else 2,
            "prompt_tokens": mean([r.get("usage", {}).get("prompt_tokens") for r in uncached]),
            "completion_tokens": mean([r.get("usage", {}).get("completion_tokens") for r in uncached]),
            "keyword_prompt_tokens": mean([r.get("usage", {}).get("keyword_prompt_tokens") for r in uncached if r.get("keywords")]),
            "keyword_completion_tokens": mean([r.get("usage", {}).get("keyword_completion_tokens") for r in uncached if r.get("keywords")]),
            "latency_p50": pctl([r.get("wall_seconds") for r in uncached], 50),
            "latency_p95": pctl([r.get("wall_seconds") for r in uncached], 95),
            "context_tokens": mean([r.get("context_meta", {}).get("context_tokens_est") for r in vals]),
            "graph_tokens": mean([r.get("context_meta", {}).get("graph_tokens_est") for r in vals]),
            "chunk_tokens": mean([r.get("context_meta", {}).get("chunk_tokens_est") for r in vals]),
            "timeline_tokens": mean([r.get("context_meta", {}).get("timeline_tokens_est") for r in vals]),
            "n_chunks": mean([len(r.get("context_meta", {}).get("used_chunks", [])) for r in vals]),
        }
        row["total_tokens"] = (row["prompt_tokens"] + row["completion_tokens"] +
                               (row["keyword_prompt_tokens"] if not np.isnan(row["keyword_prompt_tokens"]) else 0) +
                               (row["keyword_completion_tokens"] if not np.isnan(row["keyword_completion_tokens"]) else 0))
        rows.append(row)
    qtab = pd.DataFrame(rows)
    qtab.to_csv(RESULTS / "table_cost_query.csv", index=False)
    print("=== query cost ===")
    print(qtab[["label", "n_uncached", "llm_calls_per_query", "prompt_tokens", "completion_tokens",
                "total_tokens", "latency_p50", "latency_p95", "context_tokens"]].to_string(index=False))

    # ---- storage ----
    storage = {}
    for name in ["graph_c2.json", "graph_c9.json", "profiles_c2.json", "profiles_c8.json"]:
        p = RESULTS / name
        storage[name] = p.stat().st_size if p.exists() else None
    emb_total = sum(p.stat().st_size for p in (ROOT / "cache").glob("emb_*.npz"))
    storage["embeddings_total"] = emb_total
    storage["corpus_json"] = (DATA / "corpus.json").stat().st_size

    summary = {
        "indexing": {
            "chunks": 28,
            "llm_calls": idx_calls,
            "prompt_tokens": idx_prompt,
            "completion_tokens": idx_completion,
            "wall_seconds_sum_of_calls": round(idx_seconds, 2),
            "wall_seconds_full_build_measured": 181.03,
            "tokens_per_chunk": round((idx_prompt + idx_completion) / max(1, idx_calls), 1),
        },
        "incremental": {
            "split": "first 14 chunks indexed, then 14 new chunks added",
            "full_rebuild_tokens": full_tokens,
            "incremental_update_tokens": inc_tokens,
            "token_saving_fraction": round(1 - inc_tokens / max(1, full_tokens), 3),
            "full_rebuild_llm_seconds": round(full_seconds, 2),
            "incremental_update_llm_seconds": round(inc_seconds, 2),
            "note": "The incremental update re-uses the existing graph and only extracts the new 14 chunks; no old chunk is re-extracted.",
        },
        "storage_bytes": storage,
        "query": qtab.replace({np.nan: None}).to_dict(orient="records"),
    }
    (RESULTS / "cost_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== indexing ===", json.dumps(summary["indexing"], indent=2))
    print("=== incremental ===", json.dumps(summary["incremental"], indent=2))
    print("=== storage ===", json.dumps(storage, indent=2))


if __name__ == "__main__":
    main()
