#!/usr/bin/env python3
"""Merge the blind human audit with judge labels and report agreement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    audit = pd.read_csv(RESULTS / "human_audit.csv", encoding="utf-8-sig")
    key = json.loads((RESULTS / "human_audit_key.json").read_text(encoding="utf-8"))
    judged = {}
    for cond in set(v["condition"] for v in key.values()):
        p = RESULTS / f"judged_judge1_{cond}.json"
        if p.exists():
            judged[cond] = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for _, r in audit.iterrows():
        aid = r["audit_id"]
        if aid not in key:
            continue
        h = r.get("human_correctness_0_1_2")
        if pd.isna(h) or str(h).strip() == "":
            continue
        k = key[aid]
        j = judged.get(k["condition"], {}).get(k["qid"], {}).get("parsed", {}) or {}
        rows.append({
            "audit_id": aid, "condition": k["condition"], "qid": k["qid"], "type": k["type"],
            "human": int(float(h)), "judge": j.get("correctness"),
            "human_cit": r.get("human_citation_precision_0_1"),
            "judge_cit": j.get("citation_precision"),
            "question": r["question"], "answer": r["answer"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("no scored audit rows")
        return
    df["judge"] = pd.to_numeric(df["judge"], errors="coerce")
    sub = df.dropna(subset=["judge"])
    y1 = sub["human"].astype(int).to_numpy()
    y2 = sub["judge"].astype(int).to_numpy()
    summary = {
        "n": int(len(sub)),
        "agreement": float((y1 == y2).mean()),
        "kappa_3": float(cohen_kappa_score(y1, y2, weights="quadratic")),
        "kappa_binary": float(cohen_kappa_score((y1 == 2).astype(int), (y2 == 2).astype(int))),
        "mean_human": float(y1.mean()),
        "mean_judge": float(y2.mean()),
        "human_vs_judge_corr": float(np.corrcoef(y1, y2)[0, 1]) if len(set(y1) | set(y2)) > 1 else None,
    }
    cit = df.dropna(subset=["human_cit", "judge_cit"])
    if len(cit) >= 3:
        summary["citation_precision_corr"] = float(np.corrcoef(
            cit["human_cit"].astype(float), cit["judge_cit"].astype(float))[0, 1])
        summary["citation_precision_mae"] = float(np.abs(
            cit["human_cit"].astype(float) - cit["judge_cit"].astype(float)).mean())
    (RESULTS / "human_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    df.to_csv(RESULTS / "human_audit_scored.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nby condition:")
    print(df.groupby("condition").agg(n=("human", "size"), human=("human", "mean"),
                                      judge=("judge", "mean")).to_string())


if __name__ == "__main__":
    main()
