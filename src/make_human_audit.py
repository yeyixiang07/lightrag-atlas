#!/usr/bin/env python3
"""Create a blind, stratified human-audit sheet (condition labels hidden)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from evaluate import compact_context

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "human_audit.csv"
KEY = RESULTS / "human_audit_key.json"

CONDITIONS = ["c0_closed", "c1_naive", "c2_full", "c3_low", "c4_high", "c5_origin",
              "c6_empty", "c7_shuffled", "c8_extractive", "c9_temporal",
              "c10_naive_temporal"]
TYPES = ["factual", "multihop", "thematic", "temporal"]


def main() -> None:
    gold = {q["id"]: q for q in json.load(open(ROOT / "data" / "gold_qa.json"))}
    rng = random.Random(2026)
    rows = []
    key = {}
    for cond in CONDITIONS:
        p = RESULTS / f"answers_{cond}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for typ in TYPES:
            qids = [qid for qid, r in d.items() if r.get("type") == typ]
            if not qids:
                continue
            qid = rng.choice(sorted(qids))
            r = d[qid]
            aid = f"A{len(rows)+1:03d}"
            key[aid] = {"condition": cond, "qid": qid, "type": typ}
            rows.append({
                "audit_id": aid, "qid": qid, "type": typ,
                "question": gold[qid]["question"],
                "gold_answer": gold[qid]["gold_answer"],
                "key_points_or_order": json.dumps(
                    gold[qid].get("key_points") or gold[qid].get("order_pairs") or [],
                    ensure_ascii=False),
                "answer": r.get("answer", ""),
                "context_snippet": compact_context(r, max_chars=4200),
                "human_correctness_0_1_2": "",
                "human_citation_precision_0_1": "",
                "human_notes": "",
            })
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} blind audit items to {OUT}")
    print(f"condition key (do not read while scoring): {KEY}")


if __name__ == "__main__":
    main()
