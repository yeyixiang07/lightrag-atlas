#!/usr/bin/env python3
"""Reproduce 10 C2 answers under the post-crash capped-GPU environment."""
from __future__ import annotations

import json
import re
from pathlib import Path

from graph_core import GraphIndex, load_corpus, load_gold
from llm import OllamaClient
from run_queries import CONDITIONS, run_one

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QIDS = ["F02", "F07", "F14", "F22", "F28", "M02", "M07", "T04", "T14", "Q04"]


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def main() -> None:
    corpus = load_corpus()
    gold = {q["id"]: q for q in load_gold()}
    graph = json.loads((RESULTS / "graph_c2.json").read_text(encoding="utf-8"))
    client = OllamaClient(use_cache=False)
    idx = GraphIndex(client, graph, corpus, profile_mode="llm", cache_tag="c2")
    cond = next(c for c in CONDITIONS if c["id"] == "c2_full")
    old = json.loads((RESULTS / "answers_c2_full.json").read_text(encoding="utf-8"))
    rows = []
    for qid in QIDS:
        r = run_one(client, cond, gold[qid], corpus, {"llm": idx, "empty": idx,
                                                      "extractive": idx, "c9": idx})
        a, b = old[qid]["answer"], r["answer"]
        rows.append({
            "qid": qid, "old_chars": len(a), "new_chars": len(b),
            "exact": a.strip() == b.strip(),
            "jaccard": len(toks(a) & toks(b)) / max(1, len(toks(a) | toks(b))),
            "old_citations": old[qid].get("citations", []),
            "new_citations": r.get("citations", []),
            "old": a, "new": b,
        })
        print(f"{qid}: exact={rows[-1]['exact']} jaccard={rows[-1]['jaccard']:.2f}", flush=True)
    n_exact = sum(r["exact"] for r in rows)
    summary = {"n": len(rows), "exact": n_exact, "mean_jaccard": sum(r["jaccard"] for r in rows) / len(rows),
               "rows": rows}
    (RESULTS / "repro_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
