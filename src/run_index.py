#!/usr/bin/env python3
"""Build the knowledge graph, profile variants and embedding indexes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from graph_core import (GraphIndex, build_graph, graph_json, load_corpus,
                        make_profiles)
from llm import OllamaClient
from temporal import add_temporal_graph

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N chunks (pilot)")
    ap.add_argument("--out", default="graph_c2.json")
    args = ap.parse_args()

    corpus = load_corpus()
    client = OllamaClient()
    t0 = time.time()
    graph = build_graph(client, corpus, limit=args.limit)
    elapsed = time.time() - t0
    graph["stats"]["build_seconds"] = round(elapsed, 2)
    out = RESULTS / args.out
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(graph["stats"], ensure_ascii=False, indent=2))
    print("saved", out)

    if args.limit:
        print("\n--- sample entities ---")
        for e in list(graph["entities"].values())[:12]:
            print(e["name"], "|", e["type"], "|", (e["descriptions"][0] if e["descriptions"] else "")[:160])
        print("\n--- sample relations ---")
        for r in list(graph["relations"].values())[:12]:
            s = graph["entities"][r["source"]]["name"]
            t = graph["entities"][r["target"]]["name"]
            print(f"{s} -> {t} | {r['keywords']} | {(r['descriptions'][0] if r['descriptions'] else '')[:140]}")
        return

    for mode, tag in (("llm", "c2"), ("empty", "c6"), ("extractive", "c8")):
        profiles = make_profiles(graph, corpus, mode)
        (RESULTS / f"profiles_{tag}.json").write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        idx = GraphIndex(client, graph, corpus, profile_mode=mode, cache_tag=tag)
        print(f"[index] {tag} ({mode}): entities={len(idx.entities)} relations={len(idx.relations)} "
              f"embeddings={idx.entity_emb.shape}/{idx.relation_emb.shape}")

    tg = add_temporal_graph(graph)
    (RESULTS / "graph_c9.json").write_text(json.dumps(tg, ensure_ascii=False, indent=2), encoding="utf-8")
    tidx = GraphIndex(client, tg, corpus, profile_mode="llm", cache_tag="c9")
    print(f"[index] c9 temporal: entities={len(tidx.entities)} relations={len(tidx.relations)}")


if __name__ == "__main__":
    main()
