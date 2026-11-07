#!/usr/bin/env python3
"""Run all conditions over the gold set with identical decoding and budgets."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from graph_core import (GraphIndex, build_graph, citation_stanzas, extract_keywords,
                        is_temporal_question, load_corpus, load_gold, naive_context,
                        naive_retrieve)
from llm import OllamaClient
from prompts import (ANSWER_SYSTEM, ANSWER_USER, CLOSED_BOOK_SYSTEM,
                     CLOSED_BOOK_USER)
from temporal import timeline_text

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONDITIONS = [
    {"id": "c0_closed", "label": "Closed-book", "kind": "closed"},
    {"id": "c1_naive", "label": "NaiveRAG", "kind": "naive"},
    {"id": "c2_full", "label": "LightRAG-Full", "kind": "graph", "profile": "llm"},
    {"id": "c3_low", "label": "LightRAG-LowOnly", "kind": "graph", "profile": "llm", "use_high": False},
    {"id": "c4_high", "label": "LightRAG-HighOnly", "kind": "graph", "profile": "llm", "use_low": False},
    {"id": "c5_origin", "label": "LightRAG-NoOriginalText", "kind": "graph", "profile": "llm", "no_chunks": True},
    {"id": "c6_empty", "label": "LightRAG-EmptyProfiles", "kind": "graph", "profile": "empty"},
    {"id": "c7_shuffled", "label": "LightRAG-ShuffledStructure", "kind": "graph", "profile": "llm", "shuffle": True},
    {"id": "c8_extractive", "label": "LightRAG-ExtractiveProfiles", "kind": "graph", "profile": "extractive"},
    {"id": "c9_temporal", "label": "LightRAG-TemporalGraph", "kind": "graph", "profile": "llm", "temporal": True},
    {"id": "c10_naive_temporal", "label": "NaiveRAG-TemporalOrder", "kind": "naive", "temporal_expand": True},
]


def gpu_temp() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL)
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def load_graph(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def run_one(client: OllamaClient, cond: dict, q: dict, corpus: dict, indexes: dict,
            budget: int = 3000) -> dict:
    t_start = time.time()
    qid, question = q["id"], q["question"]
    temporal_q = is_temporal_question(question)
    result = {
        "condition": cond["id"], "label": cond["label"], "qid": qid, "type": q["type"],
        "question": question, "temporal_question": temporal_q,
        "temporal_activated": False, "keywords": None, "retrieval": None,
        "context": "", "context_meta": {}, "answer": "", "usage": {},
        "wall_seconds": 0.0, "error": None, "citations": [],
    }

    if cond["kind"] == "closed":
        res = client.generate(CLOSED_BOOK_USER.format(question=question),
                              system=CLOSED_BOOK_SYSTEM, num_predict=280, tag=f"{cond['id']}:{qid}")
        result["answer"] = res.text
        result["usage"] = {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                           "llm_seconds": res.total_duration_ns / 1e9, "cached": res.cached}
        result["error"] = res.error
        result["citations"] = citation_stanzas(res.text)
        result["wall_seconds"] = round(time.time() - t_start, 3)
        return result

    if cond["kind"] == "naive":
        use_temporal = cond.get("temporal_expand") and temporal_q
        ret = naive_retrieve(client, corpus, question, top_k=8, temporal_expand=use_temporal)
        ctx, meta = naive_context(ret, budget_tokens=budget, temporal_order=use_temporal)
        result["retrieval"] = {"chunk_ids": [c["id"] for c in ret["chunks"]],
                               "stanzas": [c["stanza_ids"] for c in ret["chunks"]]}
        result["temporal_activated"] = bool(use_temporal)
    else:
        use_c9 = cond.get("temporal") and temporal_q
        idx = indexes["c9"] if use_c9 else indexes[cond["profile"]]
        kw, kw_sec, kw_pt, kw_ct = extract_keywords(client, question)
        result["keywords"] = kw
        ret = idx.retrieve(
            kw["low_level_keywords"], kw["high_level_keywords"], question,
            use_low=cond.get("use_low", True), use_high=cond.get("use_high", True),
            shuffle_structure=cond.get("shuffle", False),
            rng=random.Random(1000 + int(qid[1:])),
        )
        timeline = timeline_text() if use_c9 else ""
        ctx, meta = idx.build_context(
            ret, condition="origin" if cond.get("no_chunks") else "graph",
            question=question, budget_tokens=budget, temporal_timeline=timeline)
        result["temporal_activated"] = bool(use_c9)
        result["retrieval"] = {
            "matched_entities": [idx.profiles["entities"][k]["name"] for k in ret["matched_entities"]],
            "matched_relations": [
                f'{idx.profiles["entities"][idx.profiles["relations"][r]["source"]]["name"]} -> '
                f'{idx.profiles["entities"][idx.profiles["relations"][r]["target"]]["name"]}'
                for r in ret["matched_relations"]],
            "onehop_entities": [idx.profiles["entities"][k]["name"] for k in ret["onehop_entities"]],
            "onehop_relations": len(ret["onehop_relations"]),
            "chunk_ids": [c["id"] for c in ret["chunks"]],
        }
        result["usage"].update({"keyword_prompt_tokens": kw_pt, "keyword_completion_tokens": kw_ct,
                                "keyword_seconds": kw_sec})

    result["context"] = ctx
    result["context_meta"] = meta
    res = client.generate(ANSWER_USER.format(context=ctx, question=question),
                          system=ANSWER_SYSTEM, num_predict=300, tag=f"{cond['id']}:{qid}")
    result["answer"] = res.text
    result["usage"].update({"prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens,
                            "answer_seconds": res.total_duration_ns / 1e9,
                            "cached": res.cached})
    result["error"] = res.error
    result["citations"] = citation_stanzas(res.text)
    result["wall_seconds"] = round(time.time() - t_start, 3)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="", help="comma-separated condition ids")
    ap.add_argument("--qids", default="", help="comma-separated question ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="bypass the LLM response cache")
    args = ap.parse_args()

    corpus = load_corpus()
    gold = load_gold()
    if args.qids:
        wanted = set(args.qids.split(","))
        gold = [q for q in gold if q["id"] in wanted]
    if args.limit:
        gold = gold[:args.limit]
    conds = CONDITIONS
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conds = [c for c in CONDITIONS if c["id"] in wanted]

    client = OllamaClient(use_cache=not args.no_cache)
    graph = load_graph("graph_c2.json")
    tgraph = load_graph("graph_c9.json")
    indexes = {
        "llm": GraphIndex(client, graph, corpus, profile_mode="llm", cache_tag="c2"),
        "empty": GraphIndex(client, graph, corpus, profile_mode="empty", cache_tag="c6"),
        "extractive": GraphIndex(client, graph, corpus, profile_mode="extractive", cache_tag="c8"),
        "c9": GraphIndex(client, tgraph, corpus, profile_mode="llm", cache_tag="c9"),
    }
    print("indexes ready", {k: (len(v.entities), len(v.relations)) for k, v in indexes.items()}, flush=True)

    for cond in conds:
        out_path = RESULTS / f"answers_{cond['id']}.json"
        done = {}
        if out_path.exists() and not args.force:
            done = json.loads(out_path.read_text(encoding="utf-8"))
        todo = [q for q in gold if q["id"] not in done]
        print(f"[run] {cond['id']} {cond['label']}: {len(done)} done, {len(todo)} to run", flush=True)
        for i, q in enumerate(todo, 1):
            r = run_one(client, cond, q, corpus, indexes, budget=args.budget)
            done[q["id"]] = r
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(out_path)
            status = "ERR " + str(r["error"]) if r["error"] else f"{len(r['answer'])} chars, {r['wall_seconds']}s"
            if i % 5 == 0:
                t = gpu_temp()
                if t is None:
                    print("[gpu] MISSING; waiting 30s", flush=True)
                    time.sleep(30)
                elif t >= 78:
                    print(f"[gpu] {t}C; cooling 20s", flush=True)
                    time.sleep(20)
                else:
                    print(f"[gpu] {t}C", flush=True)
            print(f"[{cond['id']}] {i}/{len(todo)} {q['id']} {status}", flush=True)


if __name__ == "__main__":
    main()
