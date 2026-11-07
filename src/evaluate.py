#!/usr/bin/env python3
"""Deterministic metrics + structured LLM judging + profile faithfulness audit."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from graph_core import (embed_texts, load_corpus, load_gold, parse_json,
                        stanza_sort_key, stanzas_from_text)
from llm import OllamaClient
from prompts import (JUDGE_SECOND_SYSTEM, JUDGE_SYSTEM, JUDGE_USER,
                     PROFILE_JUDGE_SYSTEM, PROFILE_JUDGE_USER)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
_CORPUS = load_corpus()
STANZA_TEXT = {s["id"]: s["text"] for s in _CORPUS["stanzas"]}


def compact_context(r: dict, max_chars: int = 5500) -> str:
    """Give the judge the evidence it needs, not the whole noisy retrieval dump."""
    ctx = r.get("context", "") or ""
    if not ctx:
        return "(no retrieval context: closed-book condition)"
    parts = []
    if "---Narrative Timeline" in ctx:
        timeline = ctx.split("---Narrative Timeline", 1)[1]
        timeline = "---Narrative Timeline" + timeline
        if "---Entities---" in timeline:
            timeline = timeline.split("---Entities---", 1)[0]
        parts.append(timeline.strip()[:2500])
    if "---Original Text---" in ctx:
        orig = ctx.split("---Original Text---", 1)[1]
        parts.append("---Original Text---" + orig[:4000])
    else:
        parts.append(ctx[:4000])
    cited = r.get("citations", []) or []
    if cited:
        cited_blocks = [f"[{sid}] {STANZA_TEXT.get(sid, '')}" for sid in cited if sid in STANZA_TEXT]
        if cited_blocks:
            parts.append("---Cited Stanzas (full text)---\n" + "\n".join(cited_blocks))
    return "\n\n".join(parts)[:max_chars]


def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(the|a|an|of|and|to|in|on|at|by|for|with|from)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def em_score(pred: str, aliases: list[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(a) for a in aliases))


def token_f1(pred: str, aliases: list[str]) -> float:
    pt = normalize(pred).split()
    best = 0.0
    for a in aliases:
        at = normalize(a).split()
        if not pt or not at:
            continue
        common = Counter(pt) & Counter(at)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        prec, rec = overlap / len(pt), overlap / len(at)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def contains_answer(pred: str, aliases: list[str]) -> float:
    p = normalize(pred)
    return float(any(normalize(a) and normalize(a) in p for a in aliases))


def answer_metrics(pred: str, q: dict) -> dict:
    aliases = q.get("aliases") or [q["gold_answer"]]
    return {
        "em": em_score(pred, aliases),
        "f1": token_f1(pred, aliases),
        "contains": contains_answer(pred, aliases),
    }


def judge_one(client: OllamaClient, q: dict, r: dict, system: str = JUDGE_SYSTEM,
              tag: str = "judge") -> dict:
    context = compact_context(r)
    user = JUDGE_USER.format(
        question=q["question"], qtype=q["type"], gold_answer=q["gold_answer"],
        key_points=json.dumps(q.get("key_points", []), ensure_ascii=False),
        order_pairs=json.dumps(q.get("order_pairs", []), ensure_ascii=False),
        context=context, answer=r.get("answer", ""))
    res = client.generate(user, system=system, num_predict=620, fmt="json",
                          tag=f"{tag}:{r['condition']}:{q['id']}")
    parsed = None
    try:
        parsed = parse_json(res.text)
    except Exception:
        rep = client.generate("Return only valid JSON for this evaluation:\n" + res.text,
                              system="You repair malformed JSON. Output valid JSON only.",
                              num_predict=700, fmt="json", tag=f"{tag}_repair:{r['condition']}:{q['id']}")
        try:
            parsed = parse_json(rep.text)
        except Exception:
            parsed = {}
    return {"raw": res.text, "parsed": parsed,
            "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                      "seconds": res.total_duration_ns / 1e9}, "error": res.error}


def judge_condition(cond: str, judge_model: str, system: str, suffix: str = "judge1",
                    only_qids: set[str] | None = None) -> None:
    gold = {q["id"]: q for q in load_gold()}
    path = RESULTS / f"answers_{cond}.json"
    if not path.exists():
        print("missing", path)
        return
    answers = json.loads(path.read_text(encoding="utf-8"))
    out_path = RESULTS / f"judged_{suffix}_{cond}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [qid for qid in answers if qid not in done and (only_qids is None or qid in only_qids)]
    print(f"[judge {suffix}] {cond}: {len(done)} done, {len(todo)} to run", flush=True)
    client = OllamaClient(model=judge_model)
    for i, qid in enumerate(todo, 1):
        j = judge_one(client, gold[qid], answers[qid], system=system, tag=f"judge_{suffix}")
        done[qid] = j
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        p = j["parsed"] or {}
        print(f"[{suffix}:{cond}] {i}/{len(todo)} {qid} correct={p.get('correctness')}", flush=True)


def profile_audit(n_per_type: int = 30, seed: int = 42) -> None:
    corpus = load_corpus()
    stanza_text = {s["id"]: s["text"] for s in corpus["stanzas"]}
    client = OllamaClient()
    out_path = RESULTS / "profile_audit.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"samples": []}
    existing = {(s["mode"], s["kind"], s["key"]) for s in done["samples"]}
    rng = random.Random(seed)
    for mode in ("llm", "extractive"):
        prof = json.loads((RESULTS / f"profiles_{'c2' if mode == 'llm' else 'c8'}.json").read_text(encoding="utf-8"))
        ents = list(prof["entities"].items())
        rels = list(prof["relations"].items())
        rng.shuffle(ents)
        rng.shuffle(rels)
        for kind, items, limit in (("entity", ents, n_per_type), ("relation", rels, n_per_type)):
            for key, item in items[:limit]:
                if (mode, kind, key) in existing:
                    continue
                if kind == "entity":
                    profile = item["text"]
                    src_ids = item.get("source_stanzas", [])
                    label = item["name"]
                else:
                    profile = item["text"]
                    src_ids = item.get("source_stanzas", [])
                    label = f"{prof['entities'][item['source']]['name']} -> {prof['entities'][item['target']]['name']}"
                source = "\n\n".join(f"[{sid}] {stanza_text.get(sid, '')}" for sid in src_ids[:6])
                res = client.generate(
                    PROFILE_JUDGE_USER.format(ptype=kind, profile=profile, source=source),
                    system=PROFILE_JUDGE_SYSTEM, num_predict=600, fmt="json",
                    tag=f"profile_audit:{mode}:{kind}")
                try:
                    parsed = parse_json(res.text)
                except Exception:
                    parsed = {}
                done["samples"].append({
                    "mode": mode, "kind": kind, "key": key, "label": label,
                    "profile": profile, "source_stanzas": src_ids, "source": source,
                    "parsed": parsed, "raw": res.text,
                })
                out_path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[profile {mode}/{kind}] {label[:60]} -> {parsed.get('support_rate')}", flush=True)


def literal_drift() -> None:
    corpus = load_corpus()
    stanza_text = {s["id"]: s["text"] for s in corpus["stanzas"]}
    client = OllamaClient()
    rows = []
    for mode, tag in (("llm", "c2"), ("extractive", "c8")):
        prof = json.loads((RESULTS / f"profiles_{tag}.json").read_text(encoding="utf-8"))
        for kind in ("entities", "relations"):
            for key, item in prof[kind].items():
                text = item.get("text", "")
                src = " ".join(stanza_text.get(s, "") for s in item.get("source_stanzas", [])[:4])
                if not text or not src:
                    continue
                rows.append({"mode": mode, "kind": kind[:-1], "key": key,
                             "text": text, "source": src})
    texts = [r["text"] for r in rows] + [r["source"] for r in rows]
    emb = embed_texts(client, texts)
    n = len(rows)
    for i, r in enumerate(rows):
        a, b = emb[i], emb[n + i]
        r["cosine"] = float(np.dot(a, b))
        r["char_ratio"] = _ratio(r["text"], r["source"])
        r["rouge_l"] = _rouge_l(r["text"].split(), r["source"].split())
    (RESULTS / "literal_drift.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("literal drift rows:", len(rows))


def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _rouge_l(pred: list[str], gold: list[str]) -> float:
    if not pred or not gold:
        return 0.0
    dp = [[0] * (len(gold) + 1) for _ in range(len(pred) + 1)]
    for i in range(1, len(pred) + 1):
        for j in range(1, len(gold) + 1):
            if pred[i - 1].lower() == gold[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    return lcs / max(1, len(gold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["judge1", "judge2", "profiles", "drift"], required=True)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--qids", default="")
    ap.add_argument("--judge-model", default="qwen7b:latest")
    args = ap.parse_args()
    if args.mode in ("judge1", "judge2"):
        conds = args.conditions.split(",") if args.conditions else [
            "c0_closed", "c1_naive", "c2_full", "c3_low", "c4_high", "c5_origin",
            "c6_empty", "c7_shuffled", "c8_extractive", "c9_temporal",
            "c10_naive_temporal"]
        only = set(args.qids.split(",")) if args.qids else None
        system = JUDGE_SYSTEM if args.mode == "judge1" else JUDGE_SECOND_SYSTEM
        for c in conds:
            judge_condition(c, args.judge_model, system, suffix=args.mode, only_qids=only)
    elif args.mode == "profiles":
        profile_audit()
    elif args.mode == "drift":
        literal_drift()


if __name__ == "__main__":
    main()
