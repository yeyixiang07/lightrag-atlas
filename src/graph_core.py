#!/usr/bin/env python3
"""A faithful, inspectable implementation of the LightRAG indexing/retrieval core."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llm import OllamaClient
from prompts import (ANSWER_SYSTEM, ANSWER_USER, EXTRACT_SYSTEM, EXTRACT_USER,
                     KEYWORD_SYSTEM, KEYWORD_USER, TEMPORAL_TIMELINE_HEADER)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
for p in (CACHE, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

ENTITY_TYPE_SET = {"Person", "Deity", "Creature", "Place", "Object", "Concept",
                   "Event", "Group", "Work", "Other"}

STANZA_RE = re.compile(r"\b([PD]\d+)\b")
CITATION_RE = re.compile(r"\[((?:[PD]\d+)(?:\s*,\s*[PD]\d+)*)\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# Canonical aliases for the poem's recurring figures.  The extractor may return
# "the Witch", "lady-witch", "wizard lady", "the Shape", etc.; they must collapse
# to one node or the graph fragments.
ALIASES: dict[str, list[str]] = {
    "witch": ["witch", "the witch", "lady-witch", "lady witch", "wizard lady",
              "wizard-lady", "wizard maiden", "wizard-maiden", "wizard maid",
              "the lady", "lady", "witch of atlas", "the witch of atlas"],
    "hermaphroditus": ["hermaphroditus", "hermaphrodite", "the hermaphrodite",
                       "the shape", "shape", "living image", "the living image",
                       "the image", "image", "sexless thing", "winged shape"],
    "boat": ["boat", "the boat", "pinnace", "the pinnace", "car", "the car",
             "skiff", "the skiff"],
    "nile": ["nile", "the nile", "old nilus", "nilus", "the nile river"],
    "atlas": ["atlas", "atlas' mountain", "atlas mountain", "mount atlas",
              "atlas' mountain range"],
    "atlantides": ["atlantides", "atlantide", "the atlantides", "one of the atlantides"],
    "cameleopard": ["cameleopard", "camelopard", "spotted cameleopard",
                    "spotted camelopard", "the spotted cameleopard"],
    "lioness": ["lioness", "the lioness", "brinded lioness", "the brinded lioness"],
    "mary": ["mary", "dear mary", "mary shelley", "the poet's wife"],
    "peter_bell": ["peter bell", "peter_bell", "peter"],
    "wordsworth": ["wordsworth", "wordsworth's"],
    "love": ["love", "first-born love", "first born love", "the first-born love"],
    "pan": ["pan", "universal pan", "the god pan"],
    "apollo": ["apollo"],
    "vulcan": ["vulcan", "hephaestus"],
    "venus": ["venus"],
    "silenus": ["silenus", "old silenus"],
    "ocean": ["ocean", "the ocean"],
    "time": ["time", "father time"],
    "error": ["error"],
    "truth": ["truth"],
    "death": ["death"],
    "sleep": ["sleep"],
}

DISPLAY = {
    "witch": "the Witch",
    "hermaphroditus": "Hermaphroditus",
    "boat": "the boat",
    "nile": "the Nile",
    "atlas": "Atlas",
    "atlantides": "the Atlantides",
    "cameleopard": "the spotted cameleopard",
    "lioness": "the brinded lioness",
    "mary": "Mary",
    "peter_bell": "Peter Bell",
    "wordsworth": "Wordsworth",
    "love": "Love",
    "pan": "Pan",
    "apollo": "Apollo",
    "vulcan": "Vulcan",
    "venus": "Venus",
    "silenus": "Silenus",
    "ocean": "Ocean",
    "time": "Time",
    "error": "Error",
    "truth": "Truth",
    "death": "Death",
    "sleep": "Sleep",
}


def norm_key(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for article in ("the ", "a ", "an ", "old ", "universal ", "dear ", "my "):
        if s.startswith(article):
            s = s[len(article):]
    s = s.strip(" '")
    return s


ALIAS_LOOKUP: dict[str, str] = {}
for key, names in ALIASES.items():
    ALIAS_LOOKUP[key] = key
    for n in names:
        ALIAS_LOOKUP[norm_key(n)] = key


def canonical_key(name: str) -> str:
    k = norm_key(name)
    if k in ALIAS_LOOKUP:
        return ALIAS_LOOKUP[k]
    # Light singularisation for simple plurals; never strip "s" from short names.
    if k.endswith("ies") and len(k) > 5:
        k2 = k[:-3] + "y"
        if k2 in ALIAS_LOOKUP:
            return ALIAS_LOOKUP[k2]
    if k.endswith("es") and len(k) > 5:
        k2 = k[:-2]
        if k2 in ALIAS_LOOKUP:
            return ALIAS_LOOKUP[k2]
    if k.endswith("s") and len(k) > 4:
        k2 = k[:-1]
        if k2 in ALIAS_LOOKUP:
            return ALIAS_LOOKUP[k2]
    return k


def display_name(key: str, fallback: str) -> str:
    if key in DISPLAY:
        return DISPLAY[key]
    return fallback.strip() or key


def parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


def stanzas_from_text(text: str) -> list[str]:
    seen = []
    for m in STANZA_RE.finditer(text or ""):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def load_corpus() -> dict:
    return json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))


def load_gold() -> list[dict]:
    return json.loads((DATA / "gold_qa.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def extract_chunk(client: OllamaClient, chunk: dict) -> dict:
    prompt = EXTRACT_USER.format(chunk=chunk["text"])
    res = client.generate(prompt, system=EXTRACT_SYSTEM, num_predict=1600,
                          fmt="json", tag=f"extract:{chunk['id']}")
    if res.error:
        return {"chunk": chunk["id"], "error": res.error, "entities": [], "relations": []}
    try:
        data = parse_json(res.text)
    except Exception as e:
        # One repair attempt with a minimal instruction.
        rep = client.generate(
            "Return ONLY the JSON object with keys entities and relations. No prose.\n\n" + res.text,
            system="You repair malformed JSON. Output valid JSON only.", num_predict=1600,
            fmt="json", tag=f"extract_repair:{chunk['id']}")
        try:
            data = parse_json(rep.text)
        except Exception as e2:
            return {"chunk": chunk["id"], "error": f"parse:{e}; repair:{e2}",
                    "raw": res.text[:1000], "entities": [], "relations": []}
    ents = data.get("entities", []) if isinstance(data, dict) else []
    rels = data.get("relations", []) if isinstance(data, dict) else []
    clean_ents, clean_rels = [], []
    for e in ents if isinstance(ents, list) else []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        desc = str(e.get("description", "")).strip()
        if not name or not desc:
            continue
        clean_ents.append({
            "name": name,
            "type": (str(e.get("type", "Other")).strip() if str(e.get("type", "Other")).strip() in ENTITY_TYPE_SET else "Other"),
            "description": desc, "cited_stanzas": stanzas_from_text(desc),
            "chunk_stanzas": chunk["stanza_ids"],
        })
    for r in rels if isinstance(rels, list) else []:
        if not isinstance(r, dict):
            continue
        s = str(r.get("source", "")).strip()
        t = str(r.get("target", "")).strip()
        desc = str(r.get("description", "")).strip()
        if not s or not t or not desc:
            continue
        clean_rels.append({
            "source": s, "target": t,
            "keywords": str(r.get("keywords", "")).strip(),
            "description": desc, "cited_stanzas": stanzas_from_text(desc),
            "chunk_stanzas": chunk["stanza_ids"],
        })
    return {
        "chunk": chunk["id"], "entities": clean_ents, "relations": clean_rels,
        "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                  "wall_seconds": res.total_duration_ns / 1e9},
        "raw_text": res.text[:500],
    }


def build_graph(client: OllamaClient, corpus: dict, limit: int | None = None) -> dict:
    chunks = corpus["chunks"][:limit] if limit else corpus["chunks"]
    records = []
    for i, ch in enumerate(chunks, 1):
        print(f"[extract] {i}/{len(chunks)} {ch['id']} ({ch['char_count']} chars)", flush=True)
        records.append(extract_chunk(client, ch))
    return merge_graph(records, corpus)


def merge_graph(records: list[dict], corpus: dict) -> dict:
    entities: dict[str, dict] = {}
    relations: dict[str, dict] = {}

    def ensure_entity(name: str, type_: str = "Other", desc: str = "",
                      stanzas: Iterable[str] = ()) -> str:
        key = canonical_key(name)
        if not key:
            key = norm_key(name) or "unknown"
        if key not in entities:
            entities[key] = {
                "key": key, "name": display_name(key, name), "type": type_ or "Other",
                "descriptions": [], "source_stanzas": set(), "source_chunks": set(),
                "aliases": set([name]), "mentions": 0,
            }
        e = entities[key]
        e["aliases"].add(name)
        if type_ and type_ != "Other":
            e["type"] = type_
        if desc and desc not in e["descriptions"]:
            e["descriptions"].append(desc)
        e["source_stanzas"].update(stanzas)
        e["mentions"] += 1
        return key

    for rec in records:
        chunk_stanzas = []
        for e in rec.get("entities", []):
            cited = e.get("cited_stanzas") or e.get("chunk_stanzas") or []
            key = ensure_entity(e["name"], e.get("type", "Other"), e.get("description", ""), cited)
            entities[key]["source_chunks"].add(rec["chunk"])
            for st in e.get("chunk_stanzas", []):
                entities[key]["source_stanzas"].add(st)
        for r in rec.get("relations", []):
            cited = r.get("cited_stanzas") or r.get("chunk_stanzas") or []
            s = ensure_entity(r["source"], "Other", "", cited)
            t = ensure_entity(r["target"], "Other", "", cited)
            kws = [k.strip() for k in re.split(r"[,;]", r.get("keywords", "")) if k.strip()]
            kws = sorted(set(kws))[:5]
            rkey = f"{min(s,t)}|{max(s,t)}|{','.join(kws[:2])}"
            if rkey not in relations:
                relations[rkey] = {
                    "id": f"r{len(relations):03d}", "source": s, "target": t,
                    "keywords": kws, "descriptions": [], "source_stanzas": set(),
                    "source_chunks": set(),
                }
            rr = relations[rkey]
            rr["keywords"] = sorted(set(rr["keywords"]) | set(kws))[:6]
            if r["description"] not in rr["descriptions"]:
                rr["descriptions"].append(r["description"])
            rr["source_stanzas"].update(cited)
            for st in r.get("chunk_stanzas", []):
                rr["source_stanzas"].add(st)
            rr["source_chunks"].add(rec["chunk"])

    # Prune empty stubs unless they have relations.
    edge_entities = set()
    for r in relations.values():
        edge_entities.add(r["source"])
        edge_entities.add(r["target"])
    for key in list(entities):
        if key not in edge_entities and not entities[key]["descriptions"]:
            del entities[key]
    # Drop relations whose endpoints vanished.
    relations = {k: v for k, v in relations.items()
                 if v["source"] in entities and v["target"] in entities}

    # Degrees.
    degree = Counter()
    for r in relations.values():
        degree[r["source"]] += 1
        degree[r["target"]] += 1
    for r in relations.values():
        r["source_stanzas"] = sorted(r["source_stanzas"], key=stanza_sort_key)
        r["source_chunks"] = sorted(r["source_chunks"])
    for key, e in entities.items():
        e["degree"] = degree[key]
        e["source_stanzas"] = sorted(e["source_stanzas"], key=stanza_sort_key)
        e["source_chunks"] = sorted(e["source_chunks"])
        e["aliases"] = sorted(e["aliases"])

    return {
        "entities": entities,
        "relations": relations,
        "stats": {
            "chunks": len(corpus["chunks"]),
            "entities": len(entities),
            "relations": len(relations),
            "records": len(records),
            "parse_errors": sum(1 for r in records if r.get("error")),
        },
        "records_errors": [{"chunk": r["chunk"], "error": r.get("error")} for r in records if r.get("error")],
    }


def stanza_sort_key(s: str):
    return (0 if s.startswith("D") else 1, int(s[1:]) if s[1:].isdigit() else 0)


def graph_json(graph: dict) -> str:
    return json.dumps(graph, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Profiles (the LLM-text factor of the experiment)
# ---------------------------------------------------------------------------

def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def make_profiles(graph: dict, corpus: dict, mode: str = "llm") -> dict:
    """Return a deep-ish copy with entity/relation description fields for a mode.

    mode = llm | empty | extractive
    """
    stanza_text = {s["id"]: s["text"] for s in corpus["stanzas"]}
    out = {"entities": {}, "relations": {}, "stats": dict(graph["stats"])}
    for key, e in graph["entities"].items():
        e2 = dict(e)
        e2["source_stanzas"] = list(e["source_stanzas"])
        if mode == "llm":
            e2["text"] = " ".join(e["descriptions"])[:1600]
        elif mode == "empty":
            e2["text"] = f"Entity: {e['name']} (type: {e['type']})."
        elif mode == "extractive":
            sents = []
            names = [e["name"]] + list(e.get("aliases", []))
            for st in e["source_stanzas"]:
                txt = stanza_text.get(st, "")
                for sent in sentences(txt):
                    if any(n.lower() in sent.lower() for n in names if len(n) > 2):
                        if sent not in sents:
                            sents.append(sent)
                        if len(sents) >= 3:
                            break
                if len(sents) >= 3:
                    break
            e2["text"] = " ".join(sents)[:900] if sents else e["name"]
        else:
            raise ValueError(mode)
        out["entities"][key] = e2

    for rid, r in graph["relations"].items():
        r2 = dict(r)
        r2["source_stanzas"] = list(r["source_stanzas"])
        sname = out["entities"][r["source"]]["name"]
        tname = out["entities"][r["target"]]["name"]
        kw = ", ".join(r["keywords"])
        if mode == "llm":
            r2["text"] = " ".join(r["descriptions"])[:1200]
        elif mode == "empty":
            r2["text"] = f"{sname} --[{kw}]--> {tname}."
        elif mode == "extractive":
            sents = []
            for st in r["source_stanzas"]:
                txt = stanza_text.get(st, "")
                for sent in sentences(txt):
                    low = sent.lower()
                    if (sname.lower() in low or tname.lower() in low
                            or any(k.lower() in low for k in r["keywords"])):
                        if sent not in sents:
                            sents.append(sent)
                        if len(sents) >= 2:
                            break
                if len(sents) >= 2:
                    break
            r2["text"] = " ".join(sents)[:600] if sents else f"{sname} relates to {tname}."
        out["relations"][r["id"]] = r2
    return out


# ---------------------------------------------------------------------------
# Embeddings and retrieval
# ---------------------------------------------------------------------------

def hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def embed_texts(client: OllamaClient, texts: list[str], batch: int = 16) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.extend(client.embed(texts[i:i + batch]))
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class GraphIndex:
    def __init__(self, client: OllamaClient, graph: dict, corpus: dict,
                 profile_mode: str = "llm", cache_tag: str = "c2"):
        self.client = client
        self.graph = graph
        self.corpus = corpus
        self.profile_mode = profile_mode
        self.cache_tag = cache_tag
        self.profiles = make_profiles(graph, corpus, profile_mode)
        self.entities = list(self.profiles["entities"].values())
        self.relations = list(self.profiles["relations"].values())
        self.entity_keys = [e["key"] for e in self.entities]
        self.relation_ids = [r["id"] for r in self.relations]
        self.entity_pos = {k: i for i, k in enumerate(self.entity_keys)}
        self.relation_pos = {r["id"]: i for i, r in enumerate(self.relations)}
        self.adj_entities: dict[str, set[str]] = defaultdict(set)
        self.adj_relations: dict[str, set[str]] = defaultdict(set)
        for r in self.relations:
            self.adj_entities[r["source"]].add(r["target"])
            self.adj_entities[r["target"]].add(r["source"])
            self.adj_relations[r["source"]].add(r["id"])
            self.adj_relations[r["target"]].add(r["id"])
        self._load_or_build_embeddings()

    def _load_or_build_embeddings(self):
        tag = f"{self.cache_tag}_{self.profile_mode}_{hash_obj(self.profiles)}"
        path = CACHE / f"emb_{tag}.npz"
        if path.exists():
            d = np.load(path, allow_pickle=True)
            self.entity_emb = d["entity_emb"]
            self.relation_emb = d["relation_emb"]
            self.chunk_emb = d["chunk_emb"]
            self.chunk_ids = list(d["chunk_ids"])
            return
        ent_texts = [e["name"] + " " + " ".join(e.get("aliases", [])) + " " + e["text"]
                     for e in self.entities]
        rel_texts = [" ".join(r["keywords"]) + " " + r["text"] for r in self.relations]
        chunk_texts = [c["text"] for c in self.corpus["chunks"]]
        print(f"[embed] {len(ent_texts)} entities, {len(rel_texts)} relations, {len(chunk_texts)} chunks", flush=True)
        self.entity_emb = embed_texts(self.client, ent_texts) if ent_texts else np.zeros((0, 768), np.float32)
        self.relation_emb = embed_texts(self.client, rel_texts) if rel_texts else np.zeros((0, 768), np.float32)
        self.chunk_emb = embed_texts(self.client, chunk_texts)
        self.chunk_ids = [c["id"] for c in self.corpus["chunks"]]
        np.savez(path, entity_emb=self.entity_emb, relation_emb=self.relation_emb,
                 chunk_emb=self.chunk_emb, chunk_ids=np.array(self.chunk_ids, dtype=object))

    def embed_query(self, text: str) -> np.ndarray:
        v = embed_texts(self.client, [text])[0]
        return v

    def retrieve(self, low_keywords: list[str], high_keywords: list[str],
                 question: str, top_entities: int = 8, top_relations: int = 8,
                 onehop_cap: int = 30, use_low: bool = True, use_high: bool = True,
                 shuffle_structure: bool = False, rng: random.Random | None = None):
        matched_e, matched_r = [], []
        if use_low and low_keywords:
            q = self.embed_query(" ".join(low_keywords))
            sims = self.entity_emb @ q if len(self.entity_emb) else np.zeros(0)
            scores = []
            for i, e in enumerate(self.entities):
                s = float(sims[i]) if i < len(sims) else 0.0
                low = e["name"].lower()
                lk = " ".join(low_keywords).lower()
                if any(k.lower() in low or low in k.lower() for k in low_keywords):
                    s += 0.35
                if e["name"].lower() in lk:
                    s += 0.25
                scores.append((s, i))
            scores.sort(reverse=True)
            matched_e = [self.entity_keys[i] for _, i in scores[:top_entities]]
        if use_high and high_keywords:
            q = self.embed_query(" ".join(high_keywords))
            sims = self.relation_emb @ q if len(self.relation_emb) else np.zeros(0)
            scores = []
            for i, r in enumerate(self.relations):
                s = float(sims[i]) if i < len(sims) else 0.0
                kw = " ".join(r["keywords"]).lower()
                if any(k.lower() in kw or kw in k.lower() for k in high_keywords):
                    s += 0.3
                scores.append((s, i))
            scores.sort(reverse=True)
            matched_r = [self.relation_ids[i] for _, i in scores[:top_relations]]

        # One-hop expansion.  In the structure ablation, the matched nodes are
        # unchanged but their neighbours are drawn from a random graph with the
        # same degree sequence (configuration-model style rewiring).
        onehop_e, onehop_r = set(), set()
        if shuffle_structure:
            rng = rng or random.Random(1234)
            deg_e = [len(self.adj_entities[k]) for k in self.entity_keys]
            deg_r = [len(self.adj_relations[k]) for k in self.entity_keys]
            for k in matched_e:
                for _ in range(min(3, len(self.entity_keys))):
                    onehop_e.add(rng.choice(self.entity_keys))
                for _ in range(min(3, len(self.relations))):
                    onehop_r.add(rng.choice(self.relation_ids))
            for rid in matched_r:
                r = self.profiles["relations"][rid]
                onehop_e.update([r["source"], r["target"]])
        else:
            for k in matched_e:
                onehop_e.update(self.adj_entities.get(k, set()))
                onehop_r.update(self.adj_relations.get(k, set()))
            for rid in matched_r:
                r = self.profiles["relations"][rid]
                for k in (r["source"], r["target"]):
                    onehop_e.add(k)
                    onehop_r.update(self.adj_relations.get(k, set()))
        onehop_e -= set(matched_e)
        onehop_r -= set(matched_r)
        onehop_e = sorted(onehop_e, key=lambda k: -self.profiles["entities"][k].get("degree", 0))[:onehop_cap]
        onehop_r = sorted(onehop_r, key=lambda rid: -len(self.profiles["relations"][rid]["keywords"]))[:onehop_cap]

        # Chunks by query similarity + keyword overlap.
        q = self.embed_query(question)
        sims = self.chunk_emb @ q
        q_terms = set(t.lower() for t in TOKEN_RE.findall(question) if len(t) > 2)
        chunk_scores = []
        for i, c in enumerate(self.corpus["chunks"]):
            c_terms = set(t.lower() for t in TOKEN_RE.findall(c["text"]))
            overlap = len(q_terms & c_terms) / max(1, len(q_terms))
            chunk_scores.append((float(sims[i]) + 0.15 * overlap, i))
        chunk_scores.sort(reverse=True)
        top_chunks = [self.corpus["chunks"][i] for _, i in chunk_scores[:8]]

        return {
            "matched_entities": matched_e,
            "matched_relations": matched_r,
            "onehop_entities": onehop_e,
            "onehop_relations": onehop_r,
            "chunks": top_chunks,
            "chunk_scores": [s for s, _ in chunk_scores[:8]],
            "entity_scores": {},
        }

    def build_context(self, retrieval: dict, condition: str, question: str,
                      budget_tokens: int = 3000, temporal_timeline: str = "") -> tuple[str, dict]:
        """Assemble context in a fixed order; return text and provenance metadata.

        Conditions that include original text reserve up to 1200 tokens for it,
        so verbose LLM profiles cannot crowd out the primary evidence.
        """
        parts = []
        used = 0
        used_entities, used_relations, used_chunks = [], [], []
        used_chunk_tokens = used_graph_tokens = used_timeline_tokens = 0
        has_chunks = condition != "origin"
        chunk_reserve = min(1200, budget_tokens // 2) if has_chunks else 0
        graph_limit = max(0, budget_tokens - chunk_reserve)

        def add(text: str, kind: str, ident: str, limit: int) -> bool:
            nonlocal used, used_chunk_tokens, used_graph_tokens, used_timeline_tokens
            t = estimate_tokens(text) + 2
            if used + t > limit:
                return False
            parts.append(text)
            used += t
            if kind == "e":
                used_entities.append(ident)
                used_graph_tokens += t
            elif kind == "r":
                used_relations.append(ident)
                used_graph_tokens += t
            elif kind == "c":
                used_chunks.append(ident)
                used_chunk_tokens += t
            elif kind == "t":
                used_timeline_tokens += t
            return True

        if temporal_timeline:
            add(temporal_timeline, "t", "timeline", budget_tokens)

        if condition != "naive" and condition != "naive_temporal":
            parts.append("---Entities---")
            for key in retrieval["matched_entities"] + retrieval["onehop_entities"]:
                e = self.profiles["entities"][key]
                line = f"- {e['name']} ({e['type']}): {e['text']} [source: {', '.join(e['source_stanzas'][:6])}]"
                if not add(line, "e", key, graph_limit):
                    break
            parts.append("---Relationships---")
            for rid in retrieval["matched_relations"] + retrieval["onehop_relations"]:
                r = self.profiles["relations"][rid]
                s = self.profiles["entities"][r["source"]]["name"]
                t = self.profiles["entities"][r["target"]]["name"]
                line = f"- {s} -> {t} | keywords: {', '.join(r['keywords'])} | {r['text']} [source: {', '.join(r['source_stanzas'][:6])}]"
                if not add(line, "r", rid, graph_limit):
                    break

        if has_chunks:
            add("---Original Text---", "h", "header", budget_tokens)
            for c in retrieval["chunks"]:
                block = f"[{c['id']}] (stanzas {', '.join(c['stanza_ids'])})\n{c['text']}"
                if not add(block, "c", c["id"], budget_tokens):
                    break

        context = "\n".join(parts)
        meta = {
            "context_tokens_est": used,
            "graph_tokens_est": used_graph_tokens,
            "chunk_tokens_est": used_chunk_tokens,
            "timeline_tokens_est": used_timeline_tokens,
            "used_entities": used_entities,
            "used_relations": used_relations,
            "used_chunks": used_chunks,
        }
        return context, meta


def naive_retrieve(client: OllamaClient, corpus: dict, question: str, top_k: int = 8,
                   temporal_expand: bool = False) -> dict:
    chunks = corpus["chunks"]
    texts = [c["text"] for c in chunks]
    tag = "naive"
    path = CACHE / f"emb_chunks_{hash_obj(texts)}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        emb = d["emb"]
    else:
        emb = embed_texts(client, texts)
        np.savez(path, emb=emb)
    q = embed_texts(client, [question])[0]
    sims = emb @ q
    order = np.argsort(-sims)[:top_k]
    top_chunks = [chunks[i] for i in order]
    top_ids = [c["id"] for c in top_chunks]
    selected = list(top_chunks)
    if temporal_expand:
        # Keep the retrieved set local: top-k chunks plus their immediate
        # narrative neighbours.  Expanding via shared stanza IDs would pull in
        # the whole poem because consecutive chunks overlap by one stanza.
        by_id = {c["id"]: c for c in chunks}
        idx_by_id = {c["id"]: i for i, c in enumerate(chunks)}
        expanded = set()
        for cid in [c["id"] for c in selected]:
            i = idx_by_id[cid]
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(chunks):
                    expanded.add(j)
        selected = [chunks[i] for i in sorted(expanded)]
    return {"chunks": selected, "top_ids": top_ids,
            "scores": [float(sims[i]) for i in order],
            "chunk_ids": [c["id"] for c in selected]}


def naive_context(retrieval: dict, budget_tokens: int = 3000,
                  temporal_order: bool = False) -> tuple[str, dict]:
    """Assemble a NaiveRAG context.

    When temporal_order is true, the top-scoring chunks are admitted first and
    their narrative neighbours second; the admitted set is then presented in
    stanza order.  Without this, a fixed stanza-order truncation can drop the
    late-poem evidence for late temporal questions.
    """
    chunks = retrieval["chunks"]
    if temporal_order:
        by_id = {c["id"]: c for c in chunks}
        top_ids = [i for i in retrieval.get("top_ids", []) if i in by_id]
        top_set = set(top_ids)
        candidates = [by_id[i] for i in top_ids] + [c for c in chunks if c["id"] not in top_set]
    else:
        candidates = chunks
    used = 0
    selected = []
    for c in candidates:
        block = f"[{c['id']}] (stanzas {', '.join(c['stanza_ids'])})\n{c['text']}"
        t = estimate_tokens(block) + 2
        if used + t > budget_tokens:
            if temporal_order:
                continue
            break
        selected.append(c)
        used += t
    if temporal_order:
        selected.sort(key=lambda c: stanza_sort_key(c["stanza_ids"][0]))
    parts = ["---Original Text---"] + [
        f"[{c['id']}] (stanzas {', '.join(c['stanza_ids'])})\n{c['text']}" for c in selected]
    return "\n".join(parts), {"context_tokens_est": used,
                               "used_chunks": [c["id"] for c in selected]}


def extract_keywords(client: OllamaClient, question: str) -> tuple[dict, float, int, int]:
    res = client.generate(KEYWORD_USER.format(question=question), system=KEYWORD_SYSTEM,
                          num_predict=220, fmt="json", tag="keywords")
    try:
        d = parse_json(res.text)
        high = [str(x) for x in d.get("high_level_keywords", [])][:6]
        low = [str(x) for x in d.get("low_level_keywords", [])][:8]
    except Exception:
        high, low = [], []
    if not high and not low:
        # Deterministic fallback: content words from the question.
        toks = [t for t in TOKEN_RE.findall(question) if len(t) > 3]
        low = toks[:6]
        high = toks[:3]
    # Augment with capitalised names from the question.  A 7B keyword extractor
    # often drops the main entity ("Witch") while keeping the attribute
    # ("mother"); this deterministic augmentation is applied to every
    # graph-based condition, so it cannot favour one ablation.
    stop_proper = {"who", "what", "which", "when", "where", "why", "how", "in",
                   "on", "of", "the", "a", "an", "did", "does", "is", "are",
                   "was", "were", "put", "and", "or", "for", "to", "with", "by",
                   "from", "do", "tell", "list", "explain", "describe", "order"}
    proper = []
    for w in re.findall(r"\b[A-Z][a-zA-Z']{2,}\b", question):
        w = re.sub(r"'s$", "", w)
        if w.lower() not in stop_proper and w not in proper:
            proper.append(w)
    def uniq(xs):
        out = []
        for x in xs:
            x = str(x).strip()
            if x and x not in out:
                out.append(x)
        return out
    low = uniq(low + proper)[:10]
    high = uniq(high + proper)[:6]
    return {"high_level_keywords": high, "low_level_keywords": low}, \
        res.total_duration_ns / 1e9, res.prompt_tokens, res.completion_tokens


def is_temporal_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in [" in order", "order did", "which happened first",
                                "which came first", "how many times", "how long",
                                "how old", "put the", "before or after", "next day",
                                "tenth moon", "all day", "all night", "when did",
                                "first:", "sequence"])


def citation_stanzas(answer: str) -> list[str]:
    out = []
    for m in CITATION_RE.finditer(answer):
        for s in re.split(r"\s*,\s*", m.group(1)):
            if s and s not in out:
                out.append(s)
    return out
