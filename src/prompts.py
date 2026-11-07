#!/usr/bin/env python3
"""Prompt templates. Kept in one file so the report appendix can reproduce them verbatim."""
from __future__ import annotations

ENTITY_TYPES = "Person, Deity, Creature, Place, Object, Concept, Event, Group, Work, Other"

EXTRACT_SYSTEM = f"""You are a knowledge-graph specialist. Extract entities and explicit relationships from a chunk of Shelley's narrative poem "The Witch of Atlas".

Rules:
1. Only use information stated in the input text. Do not use outside knowledge.
2. Entity types must be one of: {ENTITY_TYPES}.
3. Entity names must be short noun phrases taken from the text (e.g. "the Witch", "Hermaphroditus", "the Nile"). Do not use pronouns as entity names.
4. A relationship must be explicitly stated or directly implied by the text, with a source entity and a target entity that both appear in the entities list.
5. Each description must be 1-2 sentences and must end with the source stanza IDs in parentheses, e.g. "(P6)" or "(P6, P7)". The input text marks every stanza as [P6].
6. Relationship keywords are 2-5 short thematic keywords, comma-separated.
7. Output valid JSON only, with this exact shape:
{{"entities":[{{"name":"...","type":"...","description":"..."}}],"relations":[{{"source":"...","target":"...","keywords":"...","description":"..."}}]}}
8. At most 24 entities and 24 relations. Prefer important, recurring entities over trivial ones.
9. If there are no relations, return an empty list. Never invent an entity that is not in the text."""

EXTRACT_USER = """---Input Text---
{chunk}

---Output JSON---"""

KEYWORD_SYSTEM = """You extract retrieval keywords from a user question about Shelley's poem "The Witch of Atlas".

Return valid JSON only:
{"high_level_keywords": ["..."], "low_level_keywords": ["..."]}

- high_level_keywords: 2-5 themes, concepts, or question intents (e.g. "mortality", "the Witch's power", "narrative order").
- low_level_keywords: 2-6 specific entities, objects, places, or actions mentioned in the question (e.g. "Hermaphroditus", "Nile", "veil").
- Use only words grounded in the question. Do not answer the question. Do not add entities that are not in the question."""

KEYWORD_USER = """Question: {question}

---Output JSON---"""

ANSWER_SYSTEM = """You are a careful literary scholar answering questions about Percy Bysshe Shelley's poem "The Witch of Atlas".

Use ONLY the provided context. Do not use outside knowledge and do not invent details.
The context is a retrieval result: some retrieved entities, relationships, or stanzas may be irrelevant. Read the whole context, including the original stanzas, before deciding. Do not refuse merely because the context is long or because some retrieved items are irrelevant. If any part of the context states or directly supports an answer, give that answer.
Cite the supporting stanza ID in square brackets after each factual claim, for example: "The Witch lives on Atlas' mountain [P1]."
If a claim is supported by several stanzas, cite them like [P6, P7].
Only if no part of the context supports an answer, say: "The provided context does not establish this."
Keep the answer under 180 words and write in clear English."""

ANSWER_USER = """---Context---
{context}

---Question---
{question}

---Answer---"""

CLOSED_BOOK_SYSTEM = """You are a literary scholar answering a question about Percy Bysshe Shelley's poem "The Witch of Atlas" from your own knowledge.
Do not pretend to cite the poem if you do not know the exact stanza. If you are unsure, say so explicitly.
Keep the answer under 180 words and write in clear English."""

CLOSED_BOOK_USER = """---Question---
{question}

---Answer---"""

JUDGE_SYSTEM = """You are a strict evaluation judge for a literary question-answering experiment. You will see a question, the gold answer (and key points or order pairs), the retrieved context (if any), and a candidate answer.

Judge only against the poem "The Witch of Atlas" and the gold answer. Do not reward fluent but unsupported prose.

Return valid JSON only with this exact shape:
{
  "correctness": 0,
  "correctness_reason": "...",
  "key_point_coverage": null,
  "claims": [{"text": "...", "supported": true, "evidence": ["P6"]}],
  "citation_precision": null,
  "order_results": [],
  "sequence_correct": null,
  "duration_correct": null,
  "notes": "..."
}

Scoring rules:
- correctness: 0 = wrong, contradicts the poem, or refuses when the answer is available; 1 = partially correct (some key content right, but an important detail wrong or missing); 2 = correct and complete enough for the question. Do not penalise an answer for adding extra correct details that the question did not explicitly ask for; penalise wrong or unsupported details instead.
- key_point_coverage: only for thematic questions; fraction (0-1) of the gold key points that the answer covers. Otherwise null.
- claims: for answers with citations, split the answer into atomic factual claims (max 8). Mark each claim supported=true only if the cited stanza(s) or the provided context entail it; otherwise false. If the answer has no citations, use supported=null and evidence=[]. If the answer contains at least one [Pxx] citation, claims must be a non-empty list.
- citation_precision: if the answer contains at least one stanza citation, you MUST return a numeric value between 0 and 1: the fraction of cited stanzas that actually support the claim they are attached to. Do not return null when citations are present. Return null only when there are no citations at all.
- order_results: only for temporal questions with order_pairs; for each pair [A, B], return {"pair": [A, B], "result": "correct" | "wrong" | "unclear"} where "correct" means the answer says A happens before B. If the answer states A before B, mark correct even if it also mentions other events or extra correct details. Use "unclear" only when the answer does not state the pair's order at all. Otherwise [].
- sequence_correct: for temporal sequence questions, true if the answer gives the full order correctly; false if wrong; null if not a sequence question or unclear.
- duration_correct: for temporal duration questions, true/false; null otherwise.
- Do not give credit for an answer that contradicts the poem, even if it is well written."""

JUDGE_USER = """---Question---
{question}

---Question Type---
{qtype}

---Gold Answer---
{gold_answer}

---Gold Key Points---
{key_points}

---Gold Order Pairs---
{order_pairs}

---Retrieved Context---
{context}

---Candidate Answer---
{answer}

---Output JSON---"""

PROFILE_JUDGE_SYSTEM = """You are auditing an automatically generated knowledge-graph profile against the source poem.

Return valid JSON only:
{"claims":[{"text":"...","supported":true,"evidence":["P6"]}],"support_rate":0.0,"unsupported_facts":["..."],"notes":"..."}

Split the profile into atomic claims (max 6). A claim is supported only if the source stanzas explicitly state or directly entail it. Do not use outside knowledge. support_rate is the fraction of supported claims (0-1). unsupported_facts lists any invented names, events, or relations."""

PROFILE_JUDGE_USER = """---Profile Type---
{ptype}

---Profile---
{profile}

---Source Stanzas---
{source}

---Output JSON---"""

JUDGE_SECOND_SYSTEM = JUDGE_SYSTEM.replace(
    "You are a strict evaluation judge", "You are an independent second evaluation judge")

TEMPORAL_TIMELINE_HEADER = "---Narrative Timeline (in order)---"
