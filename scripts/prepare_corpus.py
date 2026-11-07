#!/usr/bin/env python3
"""Parse The Witch of Atlas (Gutenberg #4696) into traceable stanzas and chunks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "pg4696_raw.txt"
OUT = ROOT / "data" / "corpus.json"

START = "*** START OF THE PROJECT GUTENBERG EBOOK THE WITCH OF ATLAS ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK THE WITCH OF ATLAS ***"
SECTION = "THE WITCH OF ATLAS."

STANZA_RE = re.compile(r"^\s*(\d+)\.\s*$")
LINE_NO_RE = re.compile(r"\s+_\d+\s*$")


def parse() -> dict:
    raw = RAW.read_text(encoding="utf-8")
    body = raw.split(START, 1)[1].split(END, 1)[0]
    lines = body.splitlines()

    section = "dedication"
    stanzas: list[dict] = []
    cur_num: int | None = None
    cur_lines: list[str] = []
    cur_start_line = 0

    def flush() -> None:
        nonlocal cur_num, cur_lines, cur_start_line
        if cur_num is None or not cur_lines:
            return
        text_lines = [LINE_NO_RE.sub("", ln).strip() for ln in cur_lines]
        text_lines = [ln for ln in text_lines if ln]
        stanzas.append(
            {
                "source_section": section,
                "source_label": cur_num,
                "text": "\n".join(text_lines),
                "line_start": cur_start_line,
                "line_end": cur_start_line + len(text_lines) - 1,
            }
        )
        cur_num = None
        cur_lines = []

    for i, line in enumerate(lines, 1):
        if line.strip() == SECTION:
            flush()
            section = "poem"
            continue
        m = STANZA_RE.match(line)
        if m:
            flush()
            cur_num = int(m.group(1))
            cur_start_line = i
            continue
        if cur_num is not None:
            if line.strip():
                cur_lines.append(line)
            elif cur_lines:
                # Blank line may terminate a stanza; keep collecting only if the
                # next non-blank line is another stanza marker. Simpler: flush on
                # a blank line after at least one content line.
                flush()
    flush()

    # Canonical IDs: dedication D1..D6, poem P1..P78 (source skips 78/79).
    counters = {"dedication": 0, "poem": 0}
    for st in stanzas:
        counters[st["source_section"]] += 1
        if st["source_section"] == "dedication":
            st["id"] = f"D{counters['dedication']}"
            st["global_index"] = counters["dedication"]
        else:
            st["id"] = f"P{counters['poem']}"
            st["global_index"] = 1000 + counters["poem"]
            st["narrative_order"] = counters["poem"]
    return {"title": "The Witch of Atlas", "author": "Percy Bysshe Shelley",
            "source": "Project Gutenberg eBook #4696", "stanzas": stanzas}


def make_chunks(stanzas: list[dict], target_chars: int = 1200, overlap_stanzas: int = 1) -> list[dict]:
    chunks: list[dict] = []
    i = 0
    while i < len(stanzas):
        group = []
        size = 0
        j = i
        while j < len(stanzas) and (size < target_chars or j == i):
            piece = f"[{stanzas[j]['id']}] {stanzas[j]['text']}"
            group.append(stanzas[j])
            size += len(piece) + 2
            j += 1
        text = "\n\n".join(f"[{s['id']}]\n{s['text']}" for s in group)
        chunks.append(
            {
                "id": f"chunk-{len(chunks):03d}",
                "stanza_ids": [s["id"] for s in group],
                "text": text,
                "char_count": len(text),
            }
        )
        if j >= len(stanzas):
            break
        i = max(i + 1, j - overlap_stanzas)
    return chunks


def main() -> None:
    corpus = parse()
    corpus["chunks"] = make_chunks(corpus["stanzas"])
    for c in corpus["chunks"]:
        c["token_estimate"] = int(c["char_count"] / 4)
    OUT.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    n_d = sum(s["source_section"] == "dedication" for s in corpus["stanzas"])
    n_p = sum(s["source_section"] == "poem" for s in corpus["stanzas"])
    chars = sum(len(s["text"]) for s in corpus["stanzas"])
    print(f"stanzas: {n_d} dedication + {n_p} poem = {len(corpus['stanzas'])}")
    print(f"poem characters: {chars}, estimated tokens: {chars//4}")
    print(f"chunks: {len(corpus['chunks'])}, sizes: {[c['char_count'] for c in corpus['chunks']]}")
    print("last poem stanza source label:", corpus["stanzas"][-1]["source_label"])


if __name__ == "__main__":
    main()
