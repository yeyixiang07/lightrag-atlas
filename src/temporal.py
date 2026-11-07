#!/usr/bin/env python3
"""Curated narrative timeline for the temporal-aware condition (C9).

The poem is a narrative in 78 stanzas; this annotation records the order in
which the narrated events are presented.  It is a treatment variable: C2 does
not receive it, C9 does.  The annotation is derived from the poem itself and is
included in the report's appendix.
"""
from __future__ import annotations

EVENTS = [
    ("E01", "the Witch lives on Atlas' mountain in a cavern by a secret fountain", ["P1"]),
    ("E02", "the Witch's mother becomes a vapour", ["P3"]),
    ("E03", "the mother becomes a cloud", ["P3"]),
    ("E04", "the mother becomes a meteor", ["P3"]),
    ("E05", "the mother becomes a mysterious star", ["P3"]),
    ("E06", "the Witch takes shape in the cave", ["P4", "P5"]),
    ("E07", "the beasts come to the Witch", ["P6", "P7"]),
    ("E08", "Silenus and the wood-gods visit", ["P8"]),
    ("E09", "Pan visits unseen", ["P9"]),
    ("E10", "nymphs, Ocean and Priapus come", ["P10", "P11"]),
    ("E11", "the Witch weaves her veil", ["P13"]),
    ("E12", "the cave is filled with magic treasures", ["P14", "P20"]),
    ("E13", "the Witch refuses the nymphs' service", ["P22", "P23", "P24"]),
    ("E14", "the Witch weeps and the nymphs depart", ["P25"]),
    ("E15", "by day the Witch reads scrolls and embroiders", ["P26"]),
    ("E16", "by night the Witch lies in a trance in the fountain", ["P28"]),
    ("E17", "the Witch lies in the crimson fire while snow falls", ["P29", "P30"]),
    ("E18", "the boat is made (Vulcan, Venus, Apollo, Love's plant)", ["P31", "P33"]),
    ("E19", "the boat is moored on the fountain and animated", ["P34"]),
    ("E20", "Hermaphroditus is created from fire and snow", ["P35", "P36"]),
    ("E21", "Hermaphroditus sits at the prow", ["P37"]),
    ("E22", "the boat descends the cataracts", ["P38", "P42"]),
    ("E23", "the Witch calls Hermaphroditus", ["P43"]),
    ("E24", "Hermaphroditus's wings oar the boat upward", ["P44", "P45"]),
    ("E25", "the Witch sails to the Austral waters", ["P47"]),
    ("E26", "the Witch builds a cloud haven", ["P48", "P49", "P50"]),
    ("E27", "the Witch plays pranks on the water", ["P51"]),
    ("E28", "the Witch calls the armies of spirits", ["P52"]),
    ("E29", "the spirit pavilions are pitched", ["P53"]),
    ("E30", "the Witch sits on a throne and hears news", ["P54"]),
    ("E31", "the Witch weeps and laughs", ["P54"]),
    ("E32", "the Witch rides storms and joins the upper-air spirits", ["P55", "P56"]),
    ("E33", "the Witch glides down the Nile", ["P57"]),
    ("E34", "she passes Moeris and the Mareotid lakes", ["P58"]),
    ("E35", "she passes the great Labyrinth", ["P58"]),
    ("E36", "she wanders in the night among temples", ["P59"]),
    ("E37", "she observes mortals sleeping", ["P60", "P61", "P62"]),
    ("E38", "she sees the inner spirits and has a charm", ["P66"]),
    ("E39", "in after times her spirit knows love", ["P68"]),
    ("E40", "she gives the panacea to the most beautiful", ["P69"]),
    ("E41", "she restores the body on the burial night", ["P70"]),
    ("E42", "she shakes light from the funeral lamps", ["P70"]),
    ("E43", "she unwinds the swaddling bands", ["P70"]),
    ("E44", "she takes the coffin from its niche", ["P70"]),
    ("E45", "she throws the coffin into the ditch", ["P70"]),
    ("E46", "the undecaying bodies lie age after age", ["P71"]),
    ("E47", "she writes dreams for the less beautiful", ["P72"]),
    ("E48", "the priests dream of Apis", ["P73"]),
    ("E49", "the king dreams of an ape", ["P74"]),
    ("E50", "the soldiers dream of blacksmiths", ["P75"]),
    ("E51", "the lovers dream", ["P76"]),
    ("E52", "the lovers meet the next day", ["P76"]),
    ("E53", "the tenth moon shines", ["P76"]),
    ("E54", "the Witch lets the lovers marry", ["P77"]),
    ("E55", "the poem ends and defers further tales", ["P78"]),
]

DURATIONS = [
    ("the Mother of the Months had bent her bow ten times before the Witch took shape", ["P4"]),
    ("the first-born Love was three hours old when he leaped from his cradle", ["P32"]),
    ("the Witch took three days to dress", ["D5"]),
    ("Wordsworth took nineteen years over Peter Bell", ["D4"]),
    ("the lovers' blush lasts until the tenth moon shines", ["P76"]),
    ("the undecaying bodies lie age after age", ["P71"]),
]


def timeline_text(max_events: int = 55) -> str:
    lines = ["---Narrative Timeline (events in the order narrated)---"]
    for i, (eid, label, stanzas) in enumerate(EVENTS[:max_events], 1):
        lines.append(f"{i:02d}. {label} [{', '.join(stanzas)}]")
    lines.append("---Explicit Durations---")
    for label, stanzas in DURATIONS:
        lines.append(f"- {label} [{', '.join(stanzas)}]")
    return "\n".join(lines)


def add_temporal_graph(graph: dict) -> dict:
    """Return a copy of the C2 graph with event nodes and before/after edges."""
    import copy
    g = copy.deepcopy(graph)
    prev = None
    for i, (eid, label, stanzas) in enumerate(EVENTS):
        key = f"event_{eid.lower()}"
        g["entities"][key] = {
            "key": key, "name": f"{eid}: {label}", "type": "Event",
            "descriptions": [f"{label} ({', '.join(stanzas)})."],
            "text": f"{label} ({', '.join(stanzas)}).",
            "source_stanzas": list(stanzas), "source_chunks": [], "aliases": [label],
            "mentions": 1, "degree": 2,
        }
        if prev is not None:
            rid = f"temporal_{prev}_{eid}"
            g["relations"][rid] = {
                "id": rid, "source": prev, "target": key, "keywords": ["before", "narrative order"],
                "descriptions": [f"{EVENTS[i-1][1]} happens before {label}."],
                "text": f"before: {EVENTS[i-1][1]} -> {label}",
                "source_stanzas": sorted(set(EVENTS[i-1][2] + stanzas)), "source_chunks": [],
            }
        prev = key
    g["stats"] = dict(g.get("stats", {}))
    g["stats"]["temporal_events"] = len(EVENTS)
    return g
