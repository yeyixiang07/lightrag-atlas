#!/usr/bin/env python3
"""Publication-quality figures (matplotlib, vector PDF + 300 dpi PNG)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe palette + neutral grey.
PALETTE = {
    "c0_closed": "#999999", "c1_naive": "#0072B2", "c2_full": "#009E73",
    "c3_low": "#56B4E9", "c4_high": "#33a6b8", "c5_origin": "#D55E00",
    "c6_empty": "#8C6D31", "c7_shuffled": "#CC3311", "c8_extractive": "#CC79A7",
    "c9_temporal": "#332288", "c10_naive_temporal": "#EE7733",
}
ORDER = ["c0_closed", "c1_naive", "c2_full", "c3_low", "c4_high", "c5_origin",
         "c6_empty", "c7_shuffled", "c8_extractive", "c9_temporal", "c10_naive_temporal"]
SHORT = {"c0_closed": "C0 Closed-book", "c1_naive": "C1 NaiveRAG",
         "c2_full": "C2 LightRAG-Full", "c3_low": "C3 Low-only",
         "c4_high": "C4 High-only", "c5_origin": "C5 No-origin",
         "c6_empty": "C6 Empty profiles", "c7_shuffled": "C7 Shuffled graph",
         "c8_extractive": "C8 Extractive profiles", "c9_temporal": "C9 Temporal graph",
         "c10_naive_temporal": "C10 Naive + order"}
TYPES = ["factual", "multihop", "thematic", "temporal"]
TYPE_LABEL = {"factual": "Factual", "multihop": "Multi-hop", "thematic": "Thematic",
              "temporal": "Temporal"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


def panel(ax, letter):
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def read_csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def fig1():
    main = read_csv("table_main.csv")
    by = read_csv("table_by_type.csv")
    if main is None:
        return
    main["order"] = main["condition"].map({c: i for i, c in enumerate(ORDER)})
    main = main.sort_values("order")
    fig = plt.figure(figsize=(7.2, 3.35))
    gs = GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(main))
    x = main["correct_norm"].to_numpy()
    lo = main["correct_norm_lo"].to_numpy()
    hi = main["correct_norm_hi"].to_numpy()
    colors = [PALETTE.get(c, "#555555") for c in main["condition"]]
    ax.hlines(y, lo, hi, color=colors, lw=1.2, alpha=0.9)
    ax.scatter(x, y, s=28, color=colors, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(c, c) for c in main["condition"]], fontsize=6.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy (judge correctness / 2)")
    ax.set_title("Overall accuracy (100 questions, 95% CI)")
    ax.axvline(0, color="black", lw=0.5)
    for yi, xi, hi_i in zip(y, x, hi):
        ax.text(min(0.985, hi_i + 0.012), yi, f"{xi:.2f}", va="center", fontsize=6.5, color="#333333")
    ax.grid(axis="x", color="#dddddd", lw=0.4, linestyle=":")
    panel(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    mat = np.full((len(ORDER), len(TYPES)), np.nan)
    if by is not None:
        for i, c in enumerate(ORDER):
            sub = by[by["condition"] == c]
            for j, t in enumerate(TYPES):
                row = sub[sub["type"] == t]
                if not row.empty:
                    mat[i, j] = row.iloc[0]["correct_norm"]
    im = ax2.imshow(mat, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    ax2.set_xticks(range(len(TYPES)))
    ax2.set_xticklabels([TYPE_LABEL[t] for t in TYPES], rotation=25, ha="right", fontsize=7)
    ax2.set_yticks(range(len(ORDER)))
    ax2.set_yticklabels([SHORT[c].split(" ", 1)[0] for c in ORDER], fontsize=6.8)
    ax2.set_title("Accuracy by question type")
    for i in range(len(ORDER)):
        for j in range(len(TYPES)):
            if not np.isnan(mat[i, j]):
                ax2.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.2, color="white" if mat[i, j] < 0.55 else "black")
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Accuracy", fontsize=7)
    panel(ax2, "b")
    save(fig, "fig1_main")


def fig2():
    paired = read_csv("table_paired.csv")
    cost = json.loads((RESULTS / "cost_summary.json").read_text(encoding="utf-8"))["query"]
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    if paired is not None:
        h1 = paired[(paired["family"] == "H1") & (paired["scope"] == "all")].copy()
        h1["comp"] = h1["a"].map(lambda x: SHORT.get(x, x).split(" ", 1)[0]) + " vs " + \
                     h1["b"].map(lambda x: SHORT.get(x, x).split(" ", 1)[0])
        h1 = h1.iloc[::-1]
        y = np.arange(len(h1))
        for yi, (_, r) in zip(y, h1.iterrows()):
            sig = (not pd.isna(r["p_holm"])) and r["p_holm"] < 0.05
            color = "#CC3311" if sig else "#666666"
            ax.hlines(yi, r["lo"], r["hi"], color=color, lw=1.3)
            ax.scatter(r["diff"], yi, color=color, s=26, zorder=3,
                       edgecolor="white", linewidth=0.5)
            ax.text(r["hi"] + 0.015, yi, f"{r['diff']:+.2f}", va="center", fontsize=6.4)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(h1["comp"], fontsize=6.8)
        ax.set_xlabel("$\\Delta$ accuracy vs. C2 (95% CI)")
        ax.set_title("Attribution: paired differences")
        ax.grid(axis="x", color="#dddddd", lw=0.4, linestyle=":")
    panel(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    rows = [r for r in cost if r["condition"] != "c0_closed"]
    rows = sorted(rows, key=lambda r: ORDER.index(r["condition"]))
    y = np.arange(len(rows))
    g = np.array([r.get("graph_tokens") or 0 for r in rows], float)
    c = np.array([r.get("chunk_tokens") or 0 for r in rows], float)
    t = np.array([r.get("timeline_tokens") or 0 for r in rows], float)
    # Naive conditions store only the total context; the rest is original text.
    for k, r in enumerate(rows):
        if c[k] == 0 and r["condition"] in ("c1_naive", "c10_naive_temporal"):
            c[k] = max(0.0, (r.get("context_tokens") or 0) - g[k] - t[k])
    ax2.barh(y, g, color="#4C72B0", label="Graph profiles")
    ax2.barh(y, c, left=g, color="#55A868", label="Original stanzas")
    ax2.barh(y, t, left=g + c, color="#8172B2", label="Timeline")
    ax2.set_yticks(y)
    ax2.set_yticklabels([SHORT[r["condition"]].split(" ", 1)[0] for r in rows], fontsize=6.8)
    ax2.set_xlabel("Estimated context tokens")
    ax2.set_title("Context budget composition")
    ax2.legend(frameon=False, fontsize=6.4, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax2.grid(axis="x", color="#dddddd", lw=0.4, linestyle=":")
    panel(ax2, "b")
    save(fig, "fig2_attribution")


def fig3():
    prof = read_csv("table_profile_audit.csv")
    main = read_csv("table_main.csv")
    drift = json.loads((RESULTS / "literal_drift.json").read_text(encoding="utf-8")) if (RESULTS / "literal_drift.json").exists() else []
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    if prof is not None:
        kinds = ["entity", "relation"]
        x = np.arange(len(kinds)); w = 0.34
        for k, mode in enumerate(["llm", "extractive"]):
            vals, errs = [], []
            for kind in kinds:
                row = prof[(prof["mode"] == mode) & (prof["kind"] == kind)]
                vals.append(row.iloc[0]["mean_support"] if not row.empty else np.nan)
                errs.append(row.iloc[0]["sd"] / np.sqrt(row.iloc[0]["n"]) if not row.empty else 0)
            ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=2.5,
                   color="#6A3D9A" if mode == "llm" else "#1B9E77",
                   label="LLM profiles" if mode == "llm" else "Extractive profiles")
        ax.set_xticks(x); ax.set_xticklabels(["Entity", "Relation"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Source-supported claims")
        ax.set_title("Profile factuality")
        ax.legend(frameon=False, fontsize=6.0, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=2)
    panel(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    if main is not None:
        sel = main[main["condition"].isin(["c2_full", "c8_extractive"])].copy()
        sel["order"] = sel["condition"].map({c: i for i, c in enumerate(ORDER)})
        sel = sel.sort_values("order")
        metrics = [("citation_precision", "Citation precision"),
                   ("unsupported_rate", "Unsupported claim rate")]
        x = np.arange(len(metrics)); w = 0.34
        for k, (_, r) in enumerate(sel.iterrows()):
            vals = [r[m] for m, _ in metrics]
            lo = [r[m + "_lo"] for m, _ in metrics]
            hi = [r[m + "_hi"] for m, _ in metrics]
            err = [np.nan_to_num(vals) - np.nan_to_num(lo), np.nan_to_num(hi) - np.nan_to_num(vals)]
            ax2.bar(x + (k - 0.5) * w, vals, w, yerr=err, capsize=2.5,
                    color=PALETTE[r["condition"]],
                    label=SHORT[r["condition"]].split(" ", 1)[0])
        ax2.set_xticks(x); ax2.set_xticklabels([m for _, m in metrics], rotation=15, ha="right")
        ax2.set_ylim(0, 1.0)
        ax2.set_title("Answer traceability")
        ax2.legend(frameon=False, fontsize=6.2)
    panel(ax2, "b")

    ax3 = fig.add_subplot(gs[0, 2])
    if drift:
        df = pd.DataFrame(drift)
        kinds = ["llm", "extractive"]
        x = np.arange(2); w = 0.34
        for k, metric in enumerate(["cosine", "rouge_l"]):
            vals = [df[df["mode"] == m][metric].mean() for m in kinds]
            errs = [df[df["mode"] == m][metric].std(ddof=1) / np.sqrt(max(1, (df["mode"] == m).sum())) for m in kinds]
            ax3.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=2.5,
                    color="#4C72B0" if metric == "cosine" else "#DD8452",
                    label="Cosine" if metric == "cosine" else "ROUGE-L")
        ax3.set_xticks(x); ax3.set_xticklabels(["LLM", "Extractive"])
        ax3.set_ylim(0, 1.05); ax3.set_title("Profile–source similarity")
        ax3.legend(frameon=False, fontsize=6.0, loc="upper center",
                   bbox_to_anchor=(0.5, -0.26), ncol=2)
    panel(ax3, "c")
    save(fig, "fig3_factuality")


def fig4():
    h3 = read_csv("table_h3_temporal.csv")
    if h3 is None:
        return
    sub = h3[h3["type"] == "temporal"].copy()
    sub["order"] = sub["condition"].map({c: i for i, c in enumerate(ORDER)})
    sub = sub.sort_values("order")
    conds = [c for c in ORDER if c in set(sub["condition"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    metrics = [("correct_norm", "Accuracy"), ("order_accuracy", "Pairwise order"),
               ("sequence_correct", "Full sequence")]
    for ax, (m, title) in zip(axes, metrics):
        vals = [sub[sub["condition"] == c].iloc[0][m] if not sub[sub["condition"] == c].empty else np.nan for c in conds]
        ax.bar(range(len(conds)), vals, color=[PALETTE[c] for c in conds], width=0.72)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([SHORT[c].split(" ", 1)[0] for c in conds], rotation=0, ha="center", fontsize=5.4)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", lw=0.4, linestyle=":")
    axes[0].set_ylabel("Score (20 temporal questions)")
    fig.suptitle("Temporal reasoning: standard graph vs. temporal graph", fontsize=9, y=1.02)
    panel(axes[0], "a"); panel(axes[1], "b"); panel(axes[2], "c")
    fig.tight_layout()
    save(fig, "fig4_temporal")


def fig5():
    judge = read_csv("table_judge_agreement.csv")
    cost = json.loads((RESULTS / "cost_summary.json").read_text(encoding="utf-8"))["query"]
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    if judge is not None and not judge.empty:
        j = judge.copy(); j["order"] = j["condition"].map({c: i for i, c in enumerate(ORDER)})
        j = j.sort_values("order")
        x = np.arange(len(j)); w = 0.36
        ax.bar(x - w / 2, j["agreement"], w, color="#0072B2", label="Agreement")
        ax.bar(x + w / 2, j["kappa_binary"], w, color="#D55E00", label="$\\kappa$ (strict)")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[c].split(" ", 1)[0] for c in j["condition"]],
                                              rotation=0, ha="center", fontsize=5.5)
        ax.set_ylim(0, 1.05); ax.set_title("Judge agreement")
        ax.legend(frameon=False, fontsize=6.0, loc="upper center",
                  bbox_to_anchor=(0.5, -0.42), ncol=2)
    panel(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    rows = [r for r in cost if r["condition"] != "c0_closed"]
    rows = sorted(rows, key=lambda r: ORDER.index(r["condition"]))
    x = np.arange(len(rows))
    ax2.plot(x, [r["latency_p50"] for r in rows], "o-", color="#0072B2", ms=3.2, lw=0.9, label="P50")
    ax2.plot(x, [r["latency_p95"] for r in rows], "s--", color="#D55E00", ms=2.8, lw=0.8, label="P95")
    ax2.set_xticks(x); ax2.set_xticklabels([SHORT[r["condition"]].split(" ", 1)[0] for r in rows],
                                            rotation=0, ha="center", fontsize=5.5)
    ax2.set_ylabel("Latency (s, uncached)")
    ax2.set_title("Query tail latency")
    ax2.legend(frameon=False, fontsize=6.2, loc="upper center",
               bbox_to_anchor=(0.5, -0.42), ncol=2)
    panel(ax2, "b")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x, [r["total_tokens"] for r in rows], color="#009E73", width=0.7)
    ax3.set_xticks(x); ax3.set_xticklabels([SHORT[r["condition"]].split(" ", 1)[0] for r in rows],
                                            rotation=0, ha="center", fontsize=5.5)
    ax3.set_ylabel("Tokens per query")
    ax3.set_title("Query token cost")
    panel(ax3, "c")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.36)
    save(fig, "fig5_cost_judge")


def fig6():
    rows = read_csv("all_answer_rows_with_metrics.csv")
    if rows is None:
        return
    agg = rows.groupby("condition").agg(ev=("evidence_recall", "mean"),
                                        corr=("correct_norm", "mean"),
                                        n=("qid", "count")).reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for _, r in agg.iterrows():
        c = r["condition"]
        ax.scatter(r["ev"], r["corr"], s=34, color=PALETTE.get(c, "#555555"),
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(SHORT.get(c, c).split(" ", 1)[0], (r["ev"], r["corr"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Gold-evidence recall in context")
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval coverage and answer accuracy")
    ax.grid(color="#dddddd", lw=0.4, linestyle=":")
    save(fig, "fig6_evidence")


def main():
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, repr(e))


if __name__ == "__main__":
    main()
