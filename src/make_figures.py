#!/usr/bin/env python3
"""Generate publication-style figures from the result tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 200,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 10,
})

COND_ORDER = [
    ("c0_closed", "C0 Closed-book"), ("c1_naive", "C1 NaiveRAG"),
    ("c2_full", "C2 LightRAG-Full"), ("c3_low", "C3 Low-only"),
    ("c4_high", "C4 High-only"), ("c5_origin", "C5 No original text"),
    ("c6_empty", "C6 Empty profiles"), ("c7_shuffled", "C7 Shuffled graph"),
    ("c8_extractive", "C8 Extractive profiles"), ("c9_temporal", "C9 Temporal graph"),
    ("c10_naive_temporal", "C10 Naive + order"),
]
COLORS = {
    "c0_closed": "#9e9e9e", "c1_naive": "#64b5f6", "c2_full": "#2e7d32",
    "c3_low": "#4db6ac", "c4_high": "#26a69a", "c5_origin": "#ef6c00",
    "c6_empty": "#8d6e63", "c7_shuffled": "#c62828", "c8_extractive": "#6a1b9a",
    "c9_temporal": "#4527a0", "c10_naive_temporal": "#ec407a",
}
SHORT = dict(COND_ORDER)


def save(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def fig_main():
    df = pd.read_csv(RESULTS / "table_main.csv")
    df["order"] = df["condition"].map({c: i for i, (c, _) in enumerate(COND_ORDER)})
    df = df.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(df))
    x = df["correct_norm"].to_numpy()
    lo = df["correct_norm_lo"].to_numpy()
    hi = df["correct_norm_hi"].to_numpy()
    colors = [COLORS.get(c, "#607d8b") for c in df["condition"]]
    ax.barh(y, x, xerr=[x - lo, hi - x], color=colors, alpha=0.9, capsize=3,
            error_kw={"lw": 1.0, "ecolor": "#333333"})
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(c, c) for c in df["condition"]])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("正确率（judge correctness / 2，95% bootstrap CI）")
    ax.set_title("各条件的总体正确率（100 题，Qwen2.5-7B 生成 + 同模型 judge）")
    for yi, xi in zip(y, x):
        ax.text(min(0.985, xi + 0.012), yi, f"{xi:.2f}", va="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_main_accuracy.png")


def fig_by_type():
    df = pd.read_csv(RESULTS / "table_by_type.csv")
    conds = ["c0_closed", "c1_naive", "c2_full", "c3_low", "c4_high", "c5_origin",
             "c6_empty", "c7_shuffled", "c8_extractive", "c9_temporal", "c10_naive_temporal"]
    types = ["factual", "multihop", "thematic", "temporal"]
    tlabel = {"factual": "事实细节", "multihop": "多跳", "thematic": "主题", "temporal": "时序"}
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    width = 0.8 / len(conds)
    for j, cond in enumerate(conds):
        sub = df[df["condition"] == cond].set_index("type")
        vals = [sub.loc[t, "correct_norm"] if t in sub.index else np.nan for t in types]
        lo = [sub.loc[t, "correct_norm_lo"] if t in sub.index else np.nan for t in types]
        hi = [sub.loc[t, "correct_norm_hi"] if t in sub.index else np.nan for t in types]
        x = np.arange(len(types)) + (j - (len(conds) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.92, color=COLORS[cond], alpha=0.9,
               label=SHORT[cond],
               yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
               error_kw={"lw": 0.7, "ecolor": "#333333"}, capsize=1.5)
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels([tlabel[t] for t in types])
    ax.set_ylabel("正确率")
    ax.set_ylim(0, 1.0)
    ax.set_title("按题型拆分的正确率（均值 + 95% CI）")
    ax.legend(ncol=3, fontsize=7.2, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.tight_layout()
    save(fig, "fig_by_type.png")


def fig_context():
    data = json.loads((RESULTS / "cost_summary.json").read_text(encoding="utf-8"))["query"]
    rows = [r for r in data if r["condition"] != "c0_closed"]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(rows))
    g = np.array([r.get("graph_tokens") or 0 for r in rows], dtype=float)
    c = np.array([r.get("chunk_tokens") or 0 for r in rows], dtype=float)
    t = np.array([r.get("timeline_tokens") or 0 for r in rows], dtype=float)
    ax.bar(x, g, label="图画像（实体/关系）", color="#5c6bc0")
    ax.bar(x, c, bottom=g, label="原始诗节", color="#66bb6a")
    ax.bar(x, t, bottom=g + c, label="时序时间线", color="#ab47bc")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(r["condition"], r["condition"]) for r in rows], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("平均上下文 token（估算）")
    ax.set_title("上下文预算的构成（固定 3000 token 上限）")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_context_composition.png")


def fig_profile():
    path = RESULTS / "table_profile_audit.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels, vals, errs, colors = [], [], [], []
    for _, r in df.iterrows():
        labels.append(("LLM 画像" if r["mode"] == "llm" else "抽取式画像") + " · " + ("实体" if r["kind"] == "entity" else "关系"))
        vals.append(r["mean_support"])
        errs.append(r["sd"] / max(1, np.sqrt(r["n"])))
        colors.append("#5e35b1" if r["mode"] == "llm" else "#00897b")
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("画像断言被源诗节支持的比例")
    ax.set_title("画像事实性审计（每类 30 条，人工/LLM 双通道抽样）")
    for xi, vi in zip(x, vals):
        ax.text(xi, min(0.98, vi + 0.03), f"{vi:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_profile_audit.png")


def fig_temporal():
    path = RESULTS / "table_h3_temporal.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["type"] == "temporal"].set_index("condition")
    conds = [c for c, _ in COND_ORDER if c in sub.index]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    metrics = [("correct_norm", "正确率"), ("order_accuracy", "顺序对准确率"), ("sequence_correct", "全序正确率")]
    for ax, (m, title) in zip(axes, metrics):
        vals = [sub.loc[c, m] if m in sub.columns else np.nan for c in conds]
        ax.bar(range(len(conds)), vals, color=[COLORS[c] for c in conds], alpha=0.9)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([SHORT[c].split(" ", 1)[0] for c in conds], fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("得分")
    fig.suptitle("时序题（20 题）的表现：标准图 vs 时序图 vs 仅时序元数据", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_temporal.png")


def fig_evidence_scatter():
    path = RESULTS / "all_answer_rows_with_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    agg = df.groupby("condition").agg(
        evidence=("evidence_recall", "mean"),
        correct=("correct_norm", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for _, r in agg.iterrows():
        c = r["condition"]
        ax.scatter(r["evidence"], r["correct"], s=90, color=COLORS.get(c, "#607d8b"),
                   edgecolor="white", zorder=3)
        ax.annotate(SHORT.get(c, c).split(" ", 1)[0], (r["evidence"], r["correct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("gold 证据诗节在上下文中的召回率")
    ax.set_ylabel("正确率")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.0)
    ax.set_title("检索证据覆盖与最终正确率")
    fig.tight_layout()
    save(fig, "fig_evidence_vs_accuracy.png")


def fig_cost():
    data = json.loads((RESULTS / "cost_summary.json").read_text(encoding="utf-8"))
    q = pd.DataFrame(data["query"])
    q = q[q["condition"] != "c0_closed"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    x = np.arange(len(q))
    ax.bar(x, q["latency_p50"], color="#42a5f5", label="P50")
    ax.plot(x, q["latency_p95"], "o-", color="#c62828", ms=4, lw=1, label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(c, c) for c in q["condition"]], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("查询延迟（秒，未命中缓存）")
    ax.set_title("查询尾延迟")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.bar(x, q["total_tokens"], color="#7e57c2")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(c, c) for c in q["condition"]], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("每 query token（prompt + completion）")
    ax.set_title("查询 token 成本")
    fig.suptitle("全周期部署成本（本地 RTX 5060，8GB）", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_cost.png")


def fig_judge():
    path = RESULTS / "table_judge_agreement.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["agreement"], width=0.38, color="#26a69a", label="一致率")
    ax.bar(x + 0.2, df["kappa_binary"], width=0.38, color="#ef6c00", label="Cohen's κ（严格）")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(c, c) for c in df["condition"]], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("judge1 vs judge2")
    ax.set_title("双 judge 一致性（同一批答案）")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_judge_agreement.png")


def main():
    for fn in [fig_main, fig_by_type, fig_context, fig_profile, fig_temporal,
               fig_evidence_scatter, fig_cost, fig_judge]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, repr(e))


if __name__ == "__main__":
    main()
