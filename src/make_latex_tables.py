#!/usr/bin/env python3
"""Generate booktabs LaTeX tables from the analysis CSV/JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["c0_closed", "c1_naive", "c2_full", "c3_low", "c4_high", "c5_origin",
         "c6_empty", "c7_shuffled", "c8_extractive", "c9_temporal", "c10_naive_temporal"]
SHORT = {"c0_closed": "C0", "c1_naive": "C1", "c2_full": "C2", "c3_low": "C3",
         "c4_high": "C4", "c5_origin": "C5", "c6_empty": "C6", "c7_shuffled": "C7",
         "c8_extractive": "C8", "c9_temporal": "C9", "c10_naive_temporal": "C10"}


def f(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.{nd}f}"


def ci(m, lo, hi, nd=2):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", OUT / name)


def main():
    main = pd.read_csv(RESULTS / "table_main.csv")
    rows = []
    for c in ORDER:
        r = main[main["condition"] == c]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(f"{SHORT[c]} & {int(r['n'])} & {ci(r['correct_norm'], r['correct_norm_lo'], r['correct_norm_hi'])} & "
                    f"{f(r['strict_correct'])} & {f(r['lenient_correct'])} & {f(r['f1'])} & "
                    f"{f(r['citation_precision'])} & {f(r['citation_recall'])} & {f(r['refusal'])} & "
                    f"{f(r['evidence_recall'])} & {f(r['context_tokens'], 0)} \\\\")
    body = r"""\begin{table}[H]
\centering
\caption{总体结果（100 题；均值与 95\% bootstrap CI）。}
\label{tab:main}
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\begin{threeparttable}
\begin{tabular}{@{}l r c c c c c c c c r@{}}
\toprule
Condition & $n$ & Accuracy [95\% CI] & Strict & Lenient & F1 & Cite-P & Cite-R & Refusal & Evid-R & Ctx \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item 正确率 = judge correctness / 2；Strict = 2；Lenient $\ge$ 1；Cite-P/R = 引用精确率/召回率；Refusal = 回答“上下文不足”的比例；Evid-R = gold 证据诗节进入上下文的比例；Ctx = 估计上下文 token。C0--C10 见表~\ref{tab:conditions}。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write("table_main.tex", body)

    by = pd.read_csv(RESULTS / "table_by_type.csv")
    rows = []
    for c in ORDER:
        sub = by[by["condition"] == c]
        if sub.empty:
            continue
        vals = []
        for t in ["factual", "multihop", "thematic", "temporal"]:
            r = sub[sub["type"] == t]
            vals.append(f(r.iloc[0]["correct_norm"]) if not r.empty else "--")
        rows.append(f"{SHORT[c]} & " + " & ".join(vals) + r" \\")
    body = r"""\begin{table}[H]
\centering
\caption{分题型正确率（judge correctness / 2）。}
\label{tab:types}
\small
\begin{threeparttable}
\begin{tabular}{@{}l cccc@{}}
\toprule
Condition & Factual (30) & Multi-hop (25) & Thematic (25) & Temporal (20) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item 数值为均值；95\% CI 见图~\ref{fig:main}(b) 与补充表。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write("table_types.tex", body)

    paired = pd.read_csv(RESULTS / "table_paired.csv")
    rows = []
    for _, r in paired.iterrows():
        scope = {"all": "All", "temporal": "Temporal",
                 "correct_norm": "Accuracy", "citation_precision": "Cite-P",
                 "unsupported_rate": "Unsupported"}.get(r["scope"], r["scope"])
        rows.append(f"{r['family']} & {SHORT.get(r['a'], r['a'])} $-$ {SHORT.get(r['b'], r['b'])} & {scope} & "
                    f"{f(r['diff'], 3)} & [{f(r['lo'], 3)}, {f(r['hi'], 3)}] & {f(r['p'], 4)} & {f(r['p_holm'], 4)} & {int(r['n'])} \\\\")
    body = r"""\begin{table}[H]
\centering
\caption{相对完整 LightRAG（C2）的配对差异、95\% CI 与 Holm 校正 $p$ 值。}
\label{tab:paired}
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\begin{threeparttable}
\begin{tabular}{@{}l l l c c c c r@{}}
\toprule
Family & Comparison & Scope & $\Delta$ accuracy & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & $n$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item 正的 $\Delta$ 表示前一个条件更准确。H1 检验图/画像归因；H2 检验改写偏差；H3 检验时序表示。C9/C10 的比较只用 20 道时序题。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write("table_paired.tex", body)

    h3 = pd.read_csv(RESULTS / "table_h3_temporal.csv")
    sub = h3[h3["type"] == "temporal"]
    rows = []
    for c in ORDER:
        r = sub[sub["condition"] == c]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(f"{SHORT[c]} & {int(r['n'])} & {f(r['correct_norm'])} & {f(r['order_accuracy'])} & "
                    f"{f(r['sequence_correct'])} & {f(r['duration_correct'])} \\\\")
    body = r"""\begin{table}[H]
\centering
\caption{时序题（20 题）：正确率、pairwise 顺序、全序与时长正确率。}
\label{tab:temporal}
\small
\begin{threeparttable}
\begin{tabular}{@{}l r c c c c@{}}
\toprule
Condition & $n$ & Accuracy & Pairwise order & Full sequence & Duration \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item pairwise 与全序分数由 judge 的 \texttt{order\_results} 计算；``unclear 对不计入 pairwise 分母。Duration 为 Q18--Q20 三道时长题的正确率。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    write("table_temporal.tex", body)

    prof_path = RESULTS / "table_profile_audit.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        rows = []
        for _, r in prof.iterrows():
            rows.append(f"{'LLM' if r['mode'] == 'llm' else 'Extractive'} & "
                        f"{'Entity' if r['kind'] == 'entity' else 'Relation'} & {int(r['n'])} & "
                        f"{f(r['mean_support'])} & {f(r['sd'])} \\\\")
        body = r"""\begin{table}[H]
\centering
\caption{画像事实性审计：被源诗节支持的原子断言比例（每格 30 条）。}
\label{tab:profiles}
\small
\begin{threeparttable}
\begin{tabular}{@{}l l r c c@{}}
\toprule
Profile & Graph element & $n$ & Support rate & SD \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item 画像从 C2（LLM）与 C8（抽取式）随机抽样；judge 将每条画像拆成至多 6 条原子断言，并逐条对照引用诗节。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
        write("table_profiles.tex", body)

    judge_path = RESULTS / "table_judge_agreement.csv"
    if judge_path.exists() and judge_path.stat().st_size > 1:
        j = pd.read_csv(judge_path)
        rows = []
        for c in ORDER:
            r = j[j["condition"] == c]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(f"{SHORT[c]} & {int(r['n'])} & {f(r['agreement'])} & {f(r['kappa_3'])} & {f(r['kappa_binary'])} & {f(r['spearman'])} \\\\")
        body = r"""\begin{table}[H]
\centering
\caption{双 judge 一致性（Qwen2.5-7B 与 Llama-3.1-8B）。}
\label{tab:judge}
\small
\begin{threeparttable}
\begin{tabular}{@{}l r c c c c@{}}
\toprule
Condition & $n$ & Agreement & $\kappa$ (quadratic) & $\kappa$ (strict) & Spearman $\rho$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item 一致率为 0/1/2 标签完全相同的比例；$\kappa$（二次加权）把 correctness 视为有序变量；$\kappa$（严格）把 correctness 2 视为阳性。若某个 judge 在该子集上为常量，$\kappa$ 退化为 0，Spearman 无法定义（表中以 --- 表示）。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
        write("table_judge.tex", body)

    qpath = RESULTS / "table_cost_query.csv"
    if qpath.exists():
        q = pd.read_csv(qpath)
        rows = []
        for c in ORDER:
            r = q[q["condition"] == c]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(f"{SHORT[c]} & {int(r['n_uncached'])} & {int(r['llm_calls_per_query'])} & "
                        f"{f(r['prompt_tokens'], 0)} & {f(r['completion_tokens'], 0)} & {f(r['total_tokens'], 0)} & "
                        f"{f(r['latency_p50'])} & {f(r['latency_p95'])} & {f(r['context_tokens'], 0)} \\\\")
        body = r"""\begin{table}[H]
\centering
\caption{查询成本与延迟（未命中缓存；本地 RTX 5060，8GB）。}
\label{tab:cost}
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\begin{threeparttable}
\begin{tabular}{@{}l r r r r r r r r@{}}
\toprule
Condition & $n$ & LLM calls & Prompt tok. & Completion tok. & Total tok. & P50 (s) & P95 (s) & Ctx tok. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\scriptsize
\item token 为每 query 均值；延迟包含关键词抽取、检索与生成。C0 为闭卷（一次调用）；C1/C10 一次；图条件两次（关键词 + 答案）。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
        write("table_cost.tex", body)


if __name__ == "__main__":
    main()
