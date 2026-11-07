# 图结构还是语言模型的改写？基于《阿特拉斯的女巫》的 LightRAG 四类不足实证研究

## 摘要

LightRAG 以实体—关系图与双层关键词检索替代扁平 chunk 检索。针对既有批评提出的四类问题——图结构增益未与 LLM 生成的画像文本分离、LLM 改写可能引入事实偏差、节点—边结构难以表示隐含时序、评测缺乏人工标注与全生命周期成本证据——本文在 Shelley 叙事长诗 *The Witch of Atlas* 上构建含 100 题、附诗节级证据的评测集，在统一模型、解码参数与上下文预算下设置 11 个消融条件，并以配对 bootstrap、McNemar 检验、Holm 校正、双评判模型与 44 条人工盲审加以检验。结果表明：归因批评（H1）与评测、部署批评（H4）得到支持，改写偏差（H2）与时序表示（H3）仅获部分支持；性能增益主要取决于是否检索到原始 chunk，而非图结构或 LLM 画像文本。完整论述见 `report/paper.pdf`。

## 实验设置

| 项目 | 说明 |
| --- | --- |
| 语料 | Shelley, *The Witch of Atlas*（Gutenberg #4696），84 节、29,026 字符、28 chunk |
| 评测集 | 100 题（事实 30、多跳 25、主题 25、时序 20），逐题绑定诗节证据 |
| 消融条件 | 11 个：闭卷、NaiveRAG、LightRAG-Full、Low/High-only、No-origin、Empty profiles、Shuffled graph、Extractive profiles、Temporal graph、Naive+order |
| 模型 | Qwen2.5-7B-Instruct（生成与评判一）、Llama-3.1-8B-Instruct（评判二），均为 Q4_K_M；嵌入为 nomic-embed-text v1.5 |
| 统计 | 配对 bootstrap（10,000 次）、McNemar、Holm 校正、评判一致性 κ、44 条人工盲审 |

## 主要结果

NaiveRAG 达 0.820，完整 LightRAG 为 0.845（+0.025，p=0.515）；抽空 LLM 画像或打乱图结构均无显著下降，而移除原始 chunk 使准确率下降 0.400（p<0.001）。LLM 改写使字面一致性下降，时序图相对标准图仅边缘显著（p=0.077）；人工盲审二次加权 κ=0.797，第二评判模型在闭卷条件下几乎不一致（κ=0.047）。索引成本 32.5k token / 195.8 s，查询 P50 4.5 s，增量更新节省 50.9%。

## 复现

```bash
python scripts/prepare_corpus.py
python scripts/build_gold.py
python src/run_index.py --out graph_c2.json
LLM_COOLDOWN=0.5 python src/run_queries.py
python src/evaluate.py --mode judge1
python src/evaluate.py --mode profiles
python src/evaluate.py --mode drift
python src/analyze.py
python src/make_latex_tables.py
python src/make_figures_paper.py
cd report && latexmk -xelatex -interaction=nonstopmode paper.tex
```

论文采用 XeLaTeX + ctexart 编译；运行环境与完整命令见 `report/appendix_repro.tex`。

## 目录

- `report/paper.tex`、`report/paper.pdf`：论文源文件与成品。
- `data/`：语料与 gold set（`corpus.json`、`gold_qa.json`）。
- `src/`、`scripts/`：索引、查询、评判、统计与绘图代码。
- `results/`：逐题答案（`answers_*.json`）、双评判结果（`judged_judge1_*.json`、`judged_judge2_*.json`）、人工盲审（`human_*.csv`）与统计表（`table_*.csv`）。
- `report/tables/`、`report/figures/`：LaTeX 表与矢量图（PDF，附 300 dpi PNG 预览）。
