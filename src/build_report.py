#!/usr/bin/env python3
"""Render report/report.md to a styled A4 PDF with WeasyPrint."""
from __future__ import annotations

import json
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MD = REPORT / "report.md"
CSS = REPORT / "style.css"
META = REPORT / "meta.json"


def main() -> None:
    body_md = MD.read_text(encoding="utf-8")
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"],
                           extension_configs={"toc": {"title": "目录", "toc_depth": "2-3"}})
    body_html = md.convert(body_md)
    title = meta.get("title", "实验报告")
    subtitle = meta.get("subtitle", "")
    abstract = meta.get("abstract", "")
    date = meta.get("date", "")
    env = meta.get("environment", "")
    cover = f"""
    <section class="cover">
      <div class="kicker">REPRODUCIBLE ABLATION STUDY · 2026</div>
      <h1>{title}</h1>
      <div class="subtitle">{subtitle}</div>
      <div class="rule"></div>
      <div class="meta">
        {date}<br>
        {env}
      </div>
      <div class="abstract">{abstract}</div>
    </section>
    """
    css = CSS.read_text(encoding="utf-8")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<style>{css}</style></head>
<body>
{cover}
{body_html}
</body></html>"""
    (REPORT / "report.html").write_text(html, encoding="utf-8")
    out = REPORT / "LightRAG_阿特拉斯的女巫_实验报告.pdf"
    HTML(string=html, base_url=str(REPORT)).write_pdf(out)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
