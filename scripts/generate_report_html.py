"""Convert Markdown reports with optional figure banks into styled HTML files."""

import argparse
import html
import re
from pathlib import Path
from typing import Optional

import markdown


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def relink_markdown(markdown_text: str, base_dir: Path) -> str:
    def is_windows_absolute(target: str) -> bool:
        return bool(re.match(r"^[A-Za-z]:[\\/]", target))

    def strip_line_ref(target: str) -> str:
        if is_windows_absolute(target):
            return re.sub(r":\d+$", "", target)
        return target.split(":", 1)[0]

    def repl(match: re.Match[str]) -> str:
        label = match.group(1)
        target = match.group(2)
        if target.startswith(("http://", "https://")):
            return match.group(0)
        clean = strip_line_ref(target)
        candidate = Path(clean)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return f"[{label}]({candidate.as_uri()})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, markdown_text)


def parse_figure_bank(figure_bank_path: Path) -> list[dict[str, str]]:
    if not figure_bank_path.exists():
        return []
    lines = read_text(figure_bank_path).splitlines()
    figures: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None
    for line in lines:
        if line.startswith("## Figure "):
            if current:
                figures.append(current)
            current = {"title": line.replace("## ", "", 1).strip(), "note": "", "analysis": "", "png": ""}
            continue
        if current is None:
            continue
        if line.startswith("- PNG: ["):
            match = re.search(r"\(([^)]+)\)", line)
            if match:
                target = match.group(1)
                if re.match(r"^[A-Za-z]:[\\/]", target):
                    current["png"] = re.sub(r":\d+$", "", target)
                else:
                    current["png"] = target.split(":", 1)[0]
            continue
        text = line.strip()
        if not text or text.startswith("-"):
            continue
        if text.startswith("分析："):
            current["analysis"] = text
            continue
        if not current["note"]:
            current["note"] = text
    if current:
        figures.append(current)
    return figures


def render_figures(figures: list[dict[str, str]]) -> str:
    if not figures:
        return ""
    blocks: list[str] = ['<section class="page-break"><h1>Figures</h1>']
    for fig in figures:
        png = fig.get("png", "")
        if not png:
            continue
        uri = Path(png).resolve().as_uri()
        blocks.append('<div class="figure-block">')
        blocks.append(f"<h2>{html.escape(fig.get('title', 'Figure'))}</h2>")
        if fig.get("note"):
            note = html.escape(fig["note"].replace("指标说明：", "", 1))
            blocks.append(f"<p class='figure-note'><strong>指标说明：</strong>{note}</p>")
        if fig.get("analysis"):
            analysis = html.escape(fig["analysis"].replace("分析：", "", 1))
            blocks.append(f"<p class='figure-analysis'><strong>解读与分析：</strong>{analysis}</p>")
        blocks.append(f'<img src="{uri}" alt="{html.escape(fig.get("title", "Figure"))}" />')
        blocks.append("</div>")
    blocks.append("</section>")
    return "\n".join(blocks)


def decorate_tables(html_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        table_html = match.group(0)
        header_match = re.search(r"<tr>(.*?)</tr>", table_html, flags=re.S)
        col_count = len(re.findall(r"<th", header_match.group(1) if header_match else table_html))
        if col_count >= 16:
            cls = "table-wrap table-ultra"
        elif col_count >= 10:
            cls = "table-wrap table-wide"
        else:
            cls = "table-wrap table-normal"
        return f'<div class="{cls}">{table_html}</div>'

    return re.sub(r"<table>.*?</table>", repl, html_text, flags=re.S)


def decorate_inline_images(html_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f'<div class="inline-figure">{match.group(1)}</div>'

    return re.sub(r"<p>(<img[^>]+>)</p>", repl, html_text, flags=re.S)


def should_append_figure_bank(report_html: str, figure_bank_md: Optional[Path]) -> bool:
    if figure_bank_md is None:
        return False
    return "<img" not in report_html


CSS = """
<style>
  body {
    font-family: "Times New Roman", "SimSun", "Songti SC", serif;
    color: #1f1f1f; line-height: 1.42; font-size: 10.2px;
    max-width: 210mm; margin: 0 auto; padding: 12px;
  }
  h1, h2, h3 {
    color: #17324d; margin-top: 0.9em; margin-bottom: 0.32em;
    font-family: "Times New Roman", "SimSun", "Songti SC", serif;
  }
  h1 { font-size: 22px; border-bottom: 1.8px solid #d6dde5; padding-bottom: 6px; }
  h2 { font-size: 17px; border-left: 4px solid #6d8aa6; padding-left: 8px; }
  h3 { font-size: 14px; }
  p, li { margin: 0.22em 0; text-align: justify; }
  ul { padding-left: 1.15em; }
  strong { font-weight: 700; color: #111; }
  code { background: #f4f6f8; padding: 1px 4px; border-radius: 3px; font-family: "Consolas", "Courier New", monospace; }
  .table-wrap { margin: 6px 0 10px; }
  table { border-collapse: collapse; width: 100%; table-layout: fixed; }
  .table-normal table { font-size: 8.8px; }
  .table-wide table { font-size: 7.2px; }
  .table-ultra table { font-size: 6.2px; }
  th, td { border: 1px solid #cfd6dd; padding: 2px 3px; vertical-align: top; word-break: break-word; overflow-wrap: anywhere; }
  th { background: #eef3f7; white-space: normal; line-height: 1.18; }
  td strong { font-weight: 800; color: #000; }
  tr:nth-child(even) td { background: #fafcfd; }
  a { color: #245b8a; text-decoration: none; }
  .inline-figure { margin: 8px 0 12px; text-align: center; }
  .inline-figure img { max-width: 100%; max-height: 118mm; border: 1px solid #d9dfe5; object-fit: contain; }
  .figure-block { margin: 10px 0 12px; }
  .figure-block img { max-width: 100%; max-height: 148mm; border: 1px solid #d9dfe5; object-fit: contain; }
  .figure-note { color: #4d4d4d; margin-bottom: 6px; }
  .figure-analysis { color: #303030; margin-top: 0; margin-bottom: 6px; }
  .page-break { page-break-before: always; break-before: page; }
  blockquote { border-left: 3px solid #c9d4df; margin: 8px 0; padding-left: 10px; color: #444; }
  @media print {
    @page { size: A4; margin: 9mm 8mm; }
    .page-break { page-break-before: always; }
    .table-wrap, .figure-block, .inline-figure { break-inside: avoid-page; }
  }
</style>
"""


def build_html(report_md: Path, figure_bank_md: Optional[Path], output_html: Path, title: str) -> None:
    report_md_text = relink_markdown(read_text(report_md), report_md.parent)
    report_html = markdown.markdown(report_md_text, extensions=["tables", "fenced_code", "toc"], output_format="html5")
    report_html = decorate_tables(report_html)
    report_html = decorate_inline_images(report_html)
    figures_html = (
        render_figures(parse_figure_bank(figure_bank_md))
        if should_append_figure_bank(report_html, figure_bank_md)
        else ""
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  {CSS}
</head>
<body>
  {report_html}
  {figures_html}
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Markdown reports with figure banks into styled HTML files.")
    parser.add_argument("--report", required=True, help="Markdown report path.")
    parser.add_argument("--figure-bank", default="", help="Optional figure-bank markdown path.")
    parser.add_argument("--out", default="", help="Output HTML path (default: <report>.html).")
    parser.add_argument("--title", default="Reviewer Report", help="HTML page title.")
    args = parser.parse_args()

    report_md = Path(args.report).resolve()
    figure_bank = Path(args.figure_bank).resolve() if args.figure_bank.strip() else None
    html_out = Path(args.out).resolve() if args.out.strip() else report_md.with_suffix(".html")

    build_html(report_md, figure_bank, html_out, title=args.title)
    print(f"[ReportHTML] {html_out}")


if __name__ == "__main__":
    main()
