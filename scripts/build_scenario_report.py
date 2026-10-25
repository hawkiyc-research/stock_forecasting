"""Build the bilingual Word version of one Markdown results report.

The Markdown report has a Chinese half and an English half separated by the ``## English``
heading (the project's report convention). Each half is converted with the shared docx helpers,
the Chinese part first, then a page break, then the English part.

Usage: python3 scripts/build_scenario_report.py <report.md> <output.docx> [footer prefix]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx_report import markdown_to_xml, page_break, write_docx


def halves(markdown: str) -> tuple[str, str]:
    zh, en = markdown.split("\n## English\n", 1)
    zh = zh.replace("\n## 中文\n", "\n", 1)
    return zh, en


def title_of(markdown: str) -> str:
    first = markdown.strip().splitlines()[0]
    return re.sub(r"^#\s+", "", first).strip()


def main() -> None:
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    markdown = source.read_text(encoding="utf-8")
    zh, en = halves(markdown)
    title = title_of(markdown)
    footer = sys.argv[3] if len(sys.argv) > 3 else f"stock_forecasting · {title}"
    body = markdown_to_xml(zh) + page_break() + markdown_to_xml(en)
    counts = write_docx(output, body, title=title, footer_prefix=footer)
    print("written", output, output.stat().st_size, "bytes; tables", counts["tables"],
          "paragraphs", counts["paragraphs"])


if __name__ == "__main__":
    main()
