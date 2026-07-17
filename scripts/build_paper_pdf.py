"""build_paper_pdf: render paper/main.md into paper/main.pdf (Week 10, EXP-013).

Pandoc/LaTeX are not installed; this uses a lightweight reportlab renderer
that handles the markdown subset the paper actually uses: ATX headings
(#..####), paragraphs, markdown pipe-tables, bold (**...**), inline code
(`...`), horizontal rules (---), and ordered reference lists. The source
markdown remains the canonical paper text (PLAN.md Week 10 assumption); the
PDF is the packaged portfolio artifact.

Usage:
    python scripts/build_paper_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "paper" / "main.md"
OUT = REPO / "paper" / "main.pdf"


def _inline(text: str) -> str:
    """Convert a markdown inline span to reportlab mini-HTML (escaped)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def build() -> Path:
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    lines = SRC.read_text(encoding="utf-8").split("\n")

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=13,
                          alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=15, leading=19, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading1"], fontSize=12, leading=15,
                        spaceBefore=10, spaceAfter=5)
    h3 = ParagraphStyle("h3", parent=styles["Heading2"], fontSize=10.5, leading=13,
                        spaceBefore=7, spaceAfter=3)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)

    flow = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue
        if stripped.startswith("---") and set(stripped) == {"-"}:
            flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.grey,
                                   spaceBefore=4, spaceAfter=8))
            i += 1
            continue

        # markdown pipe table: header row, separator row, then body rows
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for r_i, raw in enumerate(block):
                if r_i == 1:
                    continue  # separator
                cells = [c.strip() for c in raw.strip("|").split("|")]
                rows.append([Paragraph(_inline(c), cell) for c in cells])
            tbl = Table(rows, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 8))
            continue

        if stripped.startswith("#### "):
            flow.append(Paragraph(_inline(stripped[5:]), h3))
        elif stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:]), h3))
        elif stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:]), h2))
        elif stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:]), h1))
        else:
            flow.append(Paragraph(_inline(stripped), body))
        i += 1

    doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                            title="Safety-Monitored Edge Perception for ADAS",
                            author="Pruthu Parikh")
    doc.build(flow)
    return OUT


def main() -> int:
    out = build()
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
