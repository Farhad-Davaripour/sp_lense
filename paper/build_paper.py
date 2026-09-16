"""Render the same Markdown to PDF and portable LaTeX; no external service."""

import html
import json
import re

try:
    from .assets import image_asset
except ImportError:
    from assets import image_asset
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf"
for name, file in [
    ("Paper", "DejaVuSerif.ttf"),
    ("PaperBold", "DejaVuSerif-Bold.ttf"),
    ("PaperItalic", "DejaVuSerif-Italic.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / file)))
pdfmetrics.registerFontFamily(
    "Paper", normal="Paper", bold="PaperBold", italic="PaperItalic", boldItalic="PaperBold"
)
styles = {
    "body": ParagraphStyle(
        "body", fontName="Paper", fontSize=9.2, leading=12, spaceAfter=7, alignment=TA_JUSTIFY
    ),
    "title": ParagraphStyle(
        "title", fontName="PaperBold", fontSize=15.5, leading=19, alignment=TA_CENTER, spaceAfter=8
    ),
    "author": ParagraphStyle(
        "author", fontName="Paper", fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=9
    ),
    "abstract": ParagraphStyle(
        "abstract", fontName="Paper", fontSize=9.1, leading=11.6, spaceAfter=6, alignment=TA_JUSTIFY
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="PaperBold",
        fontSize=11,
        leading=14,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    ),
    "h3": ParagraphStyle(
        "h3",
        fontName="PaperBold",
        fontSize=9.6,
        leading=12,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True,
    ),
    "caption": ParagraphStyle("caption", fontName="Paper", fontSize=7.6, leading=9.4, spaceAfter=9),
    "ref": ParagraphStyle(
        "ref", fontName="Paper", fontSize=7.6, leading=9.6, spaceAfter=6, wordWrap="CJK"
    ),
}


def markup(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)
    return s


def footer(canvas, doc):
    canvas.setFont("Paper", 7)
    canvas.setFillColorRGB(0.3, 0.3, 0.3)
    canvas.drawString(44, 25, "Anonymous · exploratory diagnostic study")
    canvas.drawRightString(568, 25, str(doc.page))


def tex_escape(s):
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]:
        s = s.replace(a, b)
    return s


def main():
    text = (HERE / "manuscript.md").read_text(encoding="utf-8")
    blocks = text.split("\n\n")
    title = blocks[0][2:]
    abstract_start = text.index("## Abstract")
    intro = text.index("## 1 Introduction")
    abstract = text[abstract_start + len("## Abstract") : intro].strip()
    body = text[intro:]
    width, height = 612, 792
    margin = 44
    gap = 18
    col = (width - 2 * margin - gap) / 2
    titleheight = (
        sum(
            Paragraph(markup(value), styles[style]).wrap(width - 2 * margin, height)[1]
            + styles[style].spaceAfter
            for value, style in [
                (title, "title"),
                ("Anonymous", "author"),
                ("Abstract. " + abstract, "abstract"),
            ]
        )
        + 14
    )
    doc = BaseDocTemplate(
        str(HERE / "paper.pdf"),
        pagesize=(width, height),
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=42,
        title=title,
        author="Anonymous submission",
    )
    first = [
        Frame(
            margin,
            height - margin - titleheight,
            width - 2 * margin,
            titleheight,
            id="title",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        ),
        Frame(
            margin,
            42,
            col,
            height - margin - titleheight - 12 - 42,
            id="left",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        ),
        Frame(
            margin + col + gap,
            42,
            col,
            height - margin - titleheight - 12 - 42,
            id="right",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        ),
    ]
    other = [
        Frame(
            margin,
            42,
            col,
            height - margin - 42,
            id="left2",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        ),
        Frame(
            margin + col + gap,
            42,
            col,
            height - margin - 42,
            id="right2",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        ),
    ]
    doc.addPageTemplates(
        [
            PageTemplate(id="First", frames=first, onPage=footer),
            PageTemplate(id="Body", frames=other, onPage=footer),
        ]
    )
    story = [
        NextPageTemplate("Body"),
        Paragraph(markup(title), styles["title"]),
        Paragraph("Anonymous", styles["author"]),
        Paragraph("<b>Abstract.</b> " + markup(abstract), styles["abstract"]),
        FrameBreak(),
    ]
    refs = False
    tex = [
        r"\documentclass[10pt,twocolumn]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx,hyperref,amsmath}",
        r"\title{" + tex_escape(title) + "}",
        r"\author{Anonymous}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        tex_escape(abstract),
        r"\end{abstract}",
    ]
    for block in body.split("\n\n"):
        b = block.strip()
        if not b:
            continue
        if b.startswith("## "):
            heading = b[3:]
            refs = heading == "References"
            story.append(Paragraph(markup(heading), styles["h2"]))
            tex.append(r"\section*{" + tex_escape(heading) + "}")
            continue
        if b.startswith("### "):
            story.append(Paragraph(markup(b[4:]), styles["h3"]))
            tex.append(r"\subsection*{" + tex_escape(b[4:]) + "}")
            continue
        if b.startswith("|"):
            rows = [
                [cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in b.splitlines()
                if line.strip().startswith("|")
            ]
            rows = [
                row
                for row in rows
                if not all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in row)
            ]
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Malformed Markdown table")
            cell_style = ParagraphStyle(
                "table-cell", parent=styles["caption"], fontSize=7.4, leading=9.4, spaceAfter=0
            )
            formatted = [
                [
                    Paragraph(
                        ("<b>" + markup(cell) + "</b>") if i == 0 else markup(cell), cell_style
                    )
                    for cell in row
                ]
                for i, row in enumerate(rows)
            ]
            table = Table(
                formatted,
                colWidths=[col / len(rows[0])] * len(rows[0]),
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff5")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#59758a")),
                        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d4dce2")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(KeepTogether([table, Spacer(1, 8)]))
            fractions = "".join(
                "p{" + str(round((1 - 0.025 * len(rows[0])) / len(rows[0]), 3)) + r"\linewidth}"
                for _ in rows[0]
            )
            tex.extend(
                [
                    r"\begin{table}[t]",
                    r"\centering\footnotesize",
                    r"\setlength{\tabcolsep}{2pt}",
                    r"\begin{tabular}{" + fractions + "}",
                    r"\hline",
                ]
            )
            for i, row in enumerate(rows):
                tex.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
                if i == 0:
                    tex.append(r"\hline")
            tex.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
            continue
        match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", b, re.DOTALL)
        if match:
            caption, path = match.groups()
            im = Image(str(image_asset(HERE, path)))
            ratio = im.imageHeight / im.imageWidth
            im.drawWidth = col
            im.drawHeight = col * ratio
            story.append(
                KeepTogether([Spacer(1, 4), im, Paragraph(markup(caption), styles["caption"])])
            )
            tex.extend(
                [
                    r"\begin{figure}[t]",
                    r"\centering",
                    r"\includegraphics[width=\linewidth]{" + path + "}",
                    r"\caption{" + tex_escape(caption) + "}",
                    r"\end{figure}",
                ]
            )
            continue
        story.append(Paragraph(markup(b.replace("\n", " ")), styles["ref" if refs else "body"]))
        tex.extend([tex_escape(b.replace("\n", " ")), ""])
    tex.append(r"\end{document}")
    (HERE / "manuscript.tex").write_text("\n".join(tex), encoding="utf-8")
    doc.build(story)
    import fitz

    pdf = fitz.open(HERE / "paper.pdf")
    preview = HERE / "preview"
    preview.mkdir(exist_ok=True)
    for i, page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(preview / f"page-{i + 1:02d}.png")
    print(
        json.dumps({"pages": len(pdf), "words": len(text.split()), "pdf": str(HERE / "paper.pdf")})
    )


if __name__ == "__main__":
    main()
