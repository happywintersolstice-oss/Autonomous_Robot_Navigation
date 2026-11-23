"""Render factual evidence PDF and an editable report outline, never student analysis.

Use the bundled document Python (ReportLab). Model execution is not imported.
The outline stays incomplete until the team supplies its actual authored work.
"""

import json
from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)

APP = Path(__file__).resolve().parents[2]
OUT = APP.parent / "docs/submission"


def main():
    fonts = Path("C:/Windows/Fonts")
    for name, file in (
        ("TNR", "times.ttf"),
        ("TNRBold", "timesbd.ttf"),
        ("TNRItalic", "timesi.ttf"),
        ("TNRBoldItalic", "timesbi.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily(
        "TNR",
        normal="TNR",
        bold="TNRBold",
        italic="TNRItalic",
        boldItalic="TNRBoldItalic",
    )
    body = ParagraphStyle(
        "Body",
        fontName="TNR",
        fontSize=11,
        leading=11,
        spaceAfter=4,
        allowWidows=0,
        allowOrphans=0,
    )
    numeric = ParagraphStyle("Numeric", parent=body, alignment=1)
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="TNRBold",
        spaceBefore=8,
        spaceAfter=6,
        keepWithNext=True,
    )
    story = []
    blocks = json.loads((OUT / "support-content.json").read_text())["blocks"]
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block["type"] == "paragraph":
            style = heading if block["style"].startswith(("Heading", "Title")) else body
            story.append(Paragraph(escape(block["text"]), style))
        elif block["type"] == "table":
            rows = [
                [
                    Paragraph(escape(cell), body if column == 0 else numeric)
                    for column, cell in enumerate(row)
                ]
                for row in block["rows"]
            ]
            table = Table(
                rows,
                colWidths=[width * 72 for width in block["widths_inches"]],
                repeatRows=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D9D9")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE8EF")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([table, Spacer(1, 8)])
        elif block["type"] == "image":
            image = Image(str(OUT / block["path"]))
            image.drawHeight *= 453.6 / image.drawWidth
            image.drawWidth = 453.6
            content = [image]
            if index + 1 < len(blocks) and blocks[index + 1].get("style") == "Caption":
                index += 1
                content.append(Paragraph(escape(blocks[index]["text"]), body))
            story.append(KeepTogether(content))
        index += 1

    def footer(canvas, doc):
        canvas.setFont("TNR", 11)
        canvas.drawCentredString(letter[0] / 2, 30, str(doc.page))

    pdf = OUT / "technical-evidence-appendix.pdf"
    SimpleDocTemplate(
        str(pdf),
        pagesize=letter,
        leftMargin=72,
        rightMargin=72,
        topMargin=52,
        bottomMargin=52,
        title="Technical Evidence Appendix",
        author="AI assisted technical tooling",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    outline = json.loads((OUT / "report-outline.json").read_text())["blocks"]
    html = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Report working template</title><style>@page{size:letter;margin:1in}body{max-width:6.5in;margin:1in auto;font-family:'Times New Roman',serif;font-size:11pt;line-height:1}h1,h2,h3{font-size:11pt;color:#000}p{margin:0 0 6pt}table{border-collapse:collapse;width:100%;margin:12pt 0}td,th{border:1px solid #d9d9d9;padding:6pt;vertical-align:middle}th{background:#dde8ef}main:focus{outline:1px dashed #777}@media print{body{margin:0}header{display:none}}</style><header><p>Editable outline: click the body to edit, then save your own copy. Use the original institutional cover. This is not a completed report. Verify your own authorship, contributions and permitted AI use before submission.</p></header><main contenteditable='true'>"
    ]
    for block in outline:
        if block["type"] == "paragraph":
            tag = (
                "h1"
                if block["style"] == "Title"
                else "h2" if block["style"].startswith("Heading") else "p"
            )
            html.append(f"<{tag}>{escape(block['text'])}</{tag}>")
        elif block["type"] == "table":
            html.append("<table>")
            for i, row in enumerate(block["rows"]):
                tag = "th" if i == 0 else "td"
                html.append(
                    "<tr>"
                    + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row)
                    + "</tr>"
                )
            html.append("</table>")
    html.append(
        "</main><script>function saveOutline(){const blob=new Blob(['<!doctype html>'+document.documentElement.outerHTML],{type:'text/html'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='team-report-working-copy.html';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}</script></html>"
    )
    html[0] = html[0].replace(
        "<main contenteditable='true'>",
        "<button onclick='saveOutline()'>Download your edited outline</button><button onclick='window.print()'>Print your draft</button><p>Use the download button to retain your edits. Review final pagination, disable browser print headers/footers, and keep the rubric separate.</p><main contenteditable='true' aria-label='Editable report outline'>",
    )
    (OUT / "report-working-template.html").write_text("\n".join(html), encoding="utf-8")
    print(pdf)


if __name__ == "__main__":
    main()
