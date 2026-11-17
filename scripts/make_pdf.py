#!/usr/bin/env python3
"""Build AIAssignment3_Report.pdf from the markdown report + output figures."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF
from fpdf.errors import FPDFException

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "AIAssignment3_Report.md"
OUT = ROOT / "AIAssignment3_Report.pdf"

FIGURES = [
    ROOT / "outputs" / "recognition_accuracy_curves.png",
    ROOT / "outputs" / "reconstruction_psnr_curves.png",
    ROOT / "outputs" / "eigenfaces_rsspca.png",
    ROOT / "outputs" / "eigenfaces_pca.png",
    ROOT / "outputs" / "recon_gallery_clean.png",
    ROOT / "outputs" / "recon_gallery_occluded.png",
    ROOT / "outputs" / "saltpepper_psnr_bars.png",
]


def asciiize(s: str) -> str:
    repl = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u03b7": "eta",
        "\u2264": "<=",
        "\u2248": "~",
        "\u00d7": "x",
        "\u2212": "-",
        "\u22a5": "T",
        "\u2113": "l",
        "\u2208": "in",
        "\u211d": "R",
        "\u2225": "||",
        "\u22c5": ".",
        "\u03bb": "lambda",
        "\u2011": "-",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class PDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"CISC3024 AI Assignment #3  |  page {self.page_no()}", align="C")


def safe_multi_cell(pdf: FPDF, text: str, h: float = 5) -> None:
    text = asciiize(text)
    pdf.set_x(pdf.l_margin)
    if not text.strip():
        pdf.ln(h * 0.5)
        return
    # hard-wrap long unbroken tokens (URLs, formulas)
    chunks: list[str] = []
    for word in text.split(" "):
        while len(word) > 95:
            chunks.append(word[:95])
            word = word[95:]
        chunks.append(word)
    text = " ".join(chunks)
    try:
        pdf.multi_cell(pdf.epw, h, text)
    except FPDFException:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, h, text[:80] + "...")


def main() -> None:
    md = MD.read_text()
    md = re.sub(r"!\[.*?\]\(.*?\)\n?", "", md)
    md = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 (\2)", md)
    # flatten math-ish dollars lightly
    md = md.replace("$", "")

    pdf = PDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    in_code = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            pdf.set_font("Courier" if in_code else "Helvetica", size=8 if in_code else 11)
            continue
        if in_code:
            safe_multi_cell(pdf, line, h=4)
            continue

        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            safe_multi_cell(pdf, line[2:], h=8)
            pdf.set_font("Helvetica", size=11)
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            safe_multi_cell(pdf, line[3:], h=7)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            safe_multi_cell(pdf, line[4:], h=6)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("> "):
            pdf.set_font("Helvetica", "I", 10)
            safe_multi_cell(pdf, line[2:], h=5)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("|") and "---" not in line:
            # table row -> compact text
            cells = [c.strip() for c in line.strip("|").split("|")]
            safe_multi_cell(pdf, " | ".join(cells), h=4.5)
        elif set(line.strip()) <= set("-|: ") and "|" in line:
            continue
        else:
            # strip markdown bold/code markers
            t = line.replace("**", "").replace("`", "")
            safe_multi_cell(pdf, t, h=5)

    for fp in FIGURES:
        if not fp.exists():
            continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        safe_multi_cell(pdf, fp.name, h=8)
        pdf.image(str(fp), w=174)

    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
