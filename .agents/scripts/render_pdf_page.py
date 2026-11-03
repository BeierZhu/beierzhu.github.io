"""Render one PDF page for source-layout inspection."""

import argparse
from pathlib import Path

import fitz


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pdf")
parser.add_argument("page", type=int, help="One-based page number")
parser.add_argument("output")
args = parser.parse_args()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with fitz.open(args.pdf) as document:
    page = document[args.page - 1]
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(output)
    print(page.get_text())
