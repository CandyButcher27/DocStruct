#!/usr/bin/env python3
"""
Trim a PDF to a subset of pages.
"""

import argparse
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter


def trim_pdf(input_path: str, output_path: str, start: int, end: int):
    """
    Extract pages from start → end (inclusive, 0-based indexing).
    """

    reader = PdfReader(input_path)
    writer = PdfWriter()

    total_pages = len(reader.pages)

    if start < 0 or end >= total_pages or start > end:
        raise ValueError(
            f"Invalid range: start={start}, end={end}, total_pages={total_pages}"
        )

    for i in range(start, end + 1):
        writer.add_page(reader.pages[i])

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"Saved trimmed PDF: {output_path}")
    print(f"Pages kept: {start} → {end} (total {end - start + 1})")


def main():
    parser = argparse.ArgumentParser(description="Trim PDF pages")

    parser.add_argument("input_pdf", type=str)
    parser.add_argument("output_pdf", type=str)

    parser.add_argument("--start", type=int, required=True, help="Start page (0-based)")
    parser.add_argument("--end", type=int, required=True, help="End page (inclusive)")

    args = parser.parse_args()

    if not Path(args.input_pdf).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_pdf}")

    trim_pdf(args.input_pdf, args.output_pdf, args.start, args.end)


if __name__ == "__main__":
    main()