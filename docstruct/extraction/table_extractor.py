"""Extract table cell grids and serialize them for retrieval and display."""

from __future__ import annotations

from typing import Dict, List

import pdfplumber

from docstruct.schema import Block, BoundingBox


def _crop(page, bbox: BoundingBox):
    x0 = max(0, bbox.x0)
    top = max(0, bbox.y0)
    x1 = min(page.width, bbox.x1)
    bottom = min(page.height, bbox.y1)
    if x1 <= x0 or bottom <= top:
        return None
    return page.crop((x0, top, x1, bottom))


def extract_table(page, bbox: BoundingBox) -> List[List[str]] | None:
    """Return the largest table grid found inside a bbox, or None."""
    region = _crop(page, bbox)
    if region is None:
        return None
    tables = region.extract_tables()
    if not tables:
        return None
    best = max(tables, key=lambda t: len(t) * (len(t[0]) if t else 0))
    return [[(cell or "").strip() for cell in row] for row in best]


def table_to_markdown(grid: List[List[str]]) -> str:
    """Render a cell grid as a GitHub-flavored Markdown table."""
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    norm = [row + [""] * (width - len(row)) for row in grid]
    header = norm[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in norm[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_to_plaintext(grid: List[List[str]]) -> str:
    """Render a cell grid as space-joined rows.

    Each row becomes one line of space-separated values. This preserves the
    verbatim cell content as substrings, matching how pdfplumber extract_text()
    renders table regions — which is what the Q&A gold standard is built from.
    Markdown pipes break substring-containment relevance checks.
    """
    if not grid:
        return ""
    return "\n".join("  ".join(cell for cell in row if cell) for row in grid)


def populate_tables(pdf_path: str, blocks: List[Block]) -> List[Block]:
    """Set ``table_data`` (and Markdown ``text``) on table blocks, in place."""
    by_page: Dict[int, List[Block]] = {}
    for block in blocks:
        if block.label == "table":
            by_page.setdefault(block.page_num, []).append(block)
    if not by_page:
        return blocks

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page_blocks in by_page.items():
            if page_num >= len(pdf.pages):
                continue
            page = pdf.pages[page_num]
            for block in page_blocks:
                grid = extract_table(page, block.bbox)
                if grid:
                    block.table_data = grid
                    block.text = table_to_plaintext(grid)
    return blocks
