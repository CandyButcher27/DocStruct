"""Extract table cell grids and serialize them for retrieval and display."""

from __future__ import annotations

import re
from typing import Dict, List

from docstruct import config
from docstruct.extraction.text_extractor import _pdf_context, _text_kwargs
from docstruct.schema import Block, BoundingBox


def _crop(page, bbox: BoundingBox):
    x0 = max(0, bbox.x0)
    top = max(0, bbox.y0)
    x1 = min(page.width, bbox.x1)
    bottom = min(page.height, bbox.y1)
    if x1 <= x0 or bottom <= top:
        return None
    return page.crop((x0, top, x1, bottom))


_TEXT_STRATEGY = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def extract_table(page, bbox: BoundingBox) -> List[List[str]] | None:
    """Return the largest table grid found inside a bbox, or None."""
    region = _crop(page, bbox)
    if region is None:
        return None
    tables = region.extract_tables(config.TABLE_SETTINGS or None)
    if not tables and config.TABLE_TEXT_STRATEGY_FALLBACK:
        # Borderless table: no ruled grid. Retry with the text-alignment strategy.
        tables = region.extract_tables({**config.TABLE_SETTINGS, **_TEXT_STRATEGY})
    if not tables:
        return None
    best = max(tables, key=lambda t: len(t) * (len(t[0]) if t else 0))
    return [[(cell or "").strip() for cell in row] for row in best]


def table_to_markdown(grid: List[List[str]]) -> str:
    """Render a cell grid as a GitHub-flavored Markdown table.

    Used by ``Document.markdown`` for human-readable export. Chunk text uses
    :func:`table_to_plaintext` instead — Markdown pipes break the verbatim
    substring containment that retrieval relevance is scored on.
    """
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    norm = [row + [""] * (width - len(row)) for row in grid]
    lines = [
        "| " + " | ".join(norm[0]) + " |",
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


def _looks_like_header(row: List[str]) -> bool:
    """Row 0 is a plausible header when every non-empty cell is non-numeric."""
    cells = [c for c in row if c]
    return bool(cells) and not any(re.search(r"\d", c) for c in cells)


def table_to_keyvalue(grid: List[List[str]]) -> str:
    """Render body rows as ``header: cell; ...`` pairs using row 0 as headers.

    Keeps a value's column identity ("Q3 revenue: 4.5") that space-joined rows lose,
    which is what makes a table chunk answerable for "what was X in Q3". Falls back to
    plaintext when row 0 is not a plausible header.
    """
    if not grid or len(grid) < 2 or not _looks_like_header(grid[0]):
        return table_to_plaintext(grid)
    headers = grid[0]
    lines = []
    for row in grid[1:]:
        pairs = [
            f"{headers[i]}: {cell}" if i < len(headers) and headers[i] else cell
            for i, cell in enumerate(row)
            if cell
        ]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def serialize_table(grid: List[List[str]]) -> str:
    """Render a grid to chunk text using the configured serialization."""
    if config.TABLE_SERIALIZATION == "keyvalue":
        return table_to_keyvalue(grid)
    return table_to_plaintext(grid)


def _grid_covers_region(rendered: str, raw: str) -> bool:
    """Did the extracted grid keep most of the words actually in the region?

    ``extract_tables`` is ruled-line based, so on a table where only part of the
    grid is ruled it happily returns that fragment — and rendering only the
    fragment silently drops every unruled row from the block's text. Comparing
    word counts against the raw region text catches that.
    """
    raw_words = len(raw.split())
    if not raw_words:
        return True
    return len(rendered.split()) >= config.TABLE_GRID_MIN_COVERAGE * raw_words


def populate_tables(pdf_path: str, blocks: List[Block], *, password: str | None = None,
                    pdf=None) -> List[Block]:
    """Set ``table_data`` and serialized ``text`` on table blocks, in place."""
    by_page: Dict[int, List[Block]] = {}
    for block in blocks:
        if block.label == "table":
            by_page.setdefault(block.page_num, []).append(block)
    if not by_page:
        return blocks

    with _pdf_context(pdf_path, pdf, password) as pdf:
        for page_num, page_blocks in by_page.items():
            if page_num >= len(pdf.pages):
                continue
            page = pdf.pages[page_num]
            for block in page_blocks:
                region = _crop(page, block.bbox)
                raw = (region.extract_text(**_text_kwargs()) or "").strip() if region is not None else ""
                grid = extract_table(page, block.bbox)
                if grid:
                    block.table_data = grid
                    rendered = serialize_table(grid)
                    # Prefer the structured render, but never at the cost of losing rows.
                    block.text = rendered if _grid_covers_region(rendered, raw) else raw
                elif raw:
                    block.text = raw
    return blocks
