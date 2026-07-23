"""Fill block text (and header font size) by cropping the source PDF."""

from __future__ import annotations

import contextlib
import re
import statistics
import unicodedata
from typing import Dict, List

from docstruct import config
from docstruct.errors import open_pdf
from docstruct.schema import Block, BoundingBox


def _pdf_context(pdf_path: str, pdf, password):
    """Reuse an already-open pdfplumber PDF, or open one for the caller."""
    return contextlib.nullcontext(pdf) if pdf is not None else open_pdf(pdf_path, password=password)

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")


def _text_kwargs() -> dict:
    ratio = config.TEXT_X_TOLERANCE_RATIO
    return {"x_tolerance_ratio": ratio} if ratio else {}


def _clean_text(text: str) -> str:
    """Apply the gated, deterministic text-normalization passes to extracted text."""
    if config.DEHYPHENATE:
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    if config.NORMALIZE_TEXT:
        text = unicodedata.normalize("NFKC", text).replace(chr(0x00AD), "")
    return text


def _crop(page, bbox: BoundingBox):
    x0 = max(0, bbox.x0)
    top = max(0, bbox.y0)
    x1 = min(page.width, bbox.x1)
    bottom = min(page.height, bbox.y1)
    if x1 <= x0 or bottom <= top:
        return None
    region = page.crop((x0, top, x1, bottom))
    if config.DEDUPE_CHARS:
        region = region.dedupe_chars(tolerance=1)
    # Exclude rotated/vertical glyphs so margin text doesn't pollute block text.
    return region.filter(lambda obj: obj.get("upright", True))


def extract_text(page, bbox: BoundingBox) -> str:
    """Extract text contained in a bbox on a pdfplumber page."""
    region = _crop(page, bbox)
    if region is None:
        return ""
    return _clean_text((region.extract_text(**_text_kwargs()) or "").strip())


def median_font_size(page, bbox: BoundingBox) -> float | None:
    """Median character size inside a bbox, used to rank header levels."""
    region = _crop(page, bbox)
    if region is None:
        return None
    sizes = [c.get("size") for c in region.chars if c.get("size")]
    return round(statistics.median(sizes), 2) if sizes else None


def is_bold_region(page, bbox: BoundingBox) -> bool | None:
    """Whether a majority of a bbox's characters are bold, or None if no chars."""
    region = _crop(page, bbox)
    if region is None:
        return None
    fonts = [(c.get("fontname", "") or "").lower() for c in region.chars]
    if not fonts:
        return None
    bold = sum(any(m in f for m in config.BOLD_FONT_MARKERS) for f in fonts)
    return bold > len(fonts) / 2


def populate_text(pdf_path: str, blocks: List[Block], *, password: str | None = None,
                  pdf=None) -> List[Block]:
    """Set ``text`` on text-bearing blocks and ``font_size`` on headers, in place."""
    by_page: Dict[int, List[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page_num, []).append(block)

    with _pdf_context(pdf_path, pdf, password) as pdf:
        for page_num, page_blocks in by_page.items():
            if page_num >= len(pdf.pages):
                continue
            page = pdf.pages[page_num]
            for block in page_blocks:
                if block.label in ("text", "header", "caption"):
                    block.text = extract_text(page, block.bbox)
                if block.label == "header":
                    block.font_size = median_font_size(page, block.bbox)
                    if config.HEADER_RANK_BY_WEIGHT:
                        block.is_bold = is_bold_region(page, block.bbox)
    return blocks
