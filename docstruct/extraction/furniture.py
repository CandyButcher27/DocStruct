"""Cross-page running-header/footer/page-number removal.

Runs after text population, before chunking, and only when
``config.STRIP_PAGE_FURNITURE`` is set. Deterministic and document-global: a page's
top-most or bottom-most line whose digit-normalized text repeats at nearly the same
vertical position across several pages is furniture, not content. No model.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from docstruct import config
from docstruct.schema import Block

_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", _DIGITS_RE.sub("#", text.lower())).strip()


def strip_page_furniture(blocks: List[Block]) -> List[Block]:
    """Return ``blocks`` with repeated header/footer/page-number blocks removed."""
    if not config.STRIP_PAGE_FURNITURE:
        return blocks

    by_page: Dict[int, List[Block]] = defaultdict(list)
    for block in blocks:
        if block.label in ("text", "header", "caption") and (block.text or "").strip():
            by_page[block.page_num].append(block)

    # Candidate furniture: the top-most and bottom-most line block on each page.
    candidates: List[Block] = []
    for page_blocks in by_page.values():
        candidates.append(min(page_blocks, key=lambda b: b.bbox.y0))
        candidates.append(max(page_blocks, key=lambda b: b.bbox.y1))

    # Bucket by normalized text and a coarse y band; a bucket spanning enough
    # distinct pages is furniture.
    tol = config.FURNITURE_Y_TOLERANCE
    buckets: Dict[tuple, set] = defaultdict(set)
    members: Dict[tuple, List[Block]] = defaultdict(list)
    for block in candidates:
        key = (_normalize(block.text or ""), round(block.bbox.y0 / tol))
        if not key[0]:
            continue
        buckets[key].add(block.page_num)
        members[key].append(block)

    drop_ids = {
        b.block_id
        for key, pages in buckets.items()
        if len(pages) >= config.FURNITURE_MIN_PAGES
        for b in members[key]
    }
    return [b for b in blocks if b.block_id not in drop_ids]
