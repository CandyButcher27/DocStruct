"""Post-fusion containment suppression.

Runs after fuse(), before reading_order. Removes nested detections that pollute
the block list and cause double-indexing in retrieval.

Pass order (table rule always first):
1. TABLE: drop any block fully contained within a TABLE block.
2. COMPLEX OUTER (>=3 mixed inner): drop the outer, keep inner blocks.
3. STRAY INNER same label (1-2 inner): drop the inner blobs.
4. MEANINGFUL INNER different label (1-2 inner): keep both.
"""

from __future__ import annotations

from typing import List

from docstruct import config
from docstruct.schema import Block, BoundingBox
from docstruct.utils.geometry import bbox_intersection_area


def _is_contained(inner: BoundingBox, outer: BoundingBox) -> bool:
    return (
        inner.x0 >= outer.x0
        and inner.y0 >= outer.y0
        and inner.x1 <= outer.x1
        and inner.y1 <= outer.y1
    )


def suppress_table_contained(blocks: List[Block]) -> List[Block]:
    """Drop blocks fully inside a TABLE block (Rule 1 only — safe, no recall damage).

    TABLE cells are already captured in the table chunk via extract_tables/extract_text.
    Any geometry or model block that falls inside a TABLE bbox is a duplicate.
    """
    if len(blocks) <= 1:
        return blocks

    n = len(blocks)
    to_drop: set[int] = set()
    for i, block in enumerate(blocks):
        if block.label != "table":
            continue
        for j in range(n):
            if j != i and blocks[j].label != "table" and _is_contained(blocks[j].bbox, block.bbox):
                to_drop.add(j)
    return [b for i, b in enumerate(blocks) if i not in to_drop]


def suppress_text_in_tables(blocks: List[Block]) -> List[Block]:
    """Drop a text block that is mostly inside a table whose text already covers it.

    The one containment case where content demonstrably exists twice: a text block
    (typically a model proposal) sitting inside a table region whose serialized text
    already holds the same words. Every other nested case is left alone — naive
    suppression measured a 28% content loss (decisions.md). Runs after text/table
    population, so table text is available. Gated by LABEL_AWARE_CONTAINMENT.
    """
    if not config.LABEL_AWARE_CONTAINMENT or len(blocks) <= 1:
        return blocks

    tables = [b for b in blocks if b.label == "table"]
    if not tables:
        return blocks

    to_drop: set[int] = set()
    for i, block in enumerate(blocks):
        if block.label != "text" or not (block.text or "").strip():
            continue
        area = block.bbox.area
        if area <= 0:
            continue
        words = set((block.text or "").split())
        for table in tables:
            contained = bbox_intersection_area(block.bbox, table.bbox) / area
            if contained < config.CONTAINMENT_MIN_RATIO:
                continue
            table_words = set((table.text or "").split())
            covered = len(words & table_words) / max(len(words), 1)
            if covered >= config.TABLE_GRID_MIN_COVERAGE:
                to_drop.add(i)
                break
    return [b for i, b in enumerate(blocks) if i not in to_drop]


def suppress_contained(blocks: List[Block]) -> List[Block]:
    """Return blocks with nested duplicates removed."""
    if len(blocks) <= 1:
        return blocks

    n = len(blocks)
    contains: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and _is_contained(blocks[j].bbox, blocks[i].bbox):
                contains[i].append(j)

    to_drop: set[int] = set()

    for i, block in enumerate(blocks):
        if block.label == "table":
            for j in contains[i]:
                to_drop.add(j)

    for i, block in enumerate(blocks):
        if block.label == "table" or i in to_drop:
            continue
        inner_idxs = [j for j in contains[i] if j not in to_drop]
        if not inner_idxs:
            continue
        inner_labels = {blocks[j].label for j in inner_idxs}
        if len(inner_idxs) >= 3:
            to_drop.add(i)
        elif len(inner_labels) == 1 and next(iter(inner_labels)) == block.label:
            for j in inner_idxs:
                to_drop.add(j)

    return [b for i, b in enumerate(blocks) if i not in to_drop]
