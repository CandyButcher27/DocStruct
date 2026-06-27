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

from docstruct.schema import Block, BoundingBox


def _is_contained(inner: BoundingBox, outer: BoundingBox) -> bool:
    return (
        inner.x0 >= outer.x0
        and inner.y0 >= outer.y0
        and inner.x1 <= outer.x1
        and inner.y1 <= outer.y1
    )


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
