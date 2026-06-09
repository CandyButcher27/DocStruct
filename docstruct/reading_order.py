"""Reading-order resolution and caption attachment over fused blocks.

Top-left coordinates: smaller ``y0`` is higher on the page, so blocks sort
ASCENDING by ``y0`` within a column. Captions attach one-directionally
(caption -> nearest figure/table) via ``caption_target_id``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from docstruct import config
from docstruct.schema import Block
from docstruct.utils.geometry import bbox_center

logger = logging.getLogger(__name__)


def detect_columns(blocks: List[Block], page_width: float) -> List[List[int]]:
    """Split block indices into 1 or 2 columns by the largest x-center gap."""
    positions = [(i, bbox_center(block.bbox)[0]) for i, block in enumerate(blocks)]

    if len(positions) < 2:
        return [[i for i, _ in positions]] if positions else []

    positions.sort(key=lambda p: p[1])

    gaps = [
        (idx, positions[idx][1] - positions[idx - 1][1])
        for idx in range(1, len(positions))
    ]
    split_index, max_gap = max(gaps, key=lambda g: g[1])

    if max_gap < page_width * config.COLUMN_GAP_RATIO:
        return [[i for i, _ in positions]]

    left = [i for i, _ in positions[:split_index]]
    right = [i for i, _ in positions[split_index:]]
    columns = [left, right]
    columns.sort(
        key=lambda col: sum(bbox_center(blocks[i].bbox)[0] for i in col) / len(col)
    )
    return columns


def sort_reading_order(blocks: List[Block], page_width: float) -> List[int]:
    """Return block indices in reading order (columns left->right, top->bottom)."""
    if not blocks:
        return []

    ordered: List[int] = []
    for column in detect_columns(blocks, page_width):
        column.sort(key=lambda i: blocks[i].bbox.y0)
        ordered.extend(column)
    return ordered


def find_caption_target(caption: Block, blocks: List[Block]) -> Optional[str]:
    """Nearest figure/table block id for a caption, or None if too far."""
    caption_cx, caption_cy = bbox_center(caption.bbox)
    nearest_id: Optional[str] = None
    min_distance = float("inf")

    for block in blocks:
        if block.label not in ("figure", "table"):
            continue

        block_cx, block_cy = bbox_center(block.bbox)
        distance = (
            (caption_cx - block_cx) ** 2 + (caption_cy - block_cy) ** 2
        ) ** 0.5

        # top-left origin: caption sits below target when caption.y0 >= target.y1
        if caption.bbox.y0 >= block.bbox.y1:
            distance *= 0.5

        if distance < min_distance:
            min_distance = distance
            nearest_id = block.block_id

    return nearest_id if min_distance < config.CAPTION_MAX_DISTANCE else None


def attach_captions(blocks: List[Block]) -> None:
    """Set ``caption_target_id`` on every caption block in place."""
    for block in blocks:
        if block.label != "caption":
            continue
        target_id = find_caption_target(block, blocks)
        if target_id is not None:
            block.caption_target_id = target_id


def assign_reading_order(blocks: List[Block], page_width: float) -> List[Block]:
    """Assign ``reading_order`` to each block and attach captions, in place."""
    if not blocks:
        return []

    order = sort_reading_order(blocks, page_width)
    order_map = {idx: position for position, idx in enumerate(order)}

    for i, block in enumerate(blocks):
        block.reading_order = order_map.get(i, len(blocks))

    attach_captions(blocks)
    logger.debug("Assigned reading order to %d blocks", len(blocks))
    return blocks
