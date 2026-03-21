"""
Reading order resolution stage.

Determines logical reading sequence of blocks.
Handles single and multi-column layouts robustly.
Attaches captions to nearest figures/tables.
"""

from typing import List, Dict, Any, Optional
from schemas.block import BoundingBox
from utils.geometry import bbox_center
from utils.logging import setup_logger

logger = setup_logger(__name__)


def get_block_bbox(block_data: Dict[str, Any]) -> Optional[BoundingBox]:
    if "layout_block" in block_data:
        return block_data["layout_block"].bbox
    elif "image_block" in block_data:
        return block_data["image_block"]["bbox"]
    return None


# Column Detection 
def detect_columns_from_blocks(
    classified_blocks: List[Dict[str, Any]],
    page_width: float
) -> List[List[int]]:
    """
    Detect whether page has 1 or 2 columns using largest x-gap splitting.

    Algorithm:
    1. Collect x-centers of all blocks
    2. Sort by x
    3. Find largest gap
    4. If gap is significant → split into 2 columns
    5. Otherwise → single column
    """

    positions = []

    for i, block_data in enumerate(classified_blocks):
        bbox = get_block_bbox(block_data)
        if bbox:
            center_x, _ = bbox_center(bbox)
            positions.append((i, center_x))

    if len(positions) < 2:
        return [[i for i, _ in positions]] if positions else []

    # Sort by x-center
    positions.sort(key=lambda x: x[1])

    # Compute gaps
    gaps = []
    for idx in range(1, len(positions)):
        prev_x = positions[idx - 1][1]
        curr_x = positions[idx][1]
        gaps.append((idx, curr_x - prev_x))

    if not gaps:
        return [[i for i, _ in positions]]

    # Find largest gap
    split_index, max_gap = max(gaps, key=lambda x: x[1])

    # Heuristic: only split if gap is meaningful
    if max_gap < page_width * 0.15:
        # Single column
        return [[i for i, _ in positions]]

    # Split into two columns
    left_column = [i for i, _ in positions[:split_index]]
    right_column = [i for i, _ in positions[split_index:]]

    columns = [left_column, right_column]

    # Ensure columns are sorted left → right
    columns.sort(
        key=lambda col: sum(
            bbox_center(bbox)[0]
            for i in col
            if (bbox := get_block_bbox(classified_blocks[i])) is not None
        ) / max(len([i for i in col if get_block_bbox(classified_blocks[i]) is not None]), 1)
    )

    return columns



# Reading Order Assignment

def sort_blocks_in_reading_order(
    classified_blocks: List[Dict[str, Any]],
    page_width: float
) -> List[int]:

    if not classified_blocks:
        return []

    columns = detect_columns_from_blocks(classified_blocks, page_width)

    sorted_indices = []

    for column in columns:
        column_blocks = []

        for idx in column:
            bbox = get_block_bbox(classified_blocks[idx])
            if bbox:
                column_blocks.append((idx, bbox.y0))

        # Sort top → bottom within column.
        # In bottom-left origin, high y0 = near top of page, so sort DESCENDING.
        column_blocks.sort(key=lambda x: x[1], reverse=True)
        sorted_indices.extend([idx for idx, _ in column_blocks])

    return sorted_indices


# Caption Attachment

def find_nearest_figure_or_table(
    caption_bbox: BoundingBox,
    classified_blocks: List[Dict[str, Any]]
) -> Optional[int]:

    nearest_idx = None
    min_distance = float("inf")

    caption_center = bbox_center(caption_bbox)

    for i, block_data in enumerate(classified_blocks):
        if block_data.get("block_type") not in ["figure", "table"]:
            continue

        bbox = get_block_bbox(block_data)
        if not bbox:
            continue

        block_center = bbox_center(bbox)

        distance = (
            (caption_center[0] - block_center[0]) ** 2 +
            (caption_center[1] - block_center[1]) ** 2
        ) ** 0.5

        # Prefer captions below figures/tables in bottom-left coordinates.
        if caption_bbox.y1 <= bbox.y0:
            distance *= 0.5

        if distance < min_distance:
            min_distance = distance
            nearest_idx = i

    return nearest_idx if min_distance < 100 else None


def attach_captions(classified_blocks: List[Dict[str, Any]]) -> None:

    for i, block_data in enumerate(classified_blocks):
        if block_data.get("block_type") != "caption":
            continue

        bbox = get_block_bbox(block_data)
        if not bbox:
            continue

        target_idx = find_nearest_figure_or_table(bbox, classified_blocks)

        if target_idx is not None:
            block_data["caption_for"] = target_idx
            classified_blocks[target_idx]["has_caption"] = i


def assign_reading_order(
    classified_blocks: List[Dict[str, Any]],
    page_width: float
) -> List[Dict[str, Any]]:

    if not classified_blocks:
        return []

    reading_order_indices = sort_blocks_in_reading_order(
        classified_blocks,
        page_width
    )

    order_map = {
        idx: order for order, idx in enumerate(reading_order_indices)
    }

    for i, block_data in enumerate(classified_blocks):
        block_data["reading_order"] = order_map.get(i, 999)

    attach_captions(classified_blocks)

    logger.debug(f"Assigned reading order to {len(classified_blocks)} blocks")

    return classified_blocks
