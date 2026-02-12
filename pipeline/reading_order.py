"""
Reading order resolution stage.

Determines the logical reading sequence of blocks.
Handles multi-column layouts and attaches captions to figures/tables.
"""

from typing import List, Dict, Any, Optional, Tuple
from schemas.block import BoundingBox
from utils.geometry import vertical_distance, bbox_center
from utils.logging import setup_logger


logger = setup_logger(__name__)


def detect_columns_from_blocks(classified_blocks: List[Dict[str, Any]], page_width: float) -> List[List[int]]:
    """
    Detect columns by clustering block x-positions.
    
    Args:
        classified_blocks: List of classified blocks
        page_width: Page width
        
    Returns:
        List of columns (each column is list of block indices)
    """
    if not classified_blocks:
        return []
    
    # Extract x-positions (use block centers)
    block_positions = []
    for i, block_data in enumerate(classified_blocks):
        bbox = get_block_bbox(block_data)
        if bbox:
            center_x, _ = bbox_center(bbox)
            block_positions.append((i, center_x))
    
    if not block_positions:
        return []
    
    # Sort by x-position
    block_positions.sort(key=lambda x: x[1])
    
    # Cluster into columns using gap detection
    columns = []
    current_column = [block_positions[0][0]]
    last_x = block_positions[0][1]
    
    threshold = page_width * 0.25  # 25% of page width
    
    for i, x in block_positions[1:]:
        if abs(x - last_x) < threshold:
            current_column.append(i)
            last_x = (last_x + x) / 2  # Update running average
        else:
            columns.append(current_column)
            current_column = [i]
            last_x = x
    
    if current_column:
        columns.append(current_column)
    
    return columns


def get_block_bbox(block_data: Dict[str, Any]) -> Optional[BoundingBox]:
    """
    Extract bounding box from block data.
    
    Args:
        block_data: Block data dictionary
        
    Returns:
        BoundingBox or None
    """
    if "layout_block" in block_data:
        return block_data["layout_block"].bbox
    elif "image_block" in block_data:
        return block_data["image_block"]["bbox"]
    return None


def sort_blocks_in_reading_order(
    classified_blocks: List[Dict[str, Any]],
    page_width: float
) -> List[int]:
    """
    Sort blocks into reading order.
    
    Algorithm:
    1. Detect columns
    2. Within each column, sort top-to-bottom
    3. Process columns left-to-right
    
    Args:
        classified_blocks: List of classified blocks
        page_width: Page width
        
    Returns:
        List of block indices in reading order
    """
    if not classified_blocks:
        return []
    
    # Detect columns
    columns = detect_columns_from_blocks(classified_blocks, page_width)
    
    if not columns:
        # Fallback: simple top-to-bottom sort
        indices_with_y = []
        for i, block_data in enumerate(classified_blocks):
            bbox = get_block_bbox(block_data)
            if bbox:
                indices_with_y.append((i, bbox.y0))
        
        indices_with_y.sort(key=lambda x: x[1])
        return [i for i, _ in indices_with_y]
    
    # Sort each column top-to-bottom
    sorted_indices = []
    for column in columns:
        # Get y-positions for blocks in this column
        column_blocks = []
        for idx in column:
            bbox = get_block_bbox(classified_blocks[idx])
            if bbox:
                column_blocks.append((idx, bbox.y0))
        
        # Sort by y-position
        column_blocks.sort(key=lambda x: x[1])
        sorted_indices.extend([idx for idx, _ in column_blocks])
    
    return sorted_indices


def find_nearest_figure_or_table(
    caption_bbox: BoundingBox,
    classified_blocks: List[Dict[str, Any]]
) -> Optional[int]:
    """
    Find the nearest figure or table to a caption.
    
    Args:
        caption_bbox: Bounding box of caption
        classified_blocks: All classified blocks
        
    Returns:
        Index of nearest figure/table, or None
    """
    nearest_idx = None
    min_distance = float('inf')
    
    caption_center = bbox_center(caption_bbox)
    
    for i, block_data in enumerate(classified_blocks):
        block_type = block_data.get("block_type")
        if block_type not in ["figure", "table"]:
            continue
        
        bbox = get_block_bbox(block_data)
        if not bbox:
            continue
        
        # Calculate distance between centers
        block_center = bbox_center(bbox)
        distance = ((caption_center[0] - block_center[0]) ** 2 + 
                   (caption_center[1] - block_center[1]) ** 2) ** 0.5
        
        # Prefer captions below figures/tables
        if caption_bbox.y0 > bbox.y1:  # Caption is below
            distance *= 0.5  # Give preference
        
        if distance < min_distance:
            min_distance = distance
            nearest_idx = i
    
    # Only attach if reasonably close (within 100 points)
    if min_distance < 100:
        return nearest_idx
    
    return None


def attach_captions(classified_blocks: List[Dict[str, Any]]) -> None:
    """
    Attach captions to their associated figures/tables.
    
    Modifies blocks in-place to add caption relationships.
    
    Args:
        classified_blocks: List of classified blocks
    """
    for i, block_data in enumerate(classified_blocks):
        if block_data.get("block_type") != "caption":
            continue
        
        bbox = get_block_bbox(block_data)
        if not bbox:
            continue
        
        # Find nearest figure or table
        target_idx = find_nearest_figure_or_table(bbox, classified_blocks)
        
        if target_idx is not None:
            # Store relationship
            block_data["caption_for"] = target_idx
            classified_blocks[target_idx]["has_caption"] = i
            
            logger.debug(f"Attached caption {i} to block {target_idx}")


def assign_reading_order(
    classified_blocks: List[Dict[str, Any]],
    page_width: float
) -> List[Dict[str, Any]]:
    """
    Assign reading order to all blocks and attach captions.
    
    Args:
        classified_blocks: List of classified blocks
        page_width: Page width
        
    Returns:
        List of blocks with reading_order assigned
    """
    if not classified_blocks:
        return []
    
    # Sort blocks into reading order
    reading_order_indices = sort_blocks_in_reading_order(classified_blocks, page_width)
    
    # Assign reading order
    order_map = {idx: order for order, idx in enumerate(reading_order_indices)}
    
    for i, block_data in enumerate(classified_blocks):
        block_data["reading_order"] = order_map.get(i, 999)  # Default high value for missing
    
    # Attach captions to figures/tables
    attach_captions(classified_blocks)
    
    logger.debug(f"Assigned reading order to {len(classified_blocks)} blocks")
    
    return classified_blocks