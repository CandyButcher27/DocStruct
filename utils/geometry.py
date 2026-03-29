"""
Geometric utility functions for layout analysis.

All functions are pure and deterministic.
"""

from typing import List, Tuple
from schemas.block import BoundingBox


def bbox_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate intersection over union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU score between 0 and 1
    """
    # Calculate intersection
    x_left = max(bbox1.x0, bbox2.x0)
    y_top = max(bbox1.y0, bbox2.y0)
    x_right = min(bbox1.x1, bbox2.x1)
    y_bottom = min(bbox1.y1, bbox2.y1)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    bbox1_area = bbox1.area()
    bbox2_area = bbox2.area()
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def merge_bboxes(bboxes: List[BoundingBox]) -> BoundingBox:
    """
    Merge multiple bounding boxes into one encompassing box.
    
    Args:
        bboxes: List of bounding boxes to merge
        
    Returns:
        Single bounding box encompassing all input boxes
        
    Raises:
        ValueError: If bboxes list is empty
    """
    if not bboxes:
        raise ValueError("Cannot merge empty list of bounding boxes")
    
    x0 = min(bbox.x0 for bbox in bboxes)
    y0 = min(bbox.y0 for bbox in bboxes)
    x1 = max(bbox.x1 for bbox in bboxes)
    y1 = max(bbox.y1 for bbox in bboxes)
    
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def horizontal_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate horizontal distance between two bounding boxes.
    
    Returns 0 if boxes overlap horizontally.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        Horizontal distance (0 if overlapping)
    """
    if bbox1.x1 < bbox2.x0:
        return bbox2.x0 - bbox1.x1
    elif bbox2.x1 < bbox1.x0:
        return bbox1.x0 - bbox2.x1
    else:
        return 0.0


def vertical_distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate vertical distance between two bounding boxes.
    
    Returns 0 if boxes overlap vertically.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        Vertical distance (0 if overlapping)
    """
    if bbox1.y1 < bbox2.y0:
        return bbox2.y0 - bbox1.y1
    elif bbox2.y1 < bbox1.y0:
        return bbox1.y0 - bbox2.y1
    else:
        return 0.0


def bbox_center(bbox: BoundingBox) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box
        
    Returns:
        Tuple of (center_x, center_y)
    """
    center_x = (bbox.x0 + bbox.x1) / 2
    center_y = (bbox.y0 + bbox.y1) / 2
    return (center_x, center_y)


def is_column_break(bbox1: BoundingBox, bbox2: BoundingBox, page_width: float, threshold: float = 0.3) -> bool:
    """
    Determine if two boxes are in different columns.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        page_width: Width of the page
        threshold: Minimum horizontal distance as fraction of page width

    Returns:
        True if boxes likely in different columns
    """
    h_distance = horizontal_distance(bbox1, bbox2)
    return h_distance > (page_width * threshold)


def refine_bbox_with_lines(bbox: BoundingBox, lines: List, padding: float = 2.0) -> BoundingBox:
    """
    Refine a bounding box by expanding it to include overlapping PDF lines.

    Used in model-first pipelines to adjust model detection boxes to better
    align with actual text content boundaries.

    Args:
        bbox: Original bounding box (typically from model detection)
        lines: List of line objects or dicts with 'bbox' key
        padding: Extra padding to add around the refined box

    Returns:
        Refined bounding box that encompasses overlapping lines
    """
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

    for line in lines:
        if isinstance(line, dict):
            lb = line.get("bbox")
            if not lb:
                continue
            if not isinstance(lb, BoundingBox):
                lb = BoundingBox(**lb)
        else:
            lb = getattr(line, "bbox", None)
            if lb is None:
                continue

        # Check for overlap
        if not (lb.x1 < x0 or lb.x0 > x1 or lb.y1 < y0 or lb.y0 > y1):
            x0 = min(x0, lb.x0)
            y0 = min(y0, lb.y0)
            x1 = max(x1, lb.x1)
            y1 = max(y1, lb.y1)

    return BoundingBox(
        x0=max(0, x0 - padding),
        y0=max(0, y0 - padding),
        x1=x1 + padding,
        y1=y1 + padding,
    )


def extract_text_from_bbox(lines: List, bbox: BoundingBox, overlap_threshold: float = 0.3) -> str:
    """
    Extract text from lines that overlap with a bounding box.

    Args:
        lines: List of line objects or dicts with 'bbox' and 'text' keys
        bbox: Bounding box to extract text from
        overlap_threshold: Minimum IoU overlap required (default 0.3)

    Returns:
        Concatenated text from overlapping lines
    """
    collected = []

    for line in lines:
        if isinstance(line, dict):
            lb = line.get("bbox")
            text = line.get("text", "")
            if not lb:
                continue
            if not isinstance(lb, BoundingBox):
                lb = BoundingBox(**lb)
        else:
            lb = getattr(line, "bbox", None)
            text = getattr(line, "text", "")
            if lb is None:
                continue

        if bbox_overlap(bbox, lb) >= overlap_threshold:
            if text:
                collected.append(text)

    return " ".join(collected).strip()