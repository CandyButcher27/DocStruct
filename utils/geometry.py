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


def bbox_contains(outer: BoundingBox, inner: BoundingBox, threshold: float = 0.9) -> bool:
    """
    Check if one bbox substantially contains another.
    
    Args:
        outer: Potentially containing bbox
        inner: Potentially contained bbox
        threshold: Minimum overlap ratio (inner area in outer / inner area)
        
    Returns:
        True if inner is substantially contained in outer
    """
    x_left = max(outer.x0, inner.x0)
    y_top = max(outer.y0, inner.y0)
    x_right = min(outer.x1, inner.x1)
    y_bottom = min(outer.y1, inner.y1)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = inner.area()
    
    if inner_area == 0:
        return False
    
    return (intersection_area / inner_area) >= threshold


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


def vertical_alignment_score(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate how well two boxes are vertically aligned.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        Score between 0 and 1 (1 = perfectly aligned)
    """
    # Calculate overlap in vertical range
    y_top = max(bbox1.y0, bbox2.y0)
    y_bottom = min(bbox1.y1, bbox2.y1)
    
    if y_bottom <= y_top:
        return 0.0
    
    overlap = y_bottom - y_top
    min_height = min(bbox1.height(), bbox2.height())
    
    if min_height == 0:
        return 0.0
    
    return min(1.0, overlap / min_height)


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