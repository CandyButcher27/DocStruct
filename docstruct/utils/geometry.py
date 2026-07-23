"""Pure, deterministic geometry helpers shared across the pipeline."""

from __future__ import annotations

from typing import List, Tuple

from docstruct.schema import BoundingBox


def bbox_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Intersection-over-union of two boxes (0.0 when disjoint)."""
    x_left = max(bbox1.x0, bbox2.x0)
    y_top = max(bbox1.y0, bbox2.y0)
    x_right = min(bbox1.x1, bbox2.x1)
    y_bottom = min(bbox1.y1, bbox2.y1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = bbox1.area + bbox2.area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def bbox_intersection_area(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Area of the overlap between two boxes (0.0 when disjoint)."""
    x_left = max(bbox1.x0, bbox2.x0)
    y_top = max(bbox1.y0, bbox2.y0)
    x_right = min(bbox1.x1, bbox2.x1)
    y_bottom = min(bbox1.y1, bbox2.y1)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def merge_bboxes(bboxes: List[BoundingBox]) -> BoundingBox:
    """Smallest box enclosing all inputs; page dims taken from the first box."""
    if not bboxes:
        raise ValueError("Cannot merge empty list of bounding boxes")

    return BoundingBox(
        x0=min(b.x0 for b in bboxes),
        y0=min(b.y0 for b in bboxes),
        x1=max(b.x1 for b in bboxes),
        y1=max(b.y1 for b in bboxes),
        page_width=bboxes[0].page_width,
        page_height=bboxes[0].page_height,
    )


def bbox_center(bbox: BoundingBox) -> Tuple[float, float]:
    """Return the (x, y) centroid of a box."""
    return ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)


def extract_text_from_bbox(
    lines: List, bbox: BoundingBox, overlap_threshold: float = 0.3
) -> str:
    """Concatenate text from line objects/dicts overlapping ``bbox``."""
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
