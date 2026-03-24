"""
Geometry-based region proposal generation for true hybrid pipeline.

This module generates block proposals from geometric signals (text clustering,
PDF ruling lines, image regions) INDEPENDENTLY of model detections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from schemas.block import BoundingBox
from utils.geometry import bbox_overlap, merge_bboxes


@dataclass
class RegionProposal:
    """A region proposal from either model or geometry source."""

    bbox: BoundingBox
    source: Literal["model", "geometry"]
    proposed_type: Optional[str]  # None for pure geometry proposals
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


def _infer_type_from_text_features(
    avg_font_size: float,
    page_avg_font_size: float,
    word_count: int,
    text: str,
    y_position: float,
    page_height: float,
) -> tuple[str, float]:
    """
    Infer block type from text features using heuristics.

    Returns:
        (proposed_type, confidence)
    """
    # Header detection
    if avg_font_size > page_avg_font_size * 1.3 and word_count < 15:
        if y_position > page_height * 0.7:  # Near top of page
            return "header", 0.7
        return "header", 0.5

    # Caption detection
    text_lower = text.lower().strip()
    caption_prefixes = ("figure", "fig.", "fig ", "table", "image", "chart", "graph")
    if text_lower.startswith(caption_prefixes) and word_count < 30:
        return "caption", 0.75

    # Short text could be header or caption
    if word_count < 8 and avg_font_size < page_avg_font_size * 0.9:
        return "caption", 0.4

    # Default to text
    confidence = 0.6 if word_count > 10 else 0.4
    return "text", confidence


def _generate_text_cluster_proposals(
    layout_blocks: List,
    page_data,
) -> List[RegionProposal]:
    """
    Generate proposals from text cluster blocks (LayoutBlocks).

    This reuses existing geometric clustering from form_layout_blocks().
    """
    proposals = []

    # Calculate page average font size
    total_font_size = 0.0
    font_count = 0
    for block in layout_blocks:
        if hasattr(block, "avg_font_size") and block.avg_font_size > 0:
            total_font_size += block.avg_font_size
            font_count += 1

    page_avg_font_size = total_font_size / font_count if font_count > 0 else 12.0

    for block in layout_blocks:
        bbox = getattr(block, "bbox", None)
        if not bbox:
            continue

        text = getattr(block, "text", "") or ""
        avg_font_size = getattr(block, "avg_font_size", page_avg_font_size)
        spans = getattr(block, "spans", [])
        word_count = len(text.split())

        # Infer type from text features
        proposed_type, type_confidence = _infer_type_from_text_features(
            avg_font_size=avg_font_size,
            page_avg_font_size=page_avg_font_size,
            word_count=word_count,
            text=text,
            y_position=bbox.y1,  # Top of bbox
            page_height=page_data.height,
        )

        # Base confidence from clustering quality
        cluster_confidence = min(1.0, 0.3 + 0.1 * min(len(spans), 5))

        # Combined confidence
        confidence = 0.6 * type_confidence + 0.4 * cluster_confidence

        proposals.append(
            RegionProposal(
                bbox=bbox,
                source="geometry",
                proposed_type=proposed_type,
                confidence=round(confidence, 4),
                evidence={
                    "strategy": "text_cluster",
                    "span_count": len(spans),
                    "word_count": word_count,
                    "avg_font_size": round(avg_font_size, 2),
                    "page_avg_font_size": round(page_avg_font_size, 2),
                },
            )
        )

    return proposals


def _generate_line_bounded_proposals(
    page_lines: List[Dict[str, Any]],
    page_data,
    min_area: float = 1000.0,
) -> List[RegionProposal]:
    """
    Detect rectangular regions bounded by PDF ruling lines.

    High confidence indicator for tables and forms.
    """
    proposals = []

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in page_lines:
        line_bbox = line.get("bbox")
        if not line_bbox:
            continue

        if isinstance(line_bbox, dict):
            x0, y0 = line_bbox.get("x0", 0), line_bbox.get("y0", 0)
            x1, y1 = line_bbox.get("x1", 0), line_bbox.get("y1", 0)
        elif hasattr(line_bbox, "x0"):
            x0, y0, x1, y1 = line_bbox.x0, line_bbox.y0, line_bbox.x1, line_bbox.y1
        else:
            continue

        width = abs(x1 - x0)
        height = abs(y1 - y0)

        if width > height * 3 and width > 20:  # Horizontal
            h_lines.append({"y": (y0 + y1) / 2, "x0": x0, "x1": x1})
        elif height > width * 3 and height > 20:  # Vertical
            v_lines.append({"x": (x0 + x1) / 2, "y0": y0, "y1": y1})

    # Need at least 2 of each to form a rectangle
    if len(h_lines) < 2 or len(v_lines) < 2:
        return proposals

    # Sort lines
    h_lines.sort(key=lambda l: l["y"])
    v_lines.sort(key=lambda l: l["x"])

    # Find rectangular regions
    seen_regions = set()

    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            h1, h2 = h_lines[i], h_lines[i + 1]
            v1, v2 = v_lines[j], v_lines[j + 1]

            # Check if lines actually bound a region
            x0 = v1["x"]
            x1 = v2["x"]
            y0 = h1["y"]
            y1 = h2["y"]

            if x1 <= x0 or y1 <= y0:
                continue

            # Check minimum area
            area = (x1 - x0) * (y1 - y0)
            if area < min_area:
                continue

            # Dedupe similar regions
            region_key = (round(x0 / 10), round(y0 / 10), round(x1 / 10), round(y1 / 10))
            if region_key in seen_regions:
                continue
            seen_regions.add(region_key)

            try:
                bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
            except Exception:
                continue

            proposals.append(
                RegionProposal(
                    bbox=bbox,
                    source="geometry",
                    proposed_type="table",  # Line-bounded = high table likelihood
                    confidence=0.7,
                    evidence={
                        "strategy": "line_bounded",
                        "h_line_count": len(h_lines),
                        "v_line_count": len(v_lines),
                        "area": round(area, 2),
                    },
                )
            )

    return proposals


def _generate_image_proposals(
    images: List[Dict[str, Any]],
    page_num: int,
) -> List[RegionProposal]:
    """
    Generate proposals from embedded images (potential figures).
    """
    proposals = []

    for img in images:
        bbox = img.get("bbox")
        if not bbox:
            continue

        if isinstance(bbox, dict):
            try:
                bbox = BoundingBox(**bbox)
            except Exception:
                continue

        # Images are high-confidence figures
        proposals.append(
            RegionProposal(
                bbox=bbox,
                source="geometry",
                proposed_type="figure",
                confidence=0.8,
                evidence={
                    "strategy": "image_region",
                    "image_width": img.get("width", 0),
                    "image_height": img.get("height", 0),
                },
            )
        )

    return proposals


def _merge_overlapping_proposals(
    proposals: List[RegionProposal],
    iou_threshold: float = 0.7,
) -> List[RegionProposal]:
    """
    Merge highly overlapping proposals from the same strategy.
    """
    if len(proposals) <= 1:
        return proposals

    # Sort by confidence descending
    sorted_props = sorted(proposals, key=lambda p: p.confidence, reverse=True)
    kept = []

    for prop in sorted_props:
        # Check if overlaps too much with any kept proposal
        overlaps = False
        for kept_prop in kept:
            if bbox_overlap(prop.bbox, kept_prop.bbox) > iou_threshold:
                overlaps = True
                break

        if not overlaps:
            kept.append(prop)

    return kept


def generate_geometry_proposals(
    page_data,
    layout_blocks: List,
    page_lines: List[Dict[str, Any]],
    min_line_bounded_area: float = 1000.0,
) -> List[RegionProposal]:
    """
    Generate geometry-based region proposals independently of model detections.

    Strategies:
    1. Text clustering (from existing LayoutBlocks)
    2. Line-bounded regions (PDF ruling lines -> tables/forms)
    3. Image regions (embedded images -> figures)

    Args:
        page_data: PageData object from decomposition
        layout_blocks: List of LayoutBlock objects from process_page_layout
        page_lines: List of PDF ruling lines

    Returns:
        List of RegionProposal objects
    """
    all_proposals = []

    # Strategy 1: Text cluster proposals
    text_proposals = _generate_text_cluster_proposals(layout_blocks, page_data)
    all_proposals.extend(text_proposals)

    # Strategy 2: Line-bounded regions (tables)
    line_proposals = _generate_line_bounded_proposals(
        page_lines, page_data, min_area=min_line_bounded_area
    )
    all_proposals.extend(line_proposals)

    # Strategy 3: Image regions (figures)
    image_proposals = _generate_image_proposals(page_data.images, page_data.page_num)
    all_proposals.extend(image_proposals)

    # Merge highly overlapping proposals
    merged = _merge_overlapping_proposals(all_proposals, iou_threshold=0.7)

    return merged


def detections_to_proposals(
    detections: List,
) -> List[RegionProposal]:
    """
    Convert model Detection objects to RegionProposal format.

    Args:
        detections: List of Detection objects from model

    Returns:
        List of RegionProposal objects with source="model"
    """
    proposals = []

    for det in detections:
        proposals.append(
            RegionProposal(
                bbox=det.bbox,
                source="model",
                proposed_type=det.detection_type,
                confidence=det.confidence,
                evidence={
                    "strategy": "model_detection",
                    "model_type": det.detection_type,
                },
            )
        )

    return proposals
