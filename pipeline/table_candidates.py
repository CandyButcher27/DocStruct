"""Table candidate detection and fusion utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from models.detector import Detection
from pipeline.layout import LayoutBlock
from schemas.block import BoundingBox
from utils.geometry import bbox_overlap, merge_bboxes


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _token_is_numeric(token: str) -> bool:
    cleaned = token.strip().replace(",", "")
    if not cleaned:
        return False
    if cleaned.count(".") <= 1:
        cleaned = cleaned.replace(".", "")
    return cleaned.isdigit()


def _line_counts_in_bbox(bbox: BoundingBox, page_lines: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    total = 0
    horizontal = 0
    vertical = 0
    for line in page_lines:
        mid_x = (float(line["x0"]) + float(line["x1"])) / 2.0
        mid_y = (float(line["y0"]) + float(line["y1"])) / 2.0
        if bbox.x0 <= mid_x <= bbox.x1 and bbox.y0 <= mid_y <= bbox.y1:
            total += 1
            if line.get("orientation") == "horizontal":
                horizontal += 1
            else:
                vertical += 1
    return total, horizontal, vertical


def _alignment_band_score(block: LayoutBlock) -> float:
    if not block.spans:
        return 0.0
    centers = sorted((span.bbox.x0 + span.bbox.x1) / 2.0 for span in block.spans)
    if len(centers) < 4:
        return 0.0

    bands = 1
    for i in range(1, len(centers)):
        if abs(centers[i] - centers[i - 1]) > 24.0:
            bands += 1
    return _clip01((bands - 1) / 4.0)


def _multirow_grid_score(block: LayoutBlock) -> float:
    if not block.spans:
        return 0.0
    rows = {}
    for span in block.spans:
        key = round(span.bbox.y0 / 8.0)
        rows[key] = rows.get(key, 0) + 1
    populated_rows = sum(1 for count in rows.values() if count >= 2)
    return _clip01(populated_rows / 6.0)


def detect_geometry_table_candidates(
    layout_blocks: List[LayoutBlock],
    page_lines: List[Dict[str, Any]],
    score_threshold: float = 0.45,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Detect geometry-driven table candidates from layout blocks."""
    candidates: List[Dict[str, Any]] = []

    for block in layout_blocks:
        if not block.bbox:
            continue

        words = block.text.split()
        if not words:
            continue

        line_total, h_lines, v_lines = _line_counts_in_bbox(block.bbox, page_lines)
        numeric_ratio = sum(1 for token in words if _token_is_numeric(token)) / max(1, len(words))
        alignment_score = _alignment_band_score(block)
        grid_score = _multirow_grid_score(block)

        line_score = _clip01((h_lines / 3.0) * 0.6 + (v_lines / 2.0) * 0.4)
        score = _clip01(
            0.45 * line_score
            + 0.25 * _clip01(numeric_ratio / 0.45)
            + 0.15 * alignment_score
            + 0.15 * grid_score
        )

        if score < score_threshold:
            continue

        candidates.append(
            {
                "bbox": block.bbox,
                "score": round(score, 4),
                "source": "geometry",
                "evidence": {
                    "line_total": line_total,
                    "horizontal_lines": h_lines,
                    "vertical_lines": v_lines,
                    "numeric_ratio": round(numeric_ratio, 4),
                    "alignment_score": round(alignment_score, 4),
                    "grid_score": round(grid_score, 4),
                },
            }
        )

    merged = non_max_suppress_candidates(candidates, iou_threshold=0.5)
    diagnostics = {
        "geometry_candidates": len(candidates),
        "geometry_candidates_kept": len(merged),
    }
    return merged, diagnostics


def model_table_candidates(
    detections: List[Detection],
    score_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    for detection in detections:
        if detection.detection_type != "table":
            continue
        if detection.confidence < score_threshold:
            continue
        candidates.append(
            {
                "bbox": detection.bbox,
                "score": round(float(detection.confidence), 4),
                "source": "model",
                "evidence": {
                    "model_confidence": round(float(detection.confidence), 4),
                },
            }
        )
    diagnostics = {
        "model_candidates": len(candidates),
    }
    return candidates, diagnostics


def non_max_suppress_candidates(
    candidates: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    ordered = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    for candidate in ordered:
        bbox = candidate["bbox"]
        if any(bbox_overlap(bbox, existing["bbox"]) > iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def fuse_table_candidates(
    geometry_candidates: List[Dict[str, Any]],
    model_candidates: List[Dict[str, Any]],
    overlap_threshold: float = 0.35,
    fusion_acceptance_threshold: float = 0.45,
    model_confidence_threshold: float = 0.7,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Fuse geometry and model table candidates into hybrid candidates."""
    fused: List[Dict[str, Any]] = []
    used_model_idxs = set()
    dropped = 0

    for geo in geometry_candidates:
        g_bbox = geo["bbox"]
        best_idx = -1
        best_overlap = 0.0
        for idx, mod in enumerate(model_candidates):
            overlap = bbox_overlap(g_bbox, mod["bbox"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        if best_idx >= 0 and best_overlap >= overlap_threshold:
            mod = model_candidates[best_idx]
            used_model_idxs.add(best_idx)
            model_score = float(mod["score"])
            geo_score = float(geo["score"])

            if best_overlap >= 0.6 and model_score >= model_confidence_threshold:
                out_bbox = mod["bbox"]
            else:
                out_bbox = merge_bboxes([g_bbox, mod["bbox"]])

            fused_score = _clip01(0.45 * geo_score + 0.45 * model_score + 0.10 * best_overlap)
            if fused_score >= fusion_acceptance_threshold:
                fused.append(
                    {
                        "bbox": out_bbox,
                        "score": round(fused_score, 4),
                        "source": "fused",
                        "evidence": {
                            "geometry_score": round(geo_score, 4),
                            "model_score": round(model_score, 4),
                            "overlap": round(best_overlap, 4),
                        },
                    }
                )
            else:
                dropped += 1
        else:
            if float(geo["score"]) >= fusion_acceptance_threshold:
                fused.append(geo)
            else:
                dropped += 1

    for idx, mod in enumerate(model_candidates):
        if idx in used_model_idxs:
            continue
        if float(mod["score"]) >= model_confidence_threshold:
            fused.append(mod)
        else:
            dropped += 1

    merged = non_max_suppress_candidates(fused, iou_threshold=0.5)
    diagnostics = {
        "fused_tables": len(merged),
        "dropped_candidates": dropped,
    }
    return merged, diagnostics


def block_table_match(
    block_bbox: BoundingBox,
    table_candidates: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[float, Dict[str, Any] | None]:
    """Find best matching table candidate for a block."""
    best_score = 0.0
    best_candidate = None
    block_area = max(1e-9, block_bbox.area())

    for candidate in table_candidates:
        candidate_bbox = candidate["bbox"]
        iou = bbox_overlap(block_bbox, candidate_bbox)

        x_left = max(block_bbox.x0, candidate_bbox.x0)
        y_low = max(block_bbox.y0, candidate_bbox.y0)
        x_right = min(block_bbox.x1, candidate_bbox.x1)
        y_high = min(block_bbox.y1, candidate_bbox.y1)
        inter = 0.0
        if x_right > x_left and y_high > y_low:
            inter = (x_right - x_left) * (y_high - y_low)

        overlap_ratio = inter / block_area
        overlap_metric = max(iou, overlap_ratio)
        weighted = overlap_metric * float(candidate.get("score", 0.0))

        if overlap_metric >= threshold and weighted > best_score:
            best_score = weighted
            best_candidate = candidate

    return best_score, best_candidate
