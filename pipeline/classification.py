"""Block classification stage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.detector import Detection
from pipeline.conflict_resolver import ConflictResolver
from pipeline.layout import LayoutBlock
from pipeline.table_candidates import block_table_match
from schemas.block import BoundingBox
from utils.geometry import bbox_overlap
from utils.logging import setup_logger

logger = setup_logger(__name__)

TEXT_BLOCK_TYPES = ["text", "header", "table", "caption"]
HYBRID_MODEL_WEIGHT = 0.60
HYBRID_RULE_WEIGHT = 0.25
HYBRID_GEO_WEIGHT = 0.15
CLASS_CONFIDENCE_THRESHOLDS = {
    "text": 0.35,
    "header": 0.50,
    "table": 0.55,
    "figure": 0.45,
    "caption": 0.45,
}


def compute_rule_score(block: LayoutBlock, block_type: str, avg_font_size: float, page_height: float) -> float:
    score = 0.0

    if block_type == "header":
        if block.avg_font_size > avg_font_size * 1.3:
            score += 0.4
        if len(block.text.split()) < 15:
            score += 0.3
        if block.bbox and block.bbox.y1 > page_top_threshold(page_height):
            score += 0.3
    elif block_type == "text":
        if 0.8 < block.avg_font_size / avg_font_size < 1.2:
            score += 0.5
        if len(block.text.split()) > 10:
            score += 0.5
    elif block_type == "caption":
        if block.avg_font_size < avg_font_size * 0.9:
            score += 0.4
        if block.text.lower().startswith(("fig", "figure", "table", "image")):
            score += 0.6
    elif block_type == "table":
        if any(char in block.text for char in ["\t", "|"]):
            score += 0.3
        words = block.text.split()
        if len(words) > 5 and sum(1 for w in words if w.replace(".", "").isdigit()) > len(words) * 0.3:
            score += 0.3
    return min(1.0, score)


def compute_model_score(
    block_bbox: BoundingBox,
    detections: List[Detection],
    target_type: str,
    overlap_threshold: float = 0.4,
) -> float:
    best_score = 0.0
    for detection in detections:
        if detection.detection_type != target_type:
            continue
        overlap = bbox_overlap(block_bbox, detection.bbox)
        if overlap < overlap_threshold:
            continue
        weighted = detection.confidence * (0.5 + 0.5 * overlap)
        best_score = max(best_score, weighted)
    return best_score


def page_top_threshold(page_height: float) -> float:
    return page_height * 0.85


def compute_geometric_score(block: LayoutBlock, page_width: float, page_height: float) -> float:
    if not block.bbox:
        return 0.0
    score = 1.0
    area_ratio = block.bbox.area() / (page_width * page_height)
    if area_ratio < 0.001:
        score *= 0.5
    elif area_ratio > 0.8:
        score *= 0.7

    aspect_ratio = block.bbox.width() / block.bbox.height() if block.bbox.height() > 0 else 1.0
    if aspect_ratio > 10 or aspect_ratio < 0.1:
        score *= 0.8
    if block.bbox.x0 < 0 or block.bbox.y0 < 0:
        score *= 0.3
    if block.bbox.x1 > page_width or block.bbox.y1 > page_height:
        score *= 0.3
    return score


def _score_text_block(
    block: LayoutBlock,
    detections: List[Detection],
    avg_font_size: float,
    page_height: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    rule_scores = {block_type: compute_rule_score(block, block_type, avg_font_size, page_height) for block_type in TEXT_BLOCK_TYPES}
    if not block.bbox:
        model_scores = {block_type: 0.0 for block_type in TEXT_BLOCK_TYPES}
    else:
        model_scores = {block_type: compute_model_score(block.bbox, detections, block_type) for block_type in TEXT_BLOCK_TYPES}
    return rule_scores, model_scores


def _max_score(score_map: Dict[str, float]) -> tuple[str, float]:
    best_type = "text"
    best_score = -1.0
    for block_type, score in score_map.items():
        if score > best_score:
            best_type = block_type
            best_score = score
    return best_type, max(0.0, best_score)


def classify_block(
    block: LayoutBlock,
    detections: List[Detection],
    page_width: float,
    page_height: float,
    avg_font_size: float,
    page_lines: Optional[List[Dict[str, Any]]] = None,
    variant_mode: str = "hybrid",
) -> Dict[str, Any]:
    rule_scores, model_scores = _score_text_block(block, detections, avg_font_size, page_height)
    geometric_score = compute_geometric_score(block, page_width, page_height)
    best_rule_type, best_rule_score = _max_score(rule_scores)
    best_model_type, best_model_score = _max_score(model_scores)

    if variant_mode == "geometry":
        return {
            "block_type": best_rule_type,
            "confidence": {
                "rule_score": rule_scores[best_rule_type],
                "model_score": 0.0,
                "geometric_score": geometric_score,
                "final_confidence": min(1.0, 0.6 * rule_scores[best_rule_type] + 0.4 * geometric_score),
            },
            "has_model_support": False,
        }

    if variant_mode == "model":
        chosen_type = best_model_type if best_model_score > 0 else "text"
        return {
            "block_type": chosen_type,
            "confidence": {
                "rule_score": 0.0,
                "model_score": model_scores.get(chosen_type, 0.0),
                "geometric_score": geometric_score,
                "final_confidence": min(1.0, 0.9 * model_scores.get(chosen_type, 0.0) + 0.1 * geometric_score),
            },
            "has_model_support": best_model_score > 0,
        }

    if best_model_score <= 0.0:
        return {
            "block_type": best_rule_type,
            "confidence": {
                "rule_score": rule_scores[best_rule_type],
                "model_score": 0.0,
                "geometric_score": geometric_score,
                "final_confidence": min(1.0, 0.75 * rule_scores[best_rule_type] + 0.25 * geometric_score),
            },
            "has_model_support": False,
        }

    resolver = ConflictResolver()
    resolved_type, _ = resolver.resolve(
        rule_type=best_rule_type,
        model_type=best_model_type,
        rule_score=best_rule_score,
        model_score=best_model_score,
        block_bbox=block.bbox,
        page_lines=page_lines,
        page_width=page_width,
        page_height=page_height,
    )
    resolved_rule = rule_scores.get(resolved_type, 0.0)
    resolved_model = model_scores.get(resolved_type, 0.0)
    final_confidence = (
        HYBRID_MODEL_WEIGHT * resolved_model
        + HYBRID_RULE_WEIGHT * resolved_rule
        + HYBRID_GEO_WEIGHT * geometric_score
    )
    return {
        "block_type": resolved_type,
        "confidence": {
            "rule_score": resolved_rule,
            "model_score": resolved_model,
            "geometric_score": geometric_score,
            "final_confidence": min(1.0, final_confidence),
        },
        "has_model_support": resolved_model > 0,
    }


def classify_image_block(
    image_bbox: BoundingBox,
    detections: List[Detection],
    page_width: float,
    page_height: float,
    variant_mode: str = "hybrid",
) -> Dict[str, Any]:
    rule_score = 1.0
    model_score = compute_model_score(image_bbox, detections, "figure")
    geometric_score = 1.0
    area_ratio = image_bbox.area() / (page_width * page_height)
    if area_ratio < 0.001 or area_ratio > 0.8:
        geometric_score = 0.7

    if variant_mode == "geometry":
        final_conf = 0.6 * rule_score + 0.4 * geometric_score
        return {
            "block_type": "figure",
            "confidence": {
                "rule_score": rule_score,
                "model_score": 0.0,
                "geometric_score": geometric_score,
                "final_confidence": final_conf,
            },
            "has_model_support": False,
        }

    if variant_mode == "model":
        final_conf = 0.9 * model_score + 0.1 * geometric_score if model_score > 0 else 0.0
        return {
            "block_type": "figure",
            "confidence": {
                "rule_score": 0.0,
                "model_score": model_score,
                "geometric_score": geometric_score,
                "final_confidence": min(1.0, final_conf),
            },
            "has_model_support": model_score > 0,
        }

    if model_score == 0.0:
        final_conf = 0.6 * rule_score + 0.4 * geometric_score
    else:
        final_conf = HYBRID_MODEL_WEIGHT * model_score + HYBRID_RULE_WEIGHT * rule_score + HYBRID_GEO_WEIGHT * geometric_score
    return {
        "block_type": "figure",
        "confidence": {
            "rule_score": rule_score,
            "model_score": model_score,
            "geometric_score": geometric_score,
            "final_confidence": min(1.0, final_conf),
        },
        "has_model_support": model_score > 0,
    }


def calculate_page_avg_font_size(blocks: List[LayoutBlock]) -> float:
    if not blocks:
        return 12.0
    return sum(block.avg_font_size for block in blocks) / len(blocks)


def _apply_table_variant_override(
    block_data: Dict[str, Any],
    variant_mode: str,
    table_candidates: List[Dict[str, Any]],
    table_match_threshold: float,
) -> None:
    layout_block: LayoutBlock | None = block_data.get("layout_block")
    if not layout_block or not layout_block.bbox:
        return

    table_score, table_candidate = block_table_match(layout_block.bbox, table_candidates, table_match_threshold)
    if table_candidate is None:
        return

    conf = block_data.get("confidence", {})
    rule_score = float(conf.get("rule_score", 0.0))
    model_score = float(conf.get("model_score", 0.0))
    geometric_score = float(conf.get("geometric_score", 0.0))

    if variant_mode == "geometry":
        rule_score = max(rule_score, float(table_candidate.get("score", 0.0)))
        model_score = 0.0
        final_conf = 0.6 * rule_score + 0.4 * geometric_score
    elif variant_mode == "model":
        rule_score = 0.0
        model_score = max(model_score, float(table_candidate.get("score", 0.0)))
        final_conf = 0.9 * model_score + 0.1 * geometric_score
        block_data["has_model_support"] = True
    else:
        candidate_score = float(table_candidate.get("score", 0.0))
        rule_score = max(rule_score, 0.8 * candidate_score)
        model_score = max(model_score, candidate_score)
        final_conf = HYBRID_MODEL_WEIGHT * model_score + HYBRID_RULE_WEIGHT * rule_score + HYBRID_GEO_WEIGHT * geometric_score
        block_data["has_model_support"] = model_score > 0

    block_data["block_type"] = "table"
    block_data["confidence"] = {
        "rule_score": max(0.0, min(1.0, rule_score)),
        "model_score": max(0.0, min(1.0, model_score)),
        "geometric_score": max(0.0, min(1.0, geometric_score)),
        "final_confidence": max(0.0, min(1.0, final_conf)),
    }
    block_data["table_source"] = table_candidate.get("source", variant_mode)
    block_data["table_evidence"] = table_candidate.get("evidence", {})
    block_data["table_match_score"] = round(float(table_score), 4)


def classify_blocks(
    layout_blocks: List[LayoutBlock],
    image_blocks: List[Dict[str, Any]],
    detections: List[Detection],
    page_width: float,
    page_height: float,
    page_lines: Optional[List[Dict[str, Any]]] = None,
    variant_mode: str = "hybrid",
    table_candidates: Optional[List[Dict[str, Any]]] = None,
    table_match_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    classified_blocks = []
    avg_font_size = calculate_page_avg_font_size(layout_blocks)

    for block in layout_blocks:
        classification = classify_block(
            block,
            detections,
            page_width,
            page_height,
            avg_font_size,
            page_lines,
            variant_mode=variant_mode,
        )
        # Apply class-wise threshold gating
        btype = classification["block_type"]
        conf = classification["confidence"]["final_confidence"]
        if conf < CLASS_CONFIDENCE_THRESHOLDS.get(btype, 0.0):
            # If threshold not met, downgrade to text or skip if it's already text
            if btype != "text":
                classification["block_type"] = "text"
                btype = "text"
                # Re-check text threshold
                if conf < CLASS_CONFIDENCE_THRESHOLDS.get("text", 0.0):
                    continue
            else:
                continue

        block_data = {
            "layout_block": block,
            "block_type": btype,
            "confidence": classification["confidence"],
            "has_model_support": classification.get("has_model_support", False),
        }
        if table_candidates:
            _apply_table_variant_override(block_data, variant_mode, table_candidates, table_match_threshold)
        
        # Final check after table override
        if block_data["confidence"]["final_confidence"] < CLASS_CONFIDENCE_THRESHOLDS.get(block_data["block_type"], 0.0):
            continue

        if variant_mode == "model" and not block_data.get("has_model_support", False):
            continue
        classified_blocks.append(block_data)

    for image in image_blocks:
        classification = classify_image_block(
            image["bbox"],
            detections,
            page_width,
            page_height,
            variant_mode=variant_mode,
        )
        
        # Apply class-wise threshold gating for images (figures)
        if classification["confidence"]["final_confidence"] < CLASS_CONFIDENCE_THRESHOLDS.get("figure", 0.0):
            continue

        if variant_mode == "model" and not classification.get("has_model_support", False):
            continue
        classified_blocks.append(
            {
                "image_block": image,
                "block_type": classification["block_type"],
                "confidence": classification["confidence"],
                "has_model_support": classification.get("has_model_support", False),
            }
        )

    logger.debug(f"Classified {len(classified_blocks)} blocks")
    return classified_blocks

