"""Greedy IoU matching of model vs geometry proposals, with priority NMS.

Emits matched pairs plus the leftover unmatched proposals from each detector.
NMS lives here: overlapping candidates are suppressed by priority
(matched > model-only > geometry-only), highest confidence winning ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from docstruct import config
from docstruct.schema import Proposal
from docstruct.utils.geometry import bbox_overlap

_PRIORITY_MATCHED = 3
_PRIORITY_MODEL = 2
_PRIORITY_GEOMETRY = 1


@dataclass
class MatchedPair:
    model: Proposal
    geometry: Proposal
    iou: float


@dataclass
class MatchResult:
    matched: List[MatchedPair]
    unmatched_model: List[Proposal]
    unmatched_geometry: List[Proposal]
    diagnostics: Dict[str, int] = field(default_factory=dict)


def _greedy_match(
    model_proposals: List[Proposal],
    geometry_proposals: List[Proposal],
    iou_threshold: float,
) -> Tuple[List[MatchedPair], List[Proposal], List[Proposal]]:
    matched: List[MatchedPair] = []
    unmatched_model: List[Proposal] = []
    used_geo: set[int] = set()

    for model_prop in model_proposals:
        best_idx = -1
        best_iou = 0.0
        for idx, geo_prop in enumerate(geometry_proposals):
            if idx in used_geo:
                continue
            iou = bbox_overlap(model_prop.bbox, geo_prop.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0:
            used_geo.add(best_idx)
            matched.append(
                MatchedPair(model_prop, geometry_proposals[best_idx], best_iou)
            )
        else:
            unmatched_model.append(model_prop)

    unmatched_geometry = [
        geo for idx, geo in enumerate(geometry_proposals) if idx not in used_geo
    ]
    return matched, unmatched_model, unmatched_geometry


def _nms(
    matched: List[MatchedPair],
    unmatched_model: List[Proposal],
    unmatched_geometry: List[Proposal],
    iou_threshold: float,
) -> Tuple[List[MatchedPair], List[Proposal], List[Proposal]]:
    candidates = []
    for pair in matched:
        candidates.append(
            (_PRIORITY_MATCHED, pair.model.confidence, pair.model.bbox, "matched", pair)
        )
    for prop in unmatched_model:
        candidates.append(
            (_PRIORITY_MODEL, prop.confidence, prop.bbox, "model", prop)
        )
    for prop in unmatched_geometry:
        candidates.append(
            (_PRIORITY_GEOMETRY, prop.confidence, prop.bbox, "geometry", prop)
        )

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    kept_boxes = []
    matched_out: List[MatchedPair] = []
    model_out: List[Proposal] = []
    geometry_out: List[Proposal] = []

    for _, _, bbox, kind, obj in candidates:
        if any(bbox_overlap(bbox, kept) > iou_threshold for kept in kept_boxes):
            continue
        kept_boxes.append(bbox)
        if kind == "matched":
            matched_out.append(obj)
        elif kind == "model":
            model_out.append(obj)
        else:
            geometry_out.append(obj)

    return matched_out, model_out, geometry_out


def match_proposals(
    model_proposals: List[Proposal],
    geometry_proposals: List[Proposal],
    iou_threshold: float | None = None,
    nms_threshold: float | None = None,
) -> MatchResult:
    """Match the two proposal streams and deduplicate via priority NMS."""
    iou_threshold = (
        config.IOU_MATCH_THRESHOLD if iou_threshold is None else iou_threshold
    )
    nms_threshold = (
        config.NMS_IOU_THRESHOLD if nms_threshold is None else nms_threshold
    )

    matched, unmatched_model, unmatched_geometry = _greedy_match(
        model_proposals, geometry_proposals, iou_threshold
    )
    matched, unmatched_model, unmatched_geometry = _nms(
        matched, unmatched_model, unmatched_geometry, nms_threshold
    )

    diagnostics = {
        "matched": len(matched),
        "unmatched_model": len(unmatched_model),
        "unmatched_geometry": len(unmatched_geometry),
    }
    return MatchResult(matched, unmatched_model, unmatched_geometry, diagnostics)
