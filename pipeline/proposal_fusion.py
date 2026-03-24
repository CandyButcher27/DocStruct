"""
Proposal fusion module for true hybrid pipeline.

Matches model proposals to geometry proposals and produces fused results
with source-appropriate confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

from schemas.block import BoundingBox
from utils.geometry import bbox_overlap, merge_bboxes

from pipeline.hybrid_proposals import RegionProposal


# Source priority for NMS deduplication
SOURCE_PRIORITY = {
    "confirmed": 3,
    "model_only": 2,
    "geometry_only": 1,
}

CONFIDENCE_BOUNDS = {
    "confirmed": {"floor": 0.70, "ceiling": 1.00},
    "model_only": {"floor": 0.40, "ceiling": 0.85},
    "geometry_only": {"floor": 0.25, "ceiling": 0.65},
}

# Source-specific weights for hybrid classification
HYBRID_SOURCE_WEIGHTS = {
    "confirmed": {"model": 0.50, "rule": 0.30, "geo": 0.20},
    "model_only": {"model": 0.75, "rule": 0.15, "geo": 0.10},
    "geometry_only": {"model": 0.00, "rule": 0.55, "geo": 0.45},
}


@dataclass
class FusedProposal:
    """A proposal with evidence from both model and geometry (or just one)."""

    bbox: BoundingBox
    source: Literal["confirmed", "model_only", "geometry_only"]
    block_type: str
    model_confidence: float
    geometry_confidence: float
    agreement_score: float  # How well model and geometry agree
    final_confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


def _create_confirmed_proposal(
    model_prop: RegionProposal,
    geo_prop: RegionProposal,
    iou: float,
) -> FusedProposal:
    """
    Create a CONFIRMED proposal where model and geometry agree.

    Confidence Formula (HIGHEST TIER):
    - Base: 0.85
    - Model boost: +0.10 * model_confidence
    - Agreement boost: +0.05 * iou
    - Final range: 0.85 - 1.00
    """
    # Decide bbox: use model bbox if high confidence, else merge
    if model_prop.confidence >= 0.8 and iou >= 0.6:
        final_bbox = model_prop.bbox
    else:
        final_bbox = merge_bboxes([model_prop.bbox, geo_prop.bbox])

    # Type resolution: model type wins, geometry confirms
    block_type = model_prop.proposed_type or geo_prop.proposed_type or "text"

    # Agreement score: combination of IoU and type agreement
    type_match = model_prop.proposed_type == geo_prop.proposed_type
    type_agreement = 1.0 if type_match else 0.8
    agreement_score = (iou + type_agreement) / 2.0

    # Final confidence: HIGH because both sources agree
    base_confidence = 0.85
    model_boost = 0.10 * model_prop.confidence
    agreement_boost = 0.05 * iou
    final_confidence = min(1.0, base_confidence + model_boost + agreement_boost)

    return FusedProposal(
        bbox=final_bbox,
        source="confirmed",
        block_type=block_type,
        model_confidence=model_prop.confidence,
        geometry_confidence=geo_prop.confidence,
        agreement_score=round(agreement_score, 4),
        final_confidence=round(final_confidence, 4),
        evidence={
            "model_type": model_prop.proposed_type,
            "geometry_type": geo_prop.proposed_type,
            "type_match": type_match,
            "iou": round(iou, 4),
            "model_evidence": model_prop.evidence,
            "geometry_evidence": geo_prop.evidence,
        },
    )


def _create_model_only_proposal(model_prop: RegionProposal) -> FusedProposal:
    """
    Create a MODEL-ONLY proposal where geometry did not confirm.

    Confidence Formula (MEDIUM TIER):
    - Base: model_confidence * 0.75
    - Range: 0.40 - 0.85 (bounded)
    """
    base_confidence = model_prop.confidence * 0.75
    final_confidence = max(
        CONFIDENCE_BOUNDS["model_only"]["floor"],
        min(CONFIDENCE_BOUNDS["model_only"]["ceiling"], base_confidence),
    )

    return FusedProposal(
        bbox=model_prop.bbox,
        source="model_only",
        block_type=model_prop.proposed_type or "text",
        model_confidence=model_prop.confidence,
        geometry_confidence=0.0,
        agreement_score=0.0,
        final_confidence=round(final_confidence, 4),
        evidence={
            "model_type": model_prop.proposed_type,
            "model_evidence": model_prop.evidence,
            "note": "No geometry confirmation found",
        },
    )


def _create_geometry_only_proposal(geo_prop: RegionProposal) -> FusedProposal:
    """
    Create a GEOMETRY-ONLY proposal where model missed this region.

    Confidence Formula (LOWER TIER):
    - Base: geometry_confidence * 0.60
    - Range: 0.25 - 0.65 (bounded)
    """
    base_confidence = geo_prop.confidence * 0.60
    final_confidence = max(
        CONFIDENCE_BOUNDS["geometry_only"]["floor"],
        min(CONFIDENCE_BOUNDS["geometry_only"]["ceiling"], base_confidence),
    )

    return FusedProposal(
        bbox=geo_prop.bbox,
        source="geometry_only",
        block_type=geo_prop.proposed_type or "text",
        model_confidence=0.0,
        geometry_confidence=geo_prop.confidence,
        agreement_score=0.0,
        final_confidence=round(final_confidence, 4),
        evidence={
            "geometry_type": geo_prop.proposed_type,
            "geometry_evidence": geo_prop.evidence,
            "note": "Model did not detect this region",
        },
    )


def match_and_fuse_proposals(
    model_proposals: List[RegionProposal],
    geometry_proposals: List[RegionProposal],
    iou_threshold: float = 0.35,
) -> Tuple[List[FusedProposal], Dict[str, int]]:
    """
    Match model detections to geometry proposals and produce fused results.

    Algorithm:
    1. For each model detection, find best geometry match (IoU >= threshold)
    2. Matched pairs -> "confirmed" (highest confidence tier)
    3. Unmatched model -> "model_only" (medium tier)
    4. Unmatched geometry -> "geometry_only" (lower tier)

    Args:
        model_proposals: Proposals from model detection
        geometry_proposals: Proposals from geometry analysis
        iou_threshold: Minimum IoU to consider a match

    Returns:
        fused_proposals: List of FusedProposal with source attribution
        diagnostics: Counts of confirmed/model_only/geometry_only
    """
    fused = []
    used_geo_indices = set()
    diagnostics = {"confirmed": 0, "model_only": 0, "geometry_only": 0}

    # Step 1: For each model detection, find best geometry match
    for model_prop in model_proposals:
        best_match_idx = -1
        best_iou = 0.0

        for idx, geo_prop in enumerate(geometry_proposals):
            if idx in used_geo_indices:
                continue

            iou = bbox_overlap(model_prop.bbox, geo_prop.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match_idx = idx

        if best_match_idx >= 0:
            # CONFIRMED: Model + Geometry agree
            geo_prop = geometry_proposals[best_match_idx]
            used_geo_indices.add(best_match_idx)

            fused.append(_create_confirmed_proposal(model_prop, geo_prop, best_iou))
            diagnostics["confirmed"] += 1
        else:
            # MODEL-ONLY: No geometry support
            fused.append(_create_model_only_proposal(model_prop))
            diagnostics["model_only"] += 1

    # Step 2: Handle unmatched geometry proposals
    for idx, geo_prop in enumerate(geometry_proposals):
        if idx not in used_geo_indices:
            fused.append(_create_geometry_only_proposal(geo_prop))
            diagnostics["geometry_only"] += 1

    return fused, diagnostics


def deduplicate_proposals(
    proposals: List[FusedProposal],
    iou_threshold: float = 0.5,
) -> List[FusedProposal]:
    """
    Apply NMS to remove overlapping proposals.

    Priority order (higher wins):
    1. confirmed (priority 3)
    2. model_only (priority 2)
    3. geometry_only (priority 1)

    Within same priority, higher confidence wins.

    Args:
        proposals: List of FusedProposal to deduplicate
        iou_threshold: IoU threshold for suppression

    Returns:
        Deduplicated list of proposals
    """
    if len(proposals) <= 1:
        return proposals

    # Sort by (source_priority, final_confidence) descending
    sorted_proposals = sorted(
        proposals,
        key=lambda p: (SOURCE_PRIORITY[p.source], p.final_confidence),
        reverse=True,
    )

    kept = []
    for proposal in sorted_proposals:
        # Check overlap with already-kept proposals
        overlaps = any(
            bbox_overlap(proposal.bbox, kept_p.bbox) > iou_threshold for kept_p in kept
        )
        if not overlaps:
            kept.append(proposal)

    return kept


def get_fusion_summary(diagnostics: Dict[str, int]) -> str:
    """
    Format fusion diagnostics as a summary string.

    Args:
        diagnostics: Dict with confirmed/model_only/geometry_only counts

    Returns:
        Formatted summary string
    """
    total = sum(diagnostics.values())
    return (
        f"Total: {total} | "
        f"Confirmed: {diagnostics.get('confirmed', 0)} | "
        f"Model-only: {diagnostics.get('model_only', 0)} | "
        f"Geo-only: {diagnostics.get('geometry_only', 0)}"
    )
