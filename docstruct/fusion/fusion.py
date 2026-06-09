"""Assemble fused :class:`Block` objects from matcher + arbiter output."""

from __future__ import annotations

from typing import List

from docstruct import config
from docstruct.schema import Block, BoundingBox, ConfidenceBreakdown, Proposal, Source
from docstruct.utils.geometry import merge_bboxes
from docstruct.fusion.arbiter import ArbitratedItem, arbitrate
from docstruct.fusion.matcher import MatchResult


def _clamp(value: float, source_key: str) -> float:
    bounds = config.CONFIDENCE_BOUNDS[source_key]
    return max(bounds["floor"], min(bounds["ceiling"], value))


def _matched_bbox(item: ArbitratedItem) -> BoundingBox:
    if (
        item.model_score >= config.CONFIRMED_BBOX_MODEL_CONF
        and item.iou >= config.CONFIRMED_BBOX_IOU
    ):
        return item.model.bbox
    return merge_bboxes([item.model.bbox, item.geometry.bbox])


def _block_from_item(item: ArbitratedItem, block_id: str) -> Block:
    return Block(
        bbox=_matched_bbox(item),
        label=item.label,
        confidence=ConfidenceBreakdown(
            geometry_score=item.geometry_score,
            model_score=item.model_score,
            final=item.final,
        ),
        source=item.source,
        page_num=item.model.page_num,
        block_id=block_id,
    )


def _block_from_model(prop: Proposal, block_id: str) -> Block:
    final = round(
        _clamp(prop.confidence * config.UNILATERAL_MODEL_SCALE, "unilateral_model"), 4
    )
    return Block(
        bbox=prop.bbox,
        label=prop.label,
        confidence=ConfidenceBreakdown(
            geometry_score=0.0, model_score=prop.confidence, final=final
        ),
        source=Source.UNILATERAL_MODEL,
        page_num=prop.page_num,
        block_id=block_id,
    )


def _block_from_geometry(prop: Proposal, block_id: str) -> Block:
    final = round(
        _clamp(prop.confidence * config.UNILATERAL_GEOMETRY_SCALE, "unilateral_geometry"),
        4,
    )
    return Block(
        bbox=prop.bbox,
        label=prop.label,
        confidence=ConfidenceBreakdown(
            geometry_score=prop.confidence, model_score=0.0, final=final
        ),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=prop.page_num,
        block_id=block_id,
    )


def fuse(match_result: MatchResult) -> List[Block]:
    """Produce reading-order-naive blocks (ids assigned, ``reading_order`` unset)."""
    items = arbitrate(match_result.matched)

    blocks: List[Block] = []
    counter = 0

    for item in items:
        blocks.append(_block_from_item(item, f"block_{counter:04d}"))
        counter += 1
    for prop in match_result.unmatched_model:
        blocks.append(_block_from_model(prop, f"block_{counter:04d}"))
        counter += 1
    for prop in match_result.unmatched_geometry:
        blocks.append(_block_from_geometry(prop, f"block_{counter:04d}"))
        counter += 1

    return blocks
