"""Label arbitration for matched proposal pairs.

Same label on both detectors => CONFIRMED (high-tier confidence formula).
Different label => DISPUTED: the higher-confidence detector wins the label and
its confidence is scaled by ``DISPUTED_MULTIPLIER`` (0.85).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from docstruct import config
from docstruct.schema import Label, Proposal, Source
from docstruct.fusion.matcher import MatchedPair


@dataclass
class ArbitratedItem:
    source: Source
    label: Label
    model: Proposal
    geometry: Proposal
    iou: float
    geometry_score: float
    model_score: float
    final: float


def _confirmed_confidence(model_conf: float, iou: float) -> float:
    raw = (
        config.CONFIRMED_BASE
        + config.CONFIRMED_MODEL_BOOST * model_conf
        + config.CONFIRMED_AGREEMENT_BOOST * iou
    )
    return min(1.0, raw)


def arbitrate(matched: List[MatchedPair]) -> List[ArbitratedItem]:
    """Classify each matched pair as confirmed or disputed."""
    items: List[ArbitratedItem] = []

    for pair in matched:
        model, geometry, iou = pair.model, pair.geometry, pair.iou

        if model.label == geometry.label:
            items.append(
                ArbitratedItem(
                    source=Source.CONFIRMED,
                    label=model.label,
                    model=model,
                    geometry=geometry,
                    iou=iou,
                    geometry_score=geometry.confidence,
                    model_score=model.confidence,
                    final=round(_confirmed_confidence(model.confidence, iou), 4),
                )
            )
        else:
            if model.confidence >= geometry.confidence:
                winner_label = model.label
                winner_conf = model.confidence
            else:
                winner_label = geometry.label
                winner_conf = geometry.confidence

            items.append(
                ArbitratedItem(
                    source=Source.DISPUTED,
                    label=winner_label,
                    model=model,
                    geometry=geometry,
                    iou=iou,
                    geometry_score=geometry.confidence,
                    model_score=model.confidence,
                    final=round(winner_conf * config.DISPUTED_MULTIPLIER, 4),
                )
            )

    return items
