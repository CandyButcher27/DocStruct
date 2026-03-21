"""
Conflict resolver for the adaptive hybrid ensemble.

When geometry-based rules and the ML model assign different block types
with meaningful confidence in both, this module provides arbitration logic
using extra structural evidence (ruling lines, region size, text density).
"""

from typing import Dict, Any, List, Optional, Tuple

from schemas.block import BoundingBox
from utils.logging import setup_logger

logger = setup_logger(__name__)

# Minimum gap between scores to declare a "conflict worth resolving"
CONFLICT_THRESHOLD = 0.2
# Penalty applied to the winning score when there is a genuine conflict
CONFLICT_PENALTY = 0.10


def compute_adaptive_weights(
    rule_score: float,
    model_score: float,
) -> Tuple[float, float, float]:
    """
    Compute adaptive (rule, model, geometric) weights based on signal agreement.

    When both signals agree, we trust neither overwhelmingly and keep geometry.
    When one signal is clearly dominant, we defer to it.
    When both are uncertain, we lean more on geometry as a sanity check.

    Returns:
        Tuple (w_rule, w_model, w_geometric) that sums to 1.0.
    """
    disagreement = abs(rule_score - model_score)

    if model_score == 0.0:
        # No model running (stub) — rule-only formula
        return (0.60, 0.00, 0.40)
    elif disagreement < 0.1:
        # Strong agreement between rule and model
        return (0.35, 0.45, 0.20)
    elif model_score > 0.8:
        # Model very confident — defer to it
        return (0.10, 0.70, 0.20)
    elif rule_score > 0.8:
        # Rules very confident — defer to them
        return (0.60, 0.20, 0.20)
    else:
        # Both uncertain — lean more on geometry
        return (0.30, 0.40, 0.30)


class ConflictResolver:
    """
    Arbitrates classification conflicts between geometry rules and ML model.

    Used inside classify_block() when the rule-assigned type and the
    model-assigned type differ and both have non-trivial confidence.
    """

    def resolve(
        self,
        rule_type: str,
        model_type: str,
        rule_score: float,
        model_score: float,
        block_bbox: Optional[BoundingBox],
        page_lines: Optional[List[Dict[str, Any]]] = None,
        page_width: float = 612.0,
        page_height: float = 792.0,
    ) -> Tuple[str, float]:
        """
        Resolve a conflict between rule_type and model_type.

        Args:
            rule_type:    Block type assigned by geometry rules
            model_type:   Block type assigned by ML model
            rule_score:   Rule confidence (0-1)
            model_score:  Model confidence (0-1)
            block_bbox:   Spatial bounding box (may be None)
            page_lines:   Ruling lines from PageData.lines (for table evidence)
            page_width:   Page width in points
            page_height:  Page height in points

        Returns:
            Tuple of (resolved_block_type, resolved_confidence)
        """
        # No conflict
        if rule_type == model_type:
            return rule_type, max(rule_score, model_score)

        # Not a meaningful conflict — one side is basically zero
        if rule_score < CONFLICT_THRESHOLD and model_score > 0:
            return model_type, model_score
        if model_score < CONFLICT_THRESHOLD and rule_score > 0:
            return rule_type, rule_score

        logger.debug(
            f"Conflict: rule={rule_type}({rule_score:.2f}) "
            f"vs model={model_type}({model_score:.2f})"
        )

        # ── Structural evidence tiebreakers ───────────────────────────────

        # 1. Ruling lines → strong evidence for table
        if block_bbox and page_lines:
            table_lines = self._count_lines_in_bbox(block_bbox, page_lines)
            if table_lines >= 3:
                logger.debug(f"  → Tiebreak: {table_lines} ruling lines → 'table'")
                return "table", 0.80

        # 2. Very large region with no strong text → likely figure
        if block_bbox:
            area_ratio = block_bbox.area() / (page_width * page_height)
            if area_ratio > 0.1 and model_type == "figure":
                logger.debug(f"  → Tiebreak: large area ({area_ratio:.2f}) → 'figure'")
                return "figure", model_score * (1.0 - CONFLICT_PENALTY)

        # 3. Default: trust higher-confidence source, apply conflict penalty
        if model_score >= rule_score:
            return model_type, model_score * (1.0 - CONFLICT_PENALTY)
        else:
            return rule_type, rule_score * (1.0 - CONFLICT_PENALTY)

    def _count_lines_in_bbox(
        self,
        bbox: BoundingBox,
        lines: List[Dict[str, Any]],
    ) -> int:
        """Count ruling lines whose midpoint falls inside bbox."""
        count = 0
        for line in lines:
            mid_x = (line["x0"] + line["x1"]) / 2
            mid_y = (line["y0"] + line["y1"]) / 2
            if bbox.x0 <= mid_x <= bbox.x1 and bbox.y0 <= mid_y <= bbox.y1:
                count += 1
        return count
