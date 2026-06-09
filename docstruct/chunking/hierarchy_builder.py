"""Assign section-hierarchy levels to header blocks by font-size rank.

Document-agnostic: levels come from the relative ranking of header font sizes
within the document, never from hardcoded section names. The largest distinct
size is level 1; sizes past level :data:`docstruct.config.HEADER_LEVELS` clamp
to the deepest level.
"""

from __future__ import annotations

from typing import Dict, List

from docstruct import config
from docstruct.schema import Block


def _size_signal(block: Block) -> float:
    if block.font_size:
        return block.font_size
    return block.bbox.height


def assign_header_levels(blocks: List[Block]) -> Dict[str, int]:
    """Map each header block id to a level in ``1..HEADER_LEVELS``."""
    headers = [b for b in blocks if b.label == "header"]
    if not headers:
        return {}

    distinct = sorted({round(_size_signal(b), 1) for b in headers}, reverse=True)
    size_to_level = {
        size: min(idx + 1, config.HEADER_LEVELS) for idx, size in enumerate(distinct)
    }
    return {b.block_id: size_to_level[round(_size_signal(b), 1)] for b in headers}
