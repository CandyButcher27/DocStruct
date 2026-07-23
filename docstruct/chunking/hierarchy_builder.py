"""Assign section-hierarchy levels to header blocks.

Document-agnostic by design: levels come from signals present in the document
itself, never from hardcoded section names.

The primary signal is the relative ranking of header **font sizes** — the largest
distinct size is level 1, and sizes past :data:`docstruct.config.HEADER_LEVELS`
clamp to the deepest level.

Font size alone is not always enough. Plenty of documents set every heading at the
same size and mark depth purely by numbering ("3 Method", "3.2 Setup",
"3.2.1 Ablations"), or set a sub-subsection in the same size as a subsection with a
different weight. Where a heading carries an explicit section number, that number
*states* its depth — there is nothing left to infer — so numbering takes priority
over the font ranking. Both signals are deterministic and local to the document,
which keeps this inside the no-model contract.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from docstruct import config
from docstruct.schema import Block

# "3 Method", "3.2 Setup", "3.2.1 Ablation", and the same with a trailing dot.
# Anchored and requiring following text, so a bare page number or a list marker
# is not mistaken for a heading number.
_NUMBERING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S")


def _size_signal(block: Block) -> float:
    if block.font_size:
        return block.font_size
    return block.bbox.height


def numbering_depth(text: Optional[str]) -> Optional[int]:
    """Depth implied by a leading section number, or ``None`` if unnumbered."""
    match = _NUMBERING_RE.match(text or "")
    if not match:
        return None
    return min(match.group(1).count(".") + 1, config.HEADER_LEVELS)


def assign_header_levels(blocks: List[Block]) -> Dict[str, int]:
    """Map each header block id to a level in ``1..HEADER_LEVELS``."""
    headers = [b for b in blocks if b.label == "header"]
    if not headers:
        return {}

    distinct = sorted({round(_size_signal(b), 1) for b in headers}, reverse=True)
    size_to_level = {
        size: min(idx + 1, config.HEADER_LEVELS) for idx, size in enumerate(distinct)
    }

    levels: Dict[str, int] = {}
    for block in headers:
        numbered = numbering_depth(block.text) if config.HEADER_NUMBERING_LEVELS else None
        levels[block.block_id] = (
            numbered if numbered is not None
            else size_to_level[round(_size_signal(block), 1)]
        )
    return levels
