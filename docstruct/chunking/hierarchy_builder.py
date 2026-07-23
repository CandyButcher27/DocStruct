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

# "3 Method", "3.2 Setup", "3.2.1 Ablation"; appendix "A.", "A.1", "B.2.1"; and
# Roman "IV.", "IX.1" (Roman numerals are just capital letters here). Anchored and
# requiring following text, so a bare page number or list marker is not mistaken for
# a heading number. The leading token is digits *or* capitals; a capital-letter token
# must carry a dot ("A." / "A.1") to be accepted, so a sentence opening "A survey
# of..." or an all-caps word like "ABSTRACT" is not read as a numbered heading.
_NUMBERING_RE = re.compile(r"^\s*(?P<num>\d+|[A-Z]+)(?P<sub>(?:\.\d+)*)\.?\s+\S")


def _size_signal(block: Block) -> float:
    if block.font_size:
        return block.font_size
    return block.bbox.height


def numbering_depth(text: Optional[str]) -> Optional[int]:
    """Depth implied by a leading section number, or ``None`` if unnumbered."""
    match = _NUMBERING_RE.match(text or "")
    if not match:
        return None
    num, sub = match.group("num"), match.group("sub")
    # A capital-letter token with no ".N" sub-part must be a real appendix marker
    # ("A."), not a sentence-opening "A". The regex already required a dot; guard the
    # bare-letter-with-trailing-dot-but-no-subpart case here (that is depth 1, fine),
    # but reject a lone capital token that slipped through with neither.
    if num.isalpha() and not sub and not text.lstrip()[len(num):].startswith("."):
        return None
    depth = sub.count(".") + 1
    return min(depth, config.HEADER_LEVELS)


def assign_header_levels(blocks: List[Block]) -> Dict[str, int]:
    """Map each header block id to a level in ``1..HEADER_LEVELS``."""
    headers = [b for b in blocks if b.label == "header"]
    if not headers:
        return {}

    def _rank_key(block: Block):
        size = round(_size_signal(block), 1)
        if config.HEADER_RANK_BY_WEIGHT:
            # Bold above regular at equal size: sort descending, so True must rank
            # ahead of False -> key on the size then the weight, both reversed.
            return (size, 1 if block.is_bold else 0)
        return (size,)

    distinct = sorted({_rank_key(b) for b in headers}, reverse=True)
    key_to_level = {
        key: min(idx + 1, config.HEADER_LEVELS) for idx, key in enumerate(distinct)
    }

    levels: Dict[str, int] = {}
    for block in headers:
        numbered = numbering_depth(block.text) if config.HEADER_NUMBERING_LEVELS else None
        levels[block.block_id] = (
            numbered if numbered is not None
            else key_to_level[_rank_key(block)]
        )
    return levels
