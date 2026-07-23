"""Recursive XY-cut reading order.

The legacy ordering splits a page into at most two columns by the largest gap
between block *centres*. That mishandles any page whose layout is not uniformly
one- or two-column: a full-width title, abstract, table or figure sitting across a
two-column body has its centre near the page middle, so it is assigned to whichever
column wins the comparison and is then read in the wrong place.

Recursive XY-cut instead splits a region at genuine whitespace bands and recurses,
so full-width elements separate the flow above them from the flow below them, and
columns are only found inside the regions that actually have them.

Cut order matters. A **vertical** (column) cut is tried first: on a two-column
region the horizontal paragraph gaps are real whitespace bands, and cutting on one
of those would interleave the two columns. A full-width block spans the region, so
it leaves no vertical gutter at all — the vertical cut correctly fails there and
the horizontal cut separates it from the body, which then splits into columns one
level down.

Deterministic: same blocks in, same order out. No model, no heuristic beyond
two whitespace thresholds in ``config``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from docstruct import config
from docstruct.schema import Block

# Depth bound purely to cap pathological recursion on adversarial layouts; real
# documents settle in a handful of levels.
_MAX_DEPTH = 24


def _gaps(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Maximal empty spans between a set of 1-D intervals, in order."""
    ordered = sorted(intervals)
    out: List[Tuple[float, float]] = []
    reach = ordered[0][1]
    for lo, hi in ordered[1:]:
        if lo > reach:
            out.append((reach, lo))
        reach = max(reach, hi)
    return out


def _widest(gaps: Sequence[Tuple[float, float]]):
    return max(gaps, key=lambda g: g[1] - g[0]) if gaps else None


def _cut(
    blocks: Sequence[Block],
    indices: List[int],
    min_column_gap: float,
    min_row_gap: float,
    depth: int,
) -> List[int]:
    if len(indices) <= 1 or depth >= _MAX_DEPTH:
        return sorted(indices, key=lambda i: (blocks[i].bbox.y0, blocks[i].bbox.x0))

    # Vertical cut (columns) first — see module docstring.
    x_gap = _widest(_gaps([(blocks[i].bbox.x0, blocks[i].bbox.x1) for i in indices]))
    if x_gap is not None and (x_gap[1] - x_gap[0]) >= min_column_gap:
        mid = (x_gap[0] + x_gap[1]) / 2.0
        left = [i for i in indices if blocks[i].bbox.x1 <= mid]
        right = [i for i in indices if blocks[i].bbox.x1 > mid]
        if left and right:
            return (
                _cut(blocks, left, min_column_gap, min_row_gap, depth + 1)
                + _cut(blocks, right, min_column_gap, min_row_gap, depth + 1)
            )

    # Horizontal cut (stacked bands).
    y_gap = _widest(_gaps([(blocks[i].bbox.y0, blocks[i].bbox.y1) for i in indices]))
    if y_gap is not None and (y_gap[1] - y_gap[0]) >= min_row_gap:
        mid = (y_gap[0] + y_gap[1]) / 2.0
        top = [i for i in indices if blocks[i].bbox.y1 <= mid]
        bottom = [i for i in indices if blocks[i].bbox.y1 > mid]
        if top and bottom:
            return (
                _cut(blocks, top, min_column_gap, min_row_gap, depth + 1)
                + _cut(blocks, bottom, min_column_gap, min_row_gap, depth + 1)
            )

    return sorted(indices, key=lambda i: (blocks[i].bbox.y0, blocks[i].bbox.x0))


def xy_cut_order(blocks: Sequence[Block], page_width: float) -> List[int]:
    """Block indices in reading order, by recursive XY-cut."""
    if not blocks:
        return []
    width = page_width or max((b.bbox.page_width for b in blocks), default=0.0)
    min_column_gap = width * config.XY_CUT_MIN_COLUMN_GAP_RATIO
    return _cut(
        blocks,
        list(range(len(blocks))),
        min_column_gap,
        config.XY_CUT_MIN_ROW_GAP,
        depth=0,
    )
