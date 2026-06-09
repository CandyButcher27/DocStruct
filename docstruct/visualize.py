"""Render fused blocks as colored overlays on the source PDF (PyMuPDF).

Useful for qualitative inspection and paper figures. Boxes are colored by label
and annotated with ``label:source``. Coordinates are top-left points, matching
PyMuPDF's native space, so no transform is needed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from docstruct.schema import Block

_LABEL_COLORS: Dict[str, Tuple[float, float, float]] = {
    "text": (0.13, 0.47, 0.71),
    "header": (0.84, 0.15, 0.16),
    "table": (0.17, 0.63, 0.17),
    "figure": (0.58, 0.40, 0.74),
    "caption": (1.00, 0.50, 0.05),
}
_DEFAULT_COLOR = (0.4, 0.4, 0.4)


def render_annotated(pdf_path: str, blocks: List[Block], out_path: str) -> str:
    """Draw block boxes onto a copy of the PDF and save it to ``out_path``."""
    import fitz

    by_page: Dict[int, List[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page_num, []).append(block)

    doc = fitz.open(pdf_path)
    try:
        for page_num, page_blocks in by_page.items():
            if page_num >= len(doc):
                continue
            page = doc[page_num]
            for block in page_blocks:
                color = _LABEL_COLORS.get(block.label, _DEFAULT_COLOR)
                rect = fitz.Rect(block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
                page.draw_rect(rect, color=color, width=1.2)
                page.insert_text(
                    (block.bbox.x0 + 1, max(block.bbox.y0 - 2, 4)),
                    f"{block.label}:{block.source.value}",
                    fontsize=6,
                    color=color,
                )
        doc.save(out_path)
    finally:
        doc.close()
    return out_path
