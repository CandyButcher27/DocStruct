"""Model-assisted annotation: export a draft review session from the pipeline.

Runs the pipeline (hybrid if weights are given) and writes a single self-contained
JSON session: each page rasterized to a base64 PNG plus the predicted blocks as
draft ground-truth boxes. ``tools/annotate.html`` loads this file, lets you
correct boxes/labels, and exports the final ``data/annotations/<doc>.json`` in
the format :func:`docstruct.eval.runner.load_ground_truth` expects.

Boxes stay in top-left PDF points throughout (no pixel<->point conversion), which
removes the most common silent evaluation bug.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Tuple

from docstruct.pipeline import run_pipeline


def _block_to_box(block) -> dict:
    b = block.bbox
    return {
        "label": block.label,
        "page_num": block.page_num,
        "bbox": [b.x0, b.y0, b.x1, b.y1, b.page_width, b.page_height],
    }


def export_annotation_session(
    pdf_path: str,
    out_path: str,
    weights: str | None = None,
    dpi: int = 110,
) -> Tuple[str, int, int]:
    """Write a review session JSON. Returns (out_path, n_boxes, n_pages)."""
    import fitz

    result = run_pipeline(pdf_path, weights=weights)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            uri = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
            pages.append(
                {"width_pt": page.rect.width, "height_pt": page.rect.height, "image": uri}
            )

    session = {
        "doc": os.path.basename(pdf_path),
        "dpi": dpi,
        "mode": result.diagnostics["mode"],
        "pages": pages,
        "boxes": [_block_to_box(b) for b in result.blocks],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(session, fh, ensure_ascii=False)
    return out_path, len(session["boxes"]), len(pages)
