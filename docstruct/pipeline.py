"""End-to-end orchestration: PDF -> fused blocks -> chunks.

Detectors run independently and are reconciled per page (proposals on different
pages never match). Reading order is assigned per page then offset into a single
document-global sequence. The model detector is optional; without it the
pipeline runs geometry-only and every block is ``unilateral_geometry``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from docstruct.schema import Block, Chunk
from docstruct.geometry import detector as geometry_detector
from docstruct.fusion.matcher import match_proposals
from docstruct.fusion.fusion import fuse
from docstruct.reading_order import assign_reading_order
from docstruct.extraction.text_extractor import populate_text
from docstruct.extraction.table_extractor import populate_tables
from docstruct.chunking.hierarchy_builder import assign_header_levels
from docstruct.chunking.assembler import build_chunks

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    blocks: List[Block]
    chunks: List[Chunk]
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _group_by_page(proposals) -> Dict[int, list]:
    grouped: Dict[int, list] = defaultdict(list)
    for prop in proposals:
        grouped[prop.page_num].append(prop)
    return grouped


def run_pipeline(
    pdf_path: str,
    *,
    model_detector=None,
    weights: Optional[str] = None,
) -> PipelineResult:
    """Run the full pipeline on a single PDF."""
    geometry_props = geometry_detector.detect(pdf_path)

    model_props: list = []
    mode = "geometry-only"
    detector = model_detector
    if detector is None and weights:
        from docstruct.model.detector import ModelDetector

        detector = ModelDetector(weights)
    if detector is not None:
        model_props = detector.detect(pdf_path)
        mode = "hybrid"

    geo_by_page = _group_by_page(geometry_props)
    model_by_page = _group_by_page(model_props)
    pages = sorted(set(geo_by_page) | set(model_by_page))

    all_blocks: List[Block] = []
    totals = {"matched": 0, "unmatched_model": 0, "unmatched_geometry": 0}
    id_counter = 0
    ro_offset = 0

    for page_num in pages:
        result = match_proposals(model_by_page.get(page_num, []), geo_by_page.get(page_num, []))
        for key in totals:
            totals[key] += result.diagnostics.get(key, 0)

        blocks = fuse(result)
        for block in blocks:
            block.block_id = f"block_{id_counter:04d}"
            id_counter += 1

        page_width = blocks[0].bbox.page_width if blocks else 0.0
        assign_reading_order(blocks, page_width)
        for block in blocks:
            block.reading_order += ro_offset
        ro_offset += len(blocks)
        all_blocks.extend(blocks)

    populate_text(pdf_path, all_blocks)
    populate_tables(pdf_path, all_blocks)

    levels = assign_header_levels(all_blocks)
    chunks = build_chunks(all_blocks, levels)

    diagnostics = {
        "mode": mode,
        "pages": len(pages),
        "n_blocks": len(all_blocks),
        "n_chunks": len(chunks),
        **totals,
    }
    logger.info("pipeline complete: %s", diagnostics)
    return PipelineResult(blocks=all_blocks, chunks=chunks, diagnostics=diagnostics)
