"""Tests for geometry/model table candidate detection and fusion."""

from pipeline.classification import classify_blocks
from pipeline.decomposition import TextSpan
from pipeline.layout import LayoutBlock
from pipeline.table_candidates import (
    detect_geometry_table_candidates,
    fuse_table_candidates,
)
from schemas.block import BoundingBox


def _make_block(text: str, x0: float, y0: float, x1: float, y1: float) -> LayoutBlock:
    block = LayoutBlock(page_num=0)
    block.add_span(TextSpan(text, BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1), "Arial", 12.0, 0))
    return block


def test_geometry_table_candidate_detects_ruled_numeric_block():
    block = _make_block("1 2 3 4 5 6 7 8", 50, 200, 300, 300)
    lines = [
        {"x0": 55, "y0": 210, "x1": 295, "y1": 210, "orientation": "horizontal"},
        {"x0": 55, "y0": 240, "x1": 295, "y1": 240, "orientation": "horizontal"},
        {"x0": 55, "y0": 270, "x1": 295, "y1": 270, "orientation": "horizontal"},
        {"x0": 80, "y0": 205, "x1": 80, "y1": 295, "orientation": "vertical"},
    ]

    candidates, diag = detect_geometry_table_candidates([block], lines, score_threshold=0.2)

    assert diag["geometry_candidates"] >= 1
    assert len(candidates) >= 1


def test_geometry_table_candidate_rejects_plain_paragraph():
    block = _make_block(
        "This is a normal paragraph with prose and almost no numeric columns.",
        40,
        100,
        340,
        160,
    )
    candidates, _ = detect_geometry_table_candidates([block], [], score_threshold=0.45)
    assert len(candidates) == 0


def test_fuse_table_candidates_geometry_model_overlap():
    geometry_candidates = [
        {
            "bbox": BoundingBox(x0=50, y0=200, x1=300, y1=300),
            "score": 0.7,
            "source": "geometry",
            "evidence": {},
        }
    ]
    model_candidates = [
        {
            "bbox": BoundingBox(x0=60, y0=210, x1=295, y1=295),
            "score": 0.9,
            "source": "model",
            "evidence": {},
        }
    ]

    fused, diag = fuse_table_candidates(
        geometry_candidates,
        model_candidates,
        overlap_threshold=0.3,
        fusion_acceptance_threshold=0.4,
        model_confidence_threshold=0.7,
    )

    assert diag["fused_tables"] == 1
    assert fused[0]["source"] == "fused"


def test_classify_blocks_applies_geometry_table_override():
    block = _make_block("short text", 10, 100, 150, 150)
    table_candidates = [
        {
            "bbox": BoundingBox(x0=5, y0=95, x1=155, y1=155),
            "score": 0.85,
            "source": "geometry",
            "evidence": {"line_total": 4},
        }
    ]

    classified = classify_blocks(
        layout_blocks=[block],
        image_blocks=[],
        detections=[],
        page_width=612,
        page_height=792,
        page_lines=[],
        variant_mode="geometry",
        table_candidates=table_candidates,
        table_match_threshold=0.1,
    )

    assert classified[0]["block_type"] == "table"
    assert classified[0]["table_source"] == "geometry"
