"""Tests for block classification stage."""

from models.detector import Detection
from pipeline.classification import (
    _DEFAULT_WEIGHTS,
    classify_block,
    compute_geometric_score,
    compute_model_score,
    compute_rule_score,
)
from pipeline.decomposition import TextSpan
from pipeline.layout import LayoutBlock
from schemas.block import BoundingBox


def create_test_block(text: str, font_size: float, y_position: float) -> LayoutBlock:
    block = LayoutBlock(page_num=0)
    bbox = BoundingBox(x0=10, y0=y_position, x1=200, y1=y_position + 20)
    span = TextSpan(text, bbox, "Arial", font_size, 0)
    block.add_span(span)
    return block


def test_compute_rule_score_header():
    block = create_test_block("Introduction", font_size=18.0, y_position=700)
    score = compute_rule_score(block, "header", avg_font_size=12.0, page_height=792)
    assert score > 0.5


def test_compute_rule_score_text():
    block = create_test_block(
        "This is a paragraph with many words that form a coherent text block.",
        font_size=12.0,
        y_position=200,
    )
    score = compute_rule_score(block, "text", avg_font_size=12.0, page_height=792)
    assert score > 0.5


def test_compute_rule_score_caption():
    block = create_test_block("Figure 1: Example caption", font_size=10.0, y_position=400)
    score = compute_rule_score(block, "caption", avg_font_size=12.0, page_height=792)
    assert score > 0.5


def test_compute_model_score_no_detections():
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    score = compute_model_score(bbox, [], "table")
    assert score == 0.0


def test_compute_model_score_with_matching_detection():
    block_bbox = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    det_bbox = BoundingBox(x0=15, y0=25, x1=95, y1=75)
    detection = Detection(bbox=det_bbox, detection_type="table", confidence=0.9)
    score = compute_model_score(block_bbox, [detection], "table")
    assert score > 0.0


def test_compute_geometric_score_normal():
    block = create_test_block("Normal text", font_size=12.0, y_position=200)
    score = compute_geometric_score(block, page_width=612, page_height=792)
    assert score > 0.9


def test_compute_geometric_score_out_of_bounds():
    block = LayoutBlock(page_num=0)
    bbox = BoundingBox(x0=10, y0=20, x1=1000, y1=40)
    block.add_span(TextSpan("Test", bbox, "Arial", 12.0, 0))
    score = compute_geometric_score(block, page_width=612, page_height=792)
    assert score < 0.5


def test_classify_block_geometry_mode_prefers_rules():
    block = create_test_block("Introduction", font_size=18.0, y_position=700)
    result = classify_block(
        block,
        detections=[],
        page_width=612,
        page_height=792,
        avg_font_size=12.0,
        variant_mode="geometry",
    )
    assert result["block_type"] == "header"
    assert result["confidence"]["model_score"] == 0.0


def test_classify_block_model_mode_prefers_model_signal():
    block = create_test_block("Short text", font_size=12.0, y_position=300)
    detection = Detection(
        bbox=BoundingBox(x0=8, y0=298, x1=202, y1=325),
        detection_type="caption",
        confidence=0.95,
    )
    result = classify_block(
        block,
        detections=[detection],
        page_width=612,
        page_height=792,
        avg_font_size=12.0,
        variant_mode="model",
    )
    assert result["block_type"] == "caption"
    assert result["has_model_support"] is True


def test_hybrid_confidence_formula_uses_model_preference():
    block = create_test_block("Figure 1: caption", font_size=10.0, y_position=400)
    detection = Detection(
        bbox=BoundingBox(x0=8, y0=398, x1=202, y1=425),
        detection_type="caption",
        confidence=0.9,
    )
    result = classify_block(
        block,
        detections=[detection],
        page_width=612,
        page_height=792,
        avg_font_size=12.0,
        variant_mode="hybrid",
    )
    conf = result["confidence"]
    model_w, rule_w, geo_w = _DEFAULT_WEIGHTS
    expected = (
        model_w * conf["model_score"]
        + rule_w * conf["rule_score"]
        + geo_w * conf["geometric_score"]
    )
    assert abs(conf["final_confidence"] - expected) < 1e-6
