"""
Tests for block classification stage.
"""

import pytest
from pipeline.classification import (
    compute_rule_score,
    compute_model_score,
    compute_geometric_score,
    classify_block
)
from pipeline.layout import LayoutBlock
from pipeline.decomposition import TextSpan
from models.detector import Detection
from schemas.block import BoundingBox


def create_test_block(text: str, font_size: float, y_position: float) -> LayoutBlock:
    """Helper to create test block."""
    block = LayoutBlock(page_num=0)
    bbox = BoundingBox(x0=10, y0=y_position, x1=200, y1=y_position + 20)
    span = TextSpan(text, bbox, "Arial", font_size, 0)
    block.add_span(span)
    return block


def test_compute_rule_score_header():
    """Test rule score for header detection."""
    # Large font, short text, near top
    block = create_test_block("Introduction", font_size=18.0, y_position=50)
    
    score = compute_rule_score(block, "header", avg_font_size=12.0)
    
    assert score > 0.5  # Should be classified as header


def test_compute_rule_score_text():
    """Test rule score for text detection."""
    # Normal font, longer text
    block = create_test_block(
        "This is a paragraph with many words that form a coherent text block.",
        font_size=12.0,
        y_position=200
    )
    
    score = compute_rule_score(block, "text", avg_font_size=12.0)
    
    assert score > 0.5  # Should be classified as text


def test_compute_rule_score_caption():
    """Test rule score for caption detection."""
    # Small font, starts with "Figure"
    block = create_test_block("Figure 1: Example caption", font_size=10.0, y_position=400)
    
    score = compute_rule_score(block, "caption", avg_font_size=12.0)
    
    assert score > 0.5  # Should be classified as caption


def test_compute_model_score_no_detections():
    """Test model score with no detections."""
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    
    score = compute_model_score(bbox, [], "table")
    
    assert score == 0.0


def test_compute_model_score_with_matching_detection():
    """Test model score with matching detection."""
    block_bbox = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    
    # Create overlapping detection
    det_bbox = BoundingBox(x0=15, y0=25, x1=95, y1=75)
    detection = Detection(
        bbox=det_bbox,
        detection_type="table",
        confidence=0.9
    )
    
    score = compute_model_score(block_bbox, [detection], "table")
    
    assert score > 0.0  # Should have non-zero score


def test_compute_geometric_score_normal():
    """Test geometric score for normal block."""
    block = create_test_block("Normal text", font_size=12.0, y_position=200)
    
    score = compute_geometric_score(block, page_width=612, page_height=792)
    
    assert score > 0.9  # Should be high for normal block


def test_compute_geometric_score_out_of_bounds():
    """Test geometric score for out-of-bounds block."""
    block = LayoutBlock(page_num=0)
    # Create bbox extending beyond page
    bbox = BoundingBox(x0=10, y0=20, x1=1000, y1=40)
    span = TextSpan("Test", bbox, "Arial", 12.0, 0)
    block.add_span(span)
    
    score = compute_geometric_score(block, page_width=612, page_height=792)
    
    assert score < 0.5  # Should be penalized


def test_classify_block():
    """Test full block classification."""
    block = create_test_block("Introduction", font_size=18.0, y_position=50)
    
    result = classify_block(
        block,
        detections=[],
        page_width=612,
        page_height=792,
        avg_font_size=12.0
    )
    
    assert "block_type" in result
    assert "confidence" in result
    assert result["confidence"]["final_confidence"] >= 0.0
    assert result["confidence"]["final_confidence"] <= 1.0


def test_confidence_formula():
    """Test that confidence formula is correctly applied."""
    block = create_test_block("Test text", font_size=12.0, y_position=200)
    
    result = classify_block(
        block,
        detections=[],
        page_width=612,
        page_height=792,
        avg_font_size=12.0
    )
    
    conf = result["confidence"]
    
    # Check formula: 30% rule + 50% model + 20% geometric
    expected = 0.3 * conf["rule_score"] + 0.5 * conf["model_score"] + 0.2 * conf["geometric_score"]
    
    assert abs(conf["final_confidence"] - expected) < 0.01