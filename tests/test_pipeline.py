"""
Integration tests for the complete pipeline.
"""

import pytest
import json
import tempfile
from pathlib import Path
from main import process_pdf
from main import _variant_output_path


@pytest.fixture
def temp_output():
    """Create temporary output file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


def test_determinism(temp_output):
    """
    Test that processing the same PDF twice produces identical output.
    
    This is a critical property of DocStruct - same input must yield same output.
    """
    # This test requires a real PDF file
    # For now, we'll test the determinism of the classification logic
    
    from pipeline.classification import compute_rule_score
    from pipeline.layout import LayoutBlock
    from pipeline.decomposition import TextSpan
    from schemas.block import BoundingBox
    
    # Create identical blocks
    def create_block():
        block = LayoutBlock(page_num=0)
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=40)
        span = TextSpan("Test text", bbox, "Arial", 12.0, 0)
        block.add_span(span)
        return block
    
    block1 = create_block()
    block2 = create_block()
    
    # Compute scores
    score1 = compute_rule_score(block1, "text", 12.0)
    score2 = compute_rule_score(block2, "text", 12.0)
    
    # Should be identical
    assert score1 == score2


def test_variant_output_path_suffixing():
    base = "results/output.json"
    assert _variant_output_path(base, "geometry").endswith("output_geometry.json")


def test_geometry_sanity():
    """Test that geometric operations maintain sanity."""
    from utils.geometry import bbox_overlap, merge_bboxes
    from schemas.block import BoundingBox
    
    bbox1 = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    bbox2 = BoundingBox(x0=50, y0=40, x1=150, y1=100)
    
    # Test overlap
    overlap = bbox_overlap(bbox1, bbox2)
    assert 0 <= overlap <= 1
    
    # Test merge
    merged = merge_bboxes([bbox1, bbox2])
    assert merged.x0 <= bbox1.x0
    assert merged.x1 >= bbox2.x1
    assert merged.y0 <= min(bbox1.y0, bbox2.y0)
    assert merged.y1 >= max(bbox1.y1, bbox2.y1)


def test_no_negative_bboxes():
    """Test that no negative bounding boxes are created."""
    from schemas.block import BoundingBox
    
    # Should raise validation error
    with pytest.raises(Exception):
        BoundingBox(x0=-10, y0=20, x1=100, y1=80)


def test_blocks_do_not_overlap_excessively():
    """Test that formed blocks don't have excessive overlap."""
    from pipeline.layout import form_layout_blocks
    from pipeline.decomposition import PageData, TextSpan
    from schemas.block import BoundingBox
    from utils.geometry import bbox_overlap
    
    page_data = PageData(page_num=0, width=612, height=792)
    
    # Add well-separated spans
    bbox1 = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    span1 = TextSpan("First block", bbox1, "Arial", 12.0, 0)
    page_data.add_span(span1)
    
    bbox2 = BoundingBox(x0=10, y0=100, x1=100, y1=120)
    span2 = TextSpan("Second block", bbox2, "Arial", 12.0, 0)
    page_data.add_span(span2)
    
    blocks = form_layout_blocks(page_data)
    
    # Check blocks don't overlap
    if len(blocks) >= 2 and blocks[0].bbox is not None and blocks[1].bbox is not None:
        overlap = bbox_overlap(blocks[0].bbox, blocks[1].bbox)
        assert overlap < 0.5  # Should have minimal overlap


def test_confidence_validation():
    """Test that confidence scores are properly validated."""
    from pipeline.confidence import validate_confidence_breakdown
    
    # Valid confidence
    valid_conf = {
        "rule_score": 0.6,
        "model_score": 0.8,
        "geometric_score": 0.9,
        "final_confidence": 0.76  # 0.3*0.6 + 0.5*0.8 + 0.2*0.9
    }
    
    assert validate_confidence_breakdown(valid_conf) is True
    
    # Invalid confidence (out of bounds)
    invalid_conf = {
        "rule_score": 0.6,
        "model_score": 0.8,
        "geometric_score": 0.9,
        "final_confidence": 1.5
    }
    
    with pytest.raises(ValueError):
        validate_confidence_breakdown(invalid_conf)


def test_mock_detector_classification():
    """Test that classification uses both rule and model scores."""
    from pipeline.classification import classify_block
    from pipeline.layout import LayoutBlock
    from pipeline.decomposition import TextSpan
    from schemas.block import BoundingBox
    
    block = LayoutBlock(page_num=0)
    bbox = BoundingBox(x0=10, y0=700, x1=200, y1=720)
    span = TextSpan("Introduction", bbox, "Arial", 18.0, 0)
    block.add_span(span)
    
    result = classify_block(
        block,
        detections=[],
        page_width=612,
        page_height=792,
        avg_font_size=12.0
    )
    
    # Check both scores are computed
    conf = result["confidence"]
    assert "rule_score" in conf
    assert "model_score" in conf
    assert "geometric_score" in conf
    assert "final_confidence" in conf
    
    # Final score should be weighted combination, with geometry fallback if no detections.
    if conf["model_score"] == 0.0:
        expected = 0.75 * conf["rule_score"] + 0.25 * conf["geometric_score"]
    else:
        expected = 0.25 * conf["rule_score"] + 0.6 * conf["model_score"] + 0.15 * conf["geometric_score"]
    assert abs(conf["final_confidence"] - expected) < 0.01


def test_reading_order_assignment():
    """
    Test that reading order is correctly assigned.

    In bottom-left-origin coordinates, y0=0 is the BOTTOM of the page.
    A block at y0=700 (near top of a 792pt page) should come BEFORE
    a block at y0=100 (near bottom).
    """
    from pipeline.reading_order import assign_reading_order
    from pipeline.layout import LayoutBlock
    from pipeline.decomposition import TextSpan
    from schemas.block import BoundingBox

    classified_blocks = []

    # Block near TOP of page (high y0 in bottom-left origin)
    block1 = LayoutBlock(page_num=0)
    bbox1 = BoundingBox(x0=10, y0=700, x1=100, y1=720)  # near top
    span1 = TextSpan("First", bbox1, "Arial", 12.0, 0)
    block1.add_span(span1)
    classified_blocks.append({
        "layout_block": block1,
        "block_type": "text",
        "confidence": {
            "rule_score": 0.8,
            "model_score": 0.5,
            "geometric_score": 1.0,
            "final_confidence": 0.69  # 0.3*0.8 + 0.5*0.5 + 0.2*1.0
        }
    })

    # Block near BOTTOM of page (low y0 in bottom-left origin)
    block2 = LayoutBlock(page_num=0)
    bbox2 = BoundingBox(x0=10, y0=50, x1=100, y1=70)   # near bottom
    span2 = TextSpan("Second", bbox2, "Arial", 12.0, 0)
    block2.add_span(span2)
    classified_blocks.append({
        "layout_block": block2,
        "block_type": "text",
        "confidence": {
            "rule_score": 0.8,
            "model_score": 0.5,
            "geometric_score": 1.0,
            "final_confidence": 0.69
        }
    })

    ordered = assign_reading_order(classified_blocks, page_width=612)

    # block1 (high y0 = near top) must have a LOWER reading_order index
    # (i.e. it comes first in reading sequence)
    assert ordered[0]["reading_order"] < ordered[1]["reading_order"], (
        f"Expected block at y0=700 (top) to come before block at y0=50 (bottom). "
        f"Got reading_order: {ordered[0]['reading_order']} vs {ordered[1]['reading_order']}"
    )


def test_extract_images_uses_correct_fields():
    """extract_images should preserve image coordinates from pdfplumber."""
    from pipeline.decomposition import extract_images
    from unittest.mock import MagicMock

    # Build a mock pdfplumber page with an image using y0/y1.
    mock_img = {
        "x0": 50.0, "y0": 100.0,
        "x1": 200.0, "y1": 300.0,
        "width": 150.0, "height": 200.0,
    }
    mock_page = MagicMock()
    mock_page.images = [mock_img]
    mock_page.height = 792.0

    result = extract_images(mock_page, page_num=0)
    assert len(result) == 1
    bbox = result[0]["bbox"]
    assert abs(bbox.y0 - 100.0) < 0.1, f"y0 mismatch: {bbox.y0}"
    assert abs(bbox.y1 - 300.0) < 0.1, f"y1 mismatch: {bbox.y1}"


def test_extract_lines_coordinate_conversion():
    """Regression test for Bug 2: line y-coords must be in bottom-left origin."""
    from pipeline.decomposition import extract_lines
    from unittest.mock import MagicMock

    mock_line = {
        "x0": 72.0,
        "x1": 540.0,
        "top": 100.0,     # pdfplumber top-origin: 100pt from page top
        "bottom": 101.0,  # 1pt thick horizontal line
    }
    mock_page = MagicMock()
    mock_page.lines = [mock_line]
    mock_page.height = 792.0

    result = extract_lines(mock_page)
    assert len(result) == 1
    line = result[0]
    # Expected bottom-left origin: y0 = 792-101 = 691, y1 = 792-100 = 692
    assert abs(line["y0"] - 691.0) < 0.1, f"y0 mismatch: {line['y0']}"
    assert abs(line["y1"] - 692.0) < 0.1, f"y1 mismatch: {line['y1']}"
