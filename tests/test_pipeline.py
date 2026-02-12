"""
Integration tests for the complete pipeline.
"""

import pytest
import json
import tempfile
from pathlib import Path
from main import process_pdf


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
    if len(blocks) >= 2:
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
    
    # Invalid confidence (wrong formula)
    invalid_conf = {
        "rule_score": 0.6,
        "model_score": 0.8,
        "geometric_score": 0.9,
        "final_confidence": 0.5
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
    bbox = BoundingBox(x0=10, y0=50, x1=200, y1=70)
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
    
    # Final score should be weighted combination
    expected = 0.3 * conf["rule_score"] + 0.5 * conf["model_score"] + 0.2 * conf["geometric_score"]
    assert abs(conf["final_confidence"] - expected) < 0.01


def test_reading_order_assignment():
    """Test that reading order is correctly assigned."""
    from pipeline.reading_order import assign_reading_order
    from pipeline.layout import LayoutBlock
    from pipeline.decomposition import TextSpan
    from schemas.block import BoundingBox
    
    # Create test blocks
    classified_blocks = []
    
    # Block at top
    block1 = LayoutBlock(page_num=0)
    bbox1 = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    span1 = TextSpan("First", bbox1, "Arial", 12.0, 0)
    block1.add_span(span1)
    classified_blocks.append({
        "layout_block": block1,
        "block_type": "text",
        "confidence": {
            "rule_score": 0.8,
            "model_score": 0.5,
            "geometric_score": 1.0,
            "final_confidence": 0.66
        }
    })
    
    # Block at bottom
    block2 = LayoutBlock(page_num=0)
    bbox2 = BoundingBox(x0=10, y0=200, x1=100, y1=220)
    span2 = TextSpan("Second", bbox2, "Arial", 12.0, 0)
    block2.add_span(span2)
    classified_blocks.append({
        "layout_block": block2,
        "block_type": "text",
        "confidence": {
            "rule_score": 0.8,
            "model_score": 0.5,
            "geometric_score": 1.0,
            "final_confidence": 0.66
        }
    })
    
    ordered = assign_reading_order(classified_blocks, page_width=612)
    
    # First block should have lower reading order
    assert ordered[0]["reading_order"] < ordered[1]["reading_order"]