"""
Tests for layout formation stage.
"""

import pytest
from pipeline.layout import LayoutBlock, should_merge_spans, form_layout_blocks
from pipeline.decomposition import PageData, TextSpan
from schemas.block import BoundingBox


def test_layout_block_creation():
    """Test LayoutBlock initialization."""
    block = LayoutBlock(page_num=0)
    
    assert block.page_num == 0
    assert len(block.spans) == 0
    assert block.bbox is None


def test_layout_block_add_span():
    """Test adding span to LayoutBlock."""
    block = LayoutBlock(page_num=0)
    
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    span = TextSpan("Hello", bbox, "Arial", 12.0, 0)
    
    block.add_span(span)
    
    assert len(block.spans) == 1
    assert block.text == "Hello"
    assert block.avg_font_size == 12.0
    assert block.bbox is not None


def test_should_merge_spans_same_font():
    """Test merging spans with similar fonts."""
    bbox1 = BoundingBox(x0=10, y0=20, x1=50, y1=40)
    span1 = TextSpan("Hello", bbox1, "Arial", 12.0, 0)
    
    bbox2 = BoundingBox(x0=55, y0=20, x1=100, y1=40)
    span2 = TextSpan("World", bbox2, "Arial", 12.0, 0)
    
    should_merge = should_merge_spans(span1, span2, page_width=612)
    
    assert should_merge is True


def test_should_not_merge_different_fonts():
    """Test not merging spans with very different font sizes."""
    bbox1 = BoundingBox(x0=10, y0=20, x1=50, y1=40)
    span1 = TextSpan("Hello", bbox1, "Arial", 12.0, 0)
    
    bbox2 = BoundingBox(x0=55, y0=20, x1=100, y1=60)
    span2 = TextSpan("Title", bbox2, "Arial", 24.0, 0)
    
    should_merge = should_merge_spans(span1, span2, page_width=612)
    
    assert should_merge is False


def test_should_not_merge_distant_spans():
    """Test not merging vertically distant spans."""
    bbox1 = BoundingBox(x0=10, y0=20, x1=50, y1=40)
    span1 = TextSpan("Hello", bbox1, "Arial", 12.0, 0)
    
    bbox2 = BoundingBox(x0=10, y0=100, x1=50, y1=120)
    span2 = TextSpan("World", bbox2, "Arial", 12.0, 0)
    
    should_merge = should_merge_spans(span1, span2, page_width=612)
    
    assert should_merge is False


def test_form_layout_blocks():
    """Test forming layout blocks from spans."""
    page_data = PageData(page_num=0, width=612, height=792)
    
    # Add three spans that should merge into one block
    bbox1 = BoundingBox(x0=10, y0=20, x1=50, y1=32)
    span1 = TextSpan("Hello", bbox1, "Arial", 12.0, 0)
    page_data.add_span(span1)
    
    bbox2 = BoundingBox(x0=55, y0=20, x1=100, y1=32)
    span2 = TextSpan("beautiful", bbox2, "Arial", 12.0, 0)
    page_data.add_span(span2)
    
    bbox3 = BoundingBox(x0=105, y0=20, x1=150, y1=32)
    span3 = TextSpan("world", bbox3, "Arial", 12.0, 0)
    page_data.add_span(span3)
    
    blocks = form_layout_blocks(page_data)
    
    assert len(blocks) == 1
    assert len(blocks[0].spans) == 3
    assert "Hello" in blocks[0].text
    assert "world" in blocks[0].text


def test_form_layout_blocks_multiple():
    """Test forming multiple layout blocks."""
    page_data = PageData(page_num=0, width=612, height=792)
    
    # First block (higher on the page -> larger y0 in bottom-left origin)
    bbox1 = BoundingBox(x0=10, y0=700, x1=100, y1=712)
    span1 = TextSpan("First paragraph", bbox1, "Arial", 12.0, 0)
    page_data.add_span(span1)
    
    # Second block (far below -> smaller y0)
    bbox2 = BoundingBox(x0=10, y0=600, x1=100, y1=612)
    span2 = TextSpan("Second paragraph", bbox2, "Arial", 12.0, 0)
    page_data.add_span(span2)
    
    blocks = form_layout_blocks(page_data)
    
    assert len(blocks) == 2
    assert "First" in blocks[0].text
    assert "Second" in blocks[1].text