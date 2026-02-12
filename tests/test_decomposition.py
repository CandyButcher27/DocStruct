"""
Tests for PDF decomposition stage.
"""

import pytest
from pathlib import Path
from pipeline.decomposition import decompose_pdf, TextSpan, PageData
from schemas.block import BoundingBox


def test_text_span_creation():
    """Test TextSpan object creation."""
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    span = TextSpan(
        text="Hello World",
        bbox=bbox,
        font_name="Arial",
        font_size=12.0,
        page_num=0
    )
    
    assert span.text == "Hello World"
    assert span.font_size == 12.0
    assert span.page_num == 0


def test_page_data_initialization():
    """Test PageData container."""
    page_data = PageData(page_num=0, width=612, height=792)
    
    assert page_data.page_num == 0
    assert page_data.width == 612
    assert page_data.height == 792
    assert len(page_data.spans) == 0
    assert len(page_data.images) == 0


def test_page_data_add_span():
    """Test adding spans to PageData."""
    page_data = PageData(page_num=0, width=612, height=792)
    
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=40)
    span = TextSpan("Test", bbox, "Arial", 12.0, 0)
    
    page_data.add_span(span)
    
    assert len(page_data.spans) == 1
    assert page_data.spans[0].text == "Test"