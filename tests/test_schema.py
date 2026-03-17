"""
Tests for schema validation.
"""

import pytest
from pydantic import ValidationError
from schemas.block import BoundingBox, ConfidenceBreakdown, Block
from schemas.page import Page, PageDimensions
from schemas.document import Document, DocumentMetadata


def test_bounding_box_valid():
    """Test valid bounding box creation."""
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    
    assert bbox.x0 == 10
    assert bbox.x1 == 100
    assert bbox.area() == 90 * 60
    assert bbox.width() == 90
    assert bbox.height() == 60


def test_bounding_box_invalid_x():
    """Test that x1 must be greater than x0."""
    with pytest.raises(ValidationError):
        BoundingBox(x0=100, y0=20, x1=10, y1=80)


def test_bounding_box_invalid_y():
    """Test that y1 must be greater than y0."""
    with pytest.raises(ValidationError):
        BoundingBox(x0=10, y0=80, x1=100, y1=20)


def test_bounding_box_negative():
    """Test that negative coordinates are rejected."""
    with pytest.raises(ValidationError):
        BoundingBox(x0=-10, y0=20, x1=100, y1=80)


def test_confidence_breakdown_valid():
    """Test valid confidence breakdown."""
    conf = ConfidenceBreakdown(
        rule_score=0.6,
        model_score=0.8,
        geometric_score=0.9,
        final_confidence=0.76  # 0.3*0.6 + 0.5*0.8 + 0.2*0.9
    )
    
    assert conf.final_confidence == 0.76


def test_confidence_breakdown_invalid_formula():
    """Test that confidence formula is validated."""
    with pytest.raises(ValidationError):
        ConfidenceBreakdown(
            rule_score=0.6,
            model_score=0.8,
            geometric_score=0.9,
            final_confidence=0.5  # Incorrect
        )


def test_confidence_breakdown_out_of_range():
    """Test that scores must be in [0, 1]."""
    with pytest.raises(ValidationError):
        ConfidenceBreakdown(
            rule_score=1.5,  # Invalid
            model_score=0.8,
            geometric_score=0.9,
            final_confidence=0.95
        )


def test_block_valid():
    """Test valid block creation."""
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    conf = ConfidenceBreakdown(
        rule_score=0.6,
        model_score=0.8,
        geometric_score=0.9,
        final_confidence=0.76
    )
    
    block = Block(
        block_id="block_0_0",
        block_type="text",
        bbox=bbox,
        page_num=0,
        text="Hello world",
        confidence=conf,
        reading_order=0,
        table_data=None,
        image_metadata=None,
        caption_id=None,
        parent_id=None
    )
    
    assert block.block_type == "text"
    assert block.text == "Hello world"


def test_block_table_data_validation():
    """Test that table_data is only allowed for tables."""
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    conf = ConfidenceBreakdown(
        rule_score=0.6,
        model_score=0.8,
        geometric_score=0.9,
        final_confidence=0.76
    )
    
    # Should fail: text block with table_data
    with pytest.raises(ValidationError):
        Block(
            block_id="block_0_0",
            block_type="text",
            bbox=bbox,
            page_num=0,
            text="Hello",
            confidence=conf,
            reading_order=0,
            table_data={"num_rows": 3},
            image_metadata=None,
            caption_id=None,
            parent_id=None
        )


def test_block_image_metadata_validation():
    """Test that image_metadata is only allowed for figures."""
    bbox = BoundingBox(x0=10, y0=20, x1=100, y1=80)
    conf = ConfidenceBreakdown(
        rule_score=0.6,
        model_score=0.8,
        geometric_score=0.9,
        final_confidence=0.76
    )
    
    # Should fail: text block with image_metadata
    with pytest.raises(ValidationError):
        Block(
            block_id="block_0_0",
            block_type="text",
            bbox=bbox,
            page_num=0,
            text="Hello",
            confidence=conf,
            reading_order=0,
            image_metadata={"width": 100},
            table_data=None,
            parent_id=None,
            caption_id=None
        )


def test_page_valid():
    """Test valid page creation."""
    dims = PageDimensions(width=612, height=792)
    page = Page(page_num=0, dimensions=dims, blocks=[])
    
    assert page.page_num == 0
    assert page.dimensions.width == 612


def test_document_valid():
    """Test valid document creation."""
    metadata = DocumentMetadata(filename="test.pdf", num_pages=1)
    dims = PageDimensions(width=612, height=792)
    page = Page(page_num=0, dimensions=dims, blocks=[])
    
    doc = Document(metadata=metadata, pages=[page])
    
    assert doc.metadata.filename == "test.pdf"
    assert len(doc.pages) == 1


def test_document_to_dict():
    """Test document serialization to dict."""
    metadata = DocumentMetadata(filename="test.pdf", num_pages=1)
    dims = PageDimensions(width=612, height=792)
    page = Page(page_num=0, dimensions=dims, blocks=[])
    
    doc = Document(metadata=metadata, pages=[page])
    doc_dict = doc.to_dict()
    
    assert isinstance(doc_dict, dict)
    assert doc_dict["metadata"]["filename"] == "test.pdf"