"""
Schema validation stage.

Validates the final Document against Pydantic schemas.
"""

from typing import Dict, Any, List
from schemas.document import Document, DocumentMetadata
from schemas.page import Page, PageDimensions
from schemas.block import Block, BoundingBox, ConfidenceBreakdown
from utils.logging import setup_logger


logger = setup_logger(__name__)


def create_block_from_data(block_data: Dict[str, Any], block_id: str) -> Block:
    """
    Create a validated Block from classified block data.
    
    Args:
        block_data: Classified block data
        block_id: Unique block identifier
        
    Returns:
        Validated Block instance
    """
    # Extract bbox
    if "layout_block" in block_data:
        bbox = block_data["layout_block"].bbox
        text = block_data["layout_block"].text
    elif "image_block" in block_data:
        bbox = block_data["image_block"]["bbox"]
        text = None
    else:
        raise ValueError("Block has neither layout_block nor image_block")
    
    # Extract page number
    if "layout_block" in block_data:
        page_num = block_data["layout_block"].page_num
    else:
        page_num = block_data["image_block"]["page_num"]
    
    # Create confidence breakdown
    conf_data = block_data["confidence"]
    confidence = ConfidenceBreakdown(
        rule_score=conf_data["rule_score"],
        model_score=conf_data["model_score"],
        geometric_score=conf_data["geometric_score"],
        final_confidence=conf_data["final_confidence"]
    )
    
    # Get reading order
    reading_order = block_data.get("reading_order", 0)
    
    # Get type-specific data
    table_data = block_data.get("table_data")
    image_metadata = block_data.get("image_metadata")
    
    # Get caption relationship
    caption_id = None
    if "has_caption" in block_data:
        caption_idx = block_data["has_caption"]
        caption_id = f"block_{page_num}_{caption_idx}"
    
    # Create block
    block = Block(
        block_id=block_id,
        parent_id=None,
        block_type=block_data["block_type"],
        bbox=bbox,
        page_num=page_num,
        text=text,
        confidence=confidence,
        reading_order=reading_order,
        table_data=table_data,
        image_metadata=image_metadata,
        caption_id=caption_id
    )
    
    return block


def validate_and_build_document(
    pages_data: List[tuple],
    filename: str
) -> Document:
    """
    Validate and build final Document from pipeline output.
    
    Args:
        pages_data: List of (page_data, classified_blocks) tuples
        filename: Source PDF filename
        
    Returns:
        Validated Document instance
        
    Raises:
        ValidationError: If any schema validation fails
    """
    logger.info("Building and validating document schema")
    
    # Create pages
    pages = []
    
    for page_num, (page_data, classified_blocks) in enumerate(pages_data):
        # Create page dimensions
        dimensions = PageDimensions(
            width=page_data.width,
            height=page_data.height
        )
        
        # Create blocks
        blocks = []
        for i, block_data in enumerate(classified_blocks):
            block_id = f"block_{page_num}_{i}"
            try:
                block = create_block_from_data(block_data, block_id)
                blocks.append(block)
            except Exception as e:
                logger.error(f"Failed to create block {block_id}: {e}")
                raise
        
        # Create page
        page = Page(
            page_num=page_num,
            dimensions=dimensions,
            blocks=blocks
        )
        
        pages.append(page)
    
    # Create document metadata
    metadata = DocumentMetadata(
        filename=filename,
        num_pages=len(pages)
    )
    
    # Create and validate document
    document = Document(
        metadata=metadata,
        pages=pages
    )
    
    logger.info(f"Document validated: {len(pages)} pages, {len(document.get_all_blocks())} blocks")
    
    return document