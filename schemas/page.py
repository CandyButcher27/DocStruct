"""
Page schema definition.

A Page contains all blocks extracted from a single PDF page.
"""

from typing import List
from pydantic import BaseModel, Field
from schemas.block import Block


class PageDimensions(BaseModel):
    """Physical dimensions of a PDF page."""
    width: float = Field(..., gt=0, description="Page width in points")
    height: float = Field(..., gt=0, description="Page height in points")


class Page(BaseModel):
    """
    A single page in a PDF document.
    
    Contains all extracted blocks and page metadata.
    """
    page_num: int = Field(..., ge=0, description="Zero-indexed page number")
    dimensions: PageDimensions = Field(..., description="Page dimensions")
    blocks: List[Block] = Field(default_factory=list, description="All blocks on this page")
    
    def get_block_by_id(self, block_id: str) -> Block | None:
        """Retrieve a block by its ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None
    
    def get_blocks_by_type(self, block_type: str) -> List[Block]:
        """Retrieve all blocks of a given type."""
        return [b for b in self.blocks if b.block_type == block_type]