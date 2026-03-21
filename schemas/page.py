"""
Page schema definition.

All contained bounding boxes use a bottom-left origin.
"""

from typing import List

from pydantic import BaseModel, Field

from schemas.block import Block


class PageDimensions(BaseModel):
    """Physical dimensions of a PDF page."""

    width: float = Field(..., gt=0, description="Page width in points")
    height: float = Field(..., gt=0, description="Page height in points")


class Page(BaseModel):
    """A single page in a document."""

    page_num: int = Field(..., ge=0, description="Zero-indexed page number")
    dimensions: PageDimensions = Field(..., description="Page dimensions")
    blocks: List[Block] = Field(default_factory=list, description="Blocks on this page")

    def get_block_by_id(self, block_id: str) -> Block | None:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def get_blocks_by_type(self, block_type: str) -> List[Block]:
        return [block for block in self.blocks if block.block_type == block_type]
