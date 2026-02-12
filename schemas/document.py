"""
Document schema definition.

A Document represents the complete structured output from a PDF.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from schemas.page import Page


class DocumentMetadata(BaseModel):
    """Metadata about the source document."""
    filename: str = Field(..., description="Source PDF filename")
    num_pages: int = Field(..., ge=1, description="Total number of pages")
    processing_version: str = Field(default="1.0.0", description="DocStruct version")


class Document(BaseModel):
    """
    Complete structured representation of a PDF document.
    
    This is the top-level schema for validated output.
    """
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    pages: List[Page] = Field(default_factory=list, description="All pages in order")
    
    def get_page(self, page_num: int) -> Page | None:
        """Retrieve a page by its number."""
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None
    
    def get_all_blocks(self) -> List:
        """Retrieve all blocks across all pages."""
        blocks = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return blocks
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for JSON serialization."""
        return self.model_dump(mode='json')