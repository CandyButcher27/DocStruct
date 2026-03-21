"""
Document schema definition.

Serialized output for the full structured document.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from schemas.page import Page


class DocumentMetadata(BaseModel):
    """Metadata about the source document."""

    filename: str = Field(..., description="Source filename")
    num_pages: int = Field(..., ge=1, description="Total number of pages")
    processing_version: str = Field(default="1.0.0", description="DocStruct version")


class Document(BaseModel):
    """Top-level structured document output."""

    metadata: DocumentMetadata = Field(..., description="Document metadata")
    pages: List[Page] = Field(default_factory=list, description="Pages in order")

    def get_page(self, page_num: int) -> Page | None:
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None

    def get_all_blocks(self) -> List:
        blocks = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return blocks

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
