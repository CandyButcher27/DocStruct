"""
Block schema definition.

All coordinates use a bottom-left page origin.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


BlockType = Literal["text", "header", "table", "figure", "caption"]


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in bottom-left page coordinates."""

    x0: float = Field(..., ge=0, description="Left edge")
    y0: float = Field(..., ge=0, description="Bottom edge")
    x1: float = Field(..., gt=0, description="Right edge")
    y1: float = Field(..., gt=0, description="Top edge")

    @field_validator("x1")
    @classmethod
    def x1_must_be_greater_than_x0(cls, v: float, info) -> float:
        if "x0" in info.data and v <= info.data["x0"]:
            raise ValueError("x1 must be greater than x0")
        return v

    @field_validator("y1")
    @classmethod
    def y1_must_be_greater_than_y0(cls, v: float, info) -> float:
        if "y0" in info.data and v <= info.data["y0"]:
            raise ValueError("y1 must be greater than y0")
        return v

    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0


class ConfidenceBreakdown(BaseModel):
    """Detailed confidence scoring components."""

    model_config = ConfigDict(protected_namespaces=())

    rule_score: float = Field(..., ge=0, le=1)
    model_score: float = Field(..., ge=0, le=1)
    geometric_score: float = Field(..., ge=0, le=1)
    final_confidence: float = Field(..., ge=0, le=1)


class Block(BaseModel):
    """A semantic block extracted from a PDF page."""

    block_id: str = Field(..., description="Unique identifier")
    block_type: BlockType = Field(..., description="Semantic type")
    bbox: BoundingBox = Field(..., description="Bounding box")
    page_num: int = Field(..., ge=0, description="Zero-indexed page number")
    text: Optional[str] = Field(None, description="Extracted text content")
    confidence: ConfidenceBreakdown = Field(..., description="Confidence breakdown")
    reading_order: int = Field(..., ge=0, description="Reading sequence position")
    table_data: Optional[Dict[str, Any]] = Field(None, description="Table metadata")
    image_metadata: Optional[Dict[str, Any]] = Field(None, description="Figure metadata")
    caption_id: Optional[str] = Field(None, description="Associated caption block id")
    caption_target_id: Optional[str] = Field(
        None,
        description="Target block id for caption blocks",
    )
    parent_id: Optional[str] = Field(None, description="Parent block id")

    @field_validator("table_data")
    @classmethod
    def table_data_only_for_tables(cls, v: Optional[Dict[str, Any]], info) -> Optional[Dict[str, Any]]:
        if v is not None and info.data.get("block_type") != "table":
            raise ValueError("table_data can only be set for table blocks")
        return v

    @field_validator("image_metadata")
    @classmethod
    def image_metadata_only_for_figures(
        cls, v: Optional[Dict[str, Any]], info
    ) -> Optional[Dict[str, Any]]:
        if v is not None and info.data.get("block_type") != "figure":
            raise ValueError("image_metadata can only be set for figure blocks")
        return v
