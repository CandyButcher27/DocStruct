"""
Block schema definition.

A Block represents a coherent unit of content with spatial and semantic properties.
"""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


BlockType = Literal["text", "header", "table", "figure", "caption"]


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in page coordinates."""
    x0: float = Field(..., ge=0, description="Left edge")
    y0: float = Field(..., ge=0, description="Top edge")
    x1: float = Field(..., gt=0, description="Right edge")
    y1: float = Field(..., gt=0, description="Bottom edge")
    
    @field_validator('x1')
    @classmethod
    def x1_must_be_greater_than_x0(cls, v: float, info) -> float:
        if 'x0' in info.data and v <= info.data['x0']:
            raise ValueError('x1 must be greater than x0')
        return v
    
    @field_validator('y1')
    @classmethod
    def y1_must_be_greater_than_y0(cls, v: float, info) -> float:
        if 'y0' in info.data and v <= info.data['y0']:
            raise ValueError('y1 must be greater than y0')
        return v
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x1 - self.x0
    
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y1 - self.y0


class ConfidenceBreakdown(BaseModel):
    """Detailed confidence scoring components."""
    rule_score: float = Field(..., ge=0, le=1, description="Score from rule-based heuristics")
    model_score: float = Field(..., ge=0, le=1, description="Score from ML detector")
    geometric_score: float = Field(..., ge=0, le=1, description="Score from geometric consistency")
    final_confidence: float = Field(..., ge=0, le=1, description="Combined confidence score")
    
    @field_validator('final_confidence')
    @classmethod
    def validate_final_confidence(cls, v: float, info) -> float:
        """Ensure final confidence is computed from components."""
        if 'rule_score' in info.data and 'model_score' in info.data and 'geometric_score' in info.data:
            # Weighted average: 30% rule, 50% model, 20% geometric
            expected = (0.3 * info.data['rule_score'] + 
                       0.5 * info.data['model_score'] + 
                       0.2 * info.data['geometric_score'])
            if abs(v - expected) > 0.01:
                raise ValueError(f'final_confidence {v} does not match computed value {expected:.3f}')
        return v


class Block(BaseModel):
    """
    A semantic block extracted from a PDF page.
    
    Represents a coherent unit like a paragraph, table, or figure.
    """
    block_id: str = Field(..., description="Unique identifier")
    block_type: BlockType = Field(..., description="Semantic type of block")
    bbox: BoundingBox = Field(..., description="Bounding box")
    page_num: int = Field(..., ge=0, description="Zero-indexed page number")
    text: Optional[str] = Field(None, description="Extracted text content")
    
    # Metadata
    confidence: ConfidenceBreakdown = Field(..., description="Confidence breakdown")
    reading_order: int = Field(..., ge=0, description="Position in reading sequence")
    
    # Type-specific fields
    table_data: Optional[Dict[str, Any]] = Field(None, description="Table structure (for tables)")
    image_metadata: Optional[Dict[str, Any]] = Field(None, description="Image info (for figures)")
    
    # Relationships
    caption_id: Optional[str] = Field(None, description="ID of associated caption")
    parent_id: Optional[str] = Field(None, description="ID of parent block")
    
    @field_validator('table_data')
    @classmethod
    def table_data_only_for_tables(cls, v: Optional[Dict], info) -> Optional[Dict]:
        """Ensure table_data is only present for table blocks."""
        if v is not None and info.data.get('block_type') != 'table':
            raise ValueError('table_data can only be set for table blocks')
        return v
    
    @field_validator('image_metadata')
    @classmethod
    def image_metadata_only_for_figures(cls, v: Optional[Dict], info) -> Optional[Dict]:
        """Ensure image_metadata is only present for figure blocks."""
        if v is not None and info.data.get('block_type') != 'figure':
            raise ValueError('image_metadata can only be set for figure blocks')
        return v