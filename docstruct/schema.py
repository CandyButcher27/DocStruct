"""Core data model for the DocStruct pipeline.

All structures are plain dataclasses (no Pydantic) so they serialize with
``dataclasses.asdict`` and stay dependency-free. Coordinates use a **top-left**
origin: ``y0`` is the top edge, ``y1`` the bottom edge, and ``y`` increases
downward (matching pdfplumber's ``top``/``bottom``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Label = Literal["text", "header", "table", "figure", "caption"]
DetectorSource = Literal["geometry", "model"]


class Source(str, Enum):
    """Provenance of a fused :class:`Block`."""

    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    UNILATERAL_MODEL = "unilateral_model"
    UNILATERAL_GEOMETRY = "unilateral_geometry"


@dataclass
class BoundingBox:
    """Axis-aligned box in top-left page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass
class Proposal:
    """A single-detector region candidate (from geometry or the model)."""

    label: Label
    confidence: float
    bbox: BoundingBox
    source: DetectorSource
    page_num: int
    proposal_id: str


@dataclass
class ConfidenceBreakdown:
    """Per-source confidence plus the fused final score."""

    geometry_score: float
    model_score: float
    final: float


@dataclass
class Block:
    """A fused, reading-ordered layout region."""

    bbox: BoundingBox
    label: Label
    confidence: ConfidenceBreakdown
    source: Source
    page_num: int
    block_id: str
    reading_order: int = -1
    caption_target_id: str | None = None
    text: str | None = None
    table_data: list[list[str]] | None = None
    font_size: float | None = None


@dataclass
class SectionPath:
    """Up to three levels of section hierarchy for a chunk."""

    h1: str | None = None
    h2: str | None = None
    h3: str | None = None


@dataclass
class Chunk:
    """A retrieval-ready unit with section-hierarchy metadata."""

    chunk_id: str
    chunk_type: Literal["text", "table", "figure_caption", "abstract", "references"]
    content: str
    section_path: SectionPath
    page_num: int
    reading_order: int
    source_block_ids: list[str]
    metadata: dict
