"""DocStruct: a local, deterministic, structure-aware PDF chunking pipeline.

Two independent layout detectors (rules-based geometry + an optional vision
model) are fused into reading-ordered blocks, then assembled into
section-hierarchy-annotated chunks for retrieval. Zero LLM calls in the core
pipeline; fully offline.
"""

from __future__ import annotations

from docstruct.schema import (
    Block,
    BoundingBox,
    Chunk,
    ConfidenceBreakdown,
    Proposal,
    SectionPath,
    Source,
)

__version__ = "0.2.0"

__all__ = [
    "Block",
    "BoundingBox",
    "Chunk",
    "ConfidenceBreakdown",
    "Proposal",
    "SectionPath",
    "Source",
    "__version__",
]
