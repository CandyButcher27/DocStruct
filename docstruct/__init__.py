"""DocStruct: a local, deterministic, structure-aware PDF chunking pipeline.

Two independent layout detectors (rules-based geometry + an optional vision
model) are fused into reading-ordered blocks, then assembled into
section-hierarchy-annotated chunks for retrieval. Zero LLM calls in the core
pipeline; fully offline.

Typical use::

    import docstruct

    doc = docstruct.parse("paper.pdf")
    print(doc.text)
    for chunk in doc.chunks:
        print(chunk.section_path, chunk.content)
"""

from __future__ import annotations

from docstruct.document import Document, parse
from docstruct.schema import (
    Block,
    BoundingBox,
    Chunk,
    ConfidenceBreakdown,
    Proposal,
    SectionPath,
    Source,
)

__version__ = "0.3.0"

__all__ = [
    "parse",
    "Document",
    "Block",
    "BoundingBox",
    "Chunk",
    "ConfidenceBreakdown",
    "Proposal",
    "SectionPath",
    "Source",
    "__version__",
]
