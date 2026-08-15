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

import logging
from importlib.metadata import PackageNotFoundError, version

from docstruct.document import Document, parse, parse_bytes, parse_many
from docstruct.errors import (
    DocStructError,
    EmptyDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
)
from docstruct.pipeline import PipelineResult, run_pipeline
from docstruct.schema import (
    Block,
    BoundingBox,
    Chunk,
    ConfidenceBreakdown,
    Proposal,
    SectionPath,
    Source,
)

# Single source of truth is the installed package metadata (pyproject version).
# The fallback covers running from a source tree that was never installed.
# The distribution is `docstruct-rag` (the name `docstruct` was already taken on
# PyPI); the import name stays `docstruct`. importlib.metadata keys on the
# *distribution* name, so ask for that first and keep the old name as a fallback for
# anyone with a pre-rename editable install.
try:
    __version__ = version("docstruct-rag")
except PackageNotFoundError:  # pragma: no cover - depends on how it was installed
    try:
        __version__ = version("docstruct")
    except PackageNotFoundError:  # pragma: no cover - source-tree-only path
        __version__ = "0.0.0.dev0"

# Library logging hygiene: never emit unless the host app configures handlers.
logging.getLogger("docstruct").addHandler(logging.NullHandler())

__all__ = [
    "parse",
    "parse_bytes",
    "parse_many",
    "run_pipeline",
    "PipelineResult",
    "Document",
    "Block",
    "BoundingBox",
    "Chunk",
    "ConfidenceBreakdown",
    "Proposal",
    "SectionPath",
    "Source",
    "DocStructError",
    "InvalidPDFError",
    "EncryptedPDFError",
    "EmptyDocumentError",
    "__version__",
]
