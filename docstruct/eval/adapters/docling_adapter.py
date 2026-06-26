"""Docling baseline: DocumentConverter + HybridChunker.

Docling (IBM) parses the PDF into a structured document and the HybridChunker
produces section-aware chunks — a strong structure-aware competitor.
"""

from __future__ import annotations

import importlib.util
from typing import List

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk


class DoclingAdapter(ChunkAdapter):
    name = "docling"

    def __init__(self) -> None:
        self._converter = None
        self._chunker = None

    def available(self) -> bool:
        return importlib.util.find_spec("docling") is not None

    def _ensure(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            from docling.chunking import HybridChunker

            self._converter = DocumentConverter()
            self._chunker = HybridChunker()

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        self._ensure()
        doc = self._converter.convert(pdf_path).document
        out: List[EvalChunk] = []
        for i, ch in enumerate(self._chunker.chunk(doc)):
            text = getattr(ch, "text", "") or ""
            if text.strip():
                out.append(EvalChunk(id=f"dl_{i}", text=text))
        return out
