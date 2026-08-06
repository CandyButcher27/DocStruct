"""PyMuPDF4LLM baseline: Markdown extraction with per-page chunks.

pymupdf4llm renders the PDF to Markdown; ``page_chunks=True`` yields one chunk
per page. A common lightweight strategy that captures Markdown structure but no
section-aware boundaries.
"""

from __future__ import annotations

import importlib.util
from typing import List

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk


class PyMuPDF4LLMAdapter(ChunkAdapter):
    name = "pymupdf4llm"

    def available(self) -> bool:
        return importlib.util.find_spec("pymupdf4llm") is not None

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        import pymupdf4llm

        # pymupdf4llm's layout path defaults to use_ocr=SELECT_KEEP_OLD: an ONNX
        # classifier flags "needs OCR" pages and RapidOCR re-derives their text.
        # Our corpora are born-digital and every other tool reads the text layer,
        # so this costs 3x wall-clock for no new content (AES_2022_10K: 270.6s with
        # OCR on 24/116 pages vs 89.3s for langchain). Same fix as docling.
        # The per-page classifier verdicts are captured by scripts/ocr_audit.py.
        pages = pymupdf4llm.to_markdown(
            pdf_path, page_chunks=True, show_progress=False, use_ocr=False
        )
        chunks: List[EvalChunk] = []
        for i, page in enumerate(pages):
            text = (page.get("text") if isinstance(page, dict) else str(page)) or ""
            if text.strip():
                chunks.append(EvalChunk(id=f"pm_{i}", text=text, metadata={"page": i}))
        return chunks
