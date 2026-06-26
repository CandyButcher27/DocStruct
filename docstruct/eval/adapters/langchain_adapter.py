"""LangChain naive baseline: full text -> RecursiveCharacterTextSplitter.

This is the canonical fixed-window splitter most RAG tutorials use. Text is
extracted with pdfplumber (page by page) and split by LangChain's real
splitter, ignoring document structure entirely.
"""

from __future__ import annotations

import importlib.util
from typing import List

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk


class LangChainAdapter(ChunkAdapter):
    name = "langchain"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def available(self) -> bool:
        return importlib.util.find_spec("langchain_text_splitters") is not None

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        import pdfplumber
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        with pdfplumber.open(pdf_path) as pdf:
            text = "\n\n".join((page.extract_text() or "") for page in pdf.pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        parts = splitter.split_text(text)
        return [EvalChunk(id=f"lc_{i}", text=p) for i, p in enumerate(parts) if p.strip()]
