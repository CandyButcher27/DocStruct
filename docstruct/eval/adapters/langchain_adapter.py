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

        joiner = "\n\n"
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = [(page.extract_text() or "") for page in pdf.pages]
        text = joiner.join(page_texts)

        # Character span of each page within the concatenated text. This splitter is
        # handed the whole document precisely because it ignores structure, so its
        # chunks straddle page breaks by construction; page attribution has to be
        # recovered by offset rather than read off a page object.
        bounds, pos = [], 0
        for page_text in page_texts:
            bounds.append((pos, pos + len(page_text)))
            pos += len(page_text) + len(joiner)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        out: List[EvalChunk] = []
        search_from = 0
        for i, part in enumerate(splitter.split_text(text)):
            if not part.strip():
                continue
            # The splitter strips whitespace, so a chunk is not always findable
            # verbatim. Fall back to the running cursor rather than dropping the
            # page attribution, which would silently make the chunk unscorable.
            start = text.find(part, search_from)
            if start < 0:
                start = min(search_from, max(len(text) - len(part), 0))
            end = start + len(part)
            search_from = start + 1
            pages = [n for n, (a, b) in enumerate(bounds) if a < end and b > start]
            out.append(EvalChunk(id=f"lc_{i}", text=part, metadata={"pages": pages}))
        return out
