"""Unstructured.io baseline: partition_pdf (fast) + chunk_by_title.

Uses the text-based ``fast`` strategy (no heavy detection model) and
title-based chunking, a popular Unstructured RAG recipe.
"""

from __future__ import annotations

import importlib.util
from typing import List

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk


class UnstructuredAdapter(ChunkAdapter):
    name = "unstructured"

    def __init__(self, max_characters: int = 1000, combine_text_under_n_chars: int = 200) -> None:
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars

    def available(self) -> bool:
        return importlib.util.find_spec("unstructured") is not None

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.chunking.title import chunk_by_title

        elements = partition_pdf(filename=pdf_path, strategy="fast")
        chunks = chunk_by_title(elements, max_characters=self.max_characters,
                                combine_text_under_n_chars=self.combine_text_under_n_chars)
        out: List[EvalChunk] = []
        for i, c in enumerate(chunks):
            text = getattr(c, "text", "") or ""
            if not text.strip():
                continue
            # Unstructured numbers pages from 1; every other adapter and the gold
            # count from 0. chunk_by_title merges elements, so take the pages of
            # the originals where they survive.
            meta = getattr(c, "metadata", None)
            pages = set()
            for el in (getattr(meta, "orig_elements", None) or [c]):
                n = getattr(getattr(el, "metadata", None), "page_number", None)
                if n is not None:
                    pages.add(int(n) - 1)
            out.append(EvalChunk(id=f"un_{i}", text=text, metadata={"pages": sorted(pages)}))
        return out
