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

    def available(self) -> bool:
        return importlib.util.find_spec("unstructured") is not None

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.chunking.title import chunk_by_title

        elements = partition_pdf(filename=pdf_path, strategy="fast")
        chunks = chunk_by_title(elements, max_characters=1000, combine_text_under_n_chars=200)
        out: List[EvalChunk] = []
        for i, c in enumerate(chunks):
            text = getattr(c, "text", "") or ""
            if text.strip():
                out.append(EvalChunk(id=f"un_{i}", text=text))
        return out
