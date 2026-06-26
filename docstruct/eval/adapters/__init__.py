"""Chunker adapters for cross-tool benchmarking.

Every adapter exposes the same ``chunk(pdf_path) -> List[EvalChunk]`` interface
so the benchmark can hold the embedder + retriever constant and vary only the
chunker. ``get_adapters`` returns the adapters whose dependencies are installed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk
from docstruct.eval.adapters.docstruct_adapter import DocStructAdapter
from docstruct.eval.adapters.langchain_adapter import LangChainAdapter
from docstruct.eval.adapters.pymupdf4llm_adapter import PyMuPDF4LLMAdapter

_ALL = ["docstruct", "langchain", "pymupdf4llm", "unstructured", "docling"]


def build_adapter(
    name: str, weights: Optional[str] = None, cache_dir: Optional[str] = None
) -> ChunkAdapter:
    if name == "docstruct":
        return DocStructAdapter(weights=weights, cache_dir=cache_dir)
    if name == "langchain":
        return LangChainAdapter()
    if name == "pymupdf4llm":
        return PyMuPDF4LLMAdapter()
    if name == "unstructured":
        from docstruct.eval.adapters.unstructured_adapter import UnstructuredAdapter
        return UnstructuredAdapter()
    if name == "docling":
        from docstruct.eval.adapters.docling_adapter import DoclingAdapter
        return DoclingAdapter()
    raise ValueError(f"unknown adapter: {name}")


def get_adapters(
    names: Optional[List[str]] = None,
    weights: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, ChunkAdapter]:
    """Return {name: adapter} for requested adapters whose deps are available."""
    out: Dict[str, ChunkAdapter] = {}
    for name in names or _ALL:
        try:
            adapter = build_adapter(name, weights=weights, cache_dir=cache_dir)
        except Exception:  # noqa: BLE001 - missing optional adapter module
            continue
        if adapter.available():
            out[name] = adapter
    return out


__all__ = ["ChunkAdapter", "EvalChunk", "build_adapter", "get_adapters"]
