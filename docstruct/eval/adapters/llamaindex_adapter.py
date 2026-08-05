"""LlamaIndex baselines: SentenceSplitter and SemanticSplitterNodeParser.

LlamaIndex is the other framework practitioners reach for alongside LangChain, and
its two parsers sit on opposite sides of the argument this project is making:
``SentenceSplitter`` is a structure-blind fixed-window splitter, while
``SemanticSplitterNodeParser`` places boundaries at embedding-similarity drops --
the "semantic chunking" family whose cost the literature keeps questioning
(arXiv:2410.13070, arXiv:2606.00881).

The semantic variant needs an embedding model. It is given the same
``config.EMBEDDING_MODEL`` every tool is scored with, so the comparison stays
about where boundaries land rather than about who embeds better.
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional

import pdfplumber

from docstruct import config
from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk


def _page_texts(pdf_path: str) -> List[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


def _pages_for(bounds, start: int, end: int) -> List[int]:
    return [n for n, (a, b) in enumerate(bounds) if a < end and b > start]


class _LlamaIndexBase(ChunkAdapter):
    prefix = "li"

    def available(self) -> bool:
        return importlib.util.find_spec("llama_index.core") is not None

    def _parser(self):
        raise NotImplementedError

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        from llama_index.core import Document

        page_texts = _page_texts(pdf_path)
        text = "\n\n".join(page_texts)

        bounds, pos = [], 0
        for pt in page_texts:
            bounds.append((pos, pos + len(pt)))
            pos += len(pt) + 2

        nodes = self._parser().get_nodes_from_documents([Document(text=text)])

        out: List[EvalChunk] = []
        search_from = 0
        for i, node in enumerate(nodes):
            content = node.get_content()
            if not content.strip():
                continue
            # Node char offsets are not reliable across parsers, so recover the
            # span the same way the LangChain adapter does: by search.
            start = text.find(content, search_from)
            if start < 0:
                start = search_from
            end = start + len(content)
            search_from = start + 1
            out.append(
                EvalChunk(
                    id=f"{self.prefix}_{i}",
                    text=content,
                    metadata={"pages": _pages_for(bounds, start, end)},
                )
            )
        return out


class LlamaIndexAdapter(_LlamaIndexBase):
    """Fixed-window sentence splitter — LlamaIndex's default recipe."""

    name = "llamaindex"
    prefix = "li"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _parser(self):
        from llama_index.core.node_parser import SentenceSplitter

        return SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)


class LlamaIndexSemanticAdapter(_LlamaIndexBase):
    """Semantic splitter — boundaries at embedding-similarity breakpoints."""

    name = "llamaindex_semantic"
    prefix = "lis"

    def __init__(self, embed_model: Optional[str] = None, breakpoint_percentile: int = 95) -> None:
        self.embed_model = embed_model or config.EMBEDDING_MODEL
        self.breakpoint_percentile = breakpoint_percentile

    def available(self) -> bool:
        return super().available() and (
            importlib.util.find_spec("llama_index.embeddings.huggingface") is not None
        )

    def _parser(self):
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=self.breakpoint_percentile,
            embed_model=HuggingFaceEmbedding(model_name=self.embed_model),
        )
