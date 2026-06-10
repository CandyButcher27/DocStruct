"""Baseline chunkers for comparison against structure-aware chunking.

The naive baseline mimics fixed-window splitters (e.g. LangChain defaults): it
concatenates block text in reading order and cuts it into uniform word windows,
ignoring document structure. Same :class:`Chunk` type as the real pipeline so
both feed the identical index/retrieve path for a fair comparison.
"""

from __future__ import annotations

from typing import List

from docstruct import config
from docstruct.schema import Block, Chunk, SectionPath


def naive_chunk(blocks: List[Block], max_tokens: int = config.MAX_CHUNK_TOKENS) -> List[Chunk]:
    """Fixed-size word-window chunking with no structure awareness."""
    tokens: List[tuple] = []  # (word, page_num)
    for block in sorted(blocks, key=lambda b: b.reading_order):
        if not block.text:
            continue
        for word in block.text.split():
            tokens.append((word, block.page_num))

    chunks: List[Chunk] = []
    for index in range(0, len(tokens), max_tokens):
        window = tokens[index : index + max_tokens]
        if not window:
            continue
        content = " ".join(word for word, _ in window)
        chunks.append(
            Chunk(
                chunk_id=f"naive_{len(chunks):04d}",
                chunk_type="text",
                content=content,
                section_path=SectionPath(),
                page_num=window[0][1],
                reading_order=len(chunks),
                source_block_ids=[],
                metadata={},
            )
        )
    return chunks
