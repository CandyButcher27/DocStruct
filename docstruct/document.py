"""The library-facing surface: :func:`parse` and the :class:`Document` it returns.

``run_pipeline`` returns the pipeline's own result object — blocks, chunks and
fusion diagnostics — which is the right shape for evaluating the pipeline and the
wrong shape for using it. This module is the thin layer in between::

    import docstruct

    doc = docstruct.parse("paper.pdf")
    doc.text                      # whole document, in reading order
    doc.markdown                  # headers, tables and captions preserved
    for chunk in doc.chunks:      # retrieval-ready, section-path annotated
        print(chunk.section_path, chunk.content)

Nothing here is new behaviour; it is the existing pipeline behind names that read
like a text-extraction library rather than like its internals.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, Iterator, List, Optional

from docstruct.chunking.hierarchy_builder import assign_header_levels
from docstruct.extraction.table_extractor import table_to_markdown
from docstruct.schema import Block, Chunk


class Document:
    """A parsed PDF: its blocks, its chunks, and views over them."""

    def __init__(self, path: str, blocks: List[Block], chunks: List[Chunk], diagnostics: Dict):
        self.path = path
        self.blocks = blocks
        self.chunks = chunks
        self.diagnostics = diagnostics

    def __repr__(self) -> str:
        return (
            f"Document({self.path!r}, pages={self.diagnostics.get('pages')}, "
            f"blocks={len(self.blocks)}, chunks={len(self.chunks)}, "
            f"mode={self.diagnostics.get('mode')!r})"
        )

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    # --- text views ------------------------------------------------------

    @property
    def _ordered_blocks(self) -> List[Block]:
        return sorted(self.blocks, key=lambda b: b.reading_order)

    @property
    def text(self) -> str:
        """Plain text of the whole document in reading order."""
        return "\n\n".join(
            b.text.strip() for b in self._ordered_blocks if (b.text or "").strip()
        )

    @property
    def markdown(self) -> str:
        """Markdown with heading levels, tables and captions preserved.

        Rendered from blocks rather than chunks: chunks are sized for retrieval and
        deliberately merge across headings, which is the wrong shape for a document
        you intend to read.
        """
        levels = assign_header_levels(self.blocks)
        parts: List[str] = []
        for block in self._ordered_blocks:
            body = (block.text or "").strip()
            if not body:
                continue
            if block.label == "header":
                parts.append("#" * levels.get(block.block_id, 3) + " " + body)
            elif block.label == "caption":
                parts.append(f"*{body}*")
            elif block.label == "table":
                parts.append(table_to_markdown(block.table_data) if block.table_data else body)
            else:
                parts.append(body)
        return "\n\n".join(parts)

    def pages(self) -> Dict[int, str]:
        """Plain text per page number, in reading order within each page."""
        out: Dict[int, List[str]] = {}
        for block in self._ordered_blocks:
            body = (block.text or "").strip()
            if body:
                out.setdefault(block.page_num, []).append(body)
        return {page: "\n\n".join(bodies) for page, bodies in sorted(out.items())}

    # --- chunk views -----------------------------------------------------

    def chunks_of_type(self, chunk_type: str) -> List[Chunk]:
        """Chunks of one kind: ``text``, ``table``, ``figure_caption``, ``abstract``."""
        return [c for c in self.chunks if c.chunk_type == chunk_type]

    def sections(self) -> List[str]:
        """Distinct section paths present, in first-appearance order."""
        seen: List[str] = []
        for chunk in self.chunks:
            path = " > ".join(
                p for p in (chunk.section_path.h1, chunk.section_path.h2, chunk.section_path.h3) if p
            )
            if path and path not in seen:
                seen.append(path)
        return seen

    # --- serialization ---------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "diagnostics": self.diagnostics,
            "chunks": [asdict(c) for c in self.chunks],
        }

    def to_json(self, path: Optional[str] = None, *, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(payload)
        return payload


def parse(
    pdf_path: str,
    *,
    weights: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Document:
    """Parse a born-digital PDF into a :class:`Document`.

    ``weights`` enables hybrid mode by pointing at DocLayNet YOLOv8 weights; without
    it the pipeline runs geometry-only, which needs no model and no network.
    ``cache_dir`` caches detector output and populated blocks by PDF content hash,
    so re-parsing an unchanged file is close to free.
    """
    from docstruct.pipeline import run_pipeline

    result = run_pipeline(pdf_path, weights=weights, cache_dir=cache_dir)
    return Document(pdf_path, result.blocks, result.chunks, result.diagnostics)
