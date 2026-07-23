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
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from docstruct.chunking.hierarchy_builder import assign_header_levels
from docstruct.extraction.table_extractor import table_to_markdown
from docstruct.schema import Block, BoundingBox, Chunk, SectionPath

_MD_ESCAPE_RE = re.compile(r"([#|*_`])")


def _escape_md(text: str) -> str:
    """Escape Markdown control characters in body text so they render literally."""
    return _MD_ESCAPE_RE.sub(r"\\\1", text)


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
            if block.label == "figure":
                bb = block.bbox
                parts.append(
                    f"![figure](page={block.page_num} "
                    f"bbox={bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f})"
                )
                continue
            body = (block.text or "").strip()
            if not body:
                continue
            if block.label == "header":
                parts.append("#" * levels.get(block.block_id, 3) + " " + _escape_md(body))
            elif block.label == "caption":
                parts.append(f"*{_escape_md(body)}*")
            elif block.label == "table":
                parts.append(table_to_markdown(block.table_data) if block.table_data else _escape_md(body))
            else:
                parts.append(_escape_md(body))
        return "\n\n".join(parts)

    def to_markdown(self, path: Optional[str] = None) -> str:
        """Return :attr:`markdown`, optionally writing it to ``path``."""
        md = self.markdown
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(md)
        return md

    @property
    def tables(self) -> List[Tuple[Optional[List[List[str]]], int, SectionPath]]:
        """``(grid, page_num, section_path)`` for every table chunk."""
        by_id = {b.block_id: b for b in self.blocks}
        out = []
        for chunk in self.chunks_of_type("table"):
            block = by_id.get(chunk.source_block_ids[0]) if chunk.source_block_ids else None
            out.append((block.table_data if block else None, chunk.page_num, chunk.section_path))
        return out

    @property
    def figures(self) -> List[Tuple[int, BoundingBox]]:
        """``(page_num, bbox)`` for every detected figure block, in reading order."""
        return [(b.page_num, b.bbox) for b in self._ordered_blocks if b.label == "figure"]

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
    pdf_path: Union[str, Path],
    *,
    weights: Optional[str] = None,
    cache_dir: Optional[str] = None,
    password: Optional[str] = None,
) -> Document:
    """Parse a born-digital PDF into a :class:`Document`.

    ``weights`` enables hybrid mode by pointing at DocLayNet YOLOv8 weights; without
    it the pipeline runs geometry-only, which needs no model and no network.
    ``cache_dir`` caches detector output and populated blocks by PDF content hash,
    so re-parsing an unchanged file is close to free. ``password`` unlocks an
    encrypted PDF.
    """
    from docstruct.pipeline import run_pipeline

    pdf_path = str(pdf_path)
    result = run_pipeline(pdf_path, weights=weights, cache_dir=cache_dir, password=password)
    return Document(pdf_path, result.blocks, result.chunks, result.diagnostics)
