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
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

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

    def to_jsonl(self, path: Optional[str] = None) -> str:
        """One chunk per line -- the shape most vector-store ingest scripts expect."""
        lines = []
        for c in self.chunks:
            lines.append(json.dumps({
                "id": c.chunk_id,
                "text": c.content,
                "metadata": self._metadata(c),
            }, ensure_ascii=False))
        payload = chr(10).join(lines)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(payload + chr(10))
        return payload

    def _metadata(self, chunk: Chunk) -> Dict:
        """Flat, JSON-safe metadata. Flat because vector stores reject nested values."""
        sp = chunk.section_path
        return {
            "source": self.path,
            "chunk_type": chunk.chunk_type,
            "page": chunk.page_num,
            "reading_order": chunk.reading_order,
            "section_h1": sp.h1,
            "section_h2": sp.h2,
            "section_h3": sp.h3,
            "section_path": " > ".join(x for x in (sp.h1, sp.h2, sp.h3) if x),
        }

    # --- framework hand-off ----------------------------------------------

    def to_langchain(self) -> List:
        """Convert to ``langchain_core.documents.Document`` objects.

        The section path travels in metadata, which is the whole point: a LangChain
        retriever can then filter or display "which section did this come from",
        which a fixed-size splitter cannot tell it.
        """
        try:
            from langchain_core.documents import Document as LCDocument
        except ImportError as err:  # pragma: no cover - depends on user env
            raise ImportError(
                "to_langchain() needs langchain-core: pip install langchain-core"
            ) from err
        return [LCDocument(page_content=c.content, metadata=self._metadata(c))
                for c in self.chunks]

    def to_llamaindex(self) -> List:
        """Convert to LlamaIndex ``TextNode`` objects, section path in metadata."""
        try:
            from llama_index.core.schema import TextNode
        except ImportError as err:  # pragma: no cover - depends on user env
            raise ImportError(
                "to_llamaindex() needs llama-index-core: pip install llama-index-core"
            ) from err
        return [TextNode(text=c.content, id_=c.chunk_id, metadata=self._metadata(c))
                for c in self.chunks]

    def stats(self) -> Dict:
        """Counts a caller needs before indexing: how many chunks, how much text,
        and how much of the document's own words survived into them."""
        words = [len(c.content.split()) for c in self.chunks]
        by_type: Dict[str, int] = {}
        for c in self.chunks:
            by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
        return {
            "path": self.path,
            "n_blocks": len(self.blocks),
            "n_chunks": len(self.chunks),
            "n_pages": len({b.page_num for b in self.blocks}),
            "chunk_words_total": sum(words),
            "chunk_words_mean": round(sum(words) / len(words), 1) if words else 0.0,
            "chunk_words_min": min(words) if words else 0,
            "chunk_words_max": max(words) if words else 0,
            "chunks_by_type": by_type,
            "n_sections": len(self.sections()),
        }


def parse(
    pdf_path: Union[str, Path],
    *,
    weights: Optional[str] = None,
    cache_dir: Optional[str] = None,
    password: Optional[str] = None,
    config: Optional[Dict[str, object]] = None,
    on_page: Optional[Callable[[int, int], None]] = None,
) -> Document:
    """Parse a born-digital PDF into a :class:`Document`.

    ``weights`` enables hybrid mode by pointing at DocLayNet YOLOv8 weights; without
    it the pipeline runs geometry-only, which needs no model and no network.
    ``cache_dir`` caches detector output and populated blocks by PDF content hash,
    so re-parsing an unchanged file is close to free. ``password`` unlocks an
    encrypted PDF. ``config`` is a mapping of ``config.py`` overrides applied only for
    this call (thread-safe, no permanent global mutation), e.g.
    ``parse("x.pdf", config={"MIN_CHUNK_TOKENS": 300})``.
    """
    from docstruct.pipeline import run_pipeline

    pdf_path = str(pdf_path)
    result = run_pipeline(
        pdf_path, weights=weights, cache_dir=cache_dir, password=password,
        config=config, on_page=on_page,
    )
    return Document(pdf_path, result.blocks, result.chunks, result.diagnostics)


def parse_bytes(
    data: bytes,
    *,
    name: str = "<bytes>",
    **kwargs,
) -> Document:
    """Parse a PDF held in memory.

    A web service receiving an upload has bytes, not a path, and writing them to a
    temp file only to have the parser read them back is a round trip through the
    filesystem for nothing. pdfplumber and PyMuPDF both open file-like objects, but
    the pipeline is path-oriented throughout (the content-hash cache keys on the file,
    the detector rasterises by path), so this writes one temp file and cleans it up.
    That is honest about the current design rather than pretending to be zero-copy.

    ``name`` is what appears as ``Document.path``; give it the original filename so
    diagnostics and chunk ids stay meaningful.
    """
    import os
    import tempfile

    if not data.startswith(b"%PDF"):
        from docstruct.errors import InvalidPDFError
        raise InvalidPDFError(f"{name}: not a PDF (no %PDF header)")

    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        doc = parse(tmp, **kwargs)
        doc.path = name
        return doc
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def parse_many(
    pdf_paths,
    *,
    workers: Optional[int] = None,
    on_error: str = "raise",
    **kwargs,
):
    """Parse several PDFs across processes, yielding ``(path, Document | Exception)``.

    Processes rather than threads: parsing is CPU-bound in pdfplumber and, in hybrid
    mode, in the model, so threads would serialise on the GIL.

    ``on_error="raise"`` propagates the first failure; ``on_error="return"`` yields the
    exception in place of the Document, which is what a batch job over a real corpus
    wants -- one malformed PDF in 500 should not lose the other 499.

    Results arrive in completion order, not input order. Sort by path if you need
    determinism across the *batch*; determinism of each document's chunks is
    guaranteed either way.
    """
    import concurrent.futures as cf

    paths = [str(p) for p in pdf_paths]
    if on_error not in ("raise", "return"):
        raise ValueError("on_error must be 'raise' or 'return'")

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(parse, p, **kwargs): p for p in paths}
        for fut in cf.as_completed(futures):
            path = futures[fut]
            try:
                yield path, fut.result()
            except Exception as err:  # noqa: BLE001
                if on_error == "raise":
                    raise
                yield path, err
