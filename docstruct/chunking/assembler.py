"""Assemble reading-ordered blocks into retrieval-ready chunks.

Walks blocks in reading order maintaining a running :class:`SectionPath`:

- header     -> updates the section path; its text opens the chunk it introduces
                (``INLINE_HEADER_TEXT``), so headings stay retrievable
- text       -> accumulated under the current section up to MAX_CHUNK_TOKENS
- table      -> atomic chunk (plaintext rows), never split
- caption    -> figure_caption chunk, linked to its target figure/table
- figure     -> represented through its caption (standalone figures are skipped)
- abstract   -> emitted with chunk_type "abstract" (detected by section name)
- references -> skipped entirely

A structural boundary only *ends* the running text chunk once it holds at least
``MIN_CHUNK_TOKENS`` words; below the floor the boundary is crossed and the buffer
keeps accumulating. Tables and captions emit their own chunk without splitting the
prose around them unless ``BREAK_TEXT_ON_TABLE`` / ``BREAK_TEXT_ON_CAPTION`` say so.
Together these stop a page of prose interleaved with figures from collapsing into a
handful of unretrievable stubs.

A chunk is attributed to the section it *started* in, so crossing a header to reach
the floor never silently relabels the text that came before it.

Token counts approximate with whitespace word counts to stay tokenizer-free.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from docstruct import config
from docstruct.schema import Block, Chunk, SectionPath
from docstruct.chunking.hierarchy_builder import assign_header_levels

_REFERENCES_RE = re.compile(r"^\s*(references|bibliography|works cited)\b", re.IGNORECASE)
_ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)


def _section_names(section: SectionPath) -> List[str]:
    return [s for s in (section.h1, section.h2, section.h3) if s]


def _in_references(section: SectionPath) -> bool:
    return any(_REFERENCES_RE.match(s) for s in _section_names(section))


def _in_abstract(section: SectionPath) -> bool:
    return any(_ABSTRACT_RE.match(s) for s in _section_names(section))


def _set_section(section: SectionPath, level: int, name: str) -> None:
    if level <= 1:
        section.h1, section.h2, section.h3 = name, None, None
    elif level == 2:
        section.h2, section.h3 = name, None
    else:
        section.h3 = name


def _token_count(blocks: List[Block]) -> int:
    return sum(len((b.text or "").split()) for b in blocks)


def _table_segments(block: Block) -> List[str]:
    """Rendered text for a table block: one string, or several when TABLE_SPLIT_ROWS
    splits an oversized table by rows (repeating the header row per segment)."""
    text = block.text or ""
    grid = block.table_data
    if not (config.TABLE_SPLIT_ROWS and grid and len(grid) > 2):
        return [text] if text.strip() else []
    if len(text.split()) <= config.MAX_CHUNK_TOKENS:
        return [text]

    from docstruct.extraction.table_extractor import serialize_table

    header, body = grid[0], grid[1:]
    segments: List[str] = []
    current: List[List[str]] = [header]
    for row in body:
        trial = current + [row]
        if len(current) > 1 and len(serialize_table(trial).split()) > config.MAX_CHUNK_TOKENS:
            segments.append(serialize_table(current))
            current = [header, row]
        else:
            current = trial
    if len(current) > 1:
        segments.append(serialize_table(current))
    return segments or ([text] if text.strip() else [])


def _snapshot(section: SectionPath) -> SectionPath:
    return SectionPath(section.h1, section.h2, section.h3)


def _metadata(section: SectionPath, blocks: List[Block]) -> dict:
    finals = [b.confidence.final for b in blocks if b.confidence]
    return {
        "h1": section.h1,
        "h2": section.h2,
        "h3": section.h3,
        "mean_confidence": round(sum(finals) / len(finals), 4) if finals else None,
    }


def build_chunks(
    blocks: List[Block], levels: Optional[Dict[str, int]] = None
) -> List[Chunk]:
    """Build the chunk list from fused, reading-ordered, text-populated blocks."""
    if levels is None:
        levels = assign_header_levels(blocks)

    ordered = sorted(blocks, key=lambda b: b.reading_order)
    section = SectionPath()
    chunks: List[Chunk] = []
    buffer: List[Block] = []
    buffer_section: Optional[SectionPath] = None
    counter = 0

    def emit(
        chunk_type: str,
        content: str,
        blocks_in: List[Block],
        ids=None,
        section_path: Optional[SectionPath] = None,
    ) -> None:
        nonlocal counter
        if not content.strip():
            return
        path = section_path if section_path is not None else _snapshot(section)
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{counter:04d}",
                chunk_type=chunk_type,
                content=content.strip(),
                section_path=path,
                page_num=blocks_in[0].page_num,
                reading_order=blocks_in[0].reading_order,
                source_block_ids=ids or [b.block_id for b in blocks_in],
                metadata=_metadata(path, blocks_in),
            )
        )
        counter += 1

    def buffer_append(block: Block) -> None:
        """Add a block to the running text buffer, tracking where the chunk began."""
        nonlocal buffer_section
        if not (block.text or "").strip():
            return
        if not buffer:
            buffer_section = _snapshot(section)
        buffer.append(block)

    def flush_text(keep_overlap: bool = False, force: bool = True) -> None:
        """Emit the running text buffer.

        ``force=False`` is a structural boundary: the buffer is only cut if it
        already holds ``MIN_CHUNK_TOKENS`` words, otherwise the boundary is crossed
        and accumulation continues.
        """
        nonlocal buffer, buffer_section
        if not buffer:
            return
        path = buffer_section if buffer_section is not None else _snapshot(section)
        if _in_references(path):
            buffer = []
            buffer_section = None
            return
        if not force and _token_count(buffer) < config.MIN_CHUNK_TOKENS:
            return
        chunk_type = "abstract" if _in_abstract(path) else "text"
        content = "\n".join(b.text for b in buffer if b.text)
        emit(chunk_type, content, buffer, section_path=path)
        if keep_overlap and config.CHUNK_OVERLAP_TOKENS > 0:
            overlap: List[Block] = []
            words = 0
            for block in reversed(buffer):
                bw = len((block.text or "").split())
                if words + bw > config.CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, block)
                words += bw
            buffer = overlap
            buffer_section = _snapshot(section) if overlap else None
        else:
            buffer = []
            buffer_section = None

    def boundary_flush() -> None:
        flush_text(keep_overlap=config.OVERLAP_ON_BOUNDARY, force=False)

    for block in ordered:
        if block.label == "header":
            boundary_flush()
            _set_section(
                section,
                levels.get(block.block_id, config.HEADER_LEVELS),
                (block.text or "").strip(),
            )
            if config.INLINE_HEADER_TEXT and not _in_references(section):
                buffer_append(block)

        elif block.label == "table":
            if config.BREAK_TEXT_ON_TABLE:
                boundary_flush()
            if not _in_references(section):
                for segment in _table_segments(block):
                    emit("table", segment, [block])

        elif block.label == "caption":
            if config.BREAK_TEXT_ON_CAPTION:
                boundary_flush()
            if not _in_references(section):
                ids = [block.block_id]
                if block.caption_target_id:
                    ids.append(block.caption_target_id)
                emit("figure_caption", block.text or "", [block], ids=ids)

        elif block.label == "figure":
            continue

        else:  # text
            if _in_references(section):
                continue
            buffer_append(block)
            if _token_count(buffer) >= config.MAX_CHUNK_TOKENS:
                flush_text(keep_overlap=True)

    flush_text()

    # Tables and captions emit while the text buffer is still open, so chunks are
    # produced out of document order. Restore reading order and renumber, so
    # chunk_id ordering means what callers assume it means.
    chunks.sort(key=lambda c: c.reading_order)
    for index, chunk in enumerate(chunks):
        chunk.chunk_id = f"chunk_{index:04d}"
    return chunks
