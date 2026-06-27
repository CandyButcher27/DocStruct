"""Assemble reading-ordered blocks into retrieval-ready chunks.

Walks blocks in reading order maintaining a running :class:`SectionPath`:

- header     -> updates the section path, never its own chunk
- text       -> accumulated under the current section up to MAX_CHUNK_TOKENS,
                flushed on the limit or at any section boundary
- table      -> atomic chunk (Markdown-serialized), never split
- caption    -> figure_caption chunk, linked to its target figure/table
- figure     -> represented through its caption (standalone figures are skipped)
- abstract   -> emitted with chunk_type "abstract" (detected by section name)
- references -> skipped entirely

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
    counter = 0

    def emit(chunk_type: str, content: str, blocks_in: List[Block], ids=None) -> None:
        nonlocal counter
        if not content.strip():
            return
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{counter:04d}",
                chunk_type=chunk_type,
                content=content.strip(),
                section_path=_snapshot(section),
                page_num=blocks_in[0].page_num,
                reading_order=blocks_in[0].reading_order,
                source_block_ids=ids or [b.block_id for b in blocks_in],
                metadata=_metadata(section, blocks_in),
            )
        )
        counter += 1

    def flush_text(keep_overlap: bool = False) -> None:
        nonlocal buffer
        if not buffer or _in_references(section):
            buffer = []
            return
        chunk_type = "abstract" if _in_abstract(section) else "text"
        content = "\n".join(b.text for b in buffer if b.text)
        emit(chunk_type, content, buffer)
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
        else:
            buffer = []

    for block in ordered:
        if block.label == "header":
            flush_text()
            _set_section(section, levels.get(block.block_id, config.HEADER_LEVELS), (block.text or "").strip())

        elif block.label == "table":
            flush_text()
            if not _in_references(section):
                emit("table", block.text or "", [block])

        elif block.label == "caption":
            flush_text()
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
            buffer.append(block)
            if _token_count(buffer) >= config.MAX_CHUNK_TOKENS:
                flush_text(keep_overlap=True)

    flush_text()
    return chunks
