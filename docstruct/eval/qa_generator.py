"""LLM-generated, tool-agnostic Q&A gold for retrieval benchmarking.

For sampled DocStruct chunks, an LLM writes one specific question plus a
**verbatim** answer span copied from the chunk. The span is validated to be an
actual substring of the source (hallucinated spans are rejected), so relevance
can later be judged by containment against any tool's chunks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from docstruct import config
from docstruct.schema import Chunk
from docstruct.eval.relevance import contains_verbatim

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You write evaluation questions for a document retrieval benchmark. "
    "Given one passage, produce a single specific factual question that the "
    "passage answers, and copy the exact verbatim text span (character-for-"
    "character from the passage) that answers it. The span must be 3-20 words "
    "and appear word-for-word in the passage. Respond with JSON only: "
    '{"question": "...", "answer_span": "..."}'
)


@dataclass
class QAItem:
    question: str
    answer_span: str
    source_doc: str
    source_chunk_id: str
    page_num: int
    section_path: str


def _section_str(chunk: Chunk) -> str:
    parts = [chunk.section_path.h1, chunk.section_path.h2, chunk.section_path.h3]
    return " > ".join(p for p in parts if p)


def _sample_chunks(chunks: List[Chunk], n: int) -> List[Chunk]:
    rich = [
        c for c in chunks
        if c.chunk_type in ("text", "abstract") and len(c.content.split()) >= 40
    ]
    rich.sort(key=lambda c: len(c.content), reverse=True)
    if len(rich) <= n:
        return rich
    step = len(rich) / n
    return [rich[int(i * step)] for i in range(n)]


def generate_for_chunks(
    chunks: List[Chunk], doc_id: str, client, n: int = config.QA_PER_DOC
) -> List[QAItem]:
    """Generate up to ``n`` validated QA items from a document's chunks."""
    items: List[QAItem] = []
    for chunk in _sample_chunks(chunks, n):
        try:
            out = client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f'Passage:\n"""\n{chunk.content}\n"""'},
                ],
                temperature=0.2,
            )
            question = (out.get("question") or "").strip()
            span = (out.get("answer_span") or "").strip()
        except Exception as err:  # noqa: BLE001 - generation is best-effort
            logger.warning("QA generation failed for %s: %s", chunk.chunk_id, err)
            continue

        if not question or not span:
            continue
        if not contains_verbatim(chunk.content, span):
            logger.info("rejected non-verbatim span for %s", chunk.chunk_id)
            continue

        items.append(
            QAItem(
                question=question,
                answer_span=span,
                source_doc=doc_id,
                source_chunk_id=chunk.chunk_id,
                page_num=chunk.page_num,
                section_path=_section_str(chunk),
            )
        )
    return items


def generate_for_pdf(
    pdf_path: str,
    client,
    weights: Optional[str] = None,
    n: int = config.QA_PER_DOC,
    cache_dir: Optional[str] = None,
) -> List[QAItem]:
    """Run the pipeline on a PDF and generate QA items from its chunks."""
    from docstruct.pipeline import run_pipeline

    result = run_pipeline(pdf_path, weights=weights, cache_dir=cache_dir)
    return generate_for_chunks(result.chunks, os.path.basename(pdf_path), client, n)


def save_qa(items: List[QAItem], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(it) for it in items], fh, indent=2, ensure_ascii=False)


def load_qa(path: str) -> List[QAItem]:
    with open(path, "r", encoding="utf-8") as fh:
        return [QAItem(**d) for d in json.load(fh)]
