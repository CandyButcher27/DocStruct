"""LLM-generated, tool-agnostic Q&A gold for retrieval benchmarking.

Q&A is generated from the full concatenated raw text of each PDF — NOT from
DocStruct chunks. This ensures the gold standard is tool-agnostic: no tool gets
an unfair advantage because questions were written from its own chunking view.

The LLM sees the whole document and writes N (question, verbatim_answer_span)
pairs. Each span is validated as a substring of the raw document text, so any
tool that preserves that content can score a hit.

If a document exceeds the token budget, it is split into halves and N/2 pairs
are generated from each half.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from docstruct import config
from docstruct.eval.relevance import contains_verbatim

logger = logging.getLogger(__name__)

_MIN_LINE_WORDS = 4

_SYSTEM = (
    "You write evaluation questions for a document retrieval benchmark. "
    "Given a document, produce exactly {n} specific factual questions that the "
    "document answers, each with a verbatim answer span copied character-for-"
    "character from the document text. Each span MUST be a full clause of 8-20 "
    "words appearing word-for-word in the document — never a bare name, number or "
    "noun phrase, because a short span is contained by almost every chunk and "
    "cannot discriminate between retrieval systems. Spans shorter than 6 words are "
    'discarded. Respond with JSON only: {{"items": '
    '[{{"question": "...", "answer_span": "..."}}, ...]}}'
)


@dataclass
class QAItem:
    question: str
    answer_span: str
    source_doc: str
    source_chunk_id: str  # "fulldoc" or "half_0" / "half_1"
    page_num: int          # -1 for full-doc sourced items
    section_path: str      # empty string for raw-text sourced items


def _extract_full_text(pdf_path: str) -> str:
    """Concatenate all page text, stripping short noisy lines (headers/footers)."""
    import pdfplumber

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            for line in raw.splitlines():
                if len(line.split()) >= _MIN_LINE_WORDS:
                    lines.append(line)
    return "\n".join(lines)


def _generate_from_text(text: str, doc_id: str, chunk_id: str, client, n: int) -> List[QAItem]:
    items: List[QAItem] = []
    try:
        out = client.chat_json(
            [
                {"role": "system", "content": _SYSTEM.format(n=n)},
                {"role": "user", "content": f'Document:\n"""\n{text}\n"""'},
            ],
            temperature=0.2,
            # Bulk gold generation runs straight into per-minute token limits; each
            # retry sleeps for the provider's Retry-After, so the budget is measured
            # in rate-limit windows to wait out, not in transient network blips.
            retries=6,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("QA generation failed for %s (%s): %s", doc_id, chunk_id, err)
        return []

    pairs = out.get("items", [])
    print(f"    LLM returned {len(pairs)} pairs for {doc_id} ({chunk_id})", flush=True)
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        question = (pair.get("question") or "").strip()
        span = (pair.get("answer_span") or "").strip()
        if not question or not span:
            continue
        if len(span.split()) < config.QA_MIN_SPAN_WORDS:
            logger.info("rejected short span (%d words) for %s (%s)",
                        len(span.split()), doc_id, chunk_id)
            continue
        if not contains_verbatim(text, span):
            logger.info("rejected non-verbatim span for %s (%s)", doc_id, chunk_id)
            continue
        items.append(QAItem(
            question=question,
            answer_span=span,
            source_doc=doc_id,
            source_chunk_id=chunk_id,
            page_num=-1,
            section_path="",
        ))
    return items


def _split_evenly(total: int, parts: int) -> List[int]:
    """Distribute ``total`` questions over ``parts`` segments, remainder first."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def generate_for_pdf(
    pdf_path: str,
    client,
    weights: Optional[str] = None,
    n: int = config.QA_PER_DOC,
    cache_dir: Optional[str] = None,
    max_words: Optional[int] = None,
) -> List[QAItem]:
    """Generate Q&A from full document text — tool-agnostic, no DocStruct bias.

    Documents longer than ``max_words`` are split into consecutive segments and
    the question budget is spread across them, so questions come from the whole
    document rather than only the part that fit in one request. The limit is a
    per-request one, not a context-window one: hosted providers reject oversized
    request bodies well below the model's advertised context.
    """
    doc_id = os.path.basename(pdf_path)
    full_text = _extract_full_text(pdf_path)
    if not full_text.strip():
        logger.warning("no usable text in %s", doc_id)
        return []

    budget = max_words or config.QA_MAX_WORDS_PER_REQUEST
    words = full_text.split()
    if len(words) <= budget:
        return _generate_from_text(full_text, doc_id, "fulldoc", client, n)

    parts = -(-len(words) // budget)  # ceil
    size = -(-len(words) // parts)    # even segments, none larger than budget
    items: List[QAItem] = []
    for index, want in enumerate(_split_evenly(n, parts)):
        if want <= 0:
            continue
        segment = " ".join(words[index * size : (index + 1) * size])
        items += _generate_from_text(segment, doc_id, f"part_{index}", client, want)
    return items


def save_qa(items: List[QAItem], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(it) for it in items], fh, indent=2, ensure_ascii=False)


def load_qa(path: str) -> List[QAItem]:
    with open(path, "r", encoding="utf-8") as fh:
        return [QAItem(**d) for d in json.load(fh)]
