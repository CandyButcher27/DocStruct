"""LLM-generated, tool-agnostic Q&A gold for retrieval benchmarking.

Q&A is generated from the full concatenated raw text of each PDF — NOT from
DocStruct chunks. This ensures the gold standard is tool-agnostic: no tool gets
an unfair advantage because questions were written from its own chunking view.

The LLM sees the whole document and writes N (question, verbatim_answer_span)
pairs. Each span is validated as a substring of the raw document text, so any
tool that preserves that content can score a hit.

A document larger than one request's budget is split into consecutive segments
and the question budget is spread across them, so questions come from the whole
document rather than only the part that fit.
"""

from __future__ import annotations

import json
import logging
import os
import time
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
            # Providers charge the *reserved* completion budget against the
            # per-minute token limit, not the tokens actually generated. Left
            # unset that reservation is the model's full completion length, which
            # alone can exceed the limit and make a request that can never
            # succeed however long it waits. A few hundred words of JSON is all
            # this call ever needs.
            max_tokens=1500,
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


def _pace() -> None:
    """Wait before a generation request rather than bursting into a rate limit.

    One segment of a paper is most of a free-tier minute's token allowance, so
    consecutive requests collide by construction. Waiting up front is cheaper than
    a rejected request plus the provider's Retry-After, and it stops a long run
    from exhausting the retry ceiling partway through a document — which silently
    costs that document its questions.
    """
    if config.QA_REQUEST_PACING_SECONDS > 0:
        time.sleep(config.QA_REQUEST_PACING_SECONDS)


def _spread(total: int, take: int) -> List[int]:
    """``take`` indices spread evenly over ``range(total)``, first and last included."""
    if take >= total:
        return list(range(total))
    if take <= 1:
        return [0]
    step = (total - 1) / (take - 1)
    return sorted({int(round(i * step)) for i in range(take)})


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
    max_chars: Optional[int] = None,
) -> List[QAItem]:
    """Generate Q&A from full document text — tool-agnostic, no DocStruct bias.

    Documents larger than ``max_chars`` are split into consecutive segments and the
    question budget is spread across them, so questions come from the whole document
    rather than only the part that fit in one request. The limit is a per-minute
    rate-limit budget, not a context window; see ``QA_MAX_CHARS_PER_REQUEST`` for
    why it is measured in characters.
    """
    doc_id = os.path.basename(pdf_path)
    full_text = _extract_full_text(pdf_path)
    if not full_text.strip():
        logger.warning("no usable text in %s", doc_id)
        return []

    budget = max_chars or config.QA_MAX_CHARS_PER_REQUEST
    if len(full_text) <= budget:
        _pace()
        return _generate_from_text(full_text, doc_id, "fulldoc", client, n)

    # Split on word boundaries into segments no larger than the budget, then sample
    # at most `n` of them spread evenly across the document. Requesting every
    # segment of a long paper costs a rate-limited request each for no extra gold,
    # while requesting only the first few would draw all of a document's questions
    # from its introduction.
    words = full_text.split()
    parts = -(-len(full_text) // budget)  # ceil
    size = -(-len(words) // parts)
    sampled = _spread(parts, min(parts, n))

    items: List[QAItem] = []
    for index, want in zip(sampled, _split_evenly(n, len(sampled))):
        segment = " ".join(words[index * size : (index + 1) * size])
        if not segment:
            continue
        _pace()
        # Ask for two even where the split allocates one. Spans are rejected for
        # being short, paraphrased or not verbatim, so a request for exactly one
        # question routinely yields zero and the segment contributes nothing.
        items += _generate_from_text(segment, doc_id, f"part_{index}", client, max(want, 2))
    return items[:n]


def save_qa(items: List[QAItem], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(it) for it in items], fh, indent=2, ensure_ascii=False)


def load_qa(path: str) -> List[QAItem]:
    with open(path, "r", encoding="utf-8") as fh:
        return [QAItem(**d) for d in json.load(fh)]
