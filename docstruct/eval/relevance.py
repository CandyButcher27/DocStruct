"""Tool-agnostic relevance: is an answer span contained in a retrieved chunk?

The benchmark gold is ``(question, answer_span)`` where ``answer_span`` is a
verbatim snippet of source text. A retrieved chunk from ANY tool is judged
relevant if it contains that span (normalized substring), with a token-overlap
fallback for minor whitespace/hyphenation differences. This makes every tool
comparable regardless of how it splits the document.
"""

from __future__ import annotations

import re

from docstruct import config

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop soft hyphens."""
    text = (text or "").replace("­", "").replace("-\n", "")
    return _WS.sub(" ", text).strip().lower()


def contains_verbatim(source: str, span: str) -> bool:
    """True if span is a (normalized) substring of source — used for QA validation."""
    span_n = normalize_text(span)
    if not span_n:
        return False
    return span_n in normalize_text(source)


def _token_overlap(span: str, chunk: str) -> float:
    span_tokens = normalize_text(span).split()
    if not span_tokens:
        return 0.0
    chunk_tokens = set(normalize_text(chunk).split())
    hits = sum(1 for t in span_tokens if t in chunk_tokens)
    return hits / len(span_tokens)


def is_relevant(chunk_text: str, answer_span: str, min_overlap: float | None = None) -> bool:
    """True if the chunk contains the answer span (substring or token-overlap)."""
    if contains_verbatim(chunk_text, answer_span):
        return True
    min_overlap = config.RELEVANCE_MIN_OVERLAP if min_overlap is None else min_overlap
    return _token_overlap(answer_span, chunk_text) >= min_overlap
