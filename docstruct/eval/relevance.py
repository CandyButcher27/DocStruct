"""Tool-agnostic relevance: is an answer span contained in a retrieved chunk?

The benchmark gold is ``(question, answer_span)`` where ``answer_span`` is a
verbatim snippet of source text. A retrieved chunk from ANY tool is judged
relevant if it contains that span (normalized substring), with a whitespace-blind
comparison and then a token-overlap fallback for hyphenation differences. This
makes every tool comparable regardless of how it splits the document.

Word spacing in a PDF is inferred, not stored: extractors decide where words break
by measuring inter-character gaps, and they disagree, especially in small type. The
gold spans carry whichever extractor's guesses were current when they were
generated. Scoring spacing agreement would measure which tool matches the gold
generator's tokenizer, not which tool retrieves the right content — so the
containment check also runs with all whitespace removed from both sides.
"""

from __future__ import annotations

import re

from docstruct import config

_WS = re.compile(r"\s+")

# Unicode dash and quote variants that mean the same character as their ASCII form.
# PDFs are full of them, and a model quoting a document will happily normalise a
# non-breaking hyphen (U+2011) to a plain one or the other way round. Treating
# "FA-ISS" and "FA‑ISS" as different strings rejects a correct verbatim span and
# scores a chunk that contains the answer as a miss.
_EQUIVALENTS = {
    # Dashes: hyphen, non-breaking hyphen, figure/en/em dash, horizontal bar, minus.
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "­": "",                                    # soft hyphen
    # Quotes and primes.
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"',
    # Space variants. The narrow no-break space (U+202F) in particular turns up in
    # numbers rendered as "16 KB" and is invisible in a diff.
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", "　": " ",
    "​": "", "﻿": "",                      # zero-width, BOM
}
_EQUIV_TABLE = str.maketrans(_EQUIVALENTS)


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, fold dash/quote/space variants."""
    text = (text or "").translate(_EQUIV_TABLE).replace("-\n", "")
    return _WS.sub(" ", text).strip().lower()


def _despaced(text: str) -> str:
    return _WS.sub("", normalize_text(text))


def contains_verbatim(source: str, span: str) -> bool:
    """True if span is a (normalized) substring of source — used for QA validation."""
    span_n = normalize_text(span)
    if not span_n:
        return False
    if span_n in normalize_text(source):
        return True
    # Same content, different word-break guesses ("IreneAmerini" vs "Irene Amerini").
    return _despaced(span) in _despaced(source)


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
