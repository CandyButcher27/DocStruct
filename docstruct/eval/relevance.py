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


def _overlap_coefficient(a: str, b: str) -> float:
    """Szymkiewicz--Simpson overlap: |A n B| / min(|A|, |B|).

    Normalising by the *smaller* set is the whole point. Jaccard and the
    span-style ratio both punish a size mismatch that is expected here rather
    than informative: a 400-word chunk sitting entirely inside a 1,000-word
    evidence block is perfect evidence and scores 0.4 under either. Under the
    overlap coefficient it scores 1.0, and so does the reverse containment.
    """
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def is_relevant_region(chunk_text: str, region: str, min_overlap: float | None = None) -> bool:
    """True if the chunk substantially overlaps a *page-region* gold passage.

    Public human-annotated corpora do not mark sentence spans. FinanceBench
    evidence is the surrounding table or paragraph block — median 1,271
    characters, up to 6,362 — so `is_relevant` cannot score it. Containment
    never fires, and its token-overlap fallback cannot either: that fallback
    measures the fraction of the *gold's* tokens present in the chunk, which for
    a 1,000-word region and a 400-word chunk is capped at 0.4 no matter how
    perfectly the chunk sits inside the evidence. Lowering the threshold does not
    fix it, it just moves the arbitrary cutoff — the ratio is the wrong ratio.

    The overlap coefficient normalises by the smaller side, so a chunk contained
    in the evidence and evidence contained in a chunk both score 1.0, and a chunk
    from an unrelated part of the filing still scores near zero.

    Deliberately not page-metadata matching. That is the protocol
    `arXiv:2604.12047` uses, but it requires every adapter to attribute chunks to
    pages, and the LangChain baseline concatenates page text before splitting, so
    its chunks straddle pages by construction. Text overlap needs nothing from the
    adapter and stays tool-agnostic, which is the property the whole benchmark
    rests on.
    """
    min_overlap = config.RELEVANCE_REGION_MIN_OVERLAP if min_overlap is None else min_overlap
    return _overlap_coefficient(chunk_text, region) >= min_overlap


def is_relevant_page(chunk_pages, evidence_page: int) -> bool:
    """True if the chunk draws on the page the evidence lives on. Text-free.

    Required for gold whose spans are written against a *normalized parse* rather
    than the PDF's own text layer. OHR-Bench is the case that forced this: its
    `evidence_context` comes from human-corrected `gt_text`, which is reflowed and
    dehyphenated, so only **1.5%** of its spans appear verbatim in raw pdfplumber
    output. Every text-comparison rule then scores near-noise — the token-overlap
    fallback fires around its threshold roughly independently of chunking quality,
    which compresses the tools together and looks like a null result rather than a
    broken metric.

    Page identity survives any amount of reflowing, and it is how OHR-Bench frames
    its own evaluation. The cost is granularity: a chunk is credited for being on
    the right page, not for containing the answer, so this mode cannot distinguish
    two chunkers that both cover the page. Report it as page-level Recall@k, never
    as if it were containment.

    `chunk_pages` is whatever pages the chunk drew from — a chunk that straddles a
    page break legitimately answers for both. All page numbers are 0-based.
    """
    if chunk_pages is None:
        return False
    if isinstance(chunk_pages, int):
        chunk_pages = (chunk_pages,)
    return int(evidence_page) in {int(p) for p in chunk_pages}


# Relevance rules the benchmark can be run under:
#   span   — gold marks a verbatim sentence-level answer (our generated corpora)
#   region — gold marks the surrounding block (FinanceBench)
#   page   — gold is written against a normalized parse, so only the page id is
#            trustworthy (OHR-Bench)
RELEVANCE_MODES = {"span": is_relevant, "region": is_relevant_region, "page": is_relevant_page}

# Modes scored on page identity rather than chunk text. The benchmark has to hand
# these the chunk's pages instead of its text.
PAGE_MODES = frozenset({"page"})


def get_relevance(mode: str):
    if mode not in RELEVANCE_MODES:
        raise ValueError(f"unknown relevance mode {mode!r}; expected one of {sorted(RELEVANCE_MODES)}")
    return RELEVANCE_MODES[mode]
