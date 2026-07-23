"""Extraction fidelity: how much of the document survives into a tool's chunks.

This measures **extraction**, not retrieval, and it needs no annotation and no
LLM — the source PDF is its own ground truth. That makes it the one cross-tool
quality signal available at the scale of the whole corpus, where hand-annotated
detection boxes exist for two documents.

Two numbers per tool, both against raw pdfplumber page text:

- **coverage** — fraction of the document's word *instances* that appear in some
  chunk. Answers "what did this tool silently drop?" Dropped table rows, headings
  that live in no chunk, and skipped figures all show up here and nowhere else in
  the benchmark.
- **duplication** — total chunk words divided by document words. Above 1.0 means
  content is emitted more than once, which inflates an index and lets two chunks
  split the evidence for a query between them. Deliberate overlap raises it, so it
  is a cost to read next to coverage, not a defect on its own.

Counting is multiset-based (``Counter`` intersection), so a word that appears
five times in the document and once in the chunks counts as one of five covered,
not as covered. A set-based version would score a tool that dropped every repeat
of a term as perfect.

Comparison is whitespace- and case-normalised for the same reason the relevance
check is: word breaks in a PDF are inferred from character gaps, extractors
disagree about them, and scoring that disagreement would measure tokenizer
agreement rather than what content was preserved.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def raw_document_text(pdf_path: str) -> str:
    """The document's own text, straight from pdfplumber defaults.

    Deliberately *not* DocStruct's tuned extraction settings. The reference has to
    be independent of the tool being measured, or DocStruct would be scored against
    its own output and would trivially win.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def text_coverage(chunk_texts: Iterable[str], reference_text: str) -> Dict[str, float]:
    """Word-instance coverage and duplication of ``chunk_texts`` against a reference."""
    reference = Counter(_words(reference_text))
    total = sum(reference.values())
    produced = Counter()
    for text in chunk_texts:
        produced.update(_words(text))
    produced_total = sum(produced.values())

    if total == 0:
        return {"coverage": 0.0, "duplication": 0.0, "reference_words": 0,
                "chunk_words": produced_total}

    covered = sum((reference & produced).values())
    return {
        "coverage": round(covered / total, 4),
        "duplication": round(produced_total / total, 4),
        "reference_words": total,
        "chunk_words": produced_total,
    }


def demo() -> None:
    """Self-check: the cases that would catch this being wrong."""
    doc = "The system reaches an F1 score of 0.82. The system is fast."

    # Everything preserved, nothing repeated.
    perfect = text_coverage([doc], doc)
    assert perfect["coverage"] == 1.0, perfect
    assert perfect["duplication"] == 1.0, perfect

    # Nothing produced at all.
    assert text_coverage([], doc)["coverage"] == 0.0
    assert text_coverage([""], doc)["coverage"] == 0.0

    # Emitting the document twice is full coverage at double the cost — the two
    # numbers have to move independently or duplication is not measuring anything.
    doubled = text_coverage([doc, doc], doc)
    assert doubled["coverage"] == 1.0, doubled
    assert doubled["duplication"] == 2.0, doubled

    # Multiset, not set: dropping the second "the system" is not full coverage.
    partial = text_coverage(["The system reaches an F1 score of 0.82."], doc)
    assert partial["coverage"] < 1.0, partial

    # Word-break disagreement must not count as lost content.
    assert text_coverage(["Irene Amerini wrote it"], "IreneAmerini wrote it")["coverage"] < 1.0
    assert text_coverage(["the SYSTEM is Fast"], "the system is fast")["coverage"] == 1.0

    # A tool that invents text is not penalised on coverage, only on duplication.
    invented = text_coverage([doc + " entirely new words here"], doc)
    assert invented["coverage"] == 1.0 and invented["duplication"] > 1.0, invented

    print("coverage self-check passed")


if __name__ == "__main__":
    demo()
