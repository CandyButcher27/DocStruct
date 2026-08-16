"""Assemble a corpus containing only documents whose gold is verifiably theirs.

    python scripts/build_verified_corpus.py

`data/raw-pdfs/` is the corpus broken by the re-fetch described in notes.md Stage 24:
its filenames match `benchmark_qa_v6.json` but its *contents* are different papers, so
every v6 question scores zero against it. `data/arxiv-v6/` holds the 56 documents
recovered by arXiv id, and `benchmark_qa_v7_extra.json` covers 9 genuinely new papers
that only exist in `data/raw-pdfs/`.

This writes `data/corpus-v8/` plus `data/qa/benchmark_qa_v8.json` containing the union
of the two verified sets -- and it checks reachability itself rather than trusting
either directory, because trusting a filename is what caused the problem.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from docstruct.eval.relevance import get_relevance  # noqa: E402

SOURCES = [
    ("data/qa/benchmark_qa_v6.json", "data/arxiv-v6"),
    ("data/qa/benchmark_qa_v7_extra.json", "data/raw-pdfs"),
]


def doc_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/corpus-v8")
    ap.add_argument("--out-qa", default="data/qa/benchmark_qa_v8.json")
    ap.add_argument("--min-hits", type=int, default=1,
                    help="a document is kept only if at least this many of its gold spans are findable")
    args = ap.parse_args()

    is_rel = get_relevance("span")
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    kept_items: list[dict] = []
    kept_docs, dropped = [], []

    for qa_path, pdf_dir in SOURCES:
        gold = json.load(open(os.path.join(ROOT, qa_path), encoding="utf-8"))
        by_doc = collections.defaultdict(list)
        for g in gold:
            by_doc[g["source_doc"]].append(g)

        print(f"\n== {qa_path} against {pdf_dir}: {len(by_doc)} docs, {len(gold)} items")
        for i, (doc, items) in enumerate(sorted(by_doc.items()), 1):
            src = os.path.join(ROOT, pdf_dir, doc)
            if not os.path.exists(src):
                dropped.append((doc, "pdf absent"))
                continue
            text = doc_text(src)
            # is_relevant(chunk_text, answer_span) -- text first, span second.
            hits = sum(1 for it in items if is_rel(text, it["answer_span"]))
            if hits < args.min_hits:
                dropped.append((doc, f"{hits}/{len(items)} spans findable"))
                continue
            shutil.copyfile(src, os.path.join(out_dir, doc))
            kept_items.extend(items)
            kept_docs.append((doc, hits, len(items)))
            if i % 20 == 0:
                print(f"   {i}/{len(by_doc)} checked, {len(kept_docs)} kept", flush=True)

    with open(os.path.join(ROOT, args.out_qa), "w", encoding="utf-8") as fh:
        json.dump(kept_items, fh, indent=2, ensure_ascii=False)

    hits = sum(h for _, h, _ in kept_docs)
    tot = sum(n for _, _, n in kept_docs)
    print(f"\nkept {len(kept_docs)} documents, {len(kept_items)} gold items")
    print(f"span-reachable within kept docs: {hits}/{tot} = {100 * hits / tot:.1f}%")
    print(f"dropped {len(dropped)} documents")
    for doc, why in dropped[:12]:
        print(f"  {doc}: {why}")
    if len(dropped) > 12:
        print(f"  ... and {len(dropped) - 12} more")
    print(f"\nwrote {args.out_dir}/ and {args.out_qa}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
