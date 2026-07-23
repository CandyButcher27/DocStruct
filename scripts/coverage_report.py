"""Cross-tool extraction-fidelity report: what does each chunker actually keep?

Chunking only — no embedder, no retriever, no gold. Each PDF is its own reference
(raw pdfplumber text), so this runs on any corpus with no annotation and answers a
question the retrieval benchmark cannot: what content did a tool silently drop, and
how much did it emit twice?

    python scripts/coverage_report.py --limit-docs 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.eval.adapters import get_adapters  # noqa: E402
from docstruct.eval.coverage import raw_document_text, text_coverage  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs-dir", default="data/raw-pdfs")
    p.add_argument("--tools", default="docstruct,docstruct_geo,pymupdf4llm,langchain,unstructured,docling")
    p.add_argument("--weights", default="weights/yolov8m-doclaynet.pt")
    p.add_argument("--cache-dir", default=".bench_cache")
    p.add_argument("--limit-docs", type=int, default=20,
                   help="coverage varies little between documents; a subset is usually enough")
    p.add_argument("--out", default="reports/coverage_report.md")
    p.add_argument("--out-json", default="reports/coverage_results.json")
    args = p.parse_args()

    pdfs = sorted(glob.glob(os.path.join(args.pdfs_dir, "*.pdf")))
    if args.limit_docs:
        pdfs = pdfs[: args.limit_docs]
    if not pdfs:
        raise SystemExit(f"no PDFs in {args.pdfs_dir}")

    adapters = get_adapters(args.tools.split(","), weights=args.weights, cache_dir=args.cache_dir)
    print(f"tools: {list(adapters)}  docs: {len(pdfs)}", flush=True)

    # Extracted once and shared: the reference is a property of the PDF, not the tool.
    references = {pdf: raw_document_text(pdf) for pdf in pdfs}

    results = {}
    for name, adapter in adapters.items():
        print(f"\n=== {name} ===", flush=True)
        per_doc = []
        t0 = time.perf_counter()
        for pdf in pdfs:
            doc_id = os.path.basename(pdf)
            try:
                chunks = adapter.chunk(pdf)
            except Exception as err:  # noqa: BLE001
                print(f"  {doc_id}: ERROR — {err}", flush=True)
                continue
            stat = text_coverage([c.text for c in chunks], references[pdf])
            stat["doc"] = doc_id
            stat["n_chunks"] = len(chunks)
            per_doc.append(stat)
            print(f"  {doc_id}: coverage={stat['coverage']} duplication={stat['duplication']} "
                  f"chunks={len(chunks)}", flush=True)
        if not per_doc:
            continue
        results[name] = {
            "coverage": round(sum(d["coverage"] for d in per_doc) / len(per_doc), 4),
            "duplication": round(sum(d["duplication"] for d in per_doc) / len(per_doc), 4),
            "n_docs": len(per_doc),
            "seconds": round(time.perf_counter() - t0, 1),
            "per_doc": per_doc,
        }
        print(f"  => coverage={results[name]['coverage']}  "
              f"duplication={results[name]['duplication']}", flush=True)

    ranked = sorted(results.items(), key=lambda kv: kv[1]["coverage"], reverse=True)
    lines = [
        "# Extraction fidelity across chunkers",
        "",
        f"_{len(pdfs)} PDFs. Chunking only — no embedder, no retriever, no gold._",
        "",
        "Each document is its own reference (raw pdfplumber text, tool-independent), "
        "so this measures **extraction** rather than retrieval and needs no annotation.",
        "",
        "| Tool | Coverage | Duplication | Docs |",
        "|---|---|---|---|",
    ]
    for name, r in ranked:
        star = " **(ours)**" if name.startswith("docstruct") else ""
        lines.append(f"| {name}{star} | {r['coverage']} | {r['duplication']} | {r['n_docs']} |")
    lines += [
        "",
        "- **Coverage** — fraction of the document's word instances present in some "
        "chunk. Counted as a multiset, so dropping every repeat of a term is not "
        "scored as covered. This is where silent content loss becomes visible.",
        "- **Duplication** — chunk words over document words. Above 1.0 means content "
        "is emitted more than once, inflating the index and letting two chunks split "
        "the evidence for one query. A cost to read beside coverage, not a defect.",
        "",
        "Neither number is a ranking on its own: a tool can reach coverage 1.0 by "
        "emitting the whole document as one chunk, which is exactly the chunking "
        "failure the retrieval benchmark exists to catch. Read this table next to "
        "the leaderboard, not instead of it.",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    for name, r in ranked:
        print(f"  {name:16} coverage={r['coverage']}  duplication={r['duplication']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
