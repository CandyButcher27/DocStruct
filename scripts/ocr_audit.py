"""Per-page OCR-need verdicts for a corpus, without running OCR.

pymupdf4llm ships an ONNX classifier that decides, page by page, whether the
text layer is bad enough to warrant OCR. We disable OCR in the adapter (born-
digital corpora, and no other tool in the comparison OCRs), but the verdict is
itself a useful signal: it is a tool-agnostic, per-page measure of how
extractable a page's text is. This records it so we can cite born-digital-ness
instead of asserting it.

Runs the classifier only -- no OCR engine is invoked.

    python scripts/ocr_audit.py --pdfs-dir data/raw-pdfs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs-dir", default="data/raw-pdfs")
    p.add_argument("--limit-docs", type=int, default=0)
    p.add_argument("--out", default="reports/ocr_audit.json")
    args = p.parse_args()

    import pymupdf
    from pymupdf4llm.ocr.analyze_page import analyze_page

    pdfs = sorted(glob.glob(os.path.join(args.pdfs_dir, "*.pdf")))
    if args.limit_docs:
        pdfs = pdfs[: args.limit_docs]
    if not pdfs:
        print(f"no PDFs in {args.pdfs_dir}")
        return 1

    docs = {}
    for n, path in enumerate(pdfs, 1):
        name = os.path.basename(path)
        flagged = []
        pages = []
        with pymupdf.open(path) as doc:
            for page in doc:
                a = analyze_page(page)
                needs = bool(a.get("needs_ocr", False))
                if needs:
                    flagged.append(page.number)
                pages.append(
                    {
                        "page": page.number,
                        "needs_ocr": needs,
                        "ocr_spans": int(a.get("ocr_spans", 0)),
                        "chars_total": int(a.get("chars_total", 0)),
                        "chars_bad": int(a.get("chars_bad", 0)),
                        "img_area": round(float(a.get("img_area", 0.0)), 4),
                        "txt_area": round(float(a.get("txt_area", 0.0)), 4),
                    }
                )
        docs[name] = {
            "page_count": len(pages),
            "flagged_pages": flagged,
            "flagged_frac": round(len(flagged) / len(pages), 4) if pages else 0.0,
            "pages": pages,
        }
        print(f"  [{n}/{len(pdfs)}] {name}: {len(flagged)}/{len(pages)} pages flagged")

    total_pages = sum(d["page_count"] for d in docs.values())
    total_flagged = sum(len(d["flagged_pages"]) for d in docs.values())
    out = {
        "pdfs_dir": args.pdfs_dir,
        "doc_count": len(docs),
        "page_count": total_pages,
        "flagged_pages": total_flagged,
        "flagged_frac": round(total_flagged / total_pages, 4) if total_pages else 0.0,
        "docs": docs,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n{total_flagged}/{total_pages} pages flagged -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
