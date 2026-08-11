"""Fetch the FinanceBench open-source split (150 QA / 84 born-digital SEC PDFs).

Public human-annotated gold, CC-BY-NC-4.0. See memory/benchmark-datasets.md.
Writes PDFs to data/financebench/ and gold in DocStruct's QA schema to
data/qa/financebench.json, plus a sha256 manifest.

    python scripts/fetch_financebench.py --limit 2      # smoke test
    python scripts/fetch_financebench.py                # full 84 docs
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
GOLD_URL = f"{RAW}/data/financebench_open_source.jsonl"
PDF_URL = RAW + "/pdfs/{name}.pdf"

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "financebench"
QA_OUT = ROOT / "data" / "qa" / "financebench.json"
MANIFEST = ROOT / "reports" / "financebench_manifest.json"


def get(url: str) -> bytes:
    # raw.githubusercontent resets the connection on urllib's default User-Agent,
    # and throttles by tearing the socket down partway through the 84-file run
    # (WinError 10054), so a bare fetch dies around the 8th PDF. Back off and retry.
    req = urllib.request.Request(url, headers={"User-Agent": "docstruct-fetch/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == 5:
                raise
            wait = 2**attempt
            print(f"  {type(e).__name__}: {e} -- retry {attempt + 1}/5 in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N documents")
    args = ap.parse_args()

    rows = [json.loads(l) for l in get(GOLD_URL).decode("utf-8").splitlines() if l.strip()]

    docs = sorted({r["doc_name"] for r in rows})
    if args.limit:
        docs = docs[: args.limit]
    keep = set(docs)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in docs:
        dest = PDF_DIR / f"{name}.pdf"
        if not dest.exists():
            dest.write_bytes(get(PDF_URL.format(name=name)))
        manifest.append(
            {
                "file": dest.name,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "source": PDF_URL.format(name=name),
                "license": "CC-BY-NC-4.0",
            }
        )

    gold = [
        {
            "question": r["question"],
            "answer_span": ev["evidence_text"],
            "source_doc": f"{r['doc_name']}.pdf",
            "source_chunk_id": r["financebench_id"],
            "page_num": ev["evidence_page_num"],
            "section_path": "",
            "question_type": r["question_type"],
        }
        for r in rows
        if r["doc_name"] in keep
        for ev in r["evidence"]
    ]

    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    QA_OUT.write_text(json.dumps(gold, indent=1), encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    spans = [len(g["answer_span"]) for g in gold]
    print(f"{len(manifest)} pdfs -> {PDF_DIR}")
    print(f"{len(gold)} gold rows -> {QA_OUT}")
    print(f"span chars: min {min(spans)} max {max(spans)} mean {sum(spans) // len(spans)}")
    print("NOTE: evidence is page-region text, not a sentence span.")
    print("      Score with page-level Recall@k, not chunk containment.")


if __name__ == "__main__":
    main()
