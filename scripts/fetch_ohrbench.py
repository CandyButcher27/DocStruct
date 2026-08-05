"""Fetch OHR-Bench and reduce it to a DocStruct retrieval corpus.

OHR-Bench (arXiv:2412.02592, ICCV 2025, CC-BY-4.0, research-only) ships 1,261 PDFs
across 7 domains with human-written QA. We keep the part that is actually a
chunking benchmark: documents that have QA, live in a multi-page domain, and are
born-digital. See memory/benchmark-datasets.md for why each filter is there.

Needs `pyarrow` (parquet), which is not a core dependency:
    pip install pyarrow

    python scripts/fetch_ohrbench.py --limit 3     # smoke test
    python scripts/fetch_ohrbench.py               # full corpus
"""

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

HF = "https://huggingface.co/datasets/opendatalab/OHR-Bench/resolve/main"
# QA lives ONLY in the v1 parquet. v2 is larger and carries per-page gt_text for
# every document, but has no `qas` column at all.
QA_PARQUET = "OHR-Bench.parquet"
PDFS_ZIP = "pdfs.zip"

# The other three domains (news, textbook, administration) are overwhelmingly
# single-page. Retrieval within a one-page document is trivial and lifts every
# tool identically, which dilutes the benchmark rather than broadening it.
DOMAINS = {"finance", "law", "manual", "academic"}
# v1 names two domains differently from the zip and from v2.
DOMAIN_ALIAS = {"paper": "academic", "notes": "administration"}

# Mean chars on the first pages below which a PDF has no usable text layer. OCR is
# out of contract, so a scanned document is not a DocStruct input.
MIN_CHARS_PER_PAGE = 100

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ohrbench" / "_download"
PDF_DIR = ROOT / "data" / "ohrbench"
QA_OUT = ROOT / "data" / "qa" / "ohrbench.json"
MANIFEST = ROOT / "reports" / "ohrbench_manifest.json"


def download(name: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists():
        print(f"  cached {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{HF}/{name}", headers={"User-Agent": "docstruct-fetch/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            print(f"  fetched {dest.name}")
            return dest
        except OSError as err:
            # The HF CDN resets mid-transfer often enough that one attempt is not a test.
            if attempt == retries - 1:
                raise
            print(f"  retry {dest.name} after {type(err).__name__}")
    return dest


def flat_name(doc_name: str) -> str:
    """`law/Playboy... Agreement2` -> a filename the benchmark can glob and match.

    Document names carry slashes, spaces and commas; the benchmark keys questions
    to `os.path.basename(pdf)`, so the mapping has to be flat and stable.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", doc_name.replace("/", "__"))[:120] + ".pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only keep the first N documents")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    print("downloading (1.5 GB on first run)...")
    qa_path = download(QA_PARQUET, CACHE / QA_PARQUET)
    zip_path = download(PDFS_ZIP, CACHE / PDFS_ZIP)

    table = pq.read_table(qa_path, columns=["domain", "doc_name", "qas"]).to_pylist()

    # One document spans many page rows; questions hang off the pages.
    per_doc: dict[str, list[dict]] = {}
    for row in table:
        qas = row["qas"]
        if not qas or not qas.get("ID"):
            continue
        head, _, tail = row["doc_name"].partition("/")
        doc = f"{DOMAIN_ALIAS.get(head, head)}/{tail}"
        if doc.split("/")[0] not in DOMAINS:
            continue
        for i in range(len(qas["ID"])):
            per_doc.setdefault(doc, []).append(
                {
                    "question": qas["questions"][i],
                    "answer": qas["answers"][i],
                    "answer_span": qas["evidence_context"][i],
                    "page_num": int(qas["evidence_page_no"][i]),
                    "evidence_source": qas["evidence_source"][i],
                    "answer_form": qas["answer_form"][i],
                    "id": qas["ID"][i],
                }
            )

    docs = sorted(per_doc)
    if args.limit:
        docs = docs[: args.limit]
    print(f"{len(docs)} documents with QA in {sorted(DOMAINS)}")

    import pdfplumber

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(zip_path)
    members = {n[:-4]: n for n in zf.namelist() if n.lower().endswith(".pdf")}

    gold, manifest, skipped = [], [], []
    for doc in docs:
        member = members.get(doc)
        if member is None:
            skipped.append((doc, "no pdf in zip"))
            continue
        dest = PDF_DIR / flat_name(doc)
        if not dest.exists():
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        try:
            with pdfplumber.open(dest) as pdf:
                probe = pdf.pages[:3]
                chars = sum(len(p.chars) for p in probe) / max(len(probe), 1)
                pages = len(pdf.pages)
        except Exception as err:                       # a corrupt PDF is not a corpus member
            dest.unlink(missing_ok=True)
            skipped.append((doc, type(err).__name__))
            continue
        if chars < MIN_CHARS_PER_PAGE:
            dest.unlink()
            skipped.append((doc, "scanned"))
            continue

        for q in per_doc[doc]:
            gold.append(
                {
                    "question": q["question"],
                    "answer_span": q["answer_span"],
                    "source_doc": dest.name,
                    "source_chunk_id": q["id"],
                    "page_num": q["page_num"],
                    "section_path": "",
                    "answer": q["answer"],
                    "evidence_source": q["evidence_source"],
                    "answer_form": q["answer_form"],
                    "domain": doc.split("/")[0],
                }
            )
        manifest.append(
            {
                "file": dest.name,
                "ohr_doc_name": doc,
                "domain": doc.split("/")[0],
                "pages": pages,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()[:16],
                "source": f"{HF}/{PDFS_ZIP}",
                "license": "CC-BY-4.0 (research only)",
            }
        )

    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    QA_OUT.write_text(json.dumps(gold, indent=1), encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    spans = [len(g["answer_span"].split()) for g in gold] or [0]
    by_domain: dict[str, int] = {}
    for m in manifest:
        by_domain[m["domain"]] = by_domain.get(m["domain"], 0) + 1
    print(f"\n{len(manifest)} pdfs -> {PDF_DIR}   {by_domain}")
    print(f"{len(gold)} gold rows -> {QA_OUT}")
    print(f"{sum(m['pages'] for m in manifest)} pages")
    print(f"span words: median {sorted(spans)[len(spans) // 2]}  max {max(spans)}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    main()
