"""Re-fetch the v6 internal corpus by arXiv id, under its original filenames.

    python scripts/recover_v6_corpus.py --out data/arxiv-v6

The corpus that `benchmark_qa_v6.json` was generated against was overwritten by a
re-fetch that reused `doc<N>.pdf` names for different papers (notes.md Stage 24/25).
The PDFs were never in git, but the manifests describing them were, and
`reports/corpus_recovery_map.json` maps 56 of the 92 gold documents back to an arXiv id
-- each one verified by checking the manifest title against that document's own gold
questions rather than trusting filename order.

This downloads those 56 by id into a *new* directory. It never writes to
data/raw-pdfs/, because that corpus is the subject of a live manifest and overwriting
it again is how this happened in the first place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, "reports", "corpus_recovery_map.json")
PDF_URL = "https://arxiv.org/pdf/{aid}"
UA = "docstruct-recovery/0.1 (mailto:srivastavaaryaman555@gmail.com)"


def fetch(aid: str, tries: int = 4) -> bytes | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(PDF_URL.format(aid=aid), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as fh:
                data = fh.read()
            if data.startswith(b"%PDF"):
                return data
            return None
        except Exception as err:  # noqa: BLE001
            wait = 3 * (attempt + 1)
            print(f"    {type(err).__name__} -- retry in {wait}s", flush=True)
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/arxiv-v6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="reports/arxiv_v6_manifest.json")
    args = ap.parse_args()

    with open(MAP, encoding="utf-8") as fh:
        recovery = json.load(fh)
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)

    items = sorted(recovery.items())
    if args.limit:
        items = items[: args.limit]
    print(f"recovering {len(items)} documents into {args.out}", flush=True)

    manifest, missing = [], []
    for i, (fname, meta) in enumerate(items, 1):
        dest = os.path.join(out, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  [{i}/{len(items)}] {fname}: already present", flush=True)
        else:
            aid = meta["arxiv_id"]
            print(f"  [{i}/{len(items)}] {fname} <- arXiv:{aid}", flush=True)
            data = fetch(aid)
            if data is None:
                missing.append(fname)
                print("      FAILED", flush=True)
                continue
            with open(dest, "wb") as fh:
                fh.write(data)
            time.sleep(3)  # arXiv asks for a delay between requests
        with open(dest, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        manifest.append({
            "file": fname, "arxiv_id": meta["arxiv_id"], "title": meta.get("title", ""),
            "source": PDF_URL.format(aid=meta["arxiv_id"]), "sha256": digest,
        })

    path = os.path.join(ROOT, args.manifest)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nrecovered {len(manifest)}/{len(items)}; failed {len(missing)}")
    for f in missing:
        print(f"  missing: {f}")
    print(f"wrote {args.manifest}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
