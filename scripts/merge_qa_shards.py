"""Merge sharded `gen-qa` output into one gold file, and verify it before trusting it.

    python scripts/merge_qa_shards.py --shards data/qa/_v9_shard*.json \
        --out data/qa/benchmark_qa_v9.json

Generation is sharded across processes only for throughput; the shards hold disjoint
documents, so merging is a concatenation. The checking is the point of the script:
a gold file whose spans are not findable in the documents it names is exactly the
failure that cost this project its v6 corpus (`notes.md` Stage 24), and it is not
detectable by looking at the file.

`--verify` re-extracts every named PDF and applies the benchmark's own `span`
relevance rule -- note the argument order, `is_relevant(chunk_text, answer_span)`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from docstruct.eval.relevance import get_relevance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/qa/_v9_shard*.json")
    ap.add_argument("--out", default="data/qa/benchmark_qa_v9.json")
    ap.add_argument("--pdfs-dir", default="data/raw-pdfs")
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(ROOT, args.shards)))
    if not paths:
        raise SystemExit(f"no shards matched {args.shards}")

    items, seen = [], set()
    for p in paths:
        shard = json.load(open(p, encoding="utf-8"))
        print(f"  {os.path.basename(p)}: {len(shard)} items, "
              f"{len({i['source_doc'] for i in shard})} docs")
        for it in shard:
            key = (it["source_doc"], it["question"], it["answer_span"])
            if key in seen:
                continue
            seen.add(key)
            items.append(it)

    # data/raw-pdfs holds 102 files but only 75 distinct documents: 17 groups of
    # byte-identical PDFs stored under different doc<N>.pdf names, up to four copies
    # of one file. Left in, the same content is retrieved and scored up to four
    # times, and per-document means weight it accordingly. Keep one name per hash.
    canon, dropped = {}, []
    for name in sorted({i["source_doc"] for i in items}):
        path = os.path.join(ROOT, args.pdfs_dir, name)
        if not os.path.exists(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        digest = h.hexdigest()
        if digest in canon:
            dropped.append(f"{name} = {canon[digest]}")
        else:
            canon[digest] = name
    keep = set(canon.values())
    before = len(items)
    items = [i for i in items if i["source_doc"] in keep]
    if dropped:
        print(f"\ndeduplicated {len(dropped)} redundant files "
              f"({before - len(items)} gold items dropped):")
        for d in dropped[:20]:
            print(f"   {d}")

    docs = sorted({i["source_doc"] for i in items})
    print(f"\nmerged: {len(items)} items over {len(docs)} distinct documents")

    if args.verify:
        import pdfplumber

        is_rel = get_relevance("span")
        by_doc = collections.defaultdict(list)
        for i in items:
            by_doc[i["source_doc"]].append(i)

        hit = tot = 0
        dead = []
        for n, (doc, its) in enumerate(sorted(by_doc.items()), 1):
            path = os.path.join(ROOT, args.pdfs_dir, doc)
            if not os.path.exists(path):
                dead.append(f"{doc} (pdf absent)")
                continue
            with pdfplumber.open(path) as pdf:
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
            h = sum(1 for it in its if is_rel(text, it["answer_span"]))
            hit += h
            tot += len(its)
            if h == 0:
                dead.append(f"{doc} (0/{len(its)} findable)")
            if n % 25 == 0:
                print(f"   verified {n}/{len(by_doc)} docs, {hit}/{tot} spans", flush=True)

        print(f"\nspan-reachable: {hit}/{tot} = {100 * hit / tot:.1f}%")
        print(f"documents with no findable gold: {len(dead)}")
        for d in dead[:15]:
            print(f"   {d}")
        if hit == 0:
            raise SystemExit("nothing is reachable -- do not use this gold")

    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
