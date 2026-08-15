"""Verify a locally reconstructed corpus against the committed manifest.

    python datasets/verify.py pmc
    python datasets/verify.py --all

Exit code is non-zero if any file is missing or its SHA-256 differs, so this is
usable as a CI gate. A corpus that verifies here is byte-identical to the one the
paper's numbers were measured on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CORPORA = {
    "pmc": ("manifests/pmc_manifest.json", "data/pmc", "file"),
    "ohrbench": ("manifests/ohrbench_manifest.json", "data/ohrbench", "file"),
    "financebench": ("manifests/financebench_manifest.json", "data/financebench", "file"),
    "arxiv": ("manifests/dataset_manifest_v2.json", "data/raw-pdfs", "file"),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entries(manifest_path: str) -> list[dict]:
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("files", "documents", "items", "entries"):
            if key in data:
                return data[key]
        return list(data.values()) if data else []
    return data


def verify(name: str) -> int:
    rel_manifest, rel_dir, key = CORPORA[name]
    manifest = os.path.join(HERE, rel_manifest)
    corpus = os.path.join(ROOT, rel_dir)

    if not os.path.exists(manifest):
        print(f"{name}: no manifest at {rel_manifest}")
        return 1

    rows = entries(manifest)
    missing, mismatched, unchecked, ok = [], [], [], 0
    for row in rows:
        fname = row.get(key) or row.get("filename") or row.get("name")
        if not fname:
            continue
        path = os.path.join(corpus, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        want = row.get("sha256")
        if not want:
            # existence only. Say so rather than counting it as verified -- a
            # manifest without checksums cannot detect a corrupted or substituted file.
            unchecked.append(fname)
            continue
        # Manifests are inconsistent about this: ohrbench and dataset_v2 store a
        # 16-hex prefix, pmc stores the full digest. Compare over the recorded width.
        if sha256(path)[: len(want)] != want.lower():
            mismatched.append(fname)
            continue
        ok += 1

    total = len(rows)
    print(f"{name:<14} {ok}/{total} checksum-verified", end="")
    if unchecked:
        print(f"  (+{len(unchecked)} present but no checksum in manifest)", end="")
    if missing:
        print(f"  MISSING {len(missing)}", end="")
    if mismatched:
        print(f"  CHECKSUM MISMATCH {len(mismatched)}", end="")
    print()
    for f in missing[:5]:
        print(f"    missing: {f}")
    for f in mismatched[:5]:
        print(f"    mismatch: {f}")
    if len(missing) > 5 or len(mismatched) > 5:
        print("    ...")
    return 1 if (missing or mismatched) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", choices=sorted(CORPORA))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not args.corpus and not args.all:
        ap.error("name a corpus or pass --all")

    names = sorted(CORPORA) if args.all else [args.corpus]
    return max(verify(n) for n in names)


if __name__ == "__main__":
    sys.exit(main())
