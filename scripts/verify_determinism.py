"""Corpus-scale determinism evidence: same PDF in, same chunks out.

    python scripts/verify_determinism.py --pdfs-dir data/ohrbench --runs 2

The claim "the same PDF yields the same chunks" is the first line of this project's
contract and appears in the paper's abstract. Until now the only evidence was
`tests/test_golden_and_fuzz.py::test_parse_is_deterministic` -- one PDF, parsed twice,
*inside one process*. That cannot see anything that varies across process boundaries:
hash seeds, dict/set iteration order over addresses, thread scheduling, or a model's
kernel selection. So every parse here runs in a **fresh subprocess**.

What is hashed is the chunk *boundary structure* plus the content of each chunk, which
is what a downstream index would key on. Two runs agree only if every chunk is identical
in order, page, section path and text.

Reports per-document agreement so a single flaky document is visible rather than
collapsing the whole corpus into one FAIL.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import hashlib, json, sys
sys.path.insert(0, {root!r})
from docstruct.pipeline import run_pipeline
res = run_pipeline({pdf!r}, weights={weights!r})
h = hashlib.sha256()
for c in res.chunks:
    sp = c.section_path
    h.update(("|".join([
        c.chunk_id, c.chunk_type, str(c.page_num), str(c.reading_order),
        str(sp.h1), str(sp.h2), str(sp.h3),
        hashlib.sha256(c.content.encode("utf-8")).hexdigest(),
    ])).encode("utf-8"))
print(json.dumps({{"n_chunks": len(res.chunks), "n_blocks": len(res.blocks),
                   "digest": h.hexdigest()}}))
"""


def parse_once(pdf: str, weights: str | None, timeout: int) -> dict:
    code = CHILD.format(root=ROOT, pdf=pdf, weights=weights)
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if out.returncode != 0:
        return {"error": (out.stderr or "").strip().splitlines()[-1][:160] if out.stderr else "rc!=0"}
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return {"error": "no output"}
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError:
        return {"error": "bad json: " + line[-1][:120]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs-dir", default="data/ohrbench")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=None,
                    help="omit for geometry-only (CPU); pass weights to test the hybrid path")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="reports/determinism.json")
    args = ap.parse_args()

    pdfs = sorted(glob.glob(os.path.join(ROOT, args.pdfs_dir, "*.pdf")))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no PDFs under {args.pdfs_dir}")
        return 1

    weights = args.weights
    if weights and not os.path.isabs(weights):
        weights = os.path.join(ROOT, weights)
    mode = "hybrid (geometry + vision)" if weights else "geometry-only"
    print(f"{len(pdfs)} documents x {args.runs} runs, {mode}, "
          f"{args.workers} workers, one subprocess per parse", flush=True)

    t0 = time.perf_counter()
    results: dict[str, list[dict]] = {os.path.basename(p): [] for p in pdfs}
    jobs = [(p, r) for r in range(args.runs) for p in pdfs]
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_once, p, weights, args.timeout): p for p, _ in jobs}
        for fut in cf.as_completed(futs):
            pdf = futs[fut]
            results[os.path.basename(pdf)].append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} parses "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    identical, differing, failed = [], [], []
    for name, rs in results.items():
        errs = [r for r in rs if "error" in r]
        if errs or len(rs) < args.runs:
            failed.append((name, errs[0].get("error", "?") if errs else "missing run"))
            continue
        digests = {r["digest"] for r in rs}
        (identical if len(digests) == 1 else differing).append(name)

    scored = len(identical) + len(differing)
    total_chunks = sum(rs[0]["n_chunks"] for n, rs in results.items()
                       if n in identical or n in differing)
    summary = {
        "pdfs_dir": args.pdfs_dir, "mode": mode, "runs": args.runs,
        "n_documents": len(pdfs), "n_scored": scored,
        "identical": len(identical), "differing": len(differing),
        "failed": len(failed), "total_chunks_per_run": total_chunks,
        "agreement_pct": round(100.0 * len(identical) / scored, 2) if scored else 0.0,
        "differing_docs": sorted(differing)[:50],
        "failed_docs": [{"doc": d, "error": e} for d, e in sorted(failed)[:50]],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{mode}: {len(identical)}/{scored} documents byte-identical across "
          f"{args.runs} independent processes ({summary['agreement_pct']}%)")
    print(f"  {total_chunks} chunks per run; {len(failed)} document(s) failed to parse")
    for d in sorted(differing)[:10]:
        print(f"  DIFFERS: {d}")
    for d, e in sorted(failed)[:10]:
        print(f"  FAILED : {d} -- {e}")
    print(f"wrote {args.out}")
    return 0 if not differing else 2


if __name__ == "__main__":
    sys.exit(main())
