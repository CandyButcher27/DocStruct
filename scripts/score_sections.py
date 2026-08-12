"""Score every chunker against the publisher's own section boundaries.

Pk and WindowDiff (lower is better) plus the straddle rate, per tool, over the PMC
corpus. Unlike the section-path metric this is a *comparison*: every chunker has
boundaries, so langchain, unstructured, llamaindex and pymupdf4llm are all scored
on the same gold, on the same spine, with only the chunker varying -- the same
fair-comparison rule the retrieval leaderboard runs under.

    python scripts/score_sections.py --tools docstruct_geo,langchain --limit 5
    python scripts/score_sections.py --weights weights/yolov8m-doclaynet.pt

Run `scripts/section_reachability.py` first. Gold boundaries that cannot be found
in the PDF's own text are dropped from the reference here, so a document whose
gold is largely unreachable would otherwise be scored against a reference that is
mostly holes -- and that penalty would land on whichever tool happened to place
boundaries near them.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.eval.adapters import get_adapters  # noqa: E402
from docstruct.eval.segmentation import (  # noqa: E402
    boundary_mask,
    cached_spine,
    locate_boundaries,
    pk_windowdiff,
    straddle_rate,
)

# A document needs enough locatable gold to be worth scoring. Below this the
# reference is mostly holes and Pk measures the holes.
_MIN_LOCATED = 0.5


def _write(args, prepared, results) -> str:
    order = sorted(results, key=lambda n: results[n]["windowdiff"])
    lines = [
        "# Section-boundary agreement (PMC corpus)",
        "",
        f"{len(prepared)} documents with publisher JATS gold. Pk and WindowDiff are "
        "**error** rates -- lower is better, 0.0 is perfect agreement with the "
        "publisher's own section boundaries.",
        "",
        "| Tool | WindowDiff | Pk | Straddle rate | Mean chunks | Docs | Errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in order:
        r = results[n]
        mark = " **(ours)**" if n.startswith("docstruct") else ""
        lines.append(f"| {n}{mark} | {r['windowdiff']} | {r['pk']} | {r['straddle_rate']} | "
                     f"{r['mean_chunks']} | {r['n_docs']} | {r['errors']} |")
    lines += [
        "",
        "- **WindowDiff** compares the *number* of boundaries in each window, so it "
        "penalises a tool that puts three splits where the document has one. **Pk** "
        "only asks whether the window's ends fall in the same segment, so it forgives "
        "over-segmentation; read them together.",
        "- **Straddle rate** is the fraction of chunks crossing a gold boundary. It is "
        "not an error by itself -- 57.4% of gold sections are shorter than "
        "`MIN_CHUNK_TOKENS`, so merging them is the design working as intended -- but "
        "it bounds how meaningful a per-chunk section *label* can be.",
        "- Back matter is excluded (DocStruct drops references by design), as are "
        f"documents with under {_MIN_LOCATED:.0%} of their gold locatable in the PDF's "
        "own text; see `reports/section_reachability.json`.",
    ]
    md = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_docs": len(prepared), "gold": args.gold, "results": results}, f, indent=2)
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/qa/pmc_sections.json")
    ap.add_argument("--pdfs-dir", default="data/pmc")
    ap.add_argument("--tools", default="docstruct,docstruct_geo,langchain,pymupdf4llm,unstructured")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=".cache/section_ckpt")
    ap.add_argument("--out", default="reports/section_scores.json")
    ap.add_argument("--report-md", default="reports/section_scores.md")
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    docs = sorted(gold)
    if args.limit:
        docs = docs[: args.limit]

    names = [t.strip() for t in args.tools.split(",") if t.strip()]
    adapters = get_adapters(names=names, weights=args.weights, cache_dir=args.cache_dir)
    missing = [n for n in names if n not in adapters]
    print(f"tools: {list(adapters)}" + (f"  MISSING: {missing}" if missing else ""))
    if not adapters:
        return 1

    # The spine and the gold offsets are properties of the document, not of the
    # tool, so they are computed once and shared -- exactly like _REFERENCE_CACHE
    # in the benchmark.
    prepared = {}
    for doc in docs:
        path = os.path.join(args.pdfs_dir, doc)
        if not os.path.exists(path):
            continue
        spine = cached_spine(path)
        if len(spine) < 500:
            continue
        secs = [s for s in gold[doc] if not s["in_back_matter"]]
        offsets = locate_boundaries(spine, [s["text"] for s in secs])
        located = [o for o in offsets if o is not None]
        if not secs or len(located) / len(secs) < _MIN_LOCATED or len(located) < 3:
            continue
        prepared[doc] = (path, spine, located)
    print(f"{len(prepared)}/{len(docs)} documents have enough locatable gold "
          f"(>= {_MIN_LOCATED:.0%} of body sections)\n")
    if not prepared:
        print("nothing scoreable -- run scripts/section_reachability.py and read its output")
        return 1

    results = {}
    for name, adapter in adapters.items():
        print(f"=== {name} ===", flush=True)
        # Per document, not per tool. Flushing only between tools still lost a whole
        # tool's work when this environment killed the job 40 documents into the
        # first one -- chunking 122 PDFs outlives the kill window on its own.
        ckpt_path = os.path.join(args.ckpt_dir, f"section_ckpt_{name}.json")
        ckpt = {}
        if os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            print(f"  resuming: {len(ckpt)} documents already scored")
        errors = 0
        t0 = time.perf_counter()
        for i, (doc, (path, spine, located)) in enumerate(prepared.items(), 1):
            if doc in ckpt:
                continue
            try:
                chunks = adapter.chunk(path)
            except Exception as e:  # noqa: BLE001
                print(f"  {doc}: {type(e).__name__}: {e}")
                errors += 1
                continue
            chunk_offsets = [o for o in locate_boundaries(spine, [c.text for c in chunks])
                             if o is not None]
            if len(chunk_offsets) < 2:
                errors += 1
                continue
            ref = boundary_mask(len(spine), located)
            hyp = boundary_mask(len(spine), chunk_offsets)
            m = pk_windowdiff(ref, hyp)
            ckpt[doc] = {"pk": m["pk"], "windowdiff": m["windowdiff"],
                         "straddle": straddle_rate(located, chunk_offsets, len(spine)),
                         "n_chunks": len(chunks)}
            os.makedirs(args.ckpt_dir, exist_ok=True)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(ckpt, f)
            if i % 20 == 0:
                print(f"  {i}/{len(prepared)} docs", flush=True)

        scored = [ckpt[d] for d in prepared if d in ckpt]
        pks = [c["pk"] for c in scored]
        wds = [c["windowdiff"] for c in scored]
        strads = [c["straddle"] for c in scored]
        n_chunks = [c["n_chunks"] for c in scored]
        if not pks:
            print(f"  no document scored for {name}\n")
            continue
        results[name] = {
            "pk": round(statistics.fmean(pks), 4),
            "windowdiff": round(statistics.fmean(wds), 4),
            "straddle_rate": round(statistics.fmean(strads), 4),
            "mean_chunks": round(statistics.fmean(n_chunks), 1),
            "n_docs": len(pks), "errors": errors,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        r = results[name]
        print(f"  => Pk={r['pk']}  WindowDiff={r['windowdiff']}  "
              f"straddle={r['straddle_rate']}  ({r['n_docs']} docs)\n", flush=True)
        # Flush after every tool. This environment kills long unattended jobs, and a
        # five-tool pass that dies in the fifth should not throw away the four that
        # finished -- re-running with --tools for the survivor is then cheap.
        _write(args, prepared, results)

    md = _write(args, prepared, results)
    print(md)
    print(f"\nwrote {args.report_md} and {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
