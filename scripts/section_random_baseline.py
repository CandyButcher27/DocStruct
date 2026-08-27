"""Random and evenly-spaced boundary baselines for the PMC section metric.

Pk and WindowDiff both reward matching the gold segment count, so a tool that emits
roughly the right number of boundaries scores well before any of them land in the
right place. This puts a floor under Table 3: place N boundaries with no knowledge of
the document and score them on the same spine and the same located gold.

    python scripts/section_random_baseline.py --trials 20

Two floors per count: uniform random placement (averaged over --trials) and evenly
spaced placement. A tool only earns its row if it beats both at its own chunk count.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.eval.segmentation import (  # noqa: E402
    boundary_mask,
    cached_spine,
    locate_boundaries,
    pk_windowdiff,
)

_MIN_LOCATED = 0.5


def _score(spine_len, located, offsets):
    ref = boundary_mask(spine_len, located)
    hyp = boundary_mask(spine_len, offsets)
    return pk_windowdiff(ref, hyp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/qa/pmc_sections.json")
    ap.add_argument("--pdfs-dir", default="data/pmc")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--counts", default="17.7,26.8,29.1,37.5,42.7,85.6,106.9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/section_random_baseline.json")
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)

    prepared = {}
    for doc in sorted(gold):
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
        prepared[doc] = (len(spine), located)
    print(f"{len(prepared)} documents scoreable")

    rng = random.Random(args.seed)
    results = {}
    for count in [float(c) for c in args.counts.split(",")]:
        n = max(2, round(count))
        rand_pk, rand_wd = [], []
        even_pk, even_wd = [], []
        for spine_len, located in prepared.values():
            trials_pk, trials_wd = [], []
            for _ in range(args.trials):
                offs = sorted(rng.sample(range(1, spine_len), min(n - 1, spine_len - 2)))
                m = _score(spine_len, located, [0] + offs)
                trials_pk.append(m["pk"])
                trials_wd.append(m["windowdiff"])
            rand_pk.append(statistics.fmean(trials_pk))
            rand_wd.append(statistics.fmean(trials_wd))

            step = spine_len / n
            offs = [round(i * step) for i in range(n)]
            m = _score(spine_len, located, offs)
            even_pk.append(m["pk"])
            even_wd.append(m["windowdiff"])

        results[str(count)] = {
            "n_boundaries": n,
            "random_pk": round(statistics.fmean(rand_pk), 4),
            "random_windowdiff": round(statistics.fmean(rand_wd), 4),
            "even_pk": round(statistics.fmean(even_pk), 4),
            "even_windowdiff": round(statistics.fmean(even_wd), 4),
        }
        r = results[str(count)]
        print(f"  n={n:4d}  random Pk={r['random_pk']:.4f} WD={r['random_windowdiff']:.4f}"
              f"   even Pk={r['even_pk']:.4f} WD={r['even_windowdiff']:.4f}", flush=True)

    # The doc set is written out because the tool table and this floor must be read
    # on the same documents; a baseline scored on a different subset is not a floor.
    payload = {"n_docs": len(prepared), "trials": args.trials,
               "docs": sorted(prepared), "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
