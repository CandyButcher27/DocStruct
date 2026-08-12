"""Sweep a relevance mode's threshold offline, from a finished run.

`RELEVANCE_REGION_MIN_OVERLAP = 0.7` is `# unvalidated`, and every region number
this project has -- including its best result -- rides on it. Validating it by
re-running the benchmark once per candidate threshold costs a GPU session per
value, which is why it never happened.

It does not have to. Retrieval is the expensive half and the threshold does not
change it: the same chunks come back in the same order, and only the boolean
"is this one relevant" flips. So a run made with `--dump-scores` records the
continuous score behind each retrieved chunk, and this script re-thresholds it in
seconds, scoring with `metrics_from_flags` -- the benchmark's own arithmetic, not
a second copy of it.

    python -m docstruct.cli benchmark ... --relevance region --dump-scores
    python scripts/sweep_relevance_threshold.py --results reports/fb_results.json

Reading the output: a metric that climbs monotonically as the threshold falls is
not evidence for a low threshold -- at 0.0 every chunk is "relevant" and MRR is
1.0 by definition. Look instead for the plateau, the region where the ranking
between tools is stable, and prefer a value inside it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.eval.benchmark import metrics_from_flags  # noqa: E402


def rescore(per_question: list[dict], threshold: float) -> dict:
    mrr = hit1 = recall = ndcg = 0.0
    n = 0
    for q in per_question:
        scores = q.get("hyb_scores")
        if scores is None:
            continue
        rr, h1, rec, nd = metrics_from_flags([s >= threshold for s in scores])
        mrr += rr; hit1 += h1; recall += rec; ndcg += nd
        n += 1
    if not n:
        return {}
    return {"mrr": mrr / n, "hit1": hit1 / n, "recall": recall / n,
            "ndcg": ndcg / n, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="a results JSON run with --dump-scores")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    mode = data["meta"].get("relevance")
    tools = data["results"]

    missing = [t["name"] for t in tools
               if not any("hyb_scores" in q for q in t["per_question"])]
    if len(missing) == len(tools):
        print(f"no dumped scores in {args.results}.\n"
              f"Re-run the benchmark with --dump-scores (mode `page` has no threshold "
              f"and dumps nothing by design).")
        return 1
    if missing:
        print(f"warning: no scores for {missing} -- excluded\n")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    table = {}
    for t in tools:
        if t["name"] in missing:
            continue
        table[t["name"]] = {th: rescore(t["per_question"], th) for th in thresholds}

    print(f"MRR by threshold  (mode `{mode}`, {args.results})\n")
    names = list(table)
    width = max(len(n) for n in names)
    print(f"{'tool':{width}}  " + "  ".join(f"{th:>6.2f}" for th in thresholds))
    print("-" * (width + 2 + 8 * len(thresholds)))
    for name in names:
        print(f"{name:{width}}  " + "  ".join(
            f"{table[name][th].get('mrr', 0.0):>6.4f}" for th in thresholds))

    # Whether the *ranking* moves matters more than whether the scores do: a
    # threshold that reorders the leaderboard is a threshold the paper cannot leave
    # unvalidated, and one that never reorders it is one the result does not hinge on.
    print("\nranking by threshold (best first)")
    orders = {}
    for th in thresholds:
        order = sorted(names, key=lambda n: -table[n][th].get("mrr", 0.0))
        orders[th] = order
        print(f"  {th:.2f}  " + " > ".join(order))
    distinct = {tuple(o) for o in orders.values()}
    print(f"\n{len(distinct)} distinct ranking(s) across {len(thresholds)} thresholds"
          + ("  -- the ranking is threshold-independent over this range"
             if len(distinct) == 1 else "  -- the ranking DEPENDS on the threshold"))

    spread = {n: statistics.pstdev([table[n][th].get("mrr", 0.0) for th in thresholds])
              for n in names}
    print("\nMRR std-dev across thresholds: "
          + ", ".join(f"{n} {v:.4f}" for n, v in sorted(spread.items(), key=lambda kv: -kv[1])))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"results": args.results, "mode": mode,
                       "table": {n: {str(k): v for k, v in d.items()} for n, d in table.items()},
                       "rankings": {str(k): v for k, v in orders.items()}}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
