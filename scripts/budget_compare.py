import json
import random
from pathlib import Path

BUDGET = Path("reports/budget")
OURS = ("docstruct_geo", 5)
MATCHED = [("unstructured", 20), ("langchain", 17), ("llamaindex", 8)]
SAME_K = [("unstructured", 5), ("langchain", 5), ("llamaindex", 5)]
RESAMPLES = 2000


def load(tool, k, mode):
    r = json.loads((BUDGET / f"ohr_{tool}_k{k}_{mode}.json").read_text())["results"]
    return r[0] if isinstance(r, list) else next(iter(r.values()))


def paired(a, b, rng):
    ra = {(q["doc"], q["question"]): q["hyb_rr"] for q in a["per_question"]}
    rb = {(q["doc"], q["question"]): q["hyb_rr"] for q in b["per_question"]}
    d = [ra[q] - rb[q] for q in ra if q in rb]
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(RESAMPLES))
    return sum(d) / len(d), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1], len(d)


def main():
    rng = random.Random(0)
    lines = [
        "# Equal-context-budget comparison (OHR-Bench, 95 docs, 3,558 questions)",
        "",
        "Source: GitHub Actions run 44727565429, one CPU install for every row. "
        "Diff = docstruct_geo k=5 MRR minus the row's MRR, paired bootstrap over "
        f"questions ({RESAMPLES} resamples, seed 0), 95% interval.",
        "",
    ]
    for mode in ("span", "region"):
        ours = load(*OURS, mode)
        lines += [f"## {mode}", "", "| tool | k | context words | MRR | Recall@k | Hit@1 | diff | 95% CI | n |",
                  "|---|---|---|---|---|---|---|---|---|"]
        lines.append(f"| docstruct_geo | 5 | {ours['context_words']:.0f} | {ours['mrr']:.4f} | "
                     f"{ours['recall']:.4f} | {ours['hit1']:.4f} | | | |")
        for tool, k in MATCHED + SAME_K:
            r = load(tool, k, mode)
            diff, lo, hi, n = paired(ours, r, rng)
            lines.append(f"| {tool} | {k} | {r['context_words']:.0f} | {r['mrr']:.4f} | {r['recall']:.4f} | "
                         f"{r['hit1']:.4f} | {diff:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {n} |")
        lines.append("")
    (BUDGET / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
