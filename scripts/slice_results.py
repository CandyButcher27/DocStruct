"""Slice a benchmark run by gold metadata the leaderboard averages away.

The headline table is one number per tool. It cannot say *which questions* a tool
loses, and two of the open questions about the OHR-Bench run are exactly that:

1. `evidence_source` (text / table / equation) -- if DocStruct loses specifically
   on table-sourced questions, that justifies work on the table path. If it loses
   uniformly, it does not.
2. document position -- DocStruct drops back matter by design, so gold late in a
   document may be structurally unreachable for us and reachable for everyone
   else. Measured as MRR by decile of `page_num / doc_pages`.

Joins `per_question` rows to the gold on (source_doc, question); no re-run needed.

    python scripts/slice_results.py --results reports/ohr_results_span.json \
        --gold data/qa/ohrbench.json --out reports/ohr_slices_span.md
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d


def mean(xs):
    return round(statistics.fmean(xs), 4) if xs else None


def table(rows, head):
    widths = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
                     + [line(r) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/ohr_results_span.json")
    ap.add_argument("--gold", default="data/qa/ohrbench.json")
    ap.add_argument("--manifest", default="reports/ohrbench_manifest.json")
    ap.add_argument("--out", default="reports/ohr_slices.md")
    args = ap.parse_args()

    res = load(args.results)
    gold = load(args.gold)
    gold = gold["items"] if isinstance(gold, dict) else gold
    pages = {m["file"]: m["pages"] for m in load(args.manifest)}

    meta = {(g["source_doc"], g["question"]): g for g in gold}

    out = [f"# Slices of `{args.results}`",
           "",
           f"relevance mode: **{res['meta'].get('relevance')}**  |  "
           f"{res['meta'].get('n_questions')} questions, {res['meta'].get('n_docs')} docs",
           ""]

    sources = sorted({g.get("evidence_source", "?") for g in gold})
    domains = sorted({g.get("domain", "?") for g in gold})

    for field, keys, title in [("evidence_source", sources, "MRR by evidence source"),
                               ("domain", domains, "MRR by domain")]:
        rows = []
        for tool in res["results"]:
            by = collections.defaultdict(list)
            unmatched = 0
            for q in tool["per_question"]:
                g = meta.get((q["doc"], q["question"]))
                if g is None:
                    unmatched += 1
                    continue
                by[g.get(field, "?")].append(q["hyb_rr"])
            rows.append([tool["name"]] + [mean(by.get(k, [])) for k in keys]
                        + [len(by.get(keys[0], [])) and unmatched])
        out += [f"## {title}", "",
                table(rows, ["tool"] + [f"{k} (n={sum(1 for g in gold if g.get(field) == k)})"
                                        for k in keys] + ["unjoined"]), ""]

    # back-matter check: is gold late in a document harder for us specifically?
    rows = []
    bins = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    for tool in res["results"]:
        by = collections.defaultdict(list)
        for q in tool["per_question"]:
            g = meta.get((q["doc"], q["question"]))
            n = pages.get(q["doc"])
            if g is None or not n:
                continue
            frac = int(g["page_num"]) / n
            by[bins[min(int(frac * 5), 4)]].append(q["hyb_rr"])
        rows.append([tool["name"]] + [mean(by.get(b, [])) for b in bins])
    out += ["## MRR by relative position in the document",
            "",
            "Gold in the last fifth is the back-matter check: DocStruct drops references "
            "by design, so a falling right-hand column for us and a flat one for everyone "
            "else is that design decision showing up as a measured cost.",
            "",
            table(rows, ["tool"] + bins), ""]

    text = "\n".join(out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
