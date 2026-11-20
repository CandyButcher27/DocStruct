import glob
import json
import random
from pathlib import Path

REVIEW = Path("reports/review")
CONFIGS = [(100, 500), (150, 500), (200, 500), (300, 500), (400, 500), (200, 800)]
MODES = ["span", "region", "page"]
RESAMPLES = 2000


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rr(run):
    return {(q["doc"], q["question"]): q["hyb_rr"] for q in run["per_question"]}


def paired(a, b, rng):
    d = [a[q] - b[q] for q in a if q in b]
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(RESAMPLES))
    return sum(d) / len(d), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def fmt_ci(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def merge_shards(mode):
    runs = [load(p) for p in sorted(glob.glob(str(REVIEW / f"docling/docling_s*_{mode}.json")))]
    pq = [q for r in runs for q in r["per_question"]]
    n = len(pq)
    words = sum(r["metrics"]["context_words"] * r["metrics"]["n_questions"] for r in runs)
    return {
        "per_question": pq,
        "mrr": sum(q["hyb_rr"] for q in pq) / n,
        "n_questions": n,
        "n_chunks": sum(r["metrics"]["n_chunks"] for r in runs),
        "context_words": words / n,
        "errors": sum(r["metrics"]["errors"] for r in runs),
    }


def main():
    rng = random.Random(0)
    base = {m: load(REVIEW / f"floor_ohr/floor_min200_max500_{m}.json") for m in MODES}
    out = ["# Review 1 follow-up runs", "",
           "GitHub Actions run 44739832582 (OHR-Bench, 95 docs, 3,558 questions, one CPU install) "
           "and a local PMC run. Diffs are paired bootstrap over questions "
           f"({RESAMPLES} resamples, seed 0), 95% interval. Built by `scripts/review_summary.py`.", ""]

    out += ["## Chunk-size floor on OHR-Bench (docstruct_geo)", "",
            "Diff = row MRR minus the shipped 200/500 setting.", "",
            "| MIN/MAX | chunks | words/query (span) | span MRR | region MRR | page MRR | span diff | region diff |",
            "|---|---|---|---|---|---|---|---|"]
    for lo, hi in CONFIGS:
        r = {m: load(REVIEW / f"floor_ohr/floor_min{lo}_max{hi}_{m}.json") for m in MODES}
        diffs = ["" if (lo, hi) == (200, 500) else fmt_ci(paired(rr(r[m]), rr(base[m]), rng))
                 for m in ("span", "region")]
        out.append(f"| {lo}/{hi} | {r['span']['metrics']['n_chunks']} | {r['span']['metrics']['context_words']:.0f} | "
                   + " | ".join(f"{r[m]['metrics']['mrr']:.4f}" for m in MODES) + f" | {diffs[0]} | {diffs[1]} |")

    out += ["", "## Chunk-size floor on PMC section boundaries (docstruct_geo)", "",
            "Pk and WindowDiff are error rates (lower is better).", "",
            "| MIN/MAX | Pk | WindowDiff | straddle | chunks/doc | docs |", "|---|---|---|---|---|---|"]
    for lo, hi in CONFIGS:
        r = load(REVIEW / f"floor_pmc/pmc_min{lo}_max{hi}.json")["results"]["docstruct_geo"]
        out.append(f"| {lo}/{hi} | {r['pk']:.4f} | {r['windowdiff']:.4f} | {r['straddle_rate']:.4f} | "
                   f"{r['mean_chunks']} | {r['n_docs']} |")

    out += ["", "## Docling (8 shards merged) against docstruct_geo 200/500", "",
            "| mode | Docling MRR | chunks | words/query | errors | docstruct_geo MRR | geo minus Docling |",
            "|---|---|---|---|---|---|---|"]
    for m in MODES:
        d = merge_shards(m)
        diff = paired(rr(base[m]), {(q["doc"], q["question"]): q["hyb_rr"] for q in d["per_question"]}, rng)
        out.append(f"| {m} | {d['mrr']:.4f} | {d['n_chunks']} | {d['context_words']:.0f} | {d['errors']} | "
                   f"{base[m]['metrics']['mrr']:.4f} | {fmt_ci(diff)} |")

    out += ["", "## Baselines resized to docstruct_geo's context budget", "",
            "| tool | setting | words/query | span MRR | region MRR | geo minus tool (span) | geo minus tool (region) |",
            "|---|---|---|---|---|---|---|"]
    for tool in ("unstructured", "langchain", "llamaindex"):
        r = {m: load(REVIEW / f"resized/resized_{tool}_{m}.json") for m in ("span", "region")}
        setting = ", ".join(f"{k}={v}" for k, v in r["span"]["adapter_args"].items())
        diffs = [fmt_ci(paired(rr(base[m]), rr(r[m]), rng)) for m in ("span", "region")]
        out.append(f"| {tool} | {setting} | {r['span']['metrics']['context_words']:.0f} | "
                   f"{r['span']['metrics']['mrr']:.4f} | {r['region']['metrics']['mrr']:.4f} | {diffs[0]} | {diffs[1]} |")

    (REVIEW / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
