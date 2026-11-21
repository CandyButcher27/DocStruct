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
        "mean_chunk_words": sum(r["metrics"]["mean_chunk_words"] * r["metrics"]["n_chunks"] for r in runs)
        / sum(r["metrics"]["n_chunks"] for r in runs),
        "errors": sum(r["metrics"]["errors"] for r in runs),
    }


def budget_run(tool, mode):
    r = load(f"reports/budget/ohr_{tool}_k5_{mode}.json")["results"]
    r = r[0] if isinstance(r, list) else next(iter(r.values()))
    return {"metrics": r, "per_question": r["per_question"]}


def system_runs(name):
    if name == "docstruct_geo":
        return {m: load(REVIEW / f"floor_ohr/floor_min200_max500_{m}.json") for m in MODES}
    if name == "docling":
        merged = {m: merge_shards(m) for m in MODES}
        return {m: {"metrics": merged[m], "per_question": merged[m]["per_question"]} for m in MODES}
    if name == "lumberchunker":
        return {m: load(REVIEW / f"lumberchunker/lumberchunker_{m}.json") for m in MODES}
    if name in ("unstructured", "langchain", "llamaindex"):
        runs = {m: budget_run(name, m) for m in ("span", "region")}
        runs["page"] = load(REVIEW / f"table1/t1_{name}_page.json")
        return runs
    return {m: load(REVIEW / f"table1/t1_{name}_{m}.json") for m in MODES}


def p_two_sided(a, b, rng):
    d = [a[q] - b[q] for q in a if q in b]
    means = [sum(rng.choices(d, k=len(d))) / len(d) for _ in range(RESAMPLES)]
    tail = min(sum(x <= 0 for x in means), sum(x >= 0 for x in means))
    return sum(d) / len(d), min(1.0, 2 * tail / RESAMPLES)


def table1(rng):
    from scipy.stats import spearmanr

    names = ["docstruct_geo", "pymupdf4llm", "unstructured", "langchain", "llamaindex",
             "llamaindex_semantic", "docling"]
    if (REVIEW / "lumberchunker/lumberchunker_span.json").exists():
        names.append("lumberchunker")
    runs = {n: system_runs(n) for n in names}
    rows = {}
    for n in names:
        span = runs[n]["span"]["metrics"]
        rows[n] = {"mrr": {m: runs[n][m]["metrics"]["mrr"] for m in MODES},
                   "context_words": span["context_words"], "mean_chunk_words": span["mean_chunk_words"],
                   "n_chunks": span["n_chunks"]}
    for m in MODES:
        for i, n in enumerate(sorted(names, key=lambda n: -rows[n]["mrr"][m]), 1):
            rows[n].setdefault("rank", {})[m] = i
        ours = rr(runs["docstruct_geo"][m])
        for n in names[1:]:
            diff, p = p_two_sided(ours, rr(runs[n][m]), rng)
            rows[n].setdefault("geo_minus", {})[m] = {"diff": round(diff, 4), "p": round(p, 4)}
    widths = [rows[n]["mean_chunk_words"] for n in names]
    corr = {}
    for m in MODES:
        rho, p = spearmanr(widths, [rows[n]["mrr"][m] for n in names])
        corr[m] = {"rho": round(float(rho), 3), "p": round(float(p), 3), "n": len(names)}
    (REVIEW / "table1.json").write_text(json.dumps({"rows": rows, "spearman_width_mrr": corr}, indent=2),
                                        encoding="utf-8")
    out = ["", "## Table 1 on one install (current code)", "",
           "| system | span | region | page | ctx | geo minus (span / region / page) |",
           "|---|---|---|---|---|---|"]
    for n in names:
        r = rows[n]
        cells = " | ".join(f"{r['mrr'][m]:.4f} ({r['rank'][m]})" for m in MODES)
        sig = "" if n == "docstruct_geo" else ", ".join(
            f"{r['geo_minus'][m]['diff']:+.4f} p={r['geo_minus'][m]['p']}" for m in MODES)
        out.append(f"| {n} | {cells} | {r['context_words']:.0f} | {sig} |")
    out += ["", "Spearman(mean chunk words, MRR): " + ", ".join(
        f"{m} rho={c['rho']} p={c['p']} (n={c['n']})" for m, c in corr.items())]

    gold = {(q["source_doc"], q["question"]): q for q in load("data/qa/ohrbench.json")}
    out += ["", "## Span MRR by domain and evidence type", "",
            "| system | academic | finance | law | manual | text | table | coverage | duplication |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        acc = {}
        for q in runs[n]["span"]["per_question"]:
            g = gold.get((q["doc"], q["question"]))
            if g:
                acc.setdefault(g["domain"], []).append(q["hyb_rr"])
                acc.setdefault(g["evidence_source"], []).append(q["hyb_rr"])
        per_doc = runs[n]["span"].get("per_doc") or runs[n]["span"]["metrics"].get("per_doc") or []
        if n == "docling":
            per_doc = [d for p in sorted(glob.glob(str(REVIEW / "docling/docling_s*_span.json")))
                       for d in load(p)["per_doc"]]
        cov = sum(d["coverage"] for d in per_doc) / len(per_doc)
        dup = sum(d["duplication"] for d in per_doc) / len(per_doc)
        cells = " | ".join(f"{sum(acc[k]) / len(acc[k]):.4f}" for k in
                           ("academic", "finance", "law", "manual", "text", "table"))
        out.append(f"| {n} | {cells} | {cov:.4f} | {dup:.4f} |")
    return out


def pmc_baselines(rng):
    geo = load(REVIEW / "floor_pmc/pmc_min200_max500.json")["results"]["docstruct_geo"]["per_doc"]
    out = ["", "## PMC section boundaries: Docling and LumberChunker vs docstruct_geo (same gold)", "",
           "Gold `data/qa/pmc_sections.json` (126 papers; Actions run 44773185163, 8 shards per tool). "
           "Diff = tool minus docstruct_geo on documents both scored (lower is better, so positive = "
           "docstruct_geo ahead).", "",
           "| system | Pk | WindowDiff | chunks/doc | docs | errors | Pk diff | WindowDiff diff |",
           "|---|---|---|---|---|---|---|---|",
           f"| docstruct_geo | {sum(d['pk'] for d in geo.values()) / len(geo):.4f} | "
           f"{sum(d['windowdiff'] for d in geo.values()) / len(geo):.4f} | "
           f"{sum(d['n_chunks'] for d in geo.values()) / len(geo):.1f} | {len(geo)} | | | |"]
    for tool in ("docling", "lumberchunker"):
        per_doc, errors = {}, 0
        for p in sorted(glob.glob(str(REVIEW / f"pmc_baselines/pmc_{tool}_s*.json"))):
            r = load(p)["results"].get(tool)
            if r:
                per_doc.update(r["per_doc"])
                if not Path(p.replace(".json", "_retry.json")).exists():
                    errors += r["errors"]
        both = [d for d in per_doc if d in geo]
        cells = []
        for metric in ("pk", "windowdiff"):
            d = [per_doc[x][metric] - geo[x][metric] for x in both]
            means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(RESAMPLES))
            cells.append(f"{sum(d) / len(d):+.4f} [{means[int(0.025 * RESAMPLES)]:+.4f}, "
                         f"{means[int(0.975 * RESAMPLES) - 1]:+.4f}]")
        n = len(per_doc)
        out.append(f"| {tool} | {sum(v['pk'] for v in per_doc.values()) / n:.4f} | "
                   f"{sum(v['windowdiff'] for v in per_doc.values()) / n:.4f} | "
                   f"{sum(v['n_chunks'] for v in per_doc.values()) / n:.1f} | {n} | {errors} | "
                   f"{cells[0]} | {cells[1]} |")
    return out


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

    out += table1(rng)
    out += pmc_baselines(rng)
    (REVIEW / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
