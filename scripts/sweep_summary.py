import glob
import json
import random
from pathlib import Path

SWEEP = Path("reports/sweep")
RETRIEVERS = {"dense": "vec_rr", "bm25": "bm_rr", "hybrid": "hyb_rr"}
MODES = ["span", "region", "page"]
RESAMPLES = 2000
REF = "docstruct_geo"


def load():
    runs = {}
    for p in sorted(glob.glob(str(SWEEP / "sw_*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        emb = d["config"]["EMBEDDING_MODEL"]
        runs[(d["tool"], emb, d["relevance"])] = {(q["doc"], q["question"]): q for q in d["per_question"]}
    return runs


def mean(run, key):
    return sum(q[key] for q in run.values()) / len(run)


def p_ahead(a, b, key, rng):
    d = [a[q][key] - b[q][key] for q in a if q in b]
    means = [sum(rng.choices(d, k=len(d))) / len(d) for _ in range(RESAMPLES)]
    return min(1.0, 2 * min(sum(x <= 0 for x in means), sum(x >= 0 for x in means)) / RESAMPLES)


def main():
    rng = random.Random(0)
    runs = load()
    tools = sorted({k[0] for k in runs})
    embs = sorted({k[1] for k in runs})
    out = ["# Embedder x retriever sweep (OHR-Bench)", "",
           f"MRR per tool; rank in brackets. Last column: {REF} minus the best other tool, "
           "with the paired-bootstrap p. Positive = " + REF + " ahead.", ""]
    first = total = 0
    for mode in MODES:
        for emb in embs:
            for ret, key in RETRIEVERS.items():
                cell = {t: runs[(t, emb, mode)] for t in tools if (t, emb, mode) in runs}
                if REF not in cell:
                    continue
                score = {t: mean(r, key) for t, r in cell.items()}
                order = sorted(score, key=score.get, reverse=True)
                best = next(t for t in order if t != REF)
                diff = score[REF] - score[best]
                p = p_ahead(cell[REF], cell[best], key, rng)
                total += 1
                first += order[0] == REF
                out.append(f"- {mode} / {emb} / {ret}: " + ", ".join(
                    f"{t} {score[t]:.3f} ({order.index(t) + 1})" for t in tools if t in score)
                    + f" | vs {best} {diff:+.3f} (p={p:.3f})")
    out += ["", f"{REF} ranks first in {first} of {total} cells.", ""]
    Path("reports/sweep/summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


main()
