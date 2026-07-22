"""Fast single-tool ablation runner.

Runs the cross-tool benchmark for one adapter only, with config overrides applied
before chunking, and prints the headline metrics plus the worst documents. Used to
measure one chunking change at a time without re-running the (unchanged) baselines.

    python scripts/ablate.py --name minchunk120 --set MIN_CHUNK_TOKENS=120
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct import config  # noqa: E402
from docstruct.eval.adapters import get_adapters  # noqa: E402
from docstruct.eval.benchmark import benchmark_tool  # noqa: E402
from docstruct.eval.qa_generator import load_qa  # noqa: E402


def _coerce(raw: str):
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="label for this ablation run")
    p.add_argument("--tool", default="docstruct")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a docstruct.config value for this run")
    p.add_argument("--pdfs-dir", default="data/raw-pdfs")
    p.add_argument("--qa", default="data/qa/benchmark_qa.json")
    p.add_argument("--weights", default="weights/yolov8m-doclaynet.pt")
    p.add_argument("--cache-dir", default=".bench_cache")
    p.add_argument("--limit-docs", type=int, default=0, help="0 = all docs with questions")
    p.add_argument("--out-dir", default="reports/ablations")
    p.add_argument("--rerank-model", default=None)
    args = p.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        key = key.strip()
        if not hasattr(config, key):
            raise SystemExit(f"unknown config key: {key}")
        overrides[key] = _coerce(value)
        setattr(config, key, overrides[key])

    qa = load_qa(args.qa)
    pdfs = sorted(glob.glob(os.path.join(args.pdfs_dir, "*.pdf")))
    if args.limit_docs:
        with_qa = {q.source_doc for q in qa}
        keep = sorted(p for p in pdfs if os.path.basename(p) in with_qa)[: args.limit_docs]
        pdfs = keep
        qa = [q for q in qa if q.source_doc in {os.path.basename(p) for p in keep}]

    adapters = get_adapters([args.tool], weights=args.weights, cache_dir=args.cache_dir)
    if args.tool not in adapters:
        raise SystemExit(f"adapter unavailable: {args.tool}")

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    reranker = None
    if args.rerank_model:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(args.rerank_model)

    print(f"=== ablation '{args.name}' tool={args.tool} docs={len(pdfs)} overrides={overrides}")
    t0 = time.perf_counter()
    # cache_dir=None so the benchmark checkpoint never leaks results between ablations;
    # the adapter still gets the detector-proposal cache.
    result = benchmark_tool(adapters[args.tool], pdfs, qa, embedder,
                            cache_dir=None, reranker=reranker)
    elapsed = time.perf_counter() - t0

    print(f"\n--- {args.name} ---")
    print(f"MRR={result.mrr}  NDCG={result.ndcg}  Recall={result.recall}  Hit@1={result.hit1}")
    print(f"vector MRR={result.vec_mrr}  chunks={result.n_chunks}  "
          f"avg_words={result.mean_chunk_words}  context_words={result.context_words}  "
          f"MRR/1k={result.mrr_per_kword}  questions={result.n_questions}  "
          f"errors={result.errors}  wall={elapsed:.0f}s")
    worst = sorted(result.per_doc, key=lambda d: d["mrr"])[:10]
    print("\nworst docs:")
    for d in worst:
        print(f"  {d['doc']:12} mrr={d['mrr']:<7} recall={d['recall']:<7} "
              f"chunks={d['n_chunks']:<5} avg_words={d['avg_words_per_chunk']}")

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "name": args.name,
        "tool": args.tool,
        "overrides": overrides,
        "config": {k: v for k, v in vars(config).items() if k.isupper()},
        "metrics": {
            "mrr": result.mrr, "ndcg": result.ndcg, "recall": result.recall,
            "hit1": result.hit1, "vec_mrr": result.vec_mrr,
            "n_chunks": result.n_chunks, "mean_chunk_words": result.mean_chunk_words,
            "context_words": result.context_words, "mrr_per_kword": result.mrr_per_kword,
            "n_questions": result.n_questions, "errors": result.errors,
        },
        "per_doc": result.per_doc,
        "wall_seconds": round(elapsed, 1),
    }
    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
