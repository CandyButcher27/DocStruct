# Measurement environment — constraints, GPU, resume state

Operational notes from the 2026-07 Fable session. This is about *running* the eval
layer, not about the pipeline. Read it before attempting the ablation sweep or the
corpus-broadening gen-qa.

## The core finding: the eval layer is CPU-bound here, and it's the only thing that is

DocStruct's product (deterministic geometry/fusion/reading-order/chunking) is pure CPU
and fast (~0.1–0.2 s/doc). **Only the eval/measurement layer wants a GPU:**

| Stage | Where | CPU cost | GPU |
|---|---|---|---|
| YOLO layout detection | hybrid parse (`--weights`), `ModelDetector` | ~90 s/doc | ~10× |
| Embedding | `benchmark`, `ablate.py`, `index`, `query` (sentence-transformers) | ~7,100 chunks/run = minutes | ~10–50× |
| Cross-encoder rerank | `query --rerank-model` (optional) | slow/query | large |

Everything else — `parse()` geometry-only, unit tests, single-doc parsing, and
**gen-qa (runs on Ollama cloud, no local compute)** — is fine on CPU. Do not wait on
hardware for day-to-day work; reach for a GPU *only* for the sweep/benchmark.

## Why the sweep did not run

One `ablate.py` run on 92 docs is **~69 min** on this CPU (first pass YOLO + per-run
embedding of ~7,100 chunks). 14 flags ≈ **16 h**. This environment also **kills long
unattended background jobs (~hourly)**, and foreground bash caps at 10 min. So the
full sweep is not runnable here.

**Fix:** run it on a GPU — Google Colab free **T4** is enough (turns 16 h into < 1 h;
`sentence-transformers`/`ultralytics` auto-use CUDA). Colab recipe: clone repo,
`pip install -e ".[retrieval,benchmark,model]"`, upload `weights/yolov8m-doclaynet.pt`
+ `data/raw-pdfs/*.pdf` + the gold JSON, run `bash scripts/_sweep.sh`, compare each
`reports/ablations/ab_*.json` to `ab_baseline.json`, flip the winners' `config.py`
flags to default-on (and record in `results.md`).

## Why gen-qa did not finish — NOT quota, NOT doc size, NOT the model

The `gen-qa` log showed **9 docs / 45 questions completed with ZERO errors** — every
Ollama call returned pairs; no 429/413/rate-limit/timeout/traceback. It stopped
because the **environment killed the background process**, same as the sweep. User's
quota is intact and the model works.

**Resume:** `gen-qa` skips docs already in `--out` and appends, so re-running the same
command continues. Run in **small foreground-sized batches (~3 docs)** so the env's
job-kill doesn't hit, or run on a machine/Colab that won't kill it.

## Current resume state (as of this session)

- **Corpus:** 115 PDFs in `data/raw-pdfs/` (95 arXiv-era + ~20 non-arXiv fetched;
  legal/financial/medical/technical/govt/textbook — several sources rate-limited/404'd,
  so the fetch under-delivered the ~150 planned). Resume fetch: `scripts/_fetch.sh`.
- **New-doc gold:** `data/qa/benchmark_qa_v7_extra.json` holds **9/23** new docs
  (doc100–108). **14 remaining:** doc109–117, doc58, doc69, doc97–99. Resume:
  `scripts/_genqa.sh` (or per-batch gen-qa).
- **Next after gold is complete:** merge `benchmark_qa_v7_extra.json` into
  `benchmark_qa_v6.json` → v7, then a full multi-tool re-baseline on the broadened
  corpus (`docstruct benchmark`). Only then are the gated-flag ablations
  generalizable beyond arXiv.
- **Validated baseline:** `reports/ablations/ab_baseline.json` — MRR 0.8194,
  reproduces the headline docstruct 0.8203, so harness + config-aware cache are sound.
