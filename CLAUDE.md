# CLAUDE.md — DocStruct

Guidance for Claude Code sessions working in this repository.

## What this is

DocStruct is a **local, deterministic, structure-aware PDF chunking library for RAG**.
It is *not* a RAG framework. The core contract is:

> No LLM calls in the pipeline. Same PDF in → same chunks out. Fully local, auditable.

`indexing/`, `query/` and `eval/` exist to *prove* the chunks are retrieval-good,
not as product surfaces. `docstruct run` (bare chunking) and `docstruct.parse()`
are the primary entry points.

Any proposal that puts a model call inside the pipeline violates the contract and
should be rejected, not implemented. See `memory/decisions.md`.

## Read this first

The `memory/` folder is the durable context for this project. Read the file that
matches the task before writing code:

| File | Read it when |
|---|---|
| [`memory/architecture.md`](memory/architecture.md) | You need the module map, data model, or where a responsibility lives |
| [`memory/pipeline.md`](memory/pipeline.md) | You are changing detection, fusion, reading order, extraction or chunking |
| [`memory/evaluation.md`](memory/evaluation.md) | You are touching `eval/`, running benchmarks, or adding a metric |
| [`memory/results.md`](memory/results.md) | You need to know what a config value is worth, or what the current numbers are |
| [`memory/decisions.md`](memory/decisions.md) | Before proposing anything — it lists what was already tried and rejected, with measurements |
| [`memory/roadmap.md`](memory/roadmap.md) | You are picking the next piece of work |
| [`memory/conventions.md`](memory/conventions.md) | Always — commit style, test policy, how to run things |

`notes.md` is the chronological engineering log (what changed, what it measured,
whether it was kept). `implementation_plan.md` is the standing plan-vs-code audit.
`memory/` is the distilled, current-state version of both; when they disagree,
`memory/` is the one that was updated last.

## Hard rules for this repo

1. **Measure before claiming.** Any change to chunking, reading order or
   extraction is worthless until it has been run through `scripts/ablate.py` and
   compared against the current numbers in `memory/results.md`. "Should improve
   retrieval" is not a result.
2. **No LLM in the pipeline.** LLM use is confined to `eval/qa_generator.py`
   (gold generation) and is never on the parse path.
3. **All thresholds live in `config.py`.** No magic numbers in detector, fusion
   or chunking code. Values inherited from the v0 prototype are marked
   `# unvalidated` — do not silently trust them.
4. **Top-left coordinates everywhere** (`y0` = top, y increases downward),
   matching pdfplumber. Model output is transformed on the way in.
5. **Config changes carry their justification.** Every tuned constant in
   `config.py` has a comment naming the measurement that chose it. Keep it that
   way; a bare number is a regression waiting to happen.
6. **Every stage ends in a commit, and `notes.md` gets the entry.** The log is the
   product of this project as much as the code is.

## Running things

```bash
.venv/Scripts/python.exe -m pytest -q          # 119 tests, ~2.5 min
python -m docstruct.cli run data/raw-pdfs/doc1.pdf
python scripts/ablate.py --name try --set MIN_CHUNK_TOKENS=300
```

Always use the project `.venv`. The `docstruct` console-script shim can be stale
after the project directory moves — `python -m docstruct.cli` always works.
Full command reference: `memory/conventions.md`.
