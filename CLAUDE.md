# CLAUDE.md — DocStruct

## What this is

DocStruct is a **local, deterministic, structure-aware PDF chunking library for RAG**.
It is *not* a RAG framework. The core contract is:

> No LLM calls in the pipeline. Same PDF in → same chunks out. Fully local, auditable.

`indexing/`, `query/` and `eval/` exist to *prove* the chunks are retrieval-good,
not as product surfaces. `docstruct run` and `docstruct.parse()` are the primary entry
points. Any proposal that puts a model call inside the pipeline violates the contract and
is rejected, not implemented.

`notes.md` is the chronological engineering log (what changed, what it measured, whether
it was kept). Project memory is in `mimi/` (loaded below).

## Hard rules for this repo

1. **Measure before claiming.** A change to chunking, reading order or extraction is
   worthless until `scripts/ablate.py` compares it against the current numbers.
2. **No LLM in the pipeline.** LLM use is confined to `eval/qa_generator.py`.
3. **All thresholds live in `config.py`**, each tuned one with a comment naming the
   measurement that chose it. v0 values are marked `# unvalidated`; do not trust them.
4. **Top-left coordinates everywhere** (`y0` = top), matching pdfplumber.
5. **Every stage ends in a commit, and `notes.md` gets the entry** (what changed, why,
   what it measured, kept or not). `README.md` changes whenever headline numbers move.
   Commit prefixes also include `bench:`, `data:`, `perf:`, `tune:`, `revert:`. No
   `Co-Authored-By` or AI-authorship trailers.
6. **Check every corpus's relevance rule before trusting it** — reachability, not span
   length. Run `scripts/gold_reachability.py` before a corpus's first leaderboard. A
   single-mode leaderboard is not a result: the OHR-Bench ranking inverts between modes.
7. **Gold must be tool-agnostic, and preferably not ours.** Never generate Q&A from a
   benchmarked tool's output. Prefer public human-annotated gold for headline numbers.
8. **Report the losses** in the main table, not an appendix (list in
   `mimi/memory/paper.md`).
9. **Internal arXiv corpus numbers are withdrawn** until regenerated gold lands; every
   ablation against `data/raw-pdfs/` returns MRR=0.0. A manifest is evidence about the
   files it was written from, and nothing else.
10. **Two extraction defects are open** (cross-gutter text, full-width elements sorted
    after columns). Rule 1 applies to fixing them.

## The paper

`paper/` on the ACL 2023 template, `pdflatex → bibtex → pdflatex → pdflatex`, author-year
citations. **8 pages for the whole PDF, references included.** Run
`.claude/skills/acl-paper/scripts/gate.py paper/main.tex` on every `.tex` edit and report
its counts.

## Running things

```bash
.venv/Scripts/python.exe -m pytest -q          # 220 tests (215+5 skipped), ~4 min
python -m docstruct.cli run data/raw-pdfs/doc1.pdf
python scripts/ablate.py --name try --set MIN_CHUNK_TOKENS=300
python scripts/fetch_ohrbench.py --limit 3     # also fetch_financebench.py, fetch_pmc.py
python scripts/gold_reachability.py --gold data/qa/ohrbench.json --pdfs-dir data/ohrbench
python scripts/slice_results.py --results reports/ohr_results_span.json \
    --out reports/ohr_slices_span.md
```

Always use the project `.venv`; `python -m docstruct.cli` always works (the console shim
goes stale). **Before any GPU session, run the 3-document smoke** — a green suite has
missed real CLI failures five times.

<!-- memory-layer:start -->
@mimi/MIMI.md
If mimi/MIMI.md is not already in your context, read it and mimi/STATE.md before starting any task.
<!-- memory-layer:end -->
