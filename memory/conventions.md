# Conventions

## Environment

Always use the project venv: `.venv/Scripts/python.exe` (Windows) — Python 3.11.

The `docstruct` console-script shim can be stale after the project directory
moves (editable-install metadata still points at the old path). **`python -m
docstruct.cli` always works** and is what to use in scripts and docs.

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                 # 107 tests, ~2.5 min

python -m docstruct.cli run data/raw-pdfs/doc1.pdf    # geometry-only
python -m docstruct.cli run paper.pdf --weights weights/yolov8m-doclaynet.pt --json out.json
python -m docstruct.cli visualize paper.pdf --out annotated.pdf

python scripts/ablate.py --name <label> --set KEY=VALUE   # one-tool ablation
python scripts/fetch_dataset_v2.py --domain arxiv          # extend the corpus
```

Full benchmark invocation: see `evaluation.md`.

## Code style

- Python 3.11+, `from __future__ import annotations` at the top of every module.
- Plain dataclasses, no Pydantic. Everything must survive
  `dataclasses.asdict`.
- Core package depends on **pdfplumber + numpy only**. Everything else
  (ultralytics, chromadb, sentence-transformers, pymupdf, groq, the benchmark
  baselines) is an optional extra and must be imported **inside the function that
  uses it**, never at module top level. `get_adapters()` relies on this to skip
  uninstalled baselines silently.
- Module docstrings explain *why the design is this way*, not what the code does
  line by line. That is the house style and it is worth preserving — several of
  them are the only record of a bug's root cause.
- No magic numbers outside `config.py`, and **every tuned constant carries a
  comment naming the measurement that chose it**.

## Testing

- pytest, `tests/` mirrors the package layout, `conftest.py` holds shared fixtures
  (`make_bbox` etc).
- Tests that need optional extras, real PDFs, or an LLM **self-skip** when those
  are absent. The suite must stay green on a bare core install.
- Non-trivial logic (a new branch, an algorithm, a parser) gets a test. Renames
  and config tweaks do not.
- Current state: 107 passing, ~2.5 min warm (the block cache is why).

## Git workflow

- Feature branches off `main`, conventional-commit subjects ≤50 chars
  (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `bench:`, `data:`,
  `perf:`, `tune:`, `revert:`).
- **Small, deliberate commits that tell a story.** A reviewer reading the log in
  order should see the arc of the work without external context. This repo's
  history already does this — for example `863bdda` "recursive XY-cut reading
  order, off by default after it measured worse" says the outcome in the subject.
  Match that.
- No `Co-Authored-By` / AI-authorship trailers.
- Never commit to `main` directly for feature work.

## Documentation duties

Three files must stay in sync with the code; a change that skips them is not done:

| File | What goes in it |
|---|---|
| `notes.md` | Chronological log. Every stage: what changed, **why**, what it measured, whether it was kept. Newest at the bottom. |
| `memory/` | The distilled current state. Update the relevant file whenever its subject changes. |
| `implementation_plan.md` | The standing plan-vs-code audit. Update the audit table when a proposed item lands or is rejected. |

`README.md` carries the public claims and the leaderboard — update it whenever the
headline numbers move.

## Reporting results

Never state a metric without the run that produced it. Never claim an improvement
that has not been measured against the current numbers in `memory/results.md`.
When something loses, the losing number goes in `notes.md` with the reason it was
kept or dropped — that log is a deliverable of this project, not scratch.
