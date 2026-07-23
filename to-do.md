# TO-DO

Working scratchpad for the Fable-review work. Durable detail lives in `notes.md`
(Stages 8–11) and `memory/` (esp. `measurement-environment.md`, `decisions.md`,
`roadmap.md`). This file is the short "where we are / what's next".

## Done this session

- **§1 PyPI hardening (default ON):** typed errors + `open_pdf`, `Path`/`password`
  input, scanned-PDF diagnostic, single-sourced `__version__`, `py.typed`, NullHandler,
  `pymupdf` in `model` extra, `run_pipeline`/`PipelineResult` re-exports, CI + CHANGELOG.
- **Bug fixes (default ON):** fixed-point graphic clustering, confidence-ordered
  matching, appendix/Roman section numbering, graphic-primitive cap.
- **Perf:** open the PDF once for both population passes.
- **DX:** `Document.tables`/`figures`/`to_markdown`, Markdown escaping + figure
  placeholders, CLI `run --format json|md|text`, `on_page` progress callback, README.
- **§1.8 (pragmatic):** `config.override()` + `parse(config={...})` — thread-safe,
  non-mutating per-parse overrides, zero call-site rewrites.
- **§3.4 (gated):** multi-page table merge.
- **Config-gated features, default OFF, unit-tested (14):** DEDUPE_CHARS, DEHYPHENATE,
  NORMALIZE_TEXT, FIGURE_OVERLAP_BY_AREA, MULTI_COLUMN, BAND_SPLIT, STRIP_PAGE_FURNITURE,
  TABLE_TEXT_STRATEGY_FALLBACK, TABLE_SERIALIZATION, TABLE_SPLIT_ROWS, TABLE_SETTINGS,
  HEADER_RANK_BY_WEIGHT, KEEP_REFERENCES, LABEL_AWARE_CONTAINMENT, MERGE_MULTIPAGE_TABLES.
- **Cache bug fixed:** block/geo caches were config-blind to new flags → would have
  false-nulled every ablation. Now config-fingerprinted; model (YOLO) cache kept warm.
- **Corpus:** +20 non-arXiv docs (→115). Gold for 9/23 new docs.
- Full suite **192 passed**. All on `main` + `feat/pypi-hardening` (in sync, pushed).

## To do (next session)

1. **Run the gated-feature ablation sweep — needs a GPU (Colab T4).**
   16 h on this CPU + env kills long jobs. On GPU < 1 h. Steps in
   `memory/measurement-environment.md`. Run `scripts/_sweep.sh`, compare each
   `reports/ablations/ab_*.json` to `ab_baseline.json`, flip winning flags to
   default-on in `config.py`, record in `memory/results.md`.

2. **Finish corpus-broadening gold — 14 docs left** (doc109–117, 58, 69, 97–99).
   `gen-qa` resumes/appends; run in ~3-doc batches (env kills long jobs). NOT blocked
   on quota — the model works. Then merge `benchmark_qa_v7_extra.json` into the v6
   gold → v7 and re-baseline all tools (`docstruct benchmark`) on 115 docs.

3. **Then** the gated flags become generalizable beyond arXiv; re-run the sweep on v7.

## Deferred (with reasons — see decisions.md/roadmap.md)

- §4.3 deeper SectionPath — breaks chunk-JSON for depth arXiv never reaches (YAGNI).
- §5.3 confidence calibration — needs ~20 hand-annotated docs (user chose to skip).
- §7.2/§7.3 perf, §8 mkdocs site — low value / P3.
- Full §1.8 threaded ParseConfig — pragmatic version shipped; full refactor only if
  parallel-different-config throughput is ever needed.
