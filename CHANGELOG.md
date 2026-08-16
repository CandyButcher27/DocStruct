# Changelog

All notable changes are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-08-16

First release published to PyPI.

### Changed
- **Distribution renamed to `docstruct-rag`.** The name `docstruct` was already taken
  on PyPI by an unrelated document-tree package (smrt-co, last released 2023-06-22).
  **The import name is unchanged**: `pip install docstruct-rag`, then `import docstruct`.
- `__version__` now resolves against the new distribution name, with the old name kept
  as a fallback for pre-rename editable installs. Before this it reported
  `0.0.0.dev0` on a clean install.

### Added
- `parse_bytes(data, name=...)` — parse a PDF held in memory, for HTTP uploads and
  object stores. Raises `InvalidPDFError` immediately on a missing `%PDF` header.
- `parse_many(paths, workers=, on_error=)` — batch parsing across processes, yielding
  `(path, Document | Exception)`. `on_error="return"` keeps a corpus job alive when a
  single document fails.
- `Document.to_langchain()` and `Document.to_llamaindex()` — hand-off to both
  frameworks with the section path carried in metadata. Optional imports; each raises
  an `ImportError` naming the package to install.
- `Document.to_jsonl()` — one `{id, text, metadata}` object per line.
- `Document.stats()` — chunk, page and word counts before indexing.
- Chunk metadata is flat and JSON-safe by contract, with a test pinning it: vector
  stores reject nested values, so a regression would surface at the user's ingest call
  rather than near this code.
- `scripts/verify_determinism.py` — corpus-scale determinism check, one subprocess per
  parse. Measured: 95/95 OHR-Bench documents byte-identical across independent
  processes, 5,810 chunks per run.
- `docs/API.md` — full API reference, including what determinism does and does not
  guarantee.
- Optional extras: `[langchain]`, `[llamaindex]`.

### Known limitations
- Determinism is verified for the geometry-only path. The hybrid path runs a model
  through CUDA, whose kernel selection is not guaranteed bit-reproducible, and is
  **unverified**.
- Dense financial filings are slow: three OHR-Bench 10-K/10-Q documents (120–217
  pages) each needed over 30 minutes to parse.
- Born-digital PDFs only. No OCR.

## [Unreleased]

### Added
- Typed exception hierarchy (`DocStructError`, `InvalidPDFError`,
  `EncryptedPDFError`, `EmptyDocumentError`) and a shared `open_pdf` helper; every
  PDF-open site routes through it, so callers no longer catch pdfminer internals.
- `parse()` accepts `str | Path` and a `password=` for encrypted PDFs.
- Scanned/image-only diagnostic: `diagnostics["likely_scanned"]` plus a warning
  pointing at deterministic OCR pre-processing (`ocrmypdf`).
- `Document.tables`, `Document.figures`, `Document.to_markdown(path)`; Markdown now
  escapes control characters and renders figure placeholders.
- CLI `run --format json|md|text --out` turns the CLI into a converter.
- `run_pipeline` and `PipelineResult` are now public re-exports.
- `py.typed` marker; the package is typed for downstream checkers.
- Golden determinism test and a malformed-PDF fuzz corpus; graphic-clustering
  primitive cap so pathological pages can't hang.
- GitHub Actions CI (test matrix + wheel build / `twine check`).
- Config-gated, default-off features awaiting ablation before they are enabled:
  `MULTI_COLUMN`, `BAND_SPLIT`, `STRIP_PAGE_FURNITURE`, `DEDUPE_CHARS`,
  `DEHYPHENATE`, `NORMALIZE_TEXT`, `FIGURE_OVERLAP_BY_AREA`,
  `TABLE_TEXT_STRATEGY_FALLBACK`, `TABLE_SERIALIZATION`, `TABLE_SPLIT_ROWS`,
  `TABLE_SETTINGS`, `HEADER_RANK_BY_WEIGHT`, `KEEP_REFERENCES`,
  `LABEL_AWARE_CONTAINMENT`.

### Changed
- `__version__` is single-sourced from installed package metadata.
- `pymupdf` moved into the `model` extra (the model detector needs `fitz`).
- Library logging: a `NullHandler` is installed on the `docstruct` logger.
- Recognize appendix (`A.`, `A.1`) and Roman-numeral (`IV.`, `IX.1`) section
  numbering when assigning header depth.

### Fixed
- Graphic clusters now merge to a fixed point (order-independent figure regions).
- Proposal matching is confidence-ordered, so a low-confidence model box can no
  longer claim a geometry box a higher-confidence box needed.

### Performance
- The PDF is opened once for both population passes instead of twice.
