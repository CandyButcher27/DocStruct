# Changelog

All notable changes are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

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
