# DocStruct — Deterministic PDF Layout Analysis Pipeline

DocStruct is a PDF structure extraction system that identifies and classifies document regions — text blocks, headers, tables, figures, captions — and outputs validated, schema-enforced JSON. It supports three extraction modes: geometry-only (rule-based), model-only (neural), and hybrid, allowing controlled comparison of deterministic vs. learned approaches.

---

## Motivation & Design Philosophy

Most PDF parsing libraries treat layout as a secondary concern: they extract a stream of text and leave structure inference to the caller. Tools that do attempt layout analysis either rely entirely on heuristics (fragile, format-sensitive) or on neural models without fallback (opaque, non-auditable).

DocStruct was designed around a different set of constraints:

- **Auditability over magic** — every classification decision carries an explicit confidence score and the rule or model signal that produced it
- **Offline-first** — all neural models run locally; no API calls, no data leaving the machine
- **Composable pipelines** — geometry, model, and hybrid outputs are produced independently and can be compared side-by-side
- **No false precision** — the system explicitly declares what it does not support (see below)

---

## Extraction Modes

### 1. Geometry-Only (`--detector stub`)
Pure rule-based extraction using spatial geometry: bounding box positions, font sizes, whitespace gaps, and column detection. Fully deterministic. No ML involved.

**Best for:** clean, digitally-typeset PDFs with consistent formatting.

### 2. Model-Only (`--detector doclaynet` or `--detector table_transformer`)
Neural layout detection using locally-hosted Hugging Face models:

| Model | Architecture | Task |
|---|---|---|
| `deformable-detr-doclaynet` | Deformable DETR | General layout detection (text, figure, table, list, heading) |
| `table-transformer` | DETR-based | Table boundary and structure detection |

**Why Deformable DETR over a simpler detector?**
Standard DETR struggles with small, densely packed objects — which is precisely what academic and financial PDFs contain (footnotes, table cells, caption text). Deformable DETR's multi-scale attention resolves this by attending to sparse spatial locations rather than the full feature map, improving precision on document-scale layouts.

**Why a separate Table Transformer?**
Table detection is a qualitatively different problem from general layout detection: it requires row/column boundary inference, not just region classification. Mixing both into a single model degrades both tasks. Dedicated models kept here for that reason.

### 3. Hybrid (`--detector combined`)
Runs both the geometry pipeline and the neural model pipeline in parallel, then merges results using a confidence-weighted reconciliation step. Where model and geometry agree, confidence is high. Where they conflict, the output flags the discrepancy rather than silently picking one.

---

## Output Schema

All modes output Pydantic v2-validated JSON. Example block:

```json
{
  "block_id": "blk_0042",
  "type": "table",
  "bbox": [72.0, 341.5, 540.0, 489.0],
  "page": 3,
  "confidence": 0.91,
  "source": "hybrid",
  "text": null,
  "detector_signals": {
    "geometry": "column_span_heuristic",
    "model": "table_transformer_0.94"
  }
}
```

`detector_signals` is always populated — it's the audit trail for every classification decision.

---

## Pipeline Structure

```
input.pdf
    │
    ├──► Geometry Extractor      → bounding boxes, text spans, font metadata
    │         │
    ├──► OCR Fallback (Tesseract) → activated only on pages with <10% extractable text
    │
    ├──► Layout Detector         → DocLayNet region proposals (if --detector doclaynet/combined)
    │
    ├──► Table Detector          → Table Transformer proposals (if --detector table_transformer/combined)
    │
    └──► Reconciler + Validator  → merges signals, scores confidence, validates via Pydantic schema
                │
                └──► output.json  (geometry / model / hybrid variants)
```

---

## CLI

```bash
# Geometry only
python main.py input.pdf output.json --detector stub

# Neural layout detection
python main.py input.pdf output.json --detector doclaynet --variant model

# Full hybrid — all three output variants in one run
python main.py input.pdf output.json --detector combined --output-variants all

# Side-by-side visual overlay comparison
python visualize_overlay.py input.pdf \
  --left-json outputs/sample_geometry.json \
  --right-json outputs/sample_hybrid.json \
  --output-dir outputs/compare_overlay
```

---

## Setup

```bash
pip install -r requirements.txt
```

Install [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR fallback support.

Download and place models locally:
```
hf_models/
├── deformable-detr-doclaynet/
└── table-transformer/
```

Run the test suite:
```bash
python -m pytest -q
```

---

## Evaluation

```bash
# Against a local PubLayNet subset
python -m evaluation.runner --eval-mode local_pdf --data-dir ./data/publaynet \
  --detector combined --output results/local.csv

# Against HF page-image dataset (nielsr/publaynet-processed)
python -m evaluation.runner --eval-mode hf_image \
  --detector table_transformer --output results/hf.csv
```

---

## What DocStruct Does Not Do

- No LayoutLMv3 or DiT runtime support (dependency conflicts at stable versions)
- No Hugging Face-hosted PDF ingestion
- No benchmark numbers presented as production metrics — evaluation tooling is provided for you to run against your own data

---

## Project Structure

```
DocStruct/
├── pipeline/        # Core extraction stages
├── models/          # Reconciler, confidence scoring, Pydantic schemas
├── hf_models/       # Local HF model weights (not tracked in git)
├── schemas/         # JSON output schema definitions
├── utils/           # Geometry helpers, OCR wrappers
├── tests/           # pytest suite with fixture PDFs
├── main.py          # CLI entrypoint
├── visualize_overlay.py  # Visual diff tool
└── evaluation/      # Evaluation runner
```
