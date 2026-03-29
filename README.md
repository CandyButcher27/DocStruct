# DocStruct

PDF document layout analysis pipeline. Extracts structured content (text, headers, tables, figures, captions) from PDFs using a **true hybrid** architecture that combines deep learning detection with geometry-based heuristics.

---

## How It Works

Three pipeline modes share the same output format but differ in how they combine geometry and model signals:

| Mode | Description | When to use |
|---|---|---|
| `standard` | Geometry forms blocks → model classifies them | Baseline comparison |
| `model-first` | Model detections drive block creation → geometry refines | High-confidence detection domains |
| `true-hybrid` | Both sources generate proposals independently → IoU fusion | **Production (recommended)** |

### True-Hybrid Fusion

```
PDF → PageData (text spans)
       ↓
   Layout Extraction
       ↓
   Geometry Proposals ──┐
   Model Detections ────┤  IoU match (≥ 0.35)
                        ↓
          Proposal Fusion
        (confirmed / model_only / geo_only)
                        ↓
               NMS Deduplication
                        ↓
              Classification
     (model + geometry + agreement scoring)
                        ↓
          Text Assignment (conditional)
                        ↓
        Reading Order + Refinement
                        ↓
                     JSON
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Process a PDF (true-hybrid, recommended)
python main.py --mode true-hybrid --detector doclaynet documents/sample.pdf output.json

# Compare all three modes
python main.py --mode standard --variant geometry --detector stub documents/sample.pdf geo.json
python main.py --mode standard --variant model   --detector doclaynet documents/sample.pdf model.json
python main.py --mode true-hybrid                --detector doclaynet documents/sample.pdf hybrid.json
```

---

## Output Format

```json
{
  "block_id": "uuid",
  "block_type": "text | header | table | figure | caption",
  "bbox": { "x0": 66.8, "y0": 528.5, "x1": 395.6, "y1": 609.1 },
  "text": "...",
  "confidence": {
    "model_score": 0.91,
    "rule_score": 0.72,
    "geometric_score": 0.80,
    "final_confidence": 0.85
  },
  "proposal_source": "confirmed | model_only | geometry_only",
  "reading_order": 1
}
```

`bbox` uses PDF coordinate space: bottom-left origin, units in points.

---

## Benchmark Results

Evaluated on DocLayNet v1.2 validation split (50 pages), detector = DocLayNet DETR, threshold = 0.3:

| Variant | mAP@0.50 | mAP@0.75 | macro F1 | text F1 | table F1 | figure F1 |
|---|---|---|---|---|---|---|
| geometry | 0.028 | 0.009 | 0.021 | 0.050 | 0.000 | 0.000 |
| **model** | **0.844** | **0.716** | **0.816** | **0.795** | **0.866** | **0.927** |
| hybrid | 0.802 | 0.686 | 0.518 | 0.362 | 0.893 | 0.927 |

Geometry near-zero is expected: DocLayNet images have no embedded text layer for OCR.
Hybrid slightly below model-only because OCR geometry proposals add noise on image-only pages.

Full benchmark (600 docs): `results/benchmark.csv`

---

## Evaluation

```bash
# Download ground truth from HuggingFace (first time only)
python scripts/download_doclaynet.py --out-dir ./data/doclaynet --split validation --max-docs 200

# Run evaluation
python -m evaluation.runner \
  --eval-mode local_doclaynet \
  --data-dir ./data/doclaynet \
  --detector doclaynet \
  --max-docs 50 \
  --doclaynet-confidence-threshold 0.3 \
  --output results/my_run.csv
```

Supported eval modes: `local_pdf`, `local_doclaynet`, `hf_image`

---

## Configuration

All tunable parameters are in `config/defaults.yaml`:

```yaml
ensemble:
  model_weight: 0.50
  rule_weight:  0.30
  geo_weight:   0.20

classification_thresholds:
  text:    0.30
  header:  0.45
  table:   0.50
  figure:  0.40
  caption: 0.40

hybrid_pipeline:
  proposal_iou_threshold: 0.35
  nms_iou_threshold:      0.50
  confirmed_floor:        0.70
  model_only_floor:       0.40
  geometry_only_floor:    0.25
```

---

## Project Structure

```
DocStruct/
├── main.py                        # Entry point (3 pipeline modes)
├── visualize_overlay.py           # Render bbox overlays on PDF pages
├── pipeline/
│   ├── decomposition.py           # PDF → PageData (text spans, lines, images)
│   ├── layout.py                  # Geometry-based block clustering
│   ├── hybrid_proposals.py        # Independent geometry proposal generation
│   ├── proposal_fusion.py         # IoU-based model+geometry fusion
│   ├── classification.py          # Source-aware block classification
│   ├── table_candidates.py        # Table candidate detection
│   ├── tables_figures.py          # Table/figure post-processing
│   ├── reading_order.py           # Spatial block ordering
│   ├── confidence.py              # Confidence score post-processing
│   └── validator.py               # Pydantic schema validation → JSON
├── models/
│   ├── detector.py                # Detector ABC + factory (create_detector)
│   ├── doclaynet_detector.py      # DocLayNet DETR wrapper
│   └── table_transformer.py       # Table Transformer wrapper
├── schemas/
│   ├── block.py                   # BoundingBox, ConfidenceBreakdown, Block
│   ├── page.py                    # Page schema
│   └── document.py                # Document schema
├── utils/
│   ├── geometry.py                # IoU, bbox merge, distance utilities
│   ├── rendering.py               # PDF → PNG for model input
│   ├── ocr.py                     # Tesseract OCR fallback
│   ├── logging.py                 # Structured logging setup
│   └── config.py                  # YAML config loader (cached)
├── evaluation/
│   ├── runner.py                  # Evaluation loop (3 modes)
│   ├── ground_truth.py            # GT loaders (local JSONL, HuggingFace)
│   └── metrics.py                 # mAP@0.50, mAP@0.75, per-class F1
├── scripts/
│   ├── download_doclaynet.py      # Download DocLayNet GT from HuggingFace
│   └── prepare_publaynet_local_pdf.py
├── tests/                         # pytest test suite
├── config/
│   └── defaults.yaml
├── documents/                     # Sample PDFs for testing
├── data/                          # Downloaded evaluation datasets
├── results/                       # Benchmark CSV outputs
└── hf_models/                     # Cached model configs
```

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Hybrid pipeline unit tests (18 tests)
python -m pytest tests/test_hybrid_pipeline.py -v

# Single module
python -m pytest tests/test_classification.py -v
```

---

## Detectors

| Detector | Flag | Coverage | Notes |
|---|---|---|---|
| stub | `--detector stub` | None | Geometry-only baseline |
| doclaynet | `--detector doclaynet` | All classes | Recommended |
| table_transformer | `--detector table_transformer` | Tables only | Specialized |
| combined | `--detector combined` | All + tables | Both models ensemble |

Models are downloaded on first use and cached in `hf_models/`.

---

## Notes

- OCR fallback (pytesseract) activates on scanned pages — requires Tesseract binaries installed separately.
- Coordinates use bottom-left origin (PDF space). All IoU/merge operations in `utils/geometry.py` assume this convention.
- `load_doclaynet_hf` and `scripts/download_doclaynet.py` use `category_id` field (1-based) and `metadata.coco_height` from the `docling-project/DocLayNet-v1.2` HuggingFace dataset.
