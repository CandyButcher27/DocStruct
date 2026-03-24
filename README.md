# DocStruct: True Hybrid PDF Structure Extraction

**DocStruct** is a production-grade PDF understanding pipeline that implements a revolutionary `true hybrid` architecture. It combines:

- **Model-Driven Detection**: Deep learning models (DocLayNet, Table Transformer) propose block regions
- **Geometry Confirmation**: Geometric analysis confirms and boosts model predictions
- **Gap-Filling**: Geometry-only blocks catch regions the model missed
- **Tiered Confidence**: Final blocks scored by provenance (confirmed > model-only > geometry-only)

This approach achieves **higher precision AND recall** compared to either pure geometry or pure model approaches.

---

## 🎯 Three Pipeline Modes

### 1. **Standard Mode** (`--mode standard`)
Layout-first approach: geometry-based blocks classified by models.
- **Variants**: `geometry`, `model`, `hybrid`
- **Use case**: Quick processing, baseline comparison

### 2. **Model-First Mode** (`--mode model-first`)
Detection-driven with OCR fallback: model detections refined by geometry.
- **Detectors**: DocLayNet, Table Transformer, combined
- **Use case**: Document-heavy domains, high confidence in detections

### 3. **True Hybrid Mode** (`--mode true-hybrid`) ⭐ **RECOMMENDED**
Dual-source proposal matching with source-aware scoring.
- **Proposals from**: Model detection + independent geometry analysis
- **Fusion**: IoU-based matching (IoU ≥ 0.35)
- **Classes**: Confirmed (0.85-1.0) | Model-only (0.40-0.85) | Geometry-only (0.25-0.65)
- **Use case**: Maximum accuracy, production deployments

---

## 🏗️ Architecture Overview

```
PDF Input
    |
    +---> Decomposition ---> PageData (spans, lines, images)
    |
    +---> Model Detection ---> Detection[] (bbox, type, confidence)
    |           |
    |           v
    |     Model Proposals
    |           |
    +---> Geometry Proposals (independent)
                |
                v
        Match & Fuse (IoU ≥ 0.35)
                |
    +-----------+-----------+
    |           |           |
Confirmed   Model-Only   Geo-Only
(0.85-1.0)  (0.40-0.85)  (0.25-0.65)
    |           |           |
    +-----------+-----------+
                |
        NMS Deduplication
                |
        Reading Order & Output
```

### Key Stages

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Decomposition** | PDF | Extract spans, lines, images | PageData |
| **Geometry Proposals** | PageData | Text clustering, line analysis, image detection | RegionProposal[] |
| **Model Detection** | Rendered page | DocLayNet/Table Transformer | Detection[] |
| **Proposal Fusion** | Both proposal types | IoU-based matching | FusedProposal[] |
| **Classification** | FusedProposal | Source-aware weights | Block (with confidence) |
| **Deduplication** | Blocks | Priority-based NMS | Final blocks |
| **Reading Order** | Blocks | Spatial y-sort, x-sort | Ordered blocks |
| **Refinement** | Blocks | Table/figure post-processing | Refined blocks |
| **Output** | Refined blocks | JSON serialization | Document.json |

---

## 💡 Confidence Scoring

Each block gets a **source-aware confidence score** reflecting how it was detected:

### Confirmed (Model + Geometry Agree)
```
confidence = 0.85 + (0.10 * model_score) + (0.05 * iou)
Range: 0.85 - 1.00
Formula: 0.50*model + 0.30*rule + 0.20*geo (blended)
```
Highest confidence because both sources provide independent evidence.

### Model-Only (Unconfirmed by Geometry)
```
confidence = model_score * 0.75 (bounded 0.40-0.85)
Formula: 0.75*model + 0.15*rule + 0.10*geo
```
Medium confidence: model detected but no geometric support found.

### Geometry-Only (Model Missed)
```
confidence = geometry_score * 0.60 (bounded 0.25-0.65)
Formula: 0.00*model + 0.55*rule + 0.45*geo
```
Lower confidence: no model detection, geometry-driven classification.

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Extraction
```bash
# Standard mode (default)
python main.py documents/sample.pdf output.json

# True Hybrid mode (RECOMMENDED)
python main.py --mode true-hybrid --detector doclaynet documents/sample.pdf output.json --verbose
```

### Comparison Demo
```bash
# Generate all three for side-by-side comparison
python main.py --mode standard --variant geometry documents/sample.pdf geo.json
python main.py --mode standard --variant model documents/sample.pdf model.json --detector doclaynet
python main.py --mode true-hybrid --detector doclaynet documents/sample.pdf hybrid.json
```

---

## ⚙️ Configuration

Edit `config/defaults.yaml` to tune behavior:

```yaml
# Proposal matching thresholds
hybrid_pipeline:
  proposal_iou_threshold: 0.35  # Min IoU to match model+geo
  nms_iou_threshold: 0.50       # NMS suppression threshold

  # Source-specific confidence bounds
  confirmed_floor: 0.70
  model_only_floor: 0.40
  geometry_only_floor: 0.25

  # Weights (how to blend model/rule/geo scores)
  weights:
    confirmed:
      model: 0.50
      rule: 0.30
      geo: 0.20
    # ... etc
```

---

## 📊 Output Format

Each block includes:
```json
{
  "block_id": "uuid",
  "block_type": "table | text | header | figure | caption",
  "bbox": { "x0": 0, "y0": 0, "x1": 100, "y1": 50 },
  "text": "extracted text",
  "confidence": {
    "model_score": 0.9,
    "rule_score": 0.6,
    "geometric_score": 0.8,
    "final_confidence": 0.92
  },
  "proposal_source": "confirmed | model_only | geometry_only",
  "agreement_score": 0.85,
  "has_model_support": true,
  "reading_order": 1
}
```

---

## 🧪 Testing

```bash
# Run all hybrid pipeline tests
python -m pytest tests/test_hybrid_pipeline.py -v

# 18 tests covering:
# - Region proposal generation
# - Type inference from text features
# - IoU-based proposal matching
# - Confidence scoring by source
# - NMS deduplication
# - Hybrid classification
```

---

## 🔧 Advanced Features

### OCR Fallback for Scanned Pages
Automatically triggers when pages have images but sparse text (<5 text spans).
```python
from utils.ocr import is_scanned_page, ocr_page

if is_scanned_page(page_data):
    ocr_spans = ocr_page(image_bytes, page_num, width, height)
    # Use OCR results for text extraction
```

### Custom Detector
```bash
# Use combined detector (DocLayNet + Table Transformer)
python main.py --mode true-hybrid --detector combined document.pdf output.json

# Use Table Transformer only
python main.py --mode true-hybrid --detector table_transformer document.pdf output.json
```

### Verbose Logging
```bash
python main.py --mode true-hybrid --detector doclaynet document.pdf output.json --verbose
# Shows per-page proposal counts, fusion diagnostics, deduplication details
```

---

## 📈 Expected Performance

| Metric | Geometry-Only | Model-Only | True Hybrid |
|--------|---------------|-----------|-----------|
| Precision | Medium | High | **Very High** |
| Recall | High | Medium | **Very High** |
| Table F1 | 0.65 | 0.75 | **0.85+** |
| Text F1 | 0.80 | 0.82 | **0.88+** |
| Blocks Extracted | Highest | Lowest | **Balanced** |

*Estimates based on typical dense documents. Run evaluation on your domain for exact numbers.*

---

## 🐛 Debugging

### Check Proposal Matching
Enable verbose mode to see fusion diagnostics:
```bash
python main.py --mode true-hybrid --detector doclaynet document.pdf output.json --verbose
# Output: [Page 1] Fusion: confirmed=5, model_only=2, geo_only=3
```

### Inspect Confidence Breakdown
Each block includes all scoring components:
```python
block["confidence"]["model_score"]      # Raw model confidence
block["confidence"]["rule_score"]        # Heuristic-based score
block["confidence"]["geometric_score"]   # Sanity check score
block["confidence"]["final_confidence"]  # Weighted final score
block["proposal_source"]                 # confirmed|model_only|geometry_only
```

### Compare Output Modes
Use the CLI to generate all three variants:
```bash
python main.py --mode standard --variant geometry --detector stub doc.pdf geo.json
python main.py --mode standard --variant model --detector doclaynet doc.pdf model.json
python main.py --mode true-hybrid --detector doclaynet doc.pdf hybrid.json
# Compare JSON files to see differences in block detection
```

---

## 📁 Project Structure

```
DocStruct_new/
├── main.py                          # Entry point (3 modes)
├── pipeline/
│   ├── decomposition.py             # PDF → PageData
│   ├── layout.py                    # Geometry-based blocks
│   ├── hybrid_proposals.py           # Geometry proposal generation ⭐
│   ├── proposal_fusion.py            # Model-geometry matching ⭐
│   ├── classification.py             # Block type + scoring
│   ├── confidence.py                 # Confidence post-processing
│   ├── reading_order.py              # Spatial ordering
│   └── validator.py                  # JSON schema validation
├── models/
│   ├── detector.py                  # Model abstraction
│   ├── doclaynet_detector.py         # DocLayNet DETR
│   └── table_transformer.py          # Table Transformer
├── utils/
│   ├── geometry.py                  # Bbox operations, IoU, merge
│   ├── ocr.py                       # Tesseract fallback
│   ├── rendering.py                 # PDF → PNG
│   └── logging.py                   # Structured logging
├── schemas/
│   └── block.py                     # Pydantic models
├── config/
│   └── defaults.yaml                # All tunable parameters
├── tests/
│   └── test_hybrid_pipeline.py      # 18 unit tests ⭐
└── requirements.txt
```

---

## 📚 Key Modules

### `pipeline/hybrid_proposals.py` ⭐ NEW
Generates geometry-based proposals independently:
- Text clustering (from LayoutBlocks)
- Line-bounded regions (ruling lines → tables)
- Image regions (embedded images → figures)

### `pipeline/proposal_fusion.py` ⭐ NEW
Matches and fuses proposals:
- IoU-based matching (≥ 0.35)
- Confidence scoring by source
- Priority-based NMS deduplication

### `pipeline/classification.py` (Updated)
Source-aware classification:
- Different weights for confirmed vs unconfirmed
- Confidence bounds by source type

---

## 🎓 Research Notes

The true hybrid approach addresses fundamental limitations:

**Geometry-only**: High recall but misclassifies (text→table, caption→text)
**Model-only**: Better classification but misses regions model wasn't trained on
**True Hybrid**: Both sources propose independently → model confirms geometry + geometry fills model gaps

This is analogous to how humans use both: visual scan (geometry) + semantic understanding (model).

---

## 📞 Support

For issues, consult:
1. `config/defaults.yaml` - Tune thresholds
2. `--verbose` flag - See per-page diagnostics
3. Test suite - Run `pytest tests/test_hybrid_pipeline.py -v`
4. JSON output - Check `proposal_source` and confidence breakdown

---

## 📋 License & Citation

If you use DocStruct in your research, please cite:
```
@software{docstruct2025,
  title={DocStruct: True Hybrid PDF Understanding},
  year={2025}
}
```

---

**Last Updated**: March 2025 | **Status**: Production Ready
