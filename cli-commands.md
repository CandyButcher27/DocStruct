# DocStruct CLI Reference

## Processing a PDF

### True-Hybrid (recommended)
```bash
python main.py --mode true-hybrid --detector doclaynet input.pdf output.json
```

### All modes for comparison
```bash
python main.py --mode standard   --variant geometry --detector stub       input.pdf geo.json
python main.py --mode standard   --variant model    --detector doclaynet  input.pdf model.json
python main.py --mode true-hybrid                   --detector doclaynet  input.pdf hybrid.json
```

### All variants in one pass (standard mode)
```bash
python main.py --mode standard --detector doclaynet --output-variants all input.pdf out.json
# Produces: out_geometry.json, out_model.json, out_hybrid.json
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `standard` | `standard`, `model-first`, `true-hybrid` |
| `--detector` | `combined` | `stub`, `doclaynet`, `table_transformer`, `combined` |
| `--doclaynet-confidence-threshold` | `0.3` | Raw model score cutoff |
| `--output-variants` | _(single)_ | `all` to write all 3 variants |
| `--verbose` | off | Per-page fusion diagnostics |

---

## Inspecting Output

```bash
# Block count per page
jq '[.pages[] | {page: .page_num, blocks: (.blocks | length)}]' output.json

# Proposal source distribution
jq '[.pages[].blocks[].proposal_source] | group_by(.) | map({source:.[0], count:length})' output.json

# All tables
jq '.pages[].blocks[] | select(.block_type == "table") | {id:.block_id, conf:.confidence.final_confidence}' output.json

# High-confidence blocks only
jq '.pages[].blocks[] | select(.confidence.final_confidence > 0.85)' output.json

# Geometry-only blocks (lower confidence, no model support)
jq '.pages[].blocks[] | select(.proposal_source == "geometry_only")' output.json

# Average confidence across all blocks
jq '[.pages[].blocks[].confidence.final_confidence] | add / length' output.json
```

---

## Evaluation

### Step 1: Download ground truth (first time)
```bash
python scripts/download_doclaynet.py \
  --out-dir ./data/doclaynet \
  --split validation \
  --max-docs 200
```

### Step 2: Run evaluation
```bash
python -m evaluation.runner \
  --eval-mode local_doclaynet \
  --data-dir ./data/doclaynet \
  --detector doclaynet \
  --max-docs 50 \
  --doclaynet-confidence-threshold 0.3 \
  --output results/my_run.csv
```

### Eval flags

| Flag | Default | Description |
|---|---|---|
| `--eval-mode` | `local_pdf` | `local_pdf`, `local_doclaynet`, `hf_image` |
| `--detector` | `combined` | Same as main.py |
| `--max-docs` | `50` | Number of pages to evaluate |
| `--doclaynet-confidence-threshold` | `0.3` | Model score cutoff |
| `--model-confidence-threshold` | `0.3` | Per-class threshold |
| `--output` | `results/benchmark.csv` | CSV output path |
| `--fail-on-detector-error` | off | Raise instead of logging detector errors |

### Evaluate directly from HuggingFace (no local download)
```bash
python -m evaluation.runner \
  --eval-mode hf_image \
  --dataset doclaynet \
  --detector doclaynet \
  --max-docs 50 \
  --output results/hf_run.csv
```

### Read results
```bash
python -c "
import csv, math
rows = list(csv.DictReader(open('results/test_doclaynet_fixed.csv')))
for v in ['geometry','model','hybrid']:
    vr = [r for r in rows if r['variant']==v]
    maps = [float(r['mAP@0.50']) for r in vr if r['mAP@0.50'] and not math.isnan(float(r['mAP@0.50']))]
    print(f'{v:10s} mAP@0.50={sum(maps)/len(maps):.3f} (n={len(maps)})')
"
```

---

## Visualization

Render bounding box overlays on PDF pages:
```bash
python visualize_overlay.py \
  documents/sample.pdf \
  output.json \
  --output-dir overlays/ \
  --dpi 150
```

Produces one PNG per page in `overlays/` with colored boxes by block type.

---

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Hybrid pipeline only (18 unit tests)
python -m pytest tests/test_hybrid_pipeline.py -v

# Single file
python -m pytest tests/test_classification.py -v

# Stop on first failure
python -m pytest tests/ -x
```

---

## Tuning

Edit `config/defaults.yaml` to change behavior without code changes:

```yaml
# Lower IoU threshold → more confirmed matches (noisier)
hybrid_pipeline:
  proposal_iou_threshold: 0.25   # default 0.35

# Raise model weight → trust model more
ensemble:
  model_weight: 0.65             # default 0.50
  rule_weight:  0.20
  geo_weight:   0.15

# Per-class minimum confidence to emit a block
classification_thresholds:
  table: 0.40                    # default 0.50 (lower = more tables)
```

---

## Troubleshooting

**No model detections:**
```bash
# Check detector loaded + run with verbose
python main.py --mode true-hybrid --detector doclaynet --verbose doc.pdf out.json
# Look for: "LocalDocLayNetDetector loaded successfully"
# If missing: model weights not cached — first run will download from HuggingFace
```

**Near-zero mAP in evaluation:**
- Ensure GT was downloaded with `--split validation` (test split has no annotations)
- Check `annotations.jsonl` has non-empty `ground_truths`: `python -c "import json; r=json.loads(open('data/doclaynet/annotations.jsonl').readline()); print(len(r['ground_truths']))"`

**Geometry-only mAP always near zero:**
Expected on DocLayNet — these are image-only pages with no embedded text layer, so OCR extracts nothing and geometry proposals are empty.

**OCR not working:**
Tesseract must be installed separately: https://github.com/tesseract-ocr/tesseract
