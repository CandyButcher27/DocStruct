# DocStruct

DocStruct is a PDF structure extractor that supports geometry-only, model-only, and hybrid pipelines. It outputs structured JSON for text blocks, headers, tables, figures, and captions using deterministic PDF parsing, optional OCR for scanned pages, and local/offline Hugging Face models.

## Supported Features

- Deterministic geometry-only extraction with `--detector stub`
- Offline layout detection with `--detector doclaynet` (from `hf_models/deformable-detr-doclaynet`)
- Offline table detection with `--detector table_transformer` (from `hf_models/table-transformer`)
- Combined layout+table detection with `--detector combined`
- Three output variants in one run: `geometry`, `model`, `hybrid`
- Side-by-side overlay comparison for two JSON outputs
- OCR fallback for scanned pages through Tesseract
- Real evaluation modes for local PDF datasets and Hugging Face page-image datasets

## Not Supported

- Hugging Face-hosted PDF ingestion
- LayoutLMv3 or DiT as stable runtime detectors
- Smoke-test benchmarks presented as real metrics

## Setup

```bash
pip install -r requirements.txt
```

Install Tesseract on the host system if you want OCR fallback.

Local model directories expected by default:

- `hf_models/deformable-detr-doclaynet`
- `hf_models/table-transformer`

## CLI

Single-output geometry extraction:

```bash
python main.py input.pdf output.json --detector stub
```

Single-output model extraction using local DocLayNet model:

```bash
python main.py input.pdf output.json --detector doclaynet --variant model
```

Single-output hybrid extraction using both local models:

```bash
python main.py input.pdf output.json --detector combined --variant hybrid
```

Generate all three outputs in one run:

```bash
python main.py input.pdf output.json --detector combined --output-variants all
```

This creates:

- `output_geometry.json`
- `output_model.json`
- `output_hybrid.json`

Single JSON visual overlay:

```bash
python visualize_overlay.py input.pdf output.json --output-dir overlay_output
```

Side-by-side JSON comparison overlay:

```bash
python visualize_overlay.py input.pdf --left-json output_geometry.json --right-json output_hybrid.json --output-dir compare_overlay
```

## CLI Verification

Run these to validate all CLI entry points:

```bash
python main.py -h
python visualize_overlay.py -h
python -m evaluation.runner -h
```

Smoke run (all variants):

```bash
python main.py tests/fixtures/sample.pdf outputs/sample.json --detector combined --output-variants all --verbose
```

Visualizer smoke run:

```bash
python visualize_overlay.py tests/fixtures/sample.pdf --left-json outputs/sample_geometry.json --right-json outputs/sample_hybrid.json --output-dir outputs/compare_overlay
```

Test suite:

```bash
python -m pytest -q
```

## Evaluation

Local PDF evaluation:

```bash
python -m evaluation.runner --eval-mode local_pdf --data-dir ./data/publaynet --detector combined --output results/local.csv
```

Hugging Face page-image evaluation:

```bash
python -m evaluation.runner --eval-mode hf_image --detector table_transformer --output results/hf.csv
```

The supported Hugging Face dataset is `nielsr/publaynet-processed`. That path evaluates page images and annotations only. It does not import PDFs from Hugging Face.
