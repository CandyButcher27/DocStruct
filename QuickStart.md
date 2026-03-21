# DocStruct Quick Start

## Install

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- Tesseract installed if you want OCR fallback

## Extract a PDF

Geometry-only mode:

```bash
python main.py your_document.pdf output.json --detector stub
```

Model-only layout mode (local DocLayNet model):

```bash
python main.py your_document.pdf output.json --detector doclaynet --variant model
```

Combined hybrid mode (DocLayNet + TableTransformer):

```bash
python main.py your_document.pdf output.json --detector combined --variant hybrid
```

Generate all three output JSONs (`geometry`, `model`, `hybrid`) in one run:

```bash
python main.py your_document.pdf output.json --detector combined --output-variants all
```

Side-by-side visual comparison of two output JSONs:

```bash
python visualize_overlay.py your_document.pdf --left-json output_geometry.json --right-json output_hybrid.json --output-dir compare_overlay
```

## Evaluate

Local PDF evaluation:

```bash
python -m evaluation.runner --eval-mode local_pdf --data-dir ./data/publaynet --detector combined --output results/local.csv
```

Hugging Face page-image evaluation:

```bash
python -m evaluation.runner --eval-mode hf_image --detector table_transformer --output results/hf.csv
```

Notes:
- The supported HF dataset is `nielsr/publaynet-processed`.
- That path evaluates page images and annotations only. It does not import PDFs from Hugging Face.
- Supported detectors in CLI: `stub`, `table_transformer`, `doclaynet`, `combined`.

## Quick CLI Health Check

```bash
python main.py -h
python visualize_overlay.py -h
python -m evaluation.runner -h
python -m pytest -q
```
