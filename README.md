# DocStruct: Hybrid PDF Structure Extraction

DocStruct is a research-grade PDF understanding pipeline that fuses deterministic geometry parsing with deep learning-based vision models. It is designed to extract high-fidelity structured JSON from complex documents, supporting multiple extraction strategies and multi-layer ensemble reasoning.

## 🚀 Key Features

- **Hybrid Intelligence**: Combines `pdfplumber` geometry rules with `DocLayNet` and `Table Transformer` models.
- **Ensemble Benchmarking**: Run `geometry`, `model`, and `hybrid` variants side-by-side to find the best fit for your data.
- **OCR Fallback**: Automatically invokes Tesseract OCR for scanned/image-only pages with unified coordinate mapping.
- **Precision Gating**: Per-class confidence thresholds ensure high-quality extraction for sensitive fields like tables and headers.
- **Visual Overlay**: Human-in-the-loop validation with overlaid bounding boxes for visual audit.

## 🛠️ Architecture

DocStruct follows a 6-stage pipeline:
1. **Decomposition**: Atomic extraction + OCR fallback.
2. **Layout Formation**: Clustering spans into logical blocks.
3. **Classification**: Hybrid scoring (Hugging Face Models + Geometric Rules).
4. **Reading Order**: Spatial sorting for multi-column documents.
5. **Refinement**: Table and figure boundary optimisation.
6. **Validation**: Pydantic-based schema enforcement.

Refer to [docs/architecture.md](docs/architecture.md) for a detailed technical breakdown.

## 📦 Setup

```bash
pip install -r requirements.txt
# Optional: Install Tesseract for OCR fallback
```

## ⌨️ CLI Usage

### Basic Extraction (Geometry Only)
```bash
python main.py work.pdf output.json --detector stub
```

### Full Hybrid Model Run (All Variants)
```bash
python main.py work.pdf output.json --detector combined --output-variants all --verbose
```

### Visual Verification
```bash
python visualize_overlay.py work.pdf output_hybrid.json --output-dir audit_results
```

## 📊 Evaluation & Benchmarking

Benchmarking against DocLayNet using the Hugging Face hub:

```bash
python -m evaluation.runner --eval-mode hf_image --dataset doclaynet --max-docs 5 --detector combined
```

## 📜 Roadmap
Detailed in [docs/future_work.md](docs/future_work.md):
- [ ] LayoutLMv3 integration
- [ ] Table-to-Markdown parsing
- [ ] FastAPI service wrapper
