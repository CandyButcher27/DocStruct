# Future Prospects & Roadmap

This document outlines the next frontiers for the DocStruct project.

## 1. Advanced Model Integration
- **LayoutLMv3 / DiT**: Integrate state-of-the-art document foundation models as stable runtime detectors.
- **Ensemble Refining**: Improve the `ConflictResolver` to weight models based on their historical per-class precision.

## 2. Table Understanding
- **Cell Extraction**: Move beyond bounding-box detection to full row/column/cell grid parsing.
- **Table-to-Markdown**: Automatic conversion of extracted tables into clean GitHub-flavored markdown.

## 3. Performance & Scale
- **Batch Processing**: Parallelise PDF decomposition across CPU cores.
- **GPU Acceleration**: Optimise the vision model inference bottleneck.
- **API Wrapper**: A lightweight FastAPI or Gradio wrapper for "DocStruct-as-a-Service".

## 4. Evaluation & Benchmarking
- **Full HF Dataset Support**: Complete the `hf_image` evaluation mode in `runner.py`.
- **Cross-Dataset Validation**: Benchmark on PubLayNet, DocLayNet, and IIIT-AR-13K.

## 5. UI/UX
- **Interactive Visualizer**: A web UI that allows manual correction of misclassified blocks.
- **Diffing Tool**: Semantic diffing between two PDF structures to detect small textual or layout changes.
