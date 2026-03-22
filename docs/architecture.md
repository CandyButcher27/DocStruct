# DocStruct Architecture

DocStruct is a hybrid PDF structure extraction system that combines deterministic PDF parsing with modern vision models and OCR fallbacks.

## System Workflow

```mermaid
graph TD
    A[Input PDF] --> B[decomposition.py]
    B -->|Scanned Page?| C[OCR Fallback (Tesseract)]
    B -->|Native Text?| D[Raw Text Spans]
    C --> E[Unified Page Data]
    D --> E
    E --> F[layout.py - Layout Formation]
    F --> G[classification.py - Multi-Stage Scoring]
    H[Vision Models] --> G
    G -->|Geometry Scores| I[Class-wise Thresholding]
    G -->|Model Detections| I
    I --> J[reading_order.py - Spatial Sorting]
    J --> K[tables_figures.py - Block Refinement]
    K --> L[validator.py - Schema Enforcement]
    L --> M[Structured JSON Output]
```

## Core Components

### 1. Decomposition (`pipeline/decomposition.py`)
Extracts raw atomic units (text spans with fonts, image bboxes, lines) using `pdfplumber`. If a page is detected as "scanned" (empty or near-empty), it triggers a Tesseract-based OCR fallback.

### 2. Layout Formation (`pipeline/layout.py`)
Clusters atomic text spans into logical "LayoutBlocks" based on spatial proximity, alignment, and font similarity.

### 3. Classification (`pipeline/classification.py`)
The heart of DocStruct. It computes a hybrid confidence score for each block using:
- **Geometry Rules**: Heuristics based on position (e.g., headers are at the top) and font size.
- **Model Detections**: Overlap with bboxes from vision models (DocLayNet, Table Transformer).
- **Conflict Resolution**: Logic to pick the winner when rules and models disagree.

### 4. Reading Order (`pipeline/reading_order.py`)
Determines the logical flow of text blocks (left-to-right, top-to-bottom) while respecting multi-column layouts.

### 5. Validation (`pipeline/validator.py`)
Ensures the final graph of blocks adheres to the target JSON schema and produces the final serialised document.
