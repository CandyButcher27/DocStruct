# DocStruct

**Local, deterministic, structure-aware PDF chunking for RAG.**

DocStruct turns born-digital PDFs into retrieval-ready chunks that respect
document structure — section hierarchy, tables, and figure captions stay intact
instead of being shredded by fixed-size windows. It runs two independent layout
detectors (a rules-based geometry pass and an optional vision model), fuses them
with a deterministic algorithm, and emits chunks annotated with their section
path for filtered retrieval.

- **No LLM calls in the pipeline.** Same PDF in → same chunks out. Auditable.
- **Fully local.** No API, no internet at inference time. Works air-gapped.
- **Structured output.** Bounding boxes, per-source confidence, reading order,
  section hierarchy — not free-form text.

---

## Why

Naive RAG chunking (fixed token windows) destroys structure: tables split
mid-row, captions detach from figures, section headers orphan from their bodies.
DocStruct fixes this by detecting layout first and chunking along real document
boundaries, then exposing section metadata so retrieval can be filtered
(`where={"h1": "3. Methodology"}`).

| vs | DocStruct advantage |
|----|---------------------|
| LangChain naive loader | respects structure; section-path metadata |
| Unstructured.io / LlamaParse | fully local, free, deterministic, section-filtered retrieval |

---

## Architecture

```
PDF
 ├─ geometry/detector.py   pdfplumber rules ─┐  (blind to the model)
 ├─ model/detector.py      YOLOv8/DocLayNet ─┤  (optional, blind to geometry)
 │                                           ▼
 │                         fusion/matcher    greedy IoU match + priority NMS
 │                         fusion/arbiter    confirmed vs disputed (label)
 │                         fusion/fusion     → List[Block] + ConfidenceBreakdown
 ├─ reading_order.py       columns + top→bottom; caption → figure/table
 ├─ extraction/            block text + Markdown tables (pdfplumber)
 ├─ chunking/              header levels by font rank → section-aware chunks
 ├─ indexing/              sentence-transformers → ChromaDB
 └─ query/                 top-k chunks + section-path citations
```

Both detectors run fully independently; reconciliation happens only in `fusion/`.
Without model weights the pipeline runs **geometry-only** and degrades
gracefully (every block becomes `unilateral_geometry`).

### Fusion in one table

| Case | Condition | Final confidence |
|------|-----------|------------------|
| Confirmed | both detect, same label, IoU ≥ 0.35 | `0.85 + 0.10·model_conf + 0.05·IoU` |
| Disputed | both detect, different label, IoU ≥ 0.35 | `winner_conf × 0.85` |
| Unilateral | one detector only | source-scaled, bounded (see `config.py`) |

---

## Install

```bash
pip install -e .                 # core: pdfplumber + numpy (geometry-only)
pip install -e ".[model]"        # + YOLOv8 (ultralytics) + PyMuPDF  -> hybrid
pip install -e ".[retrieval]"    # + ChromaDB + sentence-transformers
pip install -e ".[viz,eval,dev]" # + visualization, eval extras, tests
pip install -e ".[all]"          # everything
```

Hybrid mode needs DocLayNet YOLOv8 weights (downloaded separately), e.g.
`hantian/yolo-doclaynet` → `weights/yolov8m-doclaynet.pt`.

---

## Usage

### CLI

```bash
# Geometry-only
docstruct run paper.pdf

# Hybrid (geometry + model)
docstruct run paper.pdf --weights weights/yolov8m-doclaynet.pt --json chunks.json

# Index a corpus and query it with section citations
docstruct index a.pdf b.pdf --db .chroma --weights weights/yolov8m-doclaynet.pt
docstruct query "what baseline did they compare against?" --db .chroma --top-k 5
docstruct query "results" --db .chroma --h1 "4. Experiments"   # filtered

# Annotated overlay for inspection / figures
docstruct visualize paper.pdf --out annotated.pdf --weights weights/yolov8m-doclaynet.pt
```

### Python

```python
from docstruct.pipeline import run_pipeline

result = run_pipeline("paper.pdf", weights="weights/yolov8m-doclaynet.pt", cache_dir=".cache")
print(result.diagnostics)            # mode, page/block/chunk counts, fusion stats
for chunk in result.chunks:
    print(chunk.chunk_type, chunk.section_path, chunk.content[:80])
```

```python
from docstruct.indexing.vector_store import VectorStore
from docstruct.query.retriever import Retriever

store = VectorStore(persist_dir=".chroma")
store.index(result.chunks, doc_id="paper")
for r in Retriever(store).retrieve("how was the corpus prepared?", top_k=3):
    print(r.citation())              # [Corpus preparation] (page 2, score 0.48)
```

---

## Evaluation

Two layers, in `docstruct.eval`:

- **Detection** — per-class precision/recall/F1 and confidence-ranked **mAP@0.5**
  against ground-truth boxes (`evaluate_detection`).
- **Retrieval** — **MRR** and **NDCG@k** over question/answer cases
  (`evaluate_retrieval`), plus `compare_chunking` to benchmark DocStruct chunks
  against a fixed-window `naive_chunk` baseline through the identical
  index/retrieve path.

Detection ground truth uses a simple per-document JSON
(`{"boxes": [{"label", "page_num", "bbox": [x0,y0,x1,y1,pw,ph]}]}`).

---

## Determinism & coordinates

Every stage is deterministic. Coordinates are **top-left** throughout
(`y0` = top, increasing downward), matching pdfplumber and PyMuPDF, so geometry,
model (after a pixel→point transform), extraction, and visualization share one
space. All thresholds live in `config.py`; v0-inherited values are flagged
`# unvalidated`.

## Scope & limitations

- **In:** born-digital, prose-structured PDFs (papers, reports, manuals, books).
- **Out:** scanned documents (no OCR — by design), slide decks, forms/invoices.
- Borderless tables are caught by the model, not by geometry (`find_tables` is
  ruled-line based) — a concrete motivation for the hybrid design.
- Geometry-only hierarchy is font-driven, so it can conflate title/author/section
  headers; the model resolves these semantically.
- Rotated/margin text is filtered out of reading flow.

## Testing

```bash
pytest            # 57 tests; retrieval/PDF-dependent ones self-skip if extras/PDFs absent
```

## License

MIT
