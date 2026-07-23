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

Chunks are sized along real boundaries but never *below* a floor
(`MIN_CHUNK_TOKENS`): a header, table or caption only ends the running chunk once
it holds enough text to be worth retrieving, so a page of prose interleaved with
figures stays one coherent chunk instead of becoming a handful of unretrievable
stubs.

| vs | DocStruct advantage |
|----|---------------------|
| LangChain naive loader | respects structure; section-path metadata |
| Unstructured.io / LlamaParse | fully local, free, deterministic, section-filtered retrieval |
| pymupdf4llm | higher MRR/NDCG/Recall/Hit@1 on less retrieved context (see below) |

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
 ├─ extraction/            block text + table grids (pdfplumber, font-scaled spacing)
 ├─ chunking/              header levels by font rank → section-aware chunks,
 │                         with a minimum-size floor on structural boundaries
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
docstruct query "results" --db .chroma --h1 "4. Experiments"   # section-filtered (dense)
docstruct query "BM25-XR-7 ablation" --db .chroma --hybrid     # dense + BM25 (RRF)

# Annotated overlay for inspection / figures
docstruct visualize paper.pdf --out annotated.pdf --weights weights/yolov8m-doclaynet.pt
```

### Python

```python
import docstruct

doc = docstruct.parse("paper.pdf")               # geometry-only: no model, no network
doc = docstruct.parse("paper.pdf", weights="weights/yolov8m-doclaynet.pt")

doc.text                                          # whole document, reading order
doc.markdown                                      # headings, tables, captions preserved
doc.pages()                                       # {page_num: text}
doc.sections()                                    # ["1. Introduction", "2. Method > 2.1 Setup", ...]
doc.to_json("chunks.json")

for chunk in doc.chunks:                          # retrieval-ready units
    print(chunk.chunk_type, chunk.section_path, chunk.content[:80])

doc.chunks_of_type("table")                       # table / text / figure_caption / abstract
```

`cache_dir=".cache"` caches detector output and populated blocks by PDF content
hash, so re-parsing an unchanged file is close to free. For the raw pipeline
result (fused blocks, fusion diagnostics) use `docstruct.pipeline.run_pipeline`.

```python
from docstruct.indexing.vector_store import VectorStore
from docstruct.query.retriever import Retriever

store = VectorStore(persist_dir=".chroma")
store.index(doc.chunks, doc_id="paper")
for r in Retriever(store).retrieve("how was the corpus prepared?", top_k=3):
    print(r.citation())              # [Corpus preparation] (page 2, score 0.48)

# hybrid (dense + BM25 via RRF), section-scoped, optionally cross-encoder reranked
retriever = Retriever(store, hybrid=True, rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever.retrieve("ablation results", top_k=5, where={"h1": "4. Experiments"})
```

Both retrieval modes honour `where`. Reranking is off unless `rerank_model` is
set — it loads a second model at query time, which the local-by-default design
otherwise avoids.

---

## Evaluation

Two layers, in `docstruct.eval`:

- **Detection** — per-class precision/recall/F1 and confidence-ranked **mAP@0.5**
  against ground-truth boxes (`evaluate_detection`). Ground truth is a simple
  per-document JSON (`{"boxes": [{"label", "page_num", "bbox": [x0,y0,x1,y1,pw,ph]}]}`);
  build it with `docstruct export-annotations` + `tools/annotate.html`.
- **Retrieval** — **MRR / NDCG@k / Recall@k / Hit@1** over LLM-generated Q&A.

### Cross-tool benchmark

A **fair** comparison against the free/local tools people actually use
(LangChain, PyMuPDF4LLM, Unstructured.io, Docling): the embedder and retriever
are held constant and **only the chunker varies**, so the result measures
chunking quality. Gold is `(question, verbatim answer_span)` generated by an LLM
(Ollama cloud, eval-only); a retrieved chunk is relevant if it **contains** the
span — tool-agnostic, so every chunker is scored identically.

Latest run — 48 born-digital PDFs, 298 LLM-generated Q&A, identical embedder and
retriever for every tool:

| Rank | Tool | MRR | NDCG@5 | Recall@5 | Hit@1 | Avg words/chunk | Context words |
|---|---|---|---|---|---|---|---|
| 1 | **docstruct** | **0.7457** | **0.7708** | **0.8859** | **0.6409** | 355.2 | 2346 |
| 2 | pymupdf4llm | 0.6941 | 0.7160 | 0.8356 | 0.6107 | 455.2 | 2576 |
| 3 | unstructured | 0.6508 | 0.6766 | 0.7886 | 0.5638 | 85.2 | 549 |
| 4 | langchain | 0.6493 | 0.6884 | 0.8221 | 0.5336 | 102.1 | 524 |
| 5 | docling | 0.5652 | 0.5814 | 0.6577 | 0.4966 | 114.2 | 674 |

**Context words** is the text actually handed to the generator per query (summed
over the retrieved top-5). MRR can always be bought by emitting bigger chunks, so
the leaderboard reports the price: DocStruct leads on every quality metric *and*
returns less text per query than the tool it beats. It does not lead on MRR per
1000 context words — the small-chunk tools do, because they retrieve very little;
that column is a tradeoff axis, not a ranking.

```bash
docstruct gen-qa data/raw-pdfs/*.pdf --out data/qa/qa.json \
  --weights weights/yolov8m-doclaynet.pt --per-doc 5 --cache-dir .bench_cache
docstruct benchmark --pdfs-dir data/raw-pdfs --qa data/qa/qa.json \
  --weights weights/yolov8m-doclaynet.pt --cache-dir .bench_cache
# -> reports/v4_report.md  (leaderboard + methodology + caveats)
```

See [`reports/v4_report.md`](reports/v4_report.md) for the full run, including
per-document breakdowns and the config snapshot that produced it, and
[`notes.md`](notes.md) for how the numbers got there.

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
pytest            # ~80 tests; retrieval/PDF/LLM-dependent ones self-skip if extras/PDFs absent
```

## License

MIT
