# Architecture

## One-line shape

```
PDF ─┬─> geometry detector ─┐
     └─> model detector ────┴─> fusion ─> reading order ─> extraction ─> chunking ─> Chunk[]
                                                                            │
                                                          indexing ─> query (eval-support only)
```

Two layout detectors run **completely independently and blind to each other**.
They never share state, never see each other's output. All reconciliation happens
in `fusion/`. That independence is the whole point of the design: it is what makes
the confidence scores mean something (agreement between two uncorrelated
observers) rather than being a single model's softmax dressed up.

Without model weights the pipeline runs **geometry-only** and degrades gracefully —
every block becomes `unilateral_geometry`, nothing crashes, no network is touched.

## Module map

| Module | Responsibility | Key entry point |
|---|---|---|
| `docstruct/pipeline.py` | Orchestration: detect → fuse → order → extract → chunk, per page then document-global | `run_pipeline(pdf_path, *, weights, cache_dir, model_detector)` |
| `docstruct/document.py` | The public `Document` view over a pipeline result | `docstruct.parse()` returns this; `.text/.markdown/.tables/.figures/.to_markdown()` |
| `docstruct/errors.py` | Typed exception hierarchy + `open_pdf` context manager every PDF-open site routes through | `DocStructError`, `InvalidPDFError`, `EncryptedPDFError`, `EmptyDocumentError`, `open_pdf()` |
| `docstruct/schema.py` | The entire data model — plain dataclasses, no Pydantic | `BoundingBox`, `Proposal`, `Block` (now incl. `is_bold`), `Chunk`, `SectionPath`, `Source` |
| `docstruct/config.py` | Every numeric threshold; `override()` for lock-guarded per-parse overrides | `config.override(**values)` |
| `docstruct/geometry/detector.py` | Rules-based layout detection from pdfplumber primitives (lines, words, rects, curves) | `detect(pdf_path) -> List[Proposal]` |
| `docstruct/model/detector.py` | Optional YOLOv8 / DocLayNet vision detection; pixel→point transform lives here | `ModelDetector(weights).detect()` |
| `docstruct/fusion/matcher.py` | Greedy IoU matching between the two proposal sets + priority NMS | `match_proposals(model, geometry)` |
| `docstruct/fusion/arbiter.py` | Label arbitration for matched pairs (confirmed vs disputed) | — |
| `docstruct/fusion/fusion.py` | Confidence formula, emits `Block` + `ConfidenceBreakdown` | `fuse(match_result)` |
| `docstruct/fusion/containment.py` | Nested-region suppression. Naive helpers unused; `suppress_text_in_tables` (gated `LABEL_AWARE_CONTAINMENT`) is the wired one | see `decisions.md` |
| `docstruct/reading_order.py` | Column split (1/2 or k via `MULTI_COLUMN`, or band-then-column via `BAND_SPLIT`) + top→bottom ordering; caption→figure/table attachment | `assign_reading_order(blocks, page_width)` |
| `docstruct/utils/xy_cut.py` | Recursive XY-cut ordering (off by default, `config.XY_CUT`) | `xy_cut_order(blocks, page_width)` |
| `docstruct/extraction/text_extractor.py` | Block text via pdfplumber, font-size-scaled spacing; gated dedupe/dehyphen/NFKC cleaning; `is_bold` for headers | `populate_text(pdf, blocks, *, password, pdf)` |
| `docstruct/extraction/table_extractor.py` | Table grids + plaintext/keyvalue/markdown rendering, borderless fallback, raw-text guard, gated multi-page merge | `populate_tables(...)`, `merge_multipage_tables(blocks)` |
| `docstruct/extraction/furniture.py` | Cross-page running header/footer/page-number removal (gated `STRIP_PAGE_FURNITURE`) | `strip_page_furniture(blocks)` |
| `docstruct/chunking/hierarchy_builder.py` | Header level assignment by font-size rank (+ optional bold, + numbering incl. appendix/Roman) | `assign_header_levels(blocks)` |
| `docstruct/chunking/assembler.py` | Blocks → `Chunk[]`, section-path tracking, size floor/ceiling | `build_chunks(blocks, levels)` |
| `docstruct/cache/` | Three caches: raw geometry proposals, model proposals, fully-populated blocks | `ProposalCache`, `ModelProposalCache`, `BlockCache` |
| `docstruct/indexing/vector_store.py` | sentence-transformers → ChromaDB | `VectorStore` |
| `docstruct/query/retriever.py` | Dense / hybrid (BM25 + RRF) retrieval, optional cross-encoder rerank | `Retriever` |
| `docstruct/eval/` | Detection metrics, retrieval metrics, cross-tool benchmark, gold Q&A generation | see `evaluation.md` |
| `docstruct/eval/adapters/` | Seven baselines: docstruct(+geo/model), langchain, pymupdf4llm, unstructured, docling, llamaindex, llamaindex_semantic. All emit `metadata["pages"]` | `get_adapters()` |
| `docstruct/llm/client.py` | OpenAI-compatible chat client. Providers: ollama (default), groq, openai. Adapts payloads the endpoint rejects on parameter grounds rather than carrying a model table | `LLMClient(provider=...)` |
| `docstruct/visualize.py` | Annotated-PDF overlay of detected blocks (PyMuPDF) | `render_annotated()` |
| `docstruct/cli.py` | `run`, `index`, `query`, `visualize`, `export-annotations`, `gen-qa`, `benchmark` | `main()` |

## Data model (`schema.py`)

Everything is a plain dataclass so `dataclasses.asdict` serializes it and the
core package stays dependency-free (pdfplumber + numpy only).

- **`BoundingBox`** — `x0, y0, x1, y1, page_width, page_height`. **Top-left
  origin**: `y0` is the top edge and y increases downward, matching pdfplumber's
  `top`/`bottom`. Every stage shares this space, including model output after its
  pixel→point transform. Getting this wrong is the single most common source of
  silently-wrong geometry.
- **`Proposal`** — one detector's candidate region. Has `source`
  (`"geometry"` | `"model"`) and a raw `confidence`.
- **`Block`** — a *fused* region. Carries `ConfidenceBreakdown`
  (`geometry_score`, `model_score`, `final`), a `Source` enum
  (`CONFIRMED` / `DISPUTED` / `UNILATERAL_MODEL` / `UNILATERAL_GEOMETRY`),
  `reading_order`, and post-extraction `text` / `table_data` / `font_size`.
- **`SectionPath`** — up to three header levels (`h1`, `h2`, `h3`). This is the
  metadata that enables filtered retrieval and is DocStruct's differentiator; no
  competing chunker in the benchmark emits it.
- **`Chunk`** — the retrieval unit. `chunk_type` ∈ `text` | `table` |
  `figure_caption` | `abstract` | `references`, plus `section_path`, `page_num`,
  `source_block_ids` (traceability back to blocks) and a `metadata` dict carrying
  `h1/h2/h3` and `mean_confidence`.

Labels are the five DocStruct classes: `text`, `header`, `table`, `figure`,
`caption`. DocLayNet's 11 classes are mapped down to these in
`config.DOCLAYNET_LABEL_MAP`.

## Caching layers

Three caches, all keyed by content hash so a changed PDF invalidates itself. The
layout-config fingerprint lives in `cache/pdf_cache.py` (`layout_config_fingerprint`,
`_LAYOUT_CONFIG_KEYS`) and is shared by the geometry and block caches:

1. `ProposalCache` — geometry proposals, keyed by PDF bytes **+ layout-config
   fingerprint** (`config_aware = True`). Geometry detection reads layout config, so
   the key must track it.
2. `ModelProposalCache` — model proposals, keyed by PDF bytes + weights identity,
   **config-independent** (`config_aware = False`). YOLO does not read layout config,
   so its expensive output is reused across config ablations.
3. `BlockCache` — the expensive one: fused, reading-ordered, **text-populated**
   blocks. Its key covers the PDF, the weights identity, and the layout-config
   fingerprint. **Chunking config keys are deliberately excluded**, because varying
   those cheaply is the entire purpose — a chunking ablation redoes zero detection
   work. Any new flag that changes *block* output MUST be added to
   `_LAYOUT_CONFIG_KEYS`, or an ablation of it silently serves stale blocks (this bug
   was found and fixed during the Fable review — see `decisions.md`).

Effect: full test suite 119 s → 30 s; a benchmark run ~18 min → ~10 min warm.
This is why iterating on chunking is affordable at all.

`BlockCache` is bypassed when a `model_detector` object is passed directly
(as opposed to a `weights` path), because the object's identity cannot be hashed.

## Public API surface

```python
import docstruct
doc = docstruct.parse("paper.pdf")            # geometry-only, no model, no network
doc = docstruct.parse(pathlib.Path("p.pdf"))  # str | Path
doc = docstruct.parse("paper.pdf", weights="weights/yolov8m-doclaynet.pt")
doc = docstruct.parse("locked.pdf", password="secret")
doc = docstruct.parse("p.pdf", config={"MIN_CHUNK_TOKENS": 300})  # per-call, no global mutation
doc = docstruct.parse("p.pdf", on_page=lambda i, n: print(f"{i+1}/{n}"))  # progress

doc.text, doc.markdown, doc.pages(), doc.sections(), doc.chunks
doc.chunks_of_type("table"); doc.tables; doc.figures
doc.to_json("chunks.json"); doc.to_markdown("paper.md")

# typed failures — never catch pdfminer internals
from docstruct import DocStructError, InvalidPDFError, EncryptedPDFError
```

`__init__` also re-exports `run_pipeline` and `PipelineResult`. The package ships
`py.typed`. `diagnostics["likely_scanned"]` flags image-only PDFs (born-digital
only; OCR is out-of-pipeline by contract).

`doc.markdown` renders from **blocks**, not chunks — chunks are sized for
retrieval and deliberately merge across headings, which is the wrong shape for a
document meant to be read by a human.

For fusion diagnostics and raw blocks, drop to
`docstruct.pipeline.run_pipeline()`, which returns `PipelineResult(blocks,
chunks, diagnostics)`.
