# DocStruct Optimization Blueprint & Implementation Plan

This document serves as a complete technical blueprint for the layout parsing, reading order, and retrieval improvements in DocStruct. It compiles all recommendations from the Codex review and our proposed optimizations, detailing the precise code structures, algorithms, and configuration keys to be updated.

---

## 1. Core Configuration Changes

### File to Modify: `docstruct/config.py`
We will introduce flags to make all new behaviors fully configurable, allowing side-by-side comparison with baseline heuristics.

```python
# --- Evaluation / Ablation Control ---
XY_CUT = True                       # Use Recursive XY-Cut instead of simplistic split
SUPPRESS_CONTAINED = True           # Enable containment suppression in run_pipeline()
PREPEND_SECTION_PATH = True         # Prepend the H1 > H2 > H3 section context to chunk text
POOLED_CORPUS = False               # Benchmark using pooled-corpus retrieval (cross-doc)
CONFIDENCE_WEIGHTED_RRF = False     # Use fusion confidence to scale retrieval ranks
RRF_CONFIDENCE_MULTIPLIER = 0.15    # Scale factor for confidence boost in RRF
```

---

## 2. Reading Order & Column Segmentation

### New File: `docstruct/utils/xy_cut.py`
This module implements the **Recursive XY-Cut** (RXYC) algorithm to replace the legacy center-based 1/2 column gap splitter.

```python
from typing import List, Tuple
from docstruct.schema import Block, BoundingBox

def recursive_xy_cut(blocks: List[Block], page_width: float, page_height: float) -> List[Block]:
    """Sort blocks recursively along horizontal and vertical whitespace bands."""
    if len(blocks) <= 1:
        return blocks

    # 1. Project bounding boxes onto X and Y axes
    # Horizontal projection (Y coordinates)
    y_intervals = sorted([(b.bbox.y0, b.bbox.y1, idx) for idx, b in enumerate(blocks)], key=lambda x: x[0])
    # Vertical projection (X coordinates)
    x_intervals = sorted([(b.bbox.x0, b.bbox.x1, idx) for idx, b in enumerate(blocks)], key=lambda x: x[0])

    # Helper: find whitespace gaps
    def find_gaps(intervals: List[Tuple[float, float, int]], limit: float) -> List[Tuple[float, float]]:
        gaps = []
        max_seen = intervals[0][1]
        for start, end, _ in intervals[1:]:
            if start > max_seen:
                gaps.append((max_seen, start))
            max_seen = max(max_seen, end)
        return gaps

    # Try horizontal split (Y-cut) first
    y_gaps = find_gaps(y_intervals, page_height)
    if y_gaps:
        # Split at the largest horizontal gap
        split_y = max(y_gaps, key=lambda g: g[1] - g[0])
        mid = (split_y[0] + split_y[1]) / 2.0
        top_half = [b for b in blocks if b.bbox.y1 <= mid]
        bottom_half = [b for b in blocks if b.bbox.y0 >= mid]
        # In case some blocks cross the cut, assign them to the half where they overlap most
        for b in blocks:
            if b not in top_half and b not in bottom_half:
                if (mid - b.bbox.y0) > (b.bbox.y1 - mid):
                    top_half.append(b)
                else:
                    bottom_half.append(b)
        
        return recursive_xy_cut(top_half, page_width, page_height) + recursive_xy_cut(bottom_half, page_width, page_height)

    # Try vertical split (X-cut) next
    x_gaps = find_gaps(x_intervals, page_width)
    if x_gaps:
        # Split at the largest vertical gap
        split_x = max(x_gaps, key=lambda g: g[1] - g[0])
        mid = (split_x[0] + split_x[1]) / 2.0
        left_half = [b for b in blocks if b.bbox.x1 <= mid]
        right_half = [b for b in blocks if b.bbox.x0 >= mid]
        for b in blocks:
            if b not in left_half and b not in right_half:
                if (mid - b.bbox.x0) > (b.bbox.x1 - mid):
                    left_half.append(b)
                else:
                    right_half.append(b)
        
        return recursive_xy_cut(left_half, page_width, page_height) + recursive_xy_cut(right_half, page_width, page_height)

    # Base case: no clean gaps; sort top-to-bottom, left-to-right
    return sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
```

### File to Modify: `docstruct/reading_order.py`
We will integrate RXYC into `assign_reading_order` when enabled:

```python
def assign_reading_order(blocks: List[Block], page_width: float) -> List[Block]:
    if not blocks:
        return []
    
    if config.XY_CUT:
        from docstruct.utils.xy_cut import recursive_xy_cut
        page_height = blocks[0].bbox.page_height if hasattr(blocks[0].bbox, 'page_height') else 800.0
        sorted_blocks = recursive_xy_cut(blocks, page_width, page_height)
        for rank, block in enumerate(sorted_blocks):
            block.reading_order = rank
    else:
        # Legacy center-based gap sort
        order = sort_reading_order(blocks, page_width)
        order_map = {idx: position for position, idx in enumerate(order)}
        for i, block in enumerate(blocks):
            block.reading_order = order_map.get(i, len(blocks))

    attach_captions(blocks)
    return blocks
```

### File to Modify: `docstruct/geometry/detector.py`
Update `_split_columns` in the rule-based geometry detector to use Recursive XY-Cut for grouping line clusters:

```python
def _split_columns(lines: List[_Line], page_width: float) -> List[List[_Line]]:
    if not lines:
        return []
        
    if config.XY_CUT:
        # Construct fake blocks to run RXYC on line clusters
        from docstruct.schema import Block, BoundingBox
        from docstruct.utils.xy_cut import recursive_xy_cut
        
        fake_blocks = []
        for idx, line in enumerate(lines):
            fake_blocks.append(Block(
                block_id=f"line_{idx}",
                label="text",
                bbox=BoundingBox(line.x0, line.top, line.x1, line.bottom, page_width, 800.0),
                page_num=0,
                reading_order=idx
            ))
            
        sorted_fake = recursive_xy_cut(fake_blocks, page_width, 800.0)
        
        # Simple clustering: if page split vertically, split sorted_fake into columns
        # Based on largest horizontal gap or midpoints
        # For simplicity, divide into columns if they lie horizontally on left/right half
        left = []
        right = []
        mid = page_width / 2
        for f in sorted_fake:
            line_idx = int(f.block_id.split("_")[1])
            line = lines[line_idx]
            if (line.x0 + line.x1) / 2 < mid:
                left.append(line)
            else:
                right.append(line)
        if left and right:
            return [left, right]
        return [lines]
    else:
        # Legacy gap-based logic...
        ...
```

---

## 3. Pipeline & Containment Suppression

### File to Modify: `docstruct/pipeline.py`
Apply containment suppression after fusing proposals to remove nested duplicate text blocks and figure components. Also add an optional `pipeline_mode` parameter to support model-only/geometry-only benchmark paths:

```python
def run_pipeline(
    pdf_path: str,
    *,
    model_detector=None,
    weights: Optional[str] = None,
    cache_dir: Optional[str] = None,
    pipeline_mode: Optional[str] = None,  # None, "geometry-only", or "model-only"
) -> PipelineResult:
    ...
    # Bypass geometry extraction in model-only mode
    if pipeline_mode == "model-only":
        geometry_props = []
    else:
        geometry_props = _cached_detect(geometry_detector.detect, pdf_path, geo_cache)

    # Bypass model extraction in geometry-only mode
    if pipeline_mode == "geometry-only":
        model_props = []
        mode = "geometry-only"
    else:
        # Standard model detection logic...
        ...

    # ... matching & fusion ...
    blocks = fuse(result)
    
    # [NEW] Apply containment suppression to strip duplicates
    if config.SUPPRESS_CONTAINED:
        blocks = suppress_contained(blocks)
        
    for block in blocks:
        block.block_id = f"block_{id_counter:04d}"
        id_counter += 1
    ...
```

---

## 4. Retrieval Section-Aware Pre-Filtering

### File to Modify: `docstruct/query/retriever.py`
We will modify the BM25 initialization in hybrid mode to respect metadata filters (`where`).

```python
    def _ensure_corpus(self, where: Optional[dict] = None):
        # Generate cache key based on the 'where' filter
        where_key = frozenset(where.items()) if where else None
        
        # Build BM25 only on the filtered subset of documents
        if self._corpus is None or self._corpus_key != where_key:
            from rank_bm25 import BM25Okapi

            got = self.store.collection.get(where=where, include=["documents", "metadatas"])
            ids = got["ids"]
            docs = got["documents"] or []
            metas = got.get("metadatas") or [{} for _ in ids]
            bm25 = BM25Okapi([(d or "").lower().split() for d in docs])
            self._corpus = (ids, docs, metas, bm25)
            self._corpus_key = where_key
            
        return self._corpus

    def _hybrid(self, query, top_k, where: Optional[dict] = None) -> List[RetrievalResult]:
        ids, docs, metas, bm25 = self._ensure_corpus(where=where)
        if not ids:
            return []
        
        pool = max(top_k, config.BM25_CANDIDATES)
        # Search Chroma with where filter
        resp = self.store.query(query, top_k=min(pool, len(ids)), where=where)
        dense_ids = resp.get("ids", [[]])[0]

        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:pool]
        bm_ids = [ids[i] for i in order]

        # RRF fusion
        fused, rrf_score = _rrf([dense_ids, bm_ids], self.rrf_k)
        
        # Inject layout confidence weighted adjustment if enabled
        if config.CONFIDENCE_WEIGHTED_RRF:
            by_id_meta = {cid: meta for cid, meta in zip(ids, metas)}
            for cid in rrf_score:
                meta = by_id_meta.get(cid) or {}
                conf = float(meta.get("mean_confidence") or 1.0)
                rrf_score[cid] *= (1.0 + config.RRF_CONFIDENCE_MULTIPLIER * conf)
            fused = sorted(rrf_score, key=lambda c: rrf_score[c], reverse=True)

        by_id = {cid: (doc, meta) for cid, doc, meta in zip(ids, docs, metas)}

        results: List[RetrievalResult] = []
        for cid in fused[:top_k]:
            doc, meta = by_id.get(cid, ("", {}))
            meta = meta or {}
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    content=doc,
                    chunk_type=meta.get("chunk_type", "text"),
                    page_num=int(meta.get("page_num", -1)),
                    section_path=_section_label(meta),
                    score=round(rrf_score[cid], 4),
                )
            )
        return results
```

---

## 5. Evaluation & Benchmark Scaling

### File to Modify: `docstruct/eval/adapters/docstruct_adapter.py`
Enable section-path prepending and pass the `pipeline_mode` directly to `run_pipeline()`:

```python
class DocStructAdapter(ChunkAdapter):
    name = "docstruct"

    def __init__(self, weights: Optional[str] = None, cache_dir: Optional[str] = None, pipeline_mode: Optional[str] = None) -> None:
        self.weights = weights
        self.cache_dir = cache_dir
        self.pipeline_mode = pipeline_mode

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        from docstruct.pipeline import run_pipeline

        result = run_pipeline(pdf_path, weights=self.weights, cache_dir=self.cache_dir, pipeline_mode=self.pipeline_mode)
        
        chunks = []
        for c in result.chunks:
            text = c.content
            sec = _section(c)
            if config.PREPEND_SECTION_PATH and sec:
                text = f"[Section: {sec}]\n\n{text}"
                
            chunks.append(EvalChunk(
                id=c.chunk_id,
                text=text,
                metadata={"section": sec, "page": c.page_num, "type": c.chunk_type, "mean_confidence": c.metadata.get("mean_confidence")},
            ))
        return chunks
```

### File to Modify: `docstruct/eval/adapters/__init__.py`
Register the ablation test adapters:

```python
def build_adapter(
    name: str, weights: Optional[str] = None, cache_dir: Optional[str] = None
) -> ChunkAdapter:
    if name == "docstruct":
        return DocStructAdapter(weights=weights, cache_dir=cache_dir)
    if name == "docstruct_geo":
        return DocStructAdapter(weights=None, cache_dir=cache_dir, pipeline_mode="geometry-only")
    if name == "docstruct_model":
        return DocStructAdapter(weights=weights, cache_dir=cache_dir, pipeline_mode="model-only")
    ...
```

### File to Modify: `docstruct/eval/benchmark.py`
Incorporate **Pooled Corpus Indexing** and **Bootstrap Significance Intervals**:

```python
# --- Bootstrap CI Helper ---
def compute_bootstrap_ci(scores: List[float], num_samples: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
    import numpy as np
    means = []
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    for _ in range(num_samples):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, ((1.0 - ci_level) / 2.0) * 100.0)
    upper = np.percentile(means, (1.0 - (1.0 - ci_level) / 2.0) * 100.0)
    return round(float(lower), 4), round(float(upper), 4)


def benchmark_tool(
    adapter: ChunkAdapter,
    pdf_paths: List[str],
    qa: List[QAItem],
    embedder,
    ...
) -> ToolResult:
    ...
    # If config.POOLED_CORPUS is enabled, index all chunks across all documents
    # into a single shared Chroma Collection prior to querying.
    if config.POOLED_CORPUS:
        # 1. Chunk all docs first
        all_chunks: List[EvalChunk] = []
        chunk_map: List[Tuple[int, str]] = [] # mapping index to source doc
        for doc_idx, pdf in enumerate(pdf_paths):
            chunks = adapter.chunk(pdf)
            all_chunks.extend(chunks)
            
        # 2. Add all chunks to one single Vector Store
        store = VectorStore(collection_name=f"bench_{adapter.name}_pooled", embedder=embedder)
        texts = [c.text for c in all_chunks]
        metas = [c.metadata for c in all_chunks]
        store.collection.add(
            ids=[f"{i}" for i in range(len(texts))],
            documents=texts,
            embeddings=embedder.encode(texts, show_progress_bar=False).tolist(),
            metadatas=metas
        )
        
        # 3. Query pooled corpus for each QA case
        # ...
    else:
        # Legacy loop for per-document isolated indexing
        ...
```

### File to Modify: `docstruct/cli.py`
Add benchmark flags for the new modes:

```python
    b_p.add_argument("--pooled", action="store_true", help="run pooled corpus evaluation")
    b_p.add_argument("--bootstrap", action="store_true", help="enable bootstrap confidence intervals")
    b_p.add_argument("--confidence-weighted", action="store_true", help="enable confidence scaling in hybrid RRF")
```

---

## 6. Verification Tests

### File to Modify: `tests/test_reading_order.py`
Add tests for XY-Cut layouts:

```python
def test_recursive_xy_cut_abstract_and_body():
    from docstruct.schema import Block
    from docstruct.utils.xy_cut import recursive_xy_cut
    from tests.conftest import make_bbox
    
    # Simulate a title block at the top, and two columns below it
    blocks = [
        Block("title", "header", make_bbox(50, 50, 550, 90), 0),
        Block("col1", "text", make_bbox(50, 100, 290, 400), 0),
        Block("col2", "text", make_bbox(310, 100, 550, 400), 0),
    ]
    
    sorted_blocks = recursive_xy_cut(blocks, 600.0, 800.0)
    assert [b.block_id for b in sorted_blocks] == ["title", "col1", "col2"]
```

---

## 7. Additional Improvements (not yet in scope above)

Found while auditing the codebase against this plan — not implemented, not scheduled elsewhere.

### 7.1 Cross-encoder reranking (highest expected ROI)
`cli.py` already declares `--rerank-model` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) but it is not
wired into the actual scoring path in `docstruct/eval/benchmark.py` or `docstruct/query/retriever.py`.
Add a rerank stage: retrieve top-20 via existing hybrid RRF, rerank with the cross-encoder, return top-5.
Confirm the flag is dead/unused before building — grep `benchmark.py` and `retriever.py` for
`rerank_model` usage first.

### 7.2 Diagnose worst-performing docs before further algorithm changes
`doc44.pdf` (MRR 0.24) and `doc19.pdf` (MRR 0.25) are outliers vs the ~0.5-1.0 range of the rest.
Dump their block/chunk output (`docstruct run doc44.pdf --json out.json`) and inspect: table-heavy
layout, multi-column edge case, malformed section hierarchy. Target the real failure mode instead of
tuning general heuristics blind.

### 7.3 Cross-boundary chunk overlap
`chunking/assembler.py`: `flush_text()` only keeps overlap (`keep_overlap=True`) on a token-limit flush.
Every flush triggered by a header/table/caption boundary calls `flush_text()` bare — the first chunk of
a new section loses the last sentence of context from the section before it. Apply `keep_overlap` on all
flush paths, or a smaller dedicated overlap for section boundaries.

### 7.4 Standalone figures produce no chunk
`chunking/assembler.py:139-140` — `figure: continue`. A figure with no caption is dropped entirely,
zero retrievable content. Add a fallback minimal `figure` chunk (page/bbox/section-path, block label) so
uncaptioned figures aren't silently lost from the index. No VLM/captioning model — that would violate the
no-LLM-calls-in-pipeline principle (see note below); metadata-only fallback stays deterministic and local.

### 7.5 Confidence formula is unvalidated
`config.py` marks `UNILATERAL_MODEL_SCALE`, `UNILATERAL_GEOMETRY_SCALE`, and both `CONFIDENCE_BOUNDS`
entries `# unvalidated`. These feed directly into `CONFIDENCE_WEIGHTED_RRF` (section 4 above) — do a
small calibration pass against the annotated eval set before trusting a confidence-weighted retrieval
rank built on untuned constants.

### 7.6 Tiny trailing chunks aren't merged
No minimum-chunk-size floor in `assembler.py`. A short leftover buffer at a section boundary becomes
its own low-context chunk (visible in per-doc data: several docs average 53-56 words/chunk). Merge a
sub-threshold trailing buffer into the next section's opening buffer instead of always cutting clean at
header boundaries.

### 7.7 Table handling gaps
- `table_extractor.py`: tables are atomic (never split) with no size cap — a table exceeding
  `MAX_CHUNK_TOKENS` produces an oversized outlier chunk with no guard.
- `table_to_markdown()` is defined but never called; `table_to_plaintext()` is what's actually wired
  into `populate_tables()`. Dead code — either remove it or use it (e.g. behind a config flag for
  Markdown-table chunk output instead of plaintext).

### 7.8 Deliberately excluded: embedding-similarity semantic chunking
Considered and rejected. DocStruct's core contract (README) is "no LLM calls in the pipeline, same PDF
in -> same chunks out, auditable, fully local." Embedding-similarity boundary detection introduces
model-version/hardware float drift (weakens determinism) and replaces the thing DocStruct is positioned
against — black-box heuristic chunking — with exactly that. It's also redundant: geometry+vision fusion
already gives ground-truth structure (headers/tables/captions), which is a better boundary signal than
inferred similarity for documents that have real structure.
If un-headered long-prose runs ever need finer splitting than the token-limit cutoff, the deterministic
fallback is sentence-boundary-aware splitting (regex/spacy sentence boundaries, no model) — not embedding
similarity. Not scheduled; only worth doing if 7.2's diagnosis surfaces it as a real failure mode.

### 7.9 Report/config provenance gap
`reports/rrf40_results.json` `meta` block only records timestamp/doc-count/question-count/model — no
snapshot of which config flags (`XY_CUT`, `SUPPRESS_CONTAINED`, `CONFIDENCE_WEIGHTED_RRF`, etc.) were
active for that run. Two reports with different MRR give no way to know *why* without manually diffing
`config.py` at that git commit. Undermines the "auditable" claim and blocks clean before/after comparison
across plan changes. Dump the full config dict into report `meta` on every benchmark run.

### 7.10 No embedding cache across benchmark reruns
Chunks get re-embedded from scratch on every rerun even when chunking logic didn't change (e.g. rerunning
after a retrieval-only change like `CONFIDENCE_WEIGHTED_RRF`). Cache embeddings keyed by chunk-text hash
to cut iteration time for retrieval-only changes; doesn't help chunking-logic reruns, which legitimately
produce different chunks.

### 7.11 Header hierarchy relies on font-size alone
`hierarchy_builder.py` ranks header levels purely by font-size rank — deterministic and document-agnostic
(good), but a doc where subsections share font size with a different weight, or use numeric prefixes
("3.2.1 Methods") that don't correlate with size, will misassign levels. Add numbering-pattern regex
(`^\d+(\.\d+)*\s`) as a secondary deterministic signal to break font-size ties. Zero-model, stays in the
structural-only contract.

### 7.12 Hybrid retrieval silently ignores `where`
`retriever.py` docstring states it outright: "Hybrid ignores `where` (BM25 runs over the whole
collection); use dense mode for section-filtered queries." Section-filtered + hybrid-quality retrieval is
not simultaneously possible today. Covered by plan section 4 as a fix, but flagged here as a correctness
gap, not just a nice-to-have — a caller assuming `where` works in hybrid mode silently gets wrong-scoped
results with no error.

**Ranked by expected ROI:** 7.1 (reranker) > 7.2 (diagnose worst docs) > 7.9 (report provenance) >
7.12 (hybrid where) > 7.3 (overlap) > 7.4 (figures) > 7.6 (tiny chunks) > 7.7 (tables) >
7.11 (header numbering) > 7.5 (confidence calibration) > 7.10 (embedding cache).

---

## 8. What DocStruct actually is

Not a RAG framework. It's a **local, deterministic chunking/extraction library** with a thin
retrieval layer (`indexing/`, `query/retriever.py`) and a benchmark harness (`eval/`) bolted on so
the chunker can be measured against RAG baselines. The core contract (no LLM calls, same PDF in →
same chunks out) lives entirely in geometry/model detection → fusion → reading order → chunking.
Indexing/query/rerank exist to prove the chunks are retrieval-good, not as a product surface in
their own right — `cli.py` exposes `run` (bare chunking) as the primary command; `index`/`query` are
secondary/eval-support commands.

## 9. Package legitimacy check (2026-07-23)

Verified, not just read from the plan:
- Installed editable, real package: `pip show docstruct` → v0.2.0, entry point `docstruct.cli:main`.
  **Bug:** editable install metadata still points at `C:\...\projects\DocStruct` (pre-move path);
  import still resolves correctly but the `docstruct.exe` console-script shim is broken (exit 1,
  no output) — use `python -m` / `from docstruct.cli import main` instead until `pip install -e .`
  is rerun from the current path.
- `pytest -q` → **107/107 passed**.
- `docstruct run` smoke-tested end to end on `data/raw-pdfs/doc1.pdf`: 22 pages → 180 blocks → 33
  chunks, section paths attached, geometry-only mode (no model weights loaded). Real, working
  pipeline, not vaporware.
- First smoke-test attempt used `data/doc1_annotated.pdf` — that file is the *output* of
  `docstruct visualize` (labels burned onto the page by `visualize.py:44`,
  `f"{block.label}:{block.source.value}"`), not a source PDF. Extracting text from it pulled the
  overlay label text in on top of the real content. That was a testing mistake on my part, not a
  DocStruct bug — retracted.
- **Real bug found on the clean PDF** (`doc1.pdf`, not the annotated one): several blocks have
  doubled interior letters — `"Trannsfer may hhave many mmeanings ass well as appplication
  doomains"`, `"preccipitation  mmm deww point  C"`, `"obsserved by Aiir Quality Monitoring
  Staation"`. Confirmed document-specific, not systemic: `doc10.pdf` extracts cleanly through the
  same code path. Likely cause: this PDF's font renders faux-bold via duplicate offset glyphs, and
  `pdfplumber.extract_text()` (`docstruct/extraction/text_extractor.py:36`) isn't deduping the
  overlap at this tolerance. Same failure family as **7.2** (doc44/doc19/doc17 MRR outliers) — worth
  checking during that diagnosis pass rather than as a separate fix.

## 10. Plan-vs-code audit (sections 1-6 above)

What the plan proposed vs. what's actually in the repo right now:

| Section | Proposed | Actual state |
|---|---|---|
| 1. Config flags | 6 new flags (`XY_CUT`, `SUPPRESS_CONTAINED`, `PREPEND_SECTION_PATH`, `POOLED_CORPUS`, `CONFIDENCE_WEIGHTED_RRF`, `RRF_CONFIDENCE_MULTIPLIER`) | Only `XY_CUT` exists (`config.py:42`), and it's `False` — commit `863bdda` shipped it, measured it worse than the legacy splitter, shipped it off by default. The other 5 flags were never added. |
| 2. Recursive XY-Cut | New `docstruct/utils/xy_cut.py`, function `recursive_xy_cut(blocks, page_width, page_height)` | File exists but the real implementation diverged from the plan's pseudocode entirely — actual function is `xy_cut_order(blocks, page_width)`, different signature, own algorithm. Wired into `reading_order.py:52` behind `config.XY_CUT`. Has real tests (`test_xy_cut_reads_two_columns_in_order`, empty/single-block edge cases). Done, just not as drafted, and disabled. |
| 3. Containment suppression | Call `suppress_contained()` in `run_pipeline()` after fusion | **Dead import.** `pipeline.py:20` imports `suppress_contained`/`suppress_table_contained` from `fusion/containment.py` — neither is ever called anywhere in the codebase (verified by grep across all `.py` files). The functions are implemented and presumably tested in isolation, but not wired in. `pipeline_mode` param (`model-only`/`geometry-only` bypass) also never landed — `run_pipeline()`'s only params are `pdf_path`, `model_detector`, `weights`, `cache_dir`. |
| 4. Hybrid retrieval `where` filter | Make `_ensure_corpus`/`_hybrid` honour `where` for BM25 | **Done**, and further along than the plan draft — `retriever.py` docstring explicitly states both dense and hybrid modes honour `where` now, `_ensure_corpus` takes `where` and caches per filter key. |
| 5. Section-path prepending in eval | `DocStructAdapter.chunk()` prepends `[Section: ...]` when `config.PREPEND_SECTION_PATH` | Not implemented. Current `docstruct/eval/adapters/docstruct_adapter.py` has no `pipeline_mode` param, no section-prepend logic, no config flag — matches the pre-plan version described in the diff, not the target. |
| 5b. `pipeline_mode` adapter routing (`docstruct_geo`/`docstruct_model`) | Register ablation adapters in `eval/adapters/__init__.py` | Not implemented — depends on 3's `pipeline_mode` param, which doesn't exist either. |
| 5c. Pooled-corpus indexing + bootstrap CI | `benchmark.py`: `compute_bootstrap_ci()`, `POOLED_CORPUS` branch in `benchmark_tool()` | Not implemented — no `bootstrap` or `POOLED_CORPUS` references anywhere in `eval/benchmark.py`. |
| 5d. CLI benchmark flags (`--pooled`, `--bootstrap`, `--confidence-weighted`) | New argparse flags | Not implemented — not in `cli.py`. |
| 6. XY-Cut verification test | `test_recursive_xy_cut_abstract_and_body` | Superseded — actual tests target `xy_cut_order`, not `recursive_xy_cut` (matches 2's divergence), but coverage intent is met. |
| 7.1 Cross-encoder reranker | Flagged "declared but not wired" | **Now wired**, contradicting the plan text above — `query/retriever.py` (`_ensure_reranker`, `_rerank`, used in both dense/hybrid before top-k truncation) and `eval/benchmark.py` (`reranker_model` → `CrossEncoder` → passed into `benchmark_tool`) both use it live. Plan section 7.1 is stale; reclassify as done. |

**Net:** reading-order and retrieval-filtering work (2, 4) shipped and is ahead of the plan.
Fusion-side cleanup (3) is half-done — built, imported, never called. Eval/benchmark tooling (5, 5b,
5c, 5d) is entirely unbuilt. 7.1 flipped from "not done" to done since the plan was written.
