# Interview Prep: DocStruct

**Description**: A local, deterministic, structure-aware PDF chunking library that fuses rule-based geometry detection with an optional YOLOv8 vision model to produce section-annotated chunks for RAG retrieval.
**Tech Stack**: Python 3.11+, pdfplumber, PyMuPDF (fitz), ultralytics (YOLOv8), ChromaDB, sentence-transformers (all-MiniLM-L6-v2), pytest
**Generated**: 2026-06-10

---

## Section 1 — High-Level System Overview

DocStruct solves a core problem with naive RAG pipelines: if you just window a PDF into 400-token chunks, you shred the structure. Tables split mid-row, figure captions detach from their figures, section headers float disconnected from the paragraphs they introduce. A retrieval system built on those chunks loses the document's semantic skeleton.

The architecture is a linear pipeline that splits concern cleanly: two detectors run independently and never see each other's output, a fusion layer reconciles them, and the downstream stages (reading order, extraction, chunking, indexing) only see the reconciled result. The geometry detector uses pdfplumber to extract word-level metadata — font sizes, bounding boxes, bold markers — and applies document-agnostic rules to classify regions as text, header, table, figure, or caption. The model detector, when weights are provided, rasterizes each PDF page at 150 DPI and runs a YOLOv8 model trained on DocLayNet's 11 layout classes, remapping them onto DocStruct's five-label taxonomy.

Fusion in `fusion/matcher.py` runs greedy IoU matching between the two proposal streams, then non-maximum suppression (NMS) to clean up duplicates, prioritizing matched pairs over unilateral detections. Matched pairs go to the arbiter, which labels them as "confirmed" (both detectors agree on the label, high confidence) or "disputed" (they disagree, the winner's confidence is penalized by 15%). Unmatched proposals from either detector become "unilateral" blocks at reduced confidence.

After fusion, blocks get reading order assigned via a column-detection heuristic (largest x-center gap as the column split point), text and table content is extracted back from the PDF by cropping each block's bounding box, and then the chunking layer assembles them into retrieval-ready `Chunk` objects with a `SectionPath` (h1/h2/h3). Section levels are inferred by ranking header font sizes — the largest distinct size becomes h1. Finally, chunks can be embedded via sentence-transformers and stored in ChromaDB with their section path as metadata, enabling filtered queries like `where={"h1": "3. Methodology"}`.

The whole pipeline is fully deterministic — same PDF always produces the same chunks. No LLMs are called at inference time. The model detector is optional; without weights the pipeline runs geometry-only and degrades gracefully, marking every block as `unilateral_geometry`.

**What I would do differently given more time:**
- The column-detection logic (both in the geometry detector and in reading order) finds only the single largest gap, so three-column layouts break. A recursive or density-based approach would handle them.
- Unilateral confidence bounds and scaling factors in `config.py` are marked `# unvalidated` — they were inherited from the prototype and need tuning against the annotated evaluation set.
- The annotation tooling (`tools/annotate.html`) is a self-contained HTML file; a proper web app with keyboard shortcuts and box editing would make producing ground truth much faster.

---

## Section 2 — File-by-File Breakdown

### `docstruct/schema.py`

Defines every data structure that flows through the pipeline.

**What it does:** Central type library using plain Python dataclasses (no Pydantic). Every stage of the pipeline passes `Proposal`, `Block`, and `Chunk` objects. `BoundingBox` uses top-left coordinates throughout, matching pdfplumber's convention. `Source` is an enum encoding the four fusion outcomes.

**How it works:** The type hierarchy is: `Proposal` (raw detector output) → `Block` (fused, with `ConfidenceBreakdown` and `Source`) → `Chunk` (retrieval-ready, with `SectionPath`). Using dataclasses means everything serializes cleanly with `dataclasses.asdict()` with no external dependency.

**Important code snippet:**
```python
class Source(str, Enum):
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    UNILATERAL_MODEL = "unilateral_model"
    UNILATERAL_GEOMETRY = "unilateral_geometry"

@dataclass
class Block:
    bbox: BoundingBox
    label: Label
    confidence: ConfidenceBreakdown
    source: Source
    page_num: int
    block_id: str
    reading_order: int = -1
    caption_target_id: str | None = None
    text: str | None = None
```

**Interview talking points:**
- Why dataclasses instead of Pydantic? Keeps the core dependency-free — `pdfplumber` and `numpy` are the only required deps. Retrieval, model, and viz are all optional extras. Pydantic would drag in a non-trivial dependency for something that doesn't need runtime validation.
- Why is `Source` a `str, Enum`? It serializes cleanly to JSON without a custom encoder, since `asdict()` just calls `str()` on it.
- What does `caption_target_id` do? Links a caption block to its nearest figure or table block so the chunker can emit them as a paired `figure_caption` chunk rather than orphaning the caption.

---

### `docstruct/config.py`

Central registry of every numeric threshold in the pipeline.

**What it does:** Single source of truth for all magic numbers — IoU thresholds, confidence formulas, font size ratios, max chunk tokens, embedding model name. No hardcoded literals anywhere else in the codebase.

**How it works:** Module-level constants. Values marked `# unvalidated` are known to be prototype-era estimates that haven't been tuned against the evaluation set.

**Important code snippet:**
```python
IOU_MATCH_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.5
CONFIRMED_BASE = 0.85
CONFIRMED_MODEL_BOOST = 0.10
CONFIRMED_AGREEMENT_BOOST = 0.05
DISPUTED_MULTIPLIER = 0.85
DOCLAYNET_LABEL_MAP = {
    "section-header": "header",
    "title": "header",
    "table": "table",
    "picture": "figure",
    "footnote": "text",
    ...
}
```

**Interview talking points:**
- Why keep all thresholds in one file rather than near the code that uses them? Makes tuning and ablation studies easy — you can change `IOU_MATCH_THRESHOLD` once and rerun the entire eval harness without hunting through multiple files.
- Why does `DOCLAYNET_LABEL_MAP` exist? DocLayNet has 11 classes; DocStruct uses 5. The map collapses `footnote`, `formula`, `list-item`, `page-footer`, `page-header` all into `text`, and `section-header` + `title` both become `header`.

---

### `docstruct/geometry/detector.py`

Rules-based layout detector using pdfplumber word metadata.

**What it does:** Produces `List[Proposal]` purely from the PDF's own text layer — no ML inference. Handles tables (pdfplumber's `find_tables()`), text blocks grouped by whitespace gaps, column splitting, and figure detection from raster images and clustered vector graphics.

**How it works:** Per page: extract tables first (they mask out their region from text and graphics detection). Extract words with font metadata, group them into `_Line` objects by vertical proximity (`LINE_Y_TOLERANCE = 3.0`), classify each line as `header` or `body` by comparing its font size to the page median (with a `1.15×` ratio threshold), split lines into columns by the largest x-center gap, then group lines into blocks by whitespace gap (gap > `1.6 × median_line_height` breaks a block). Finally detect figures by clustering raster images and vector graphic primitives within `FIGURE_CLUSTER_GAP = 10` points.

**Important code snippet:**
```python
def _line_kind(line: _Line, body_median: float) -> str:
    if body_median > 0 and line.size > body_median * config.HEADER_SIZE_RATIO:
        return "header"
    if (
        line.bold
        and len(line.words) <= config.HEADER_MAX_WORDS
        and (body_median == 0 or line.size >= body_median)
    ):
        return "header"
    return "body"
```

**Interview talking points:**
- Why rules-based at all, when you have a YOLO model? Determinism and zero-weight operation. The rules-based pass runs with no GPU, no model files, and always produces the same output given the same PDF. It's also a fallback — if the model weights are absent, the pipeline still works.
- How does the geometry detector handle two-column papers? `_split_columns()` finds the single largest x-center gap between text line centroids and treats everything left of it as column 1 and right as column 2. It works well for two-column PDFs but would mis-assign three-column layouts.
- What's the weakness of the table detector? It uses `pdfplumber.find_tables()` which relies on ruled lines. Borderless tables (common in modern academic papers) are invisible to this approach — they can only be caught by the YOLO model. That's one of the explicit motivations for the hybrid design.

---

### `docstruct/model/detector.py`

Optional YOLOv8 layout detector producing vision-model proposals.

**What it does:** Wraps an ultralytics YOLO model to run layout detection on rasterized PDF pages. Heavy deps (`ultralytics`, `pymupdf`) are imported lazily inside methods so the geometry-only path never imports them.

**How it works:** Each page is rasterized to a pixel array by PyMuPDF at `MODEL_DPI = 150` (150/72 = 2.08× zoom). YOLO returns bounding boxes in pixel coordinates; they're divided back by the zoom factor to convert to PDF points, putting them in the same coordinate space as the geometry detector. DocLayNet class names are remapped through `DOCLAYNET_LABEL_MAP`. The model is loaded lazily on first `detect()` call.

**Important code snippet:**
```python
zoom = self.dpi / 72.0
matrix = fitz.Matrix(zoom, zoom)
# ...
bbox = BoundingBox(
    x0=x0 / zoom,
    y0=y0 / zoom,
    x1=x1 / zoom,
    y1=y1 / zoom,
    page_width=page_w,
    page_height=page_h,
)
```

**Interview talking points:**
- Why 150 DPI? It's a balance: low enough that the rasterized image isn't huge, high enough that small text and fine table lines are legible to the model. 72 DPI (one pixel per PDF point) is too low; 300 DPI quadruples memory and inference time.
- Why divide by zoom when storing the bbox? The entire downstream pipeline (geometry detector, text extractor, reading order) works in PDF points. If model proposals were stored in pixels, the IoU matching in `fusion/matcher.py` would silently produce wrong results because the scale would be off by 2.08×.

---

### `docstruct/fusion/matcher.py`

Greedy IoU matching and priority NMS over the two proposal streams.

**What it does:** Takes two lists of `Proposal` objects (model and geometry) and returns a `MatchResult` with matched pairs, unmatched model proposals, and unmatched geometry proposals.

**How it works:** First pass: for each model proposal, find the highest-IoU geometry proposal above the 0.35 threshold (greedy, each geometry proposal used at most once). Second pass: NMS — all candidates (matched pairs, unmatched model, unmatched geometry) are collected and sorted by priority (matched=3 > model=2 > geometry=1), ties broken by confidence. Any candidate whose box overlaps a previously kept box above `NMS_IOU_THRESHOLD = 0.5` is suppressed.

**Important code snippet:**
```python
candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

for _, _, bbox, kind, obj in candidates:
    if any(bbox_overlap(bbox, kept) > iou_threshold for kept in kept_boxes):
        continue
    kept_boxes.append(bbox)
    # assign to matched_out, model_out, or geometry_out
```

**Interview talking points:**
- Why greedy matching? It's O(M×G) where M and G are proposal counts per page — typically under 50 each, so the quadratic cost is negligible. The alternative (Hungarian algorithm for optimal assignment) adds complexity without meaningful benefit at this scale.
- Why two different IoU thresholds (0.35 for matching, 0.5 for NMS)? Matching uses a lower threshold so two detectors that disagree slightly on box edges can still be paired. NMS uses a higher threshold so only boxes that substantially overlap get suppressed.
- What does priority mean in NMS? A confirmed pair beats a stray geometry-only proposal that happens to overlap it. This prevents the geometry detector's redundant nearby proposal from surviving NMS and creating a duplicate block.

---

### `docstruct/fusion/arbiter.py`

Classifies matched pairs as confirmed or disputed and computes final confidence.

**What it does:** For each `MatchedPair`, if both detectors agree on the label it becomes `CONFIRMED` with a boosted confidence; if they disagree, the higher-confidence detector wins but pays a 15% penalty (`DISPUTED_MULTIPLIER = 0.85`).

**How it works:** Confirmed formula: `0.85 + 0.10 × model_conf + 0.05 × IoU`. Maximum is capped at 1.0. Disputed: `winner_conf × 0.85`. The confidence formula encodes the prior that agreement between independent detectors is strong evidence — base 0.85 plus small bonuses for model confidence and box alignment quality.

**Important code snippet:**
```python
def _confirmed_confidence(model_conf: float, iou: float) -> float:
    raw = (
        config.CONFIRMED_BASE
        + config.CONFIRMED_MODEL_BOOST * model_conf
        + config.CONFIRMED_AGREEMENT_BOOST * iou
    )
    return min(1.0, raw)
```

**Interview talking points:**
- Why does the confirmed formula include IoU? A high IoU (both detectors drew nearly the same box) is evidence the region boundary is accurate, not just the label. It contributes a small bonus (up to 0.05).
- Why 0.85 base for confirmed? It reflects the prior that even when two detectors agree, neither is perfect. A confirmed block still might have a slightly wrong boundary or be a borderline case between two labels.

---

### `docstruct/fusion/fusion.py`

Assembles `Block` objects from the `MatchResult` and `ArbitratedItem` outputs.

**What it does:** Combines all three sources (matched pairs, unilateral model, unilateral geometry) into a flat `List[Block]`. Handles bbox selection for matched blocks (prefer model bbox when model confidence and IoU are both high, else take the merged union) and applies confidence floor/ceiling clamping for unilateral blocks.

**How it works:** Matched blocks: if `model_conf >= 0.8 AND iou >= 0.6`, use the model bbox (more precise); otherwise merge both bboxes into the smallest enclosing rectangle. Unilateral: multiply raw confidence by `UNILATERAL_MODEL_SCALE = 0.75` or `UNILATERAL_GEOMETRY_SCALE = 0.60`, then clamp to per-source floor/ceiling bounds.

**Important code snippet:**
```python
def _matched_bbox(item: ArbitratedItem) -> BoundingBox:
    if (
        item.model_score >= config.CONFIRMED_BBOX_MODEL_CONF
        and item.iou >= config.CONFIRMED_BBOX_IOU
    ):
        return item.model.bbox
    return merge_bboxes([item.model.bbox, item.geometry.bbox])
```

**Interview talking points:**
- Why prefer the model bbox when both conditions are met? YOLO's bounding box regression is more pixel-precise than the geometry detector's word-extent bounding. The geometry detector builds boxes from text word extents, which can be slightly loose. When the model is confident and both detectors substantially agree on location, the model bbox is the better choice.

---

### `docstruct/reading_order.py`

Assigns reading order indices and attaches captions to their figures/tables.

**What it does:** Given all blocks on a page, determines the correct reading sequence accounting for multi-column layout. Also links each caption block to its nearest figure or table via `caption_target_id`.

**How it works:** `detect_columns()` reuses the same largest-gap heuristic as the geometry detector but operates on already-fused blocks rather than raw lines. Within each column, blocks sort ascending by `y0` (top edge). Caption attachment uses Euclidean distance between block centers, with a 0.5× distance multiplier when the caption is below its target (the more common layout).

**Important code snippet:**
```python
def find_caption_target(caption: Block, blocks: List[Block]) -> Optional[str]:
    # ...
    if caption.bbox.y0 >= block.bbox.y1:
        distance *= 0.5   # caption below target is closer in reading flow
    if distance < min_distance:
        min_distance = distance
        nearest_id = block.block_id
    return nearest_id if min_distance < config.CAPTION_MAX_DISTANCE else None
```

**Interview talking points:**
- Why a distance multiplier for captions below their target? Most figure/table captions in academic PDFs appear below. Halving the distance for that case biases the match toward the correct layout without hard-coding directionality.
- Reading order is assigned per page and then offset into a global sequence in `pipeline.py`. Why? Pages are independent — a block on page 3 shouldn't influence column detection on page 1. The global offset just converts per-page indices into a document-wide sequence number.

---

### `docstruct/extraction/text_extractor.py`

Fills block text content and header font sizes by cropping the source PDF.

**What it does:** Given fused blocks (which only have bounding boxes and labels at this point), crops each block's region from the pdfplumber page and extracts the text within that box. Headers also get their `font_size` field populated (used later by the hierarchy builder).

**How it works:** Groups blocks by page number to open the PDF once. For each text/header/caption block, crops the pdfplumber page to the block's bbox and calls `extract_text()`. Rotated glyphs are filtered out so margin annotations (e.g., arXiv submission stamps) don't pollute block text.

**Interview talking points:**
- Why crop rather than re-parse from scratch? Fusion may have adjusted bbox edges (the `merge_bboxes` path in `fusion.py`). Cropping to the fused bbox is the authoritative source of text for that region.
- Why is `font_size` only populated for headers? It's only needed by `hierarchy_builder.py` to rank levels. Text blocks don't use font size downstream.

---

### `docstruct/extraction/table_extractor.py`

Extracts table cell grids and renders them as Markdown.

**What it does:** For blocks labeled `table`, crops the page, runs `pdfplumber.extract_tables()` to get the cell grid, then serializes to GitHub-flavored Markdown for storage in `block.text`.

**How it works:** If the cropped region contains multiple tables, the largest (by row × column count) is kept. Empty cells become empty strings. The Markdown serializer pads short rows to the maximum row width.

**Important code snippet:**
```python
def table_to_markdown(grid: List[List[str]]) -> str:
    header = norm[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in norm[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
```

**Interview talking points:**
- Why Markdown rather than a structured JSON for tables? Chunks go into a vector store as plain text. Markdown table syntax is still machine-parseable while being embeddable as a string. A nested JSON structure would need special-casing in the embedding/retrieval path.
- Limitation: this only works for ruled-line tables (pdfplumber's `find_tables()`). Borderless tables need the model detector — another concrete reason the hybrid mode matters.

---

### `docstruct/chunking/hierarchy_builder.py`

Maps each header block to a section level (h1/h2/h3) by font-size rank.

**What it does:** Takes all blocks, finds those labeled `header`, ranks their font sizes (or bbox height as fallback), and returns a dict mapping `block_id → level`.

**How it works:** Collects distinct rounded font sizes, sorts descending, and assigns level 1 to the largest, 2 to the next distinct size, and so on up to `HEADER_LEVELS = 3`. Any sizes beyond the third distinct group clamp to level 3.

**Interview talking points:**
- Why font-size ranking rather than hardcoded patterns like "3. Methodology"? Document-agnostic — works on any prose PDF without requiring the user to specify section naming conventions. A research paper, a manual, and a legal brief all use different section naming schemes, but all use relative font sizes to convey hierarchy.
- What happens if headers all have the same font size? All are assigned the same level (the first distinct size maps to level 1). The chunker will treat them all as h1, which may produce flat structure rather than nested, but it won't crash.

---

### `docstruct/chunking/assembler.py`

Walks blocks in reading order and assembles retrieval-ready `Chunk` objects.

**What it does:** The final transformation before indexing. Maintains a running `SectionPath` as it walks blocks, accumulates text blocks into a buffer until the token limit or a section boundary, and emits `Chunk` objects with appropriate types (`text`, `table`, `figure_caption`, `abstract`, `references` are skipped).

**How it works:** Uses word-count as an approximation of token count (`MAX_CHUNK_TOKENS = 400`) to stay tokenizer-free. Headers update the `SectionPath` but don't become chunks themselves. Tables and captions each emit their own atomic chunk immediately. Text blocks accumulate and flush either when the buffer hits 400 words or when a new header arrives. The References section is detected by name and skipped entirely.

**Important code snippet:**
```python
for block in ordered:
    if block.label == "header":
        flush_text()
        _set_section(section, levels.get(block.block_id, config.HEADER_LEVELS),
                     (block.text or "").strip())
    elif block.label == "table":
        flush_text()
        if not _in_references(section):
            emit("table", block.text or "", [block])
    elif block.label == "caption":
        flush_text()
        if not _in_references(section):
            ids = [block.block_id]
            if block.caption_target_id:
                ids.append(block.caption_target_id)
            emit("figure_caption", block.text or "", [block], ids=ids)
```

**Interview talking points:**
- Why skip references? References sections are dense citation lists that almost never contain substantive content a retrieval query would want. Including them degrades retrieval quality by introducing high-vocabulary noise.
- Why word count instead of a tokenizer? Keeps the chunking path dependency-free. A word count at 400 maps roughly to 500-600 subword tokens for typical prose, which fits comfortably within most embedding model context windows. Good enough, and avoids the `transformers` import in the core package.
- Why are figures skipped? Figures themselves have no text to embed. Their semantic content is represented through their caption, which gets a `figure_caption` chunk. The `caption_target_id` link means the caption chunk carries both its own block ID and the figure block ID as `source_block_ids`, so provenance is preserved.

---

### `docstruct/pipeline.py`

End-to-end orchestrator: PDF path in, `PipelineResult` out.

**What it does:** Coordinates all pipeline stages in sequence. The only file that knows about every other module. Handles optional model detection, per-page proposal grouping, cache interaction, and global reading order offsets.

**How it works:** Both detectors run and their proposals are grouped by page number. For each page: `match_proposals()` → `fuse()` → `assign_reading_order()`. After all pages, `populate_text()` and `populate_tables()` make a single PDF pass each (grouped by page to avoid reopening). Then `assign_header_levels()` + `build_chunks()` finish the pipeline.

**Important code snippet:**
```python
for page_num in pages:
    result = match_proposals(model_by_page.get(page_num, []),
                             geo_by_page.get(page_num, []))
    blocks = fuse(result)
    assign_reading_order(blocks, page_width)
    for block in blocks:
        block.reading_order += ro_offset
    ro_offset += len(blocks)
    all_blocks.extend(blocks)
```

**Interview talking points:**
- Why iterate per-page for fusion but per-document for extraction? Fusion must be per-page because proposals from different pages can never match each other (a block on page 1 can't fuse with one on page 2). Extraction groups by page so the PDF is opened once and all blocks on a page are extracted in a single pass.
- How does caching work? Both detectors support a `ProposalCache` keyed by SHA-256 hash of the PDF file. If the file hasn't changed, the cached proposals are returned immediately and neither pdfplumber nor YOLO runs again. The geometry and model caches are namespaced separately (`geometry` and `model.<weights-stem>`).

---

### `docstruct/indexing/vector_store.py`

ChromaDB-backed vector store with sentence-transformer embeddings.

**What it does:** Embeds chunks using `all-MiniLM-L6-v2` and stores them in a Chroma collection. Section path fields (h1, h2, h3) are flattened into Chroma's metadata dict so retrieval can filter on them directly.

**How it works:** Accepts either a persist directory (writes to disk, survives process restart) or no directory (ephemeral in-memory collection for tests). Batch-embeds all chunk contents at index time. The cosine similarity space is explicitly set on collection creation (`hnsw:space: cosine`).

**Interview talking points:**
- Why `all-MiniLM-L6-v2`? It's small (80MB), fast, and well-suited for sentence-level semantic similarity. For an interview context, mention that the model is configurable in `config.py` — the field is `EMBEDDING_MODEL`.
- Why flatten section path into scalar metadata? ChromaDB's `where` filter only works on scalar values. `h1`, `h2`, `h3` need to be separate keys to be filterable. A nested dict would require a custom filter mechanism.

---

### `docstruct/query/retriever.py`

Thin retrieval wrapper that converts Chroma responses to cited results.

**What it does:** Wraps `VectorStore.query()` and maps the raw Chroma response dict (lists of ids, documents, metadatas, distances) into typed `RetrievalResult` objects with a human-readable `.citation()` method.

**How it works:** Chroma returns cosine _distance_ (0 = identical, 2 = maximally dissimilar). The retriever converts to similarity by `score = 1.0 - distance`, which gives an intuitive 0–1 scale.

**Important code snippet:**
```python
score=round(1.0 - float(dist), 4)  # cosine distance -> similarity
```

**Interview talking points:**
- `citation()` returns `[h1 > h2] (page 3, score 0.48)`. Why include section path in citations? A RAG citation that says "page 3" is weaker evidence than one that says "3. Methodology > 3.2 Baselines, page 3". Section path citations let a reader quickly verify the chunk came from the right part of the document.

---

### `docstruct/eval/metrics.py`

Detection and retrieval metrics — mAP, MRR, NDCG.

**What it does:** Implements all-point-interpolated mean Average Precision (mAP@0.5) for detection evaluation, and Mean Reciprocal Rank (MRR) and NDCG@k for retrieval evaluation.

**How it works:** Detection: for each class, sort predictions by confidence descending, match greedily to ground truth by IoU, compute precision/recall at each rank, then integrate the precision-recall curve using the all-points interpolation (backward max then trapezoid). Per-class AP values are averaged for mAP. Retrieval: MRR computes `1/rank` for the first relevant result; NDCG@k uses discounted cumulative gain normalized by the ideal ranking.

**Interview talking points:**
- Why mAP@0.5 specifically? 0.5 IoU is the standard PASCAL VOC / COCO threshold for detection evaluation. It means "the predicted box must overlap the ground truth box by at least 50%." It's a reasonable middle ground between too strict (e.g., 0.75, which penalizes slightly off-edge boxes) and too loose.
- Why MRR for retrieval? MRR rewards finding the right answer early in the ranked list and is simple to explain. NDCG@k complements it by rewarding multiple relevant results at appropriate ranks.

---

### `docstruct/eval/runner.py`

Evaluation harness connecting pipeline output to metrics.

**What it does:** Provides `evaluate_detection()`, `evaluate_retrieval()`, and `compare_chunking()`. The `compare_chunking()` function is particularly important: it indexes both DocStruct chunks and naive baseline chunks in fresh ephemeral Chroma collections and runs the same retrieval cases against both, giving a direct apples-to-apples comparison.

**Interview talking points:**
- Ground truth format is intentionally simple: a JSON file with a `boxes` list, each with label, page_num, and a 6-element bbox array `[x0, y0, x1, y1, page_width, page_height]`. This matches `BoundingBox.__init__` parameter order exactly, so `BoundingBox(*entry["bbox"])` just works.

---

### `docstruct/eval/baselines.py`

Naive fixed-window chunker for comparison.

**What it does:** Implements the "dumb" baseline: concatenate all block text in reading order, split into 400-word windows, emit as `Chunk` objects with empty `SectionPath`. This is what most LangChain/LlamaIndex PDF loaders do by default.

**Interview talking points:**
- The naive chunker uses the exact same `Chunk` dataclass and feeds into the exact same `VectorStore`/`Retriever` path as DocStruct. This makes `compare_chunking()` a fair comparison — the only difference is how the chunks were constructed.

---

### `docstruct/eval/annotate.py`

Model-assisted annotation export for the browser-based annotation tool.

**What it does:** Runs the pipeline on a PDF, rasterizes all pages to base64 PNGs, and writes a self-contained JSON session file. `tools/annotate.html` loads this file, lets you drag/resize/relabel predicted boxes, and exports the corrected ground-truth JSON that the eval runner expects.

**Interview talking points:**
- The model-assisted workflow is important: instead of drawing boxes from scratch, the annotator corrects predicted boxes. This is faster by a significant factor. The `export-annotations` CLI command is the entry point.
- Boxes stay in PDF points throughout (never converted to pixels). This was an explicit design decision to prevent the coordinate-space confusion that's a common silent bug in annotation tools.

---

### `docstruct/cache/pdf_cache.py` and `docstruct/cache/model_cache.py`

Disk cache for detector proposals, keyed by SHA-256 of the PDF.

**What it does:** Serializes `List[Proposal]` to JSON and caches it on disk, keyed by the file's content hash. `ModelProposalCache` extends `ProposalCache` by including the weights filename in the cache namespace so different model weights don't collide.

**Interview talking points:**
- Keyed by file content hash (not path or mtime) means renaming or moving the PDF doesn't invalidate the cache. The same file at a different path gets a cache hit.

---

### `docstruct/utils/geometry.py`

Pure geometry utilities: IoU, bbox merging, centroid.

**What it does:** `bbox_overlap()` computes intersection-over-union. `merge_bboxes()` computes the smallest enclosing box. `bbox_center()` computes centroids. These are used in fusion, reading order, and eval metrics — any place that needs spatial reasoning.

**Interview talking points:**
- `bbox_overlap()` returns 0.0 for disjoint boxes without raising — `x_right < x_left` handles the non-overlapping case. This is important because it's called in hot loops (greedy matching is O(M×G) per page).

---

### `docstruct/cli.py`

Argparse CLI exposing five subcommands: `run`, `index`, `query`, `visualize`, `export-annotations`.

**What it does:** Each subcommand maps to a `_cmd_*` function. The `run` command prints a one-line diagnostics summary then previews chunks. The `index`/`query` commands chain pipeline + vector store. The `export-annotations` command drives the annotation workflow.

**Interview talking points:**
- `main()` reconfigures stdout/stderr to UTF-8 before anything else. This prevents `UnicodeEncodeError` on Windows when PDF text contains non-ASCII characters and the terminal's default encoding is cp1252.

---

### Test Suite (`tests/`)

57 tests covering all major components.

**What it does:** Unit tests for fusion (matcher, arbiter, fusion assembly), metrics (mAP, MRR, NDCG), chunking (hierarchy, assembler), geometry utilities, cache, and extraction. Integration tests for the full pipeline and retrieval that auto-skip if optional dependencies or PDFs are absent.

**How it's structured:** `conftest.py` provides `make_bbox` and `make_proposal` factory functions used across test files. Tests for fusion use synthetic proposals with known IoUs to verify exact numerical outputs (confidence formulas, NMS priority, disputed winner selection). Metric tests verify edge cases: FPs, FNs, page isolation, below-threshold IoU counts as a miss.

**Interview talking points:**
- Tests are purely synthetic — no real PDFs required for unit tests. This means the test suite runs in CI without any fixture data or model weights.
- The `test_pipeline_integration.py` and `test_retrieval_integration.py` use `pytest.importorskip()` to skip gracefully when extras aren't installed.

---

## Section 3 — Likely Interview Questions & Model Answers

### Q: Walk me through what happens when I call `docstruct run paper.pdf --weights model.pt`.

The CLI calls `run_pipeline("paper.pdf", weights="model.pt")`. First, both detectors run independently. The geometry detector uses pdfplumber to extract word bounding boxes and font metadata, applies gap-based blocking and font-size heuristics, and produces a list of `Proposal` objects labeled text/header/table/figure/caption for each page. Simultaneously, the model detector rasterizes each page to a pixel array at 150 DPI, runs YOLO inference, and produces its own proposal list with pixel coordinates converted back to PDF points. Then, per page, `match_proposals()` greedily IoU-matches the two streams and runs priority NMS. `fuse()` converts the match result into `Block` objects with fused confidence scores. After all pages, `populate_text()` and `populate_tables()` extract content back from the PDF, `assign_header_levels()` ranks headers by font size, and `build_chunks()` walks blocks in reading order to emit `Chunk` objects with section-path metadata. The result is printed as a preview plus optional JSON.

---

### Q: Why run two detectors? Why not just use the YOLO model?

Two reasons. First, the geometry detector works with no dependencies beyond pdfplumber and numpy — no GPU, no model files, fully offline. The whole pipeline degrades gracefully when no weights are provided. Second, each detector has complementary failure modes. The geometry detector is excellent at ruled-line tables and font-based headers, but can't see borderless tables or visual figures. YOLO handles those but can hallucinate or misclassify under low DPI or with unusual layouts. When both agree, confidence is high. When they disagree, that's a signal that the region is ambiguous and confidence is penalized appropriately.

---

### Q: How does the fusion confidence formula work, and why did you design it that way?

There are three cases. Confirmed (both detectors agree, IoU ≥ 0.35): `0.85 + 0.10 × model_conf + 0.05 × IoU`. The base of 0.85 reflects that agreement between two independent detectors is strong evidence but not certainty. The model confidence bonus (up to 0.10) rewards high-certainty YOLO predictions. The IoU bonus (up to 0.05) rewards boxes that overlap tightly — good box precision, not just label agreement. Disputed (both detect the region but disagree on label): winner's confidence × 0.85 — the 15% penalty reflects that a label dispute is a red flag. Unilateral (only one detector fires): raw confidence × 0.60 or 0.75 (geometry or model), clamped to floor/ceiling bounds, reflecting that a single-source detection is less reliable.

---

### Q: What are the limitations of the geometry-only mode?

Three concrete ones. First, borderless tables are invisible: pdfplumber's `find_tables()` requires ruled lines, so a table with no borders looks like a block of text. Second, figures that are pure text content (e.g., a code listing formatted as a figure) may not be detected since figure detection relies on raster images and vector graphics, not text patterns. Third, font-based header detection can conflate document title, author line, and first section header if they're all set in similar bold fonts near the same size. The model resolves many of these cases because it was trained on labeled layouts and understands semantic structure visually.

---

### Q: How does the reading order handle two-column papers?

`detect_columns()` finds the largest x-center gap between block centroids. If that gap exceeds 15% of the page width (`COLUMN_GAP_RATIO = 0.15`), blocks are split into two groups. Within each group, blocks sort ascending by their top y-coordinate. Columns are then ordered left-to-right by their average x-center. So a two-column academic paper reads left column top-to-bottom, then right column top-to-bottom. The limitation is that this only works for exactly two columns — three-column layouts would have two gaps but only the largest is found, so the split would be wrong.

---

### Q: How does the section path work and why is it useful for RAG?

Each `Chunk` carries a `SectionPath` with up to three levels: h1, h2, h3. These are populated by the assembler as it walks blocks — when it encounters a header block, it updates the current section context, and every subsequent chunk until the next header at the same or higher level carries that context. The value for RAG is two-fold: you can retrieve with semantic filtering (`where={"h1": "4. Experiments"}` returns only chunks from that section), and every result comes with a human-readable citation like `[4. Experiments > 4.2 Baselines] (page 8, score 0.61)`. This is strictly better than a naive chunker which produces no section metadata at all.

---

### Q: How do you evaluate whether DocStruct chunking is better than naive chunking?

`compare_chunking()` in `eval/runner.py` takes a set of (question, relevant_chunk_ids) test cases, indexes both DocStruct chunks and naive baseline chunks in separate ephemeral Chroma collections, runs the same retrieval queries against both, and reports MRR and NDCG@5 for each. Since both use identical embedding and retrieval code, the only variable is chunk construction. The naive baseline is implemented in `eval/baselines.py` — it produces the same `Chunk` dataclass with empty `SectionPath` and fixed 400-word windows, feeding through the identical `VectorStore`/`Retriever` path for a fully fair comparison.

---

### Q: How does the cache prevent recomputing geometry proposals?

`ProposalCache` hashes the PDF file's bytes with SHA-256 and stores the serialized proposals as a JSON file named `<hash>.geometry.json` in the cache directory. On subsequent runs with the same file, the hash matches, the JSON is loaded and deserialized back into `List[Proposal]` objects, and neither pdfplumber nor YOLO runs. Model cache adds the weights stem to the filename so `<hash>.model.yolov8m-doclaynet.json` is a separate key, meaning different weights don't collide. The cache is keyed by content hash, not path, so moving the file doesn't invalidate it.

---

### Q: Walk me through the mAP metric implementation.

For each class: sort all predictions by confidence descending. Walk down the ranked list; for each prediction, find the highest-IoU ground-truth box on the same page that hasn't been matched yet. If IoU ≥ 0.5, it's a TP; otherwise it's a FP. Track cumulative TP and FP to compute precision and recall at each rank point. Then apply all-points interpolation: for each recall level, take the maximum precision at that recall level or higher, then integrate (area under the interpolated PR curve). mAP is the mean AP across all classes. This is the standard PASCAL VOC mAP@0.5 implementation.

---

### Q: Why is the pipeline fully deterministic and why does that matter?

Every operation is deterministic: pdfplumber extraction is deterministic given the same PDF bytes, the fusion algorithm is greedy-sorted (no random tie-breaking), font-size ranking is sorted, chunking is a sequential walk. The practical implication is reproducibility — you can diff the output of two pipeline runs to see exactly what a config change did, run the eval harness and get stable numbers, and audit a specific document's chunk decomposition without worrying about nondeterministic variation. For production RAG systems, determinism also means you can cache at any stage and trust the cache is still valid as long as the input hasn't changed.

---

### Q: What would you build next if you had another two weeks?

Three things. First, tune the unvalidated confidence bounds in `config.py` using the annotated eval set — those values are inherited from the prototype and are explicitly flagged as unvalidated. Second, extend column detection to handle three-column layouts by recursively splitting instead of finding only the largest gap. Third, add an OCR path for scanned PDFs — the current design explicitly excludes them (no OCR by design), but it's the most frequent user request. The clean architecture of independent detectors means an OCR-based geometry detector could slot in without touching fusion or downstream stages.

---

## Section 4 — Glossary of Key Terms

| Term | Plain-English Definition | Where It Appears |
|------|--------------------------|------------------|
| Proposal | A raw, single-detector layout region with a label, bounding box, and confidence score — not yet reconciled with the other detector | `schema.py`, `geometry/detector.py`, `model/detector.py` |
| Block | A fused layout region after the two proposal streams have been reconciled; includes source provenance and a `ConfidenceBreakdown` | `schema.py`, `fusion/fusion.py`, `pipeline.py` |
| Chunk | A retrieval-ready unit of content with section-path metadata — the output of the pipeline that goes into the vector store | `schema.py`, `chunking/assembler.py` |
| BoundingBox | An axis-aligned rectangle in top-left PDF coordinates (y0 is the top edge, y increases downward) | `schema.py`, `utils/geometry.py` |
| Source | Enum: CONFIRMED (both detectors agree), DISPUTED (they disagree), UNILATERAL_MODEL, UNILATERAL_GEOMETRY | `schema.py`, `fusion/arbiter.py` |
| IoU | Intersection-over-Union: area of overlap divided by area of union between two bounding boxes. 0 = no overlap, 1 = identical | `utils/geometry.py`, `fusion/matcher.py` |
| NMS | Non-Maximum Suppression: after matching, remove redundant overlapping boxes by keeping the highest-priority/highest-confidence one | `fusion/matcher.py` |
| DocLayNet | An IBM dataset of 80,000+ annotated PDF pages with 11 layout classes; the YOLO model was trained on it | `config.py`, `model/detector.py` |
| SectionPath | A three-level (h1/h2/h3) section hierarchy for a chunk, inferred from font-size ranking of header blocks | `schema.py`, `chunking/hierarchy_builder.py` |
| mAP@0.5 | Mean Average Precision at IoU threshold 0.5 — the standard object detection metric, averaged across all label classes | `eval/metrics.py` |
| MRR | Mean Reciprocal Rank — average of 1/rank for the first relevant result across retrieval queries | `eval/metrics.py` |
| NDCG@k | Normalized Discounted Cumulative Gain — retrieval metric that rewards finding multiple relevant results early in the top-k list | `eval/metrics.py` |
| pdfplumber | Python library that wraps pdfminer to expose word-level bounding boxes, font sizes, and table detection from born-digital PDFs | `geometry/detector.py`, `extraction/` |
| PyMuPDF (fitz) | Fast PDF renderer; used to rasterize pages to pixel arrays for YOLO inference and to draw annotated overlays | `model/detector.py`, `visualize.py` |
| ChromaDB | Embedded vector database used to store and query chunk embeddings with metadata filtering | `indexing/vector_store.py` |
| sentence-transformers | Library for computing dense sentence embeddings; default model is `all-MiniLM-L6-v2` | `indexing/vector_store.py` |
| all-MiniLM-L6-v2 | A lightweight 80MB sentence embedding model, fast enough for local use, good at semantic similarity | `config.py`, `indexing/vector_store.py` |
| YOLOv8 / ultralytics | The vision model family used for layout detection; DocStruct uses it via the `ultralytics` Python package | `model/detector.py` |
