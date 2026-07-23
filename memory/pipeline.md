# The pipeline, stage by stage

`run_pipeline()` in `docstruct/pipeline.py` is the whole orchestration. Detection
and fusion happen **per page** (proposals on different pages can never match);
reading order is assigned per page and then offset into one document-global
sequence. Extraction and chunking run once over all blocks.

---

## 1. Geometry detection — `geometry/detector.py`

Pure rules over pdfplumber primitives. No model, no learning, fully explainable.

- Words are grouped into lines by vertical proximity (`LINE_Y_TOLERANCE = 3.0`).
- Lines are grouped into blocks; a gap larger than
  `PARAGRAPH_GAP_FACTOR (1.6) × line_height` breaks a block.
- **Headers**: font size > `HEADER_SIZE_RATIO (1.15) × body median`, or a short
  all-bold line (`HEADER_MAX_WORDS = 12`, `HEADER_MAX_LINES = 3`). Bold detection
  is by font-name substring (`BOLD_FONT_MARKERS`), which is how PDFs actually
  encode weight.
- **Tables**: pdfplumber `find_tables()` — **ruled-line based**. Borderless tables
  are invisible to geometry. This is a concrete motivation for the hybrid design,
  not an incidental gap.
- **Figures**: clusters of graphic primitives covering ≥ `FIGURE_MIN_AREA_RATIO`
  (3%) of the page and mostly text-free (`FIGURE_MAX_TEXT_OVERLAP = 0.10`).
- **Captions**: regex on universal prose markers
  (`CAPTION_PREFIX_PATTERN`: figure/fig./table/tab./scheme/algorithm + number).
- Per-label prior confidences come from `GEOMETRY_CONFIDENCE`.

Column splitting inside the detector groups line clusters; multi-column pages get
their lines assigned before block formation.

## 2. Model detection — `model/detector.py` (optional)

YOLOv8 trained on DocLayNet, run at `MODEL_DPI = 150` with
`MODEL_CONF_THRESHOLD = 0.25`. Its 11 classes collapse to DocStruct's 5 via
`DOCLAYNET_LABEL_MAP` (footnote/formula/list-item/page-header/page-footer all →
`text`; title and section-header → `header`; picture → `figure`).

**The pixel→point transform lives here.** Model output arrives in rendered-image
pixel space and must land in top-left PDF point space before anything downstream
touches it. Every other stage assumes it already has.

## 3. Fusion — `fusion/matcher.py` → `arbiter.py` → `fusion.py`

**Matching** (`matcher.py`): greedy — for each model proposal, take the
highest-IoU unused geometry proposal above `IOU_MATCH_THRESHOLD (0.35)`. Then
priority NMS over everything (matched=3 > model-only=2 > geometry-only=1,
confidence breaking ties) at `NMS_IOU_THRESHOLD (0.5)`, so nested duplicates from
the two streams collapse to one region.

**Arbitration** (`arbiter.py`): a matched pair with the same label is
`CONFIRMED`; different labels is `DISPUTED` and one label wins.

**Confidence** (`fusion.py`):

| Case | Condition | Final confidence |
|---|---|---|
| Confirmed | both detect, same label, IoU ≥ 0.35 | `0.85 + 0.10·model_conf + 0.05·IoU` |
| Disputed | both detect, different label | `winner_conf × 0.85` |
| Unilateral model | model only | `conf × 0.75`, clamped to [0.40, 0.85] |
| Unilateral geometry | geometry only | `conf × 0.60`, clamped to [0.25, 0.65] |

The unilateral scales and both clamp ranges are marked `# unvalidated` in
`config.py` — inherited from the v0 prototype, never calibrated against the
annotated set. Anything that *acts* on these numbers (e.g. confidence-weighted
retrieval ranking) is building on sand until they are.

**Bbox choice for a matched pair**: the model's box is taken outright when the
model is confident *and* the two boxes agree (`model_conf ≥ 0.8` and `IoU ≥ 0.6`);
otherwise the union of both.

## 4. Reading order — `reading_order.py`

Default (`XY_CUT = False`): split page blocks into **at most two columns** by the
largest gap between block *centres*, requiring the gap to exceed
`COLUMN_GAP_RATIO (0.15) × page_width`; then sort each column ascending by `y0`
(top-left origin — smaller `y0` is higher).

This is provably wrong for a full-width title, abstract, table or figure spanning
a two-column body: its centre sits near the page middle, so it gets assigned to
one column and read in the wrong place. The principled fix (recursive XY-cut) is
implemented in `utils/xy_cut.py` and **turned off**, because it measured worse on
this corpus. See `decisions.md`.

**Caption attachment** is one-directional: each caption gets
`caption_target_id` = nearest figure/table by centre distance, with a 0.5×
distance discount when the caption sits *below* its target (the common layout),
and a hard cutoff at `CAPTION_MAX_DISTANCE = 100.0` points.

## 5. Extraction — `extraction/`

`populate_text()` fills `block.text` from pdfplumber within each block's bbox.

**The word-spacing trap**: pdfplumber inserts a space when the inter-character gap
exceeds a flat 3pt default, which is *wider* than the real inter-word gap in small
type — so author lines, footnotes and table cells come out as
`IreneAmerini1,ElenaBalashova2`. `TEXT_X_TOLERANCE_RATIO = 0.15` scales the
tolerance with font size instead. This was worth **+0.0138 MRR** on its own, not
because it looks nicer but because BM25 cannot match a term glued to its
neighbour.

`populate_tables()` fills `table_data` (the grid) and renders `block.text`.
`extract_tables()` is ruled-line based, so on a **partly**-ruled table it returns
only the ruled fragment — which used to become the block's entire text, silently
dropping every unruled row. Now the rendered grid's word count is compared against
the raw region text and falls back to raw text below
`TABLE_GRID_MIN_COVERAGE (0.85)`. Structure is preserved in `table_data` either way.

Tables serialize as **plaintext rows**, not Markdown, for chunk content;
`table_to_markdown()` exists and is used by `Document.markdown`, which is a
human-reading surface rather than a retrieval one.

## 6. Header levels — `chunking/hierarchy_builder.py`

Two deterministic, document-local signals — no hardcoded section names, no regex
on "Introduction".

1. **Section numbering** (`HEADER_NUMBERING_LEVELS`, default on). "3 Method" → 1,
   "3.2 Setup" → 2, "3.2.1 Ablations" → 3. Where a heading carries a number, that
   number *states* its depth, so it wins outright. The regex is anchored and
   requires text after the number, so a bare page number or list marker does not
   qualify.
2. **Font-size rank**, for unnumbered headings: largest distinct size → level 1,
   clamped at `HEADER_LEVELS = 3`. Falls back to bbox height when `font_size` is
   missing.

Numbering was added because font size alone collapses the hierarchy entirely on
documents that set every heading at one size, and misassigns levels on documents
that separate depth by weight rather than size. It is **not visible in the
retrieval benchmark** — header level never moves a chunk boundary, so chunk text
is byte-identical — but `section_path` is what filtered retrieval is built on, and
it was wrong on those documents.

## 7. Chunking — `chunking/assembler.py`

Walks blocks in reading order maintaining a running `SectionPath`:

| Block label | Behaviour |
|---|---|
| `header` | boundary flush; updates the section path; its text **opens the body** of the chunk it introduces (`INLINE_HEADER_TEXT`) |
| `text` | accumulates into the buffer; hard flush at `MAX_CHUNK_TOKENS (500)` with `CHUNK_OVERLAP_TOKENS (75)` carried forward |
| `table` | atomic chunk, never split; does **not** split surrounding prose (`BREAK_TEXT_ON_TABLE = False`) |
| `caption` | `figure_caption` chunk linked to its target; does not split prose (`BREAK_TEXT_ON_CAPTION = False`) |
| `figure` | skipped — represented through its caption |
| abstract | detected by section name, emitted as `chunk_type="abstract"` |
| references | dropped entirely |

**The central rule** (and the single largest quality win in this project's
history, +0.0429 MRR):

> A structural boundary only *ends* the running chunk once it already holds
> `MIN_CHUNK_TOKENS (200)` words. Below the floor, the boundary is crossed and
> accumulation continues.

Without it, a page of prose interleaved with three figures became four
unretrievable stubs — roughly half of all chunks were under 25 words. Tiny chunks
hurt ranking twice over: diffuse embeddings, and they crowd the top-5 with
near-duplicates of each other.

**Section attribution**: a chunk is attributed to the section it *started* in
(`buffer_section` snapshot on first append), so crossing a header to reach the
floor never silently relabels the text that came before it.

Tables and captions emit while the text buffer is still open, so chunks are
produced out of document order — the assembler re-sorts by `reading_order` and
renumbers `chunk_id` at the end, so id ordering means what callers assume.

Token counts are whitespace word counts. Deliberately tokenizer-free: a real
tokenizer would tie chunk boundaries to a model version and break determinism.

---

## Where the bodies are buried

- `fusion/containment.py` — `suppress_contained` / `suppress_table_contained` are
  implemented, imported by `pipeline.py`, and **never called**. Both were wired in
  and reverted after measurement; see `decisions.md` before re-enabling.
- Standalone figures with no caption produce no chunk at all.
- The `# unvalidated` confidence constants feed nothing today, but anything that
  starts consuming them (confidence-weighted ranking) is built on untuned numbers.
