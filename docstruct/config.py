"""Central configuration: every numeric threshold the pipeline uses.

No magic numbers live in detector or fusion code. Values marked ``# unvalidated``
are inherited from the v0 prototype and have not yet been tuned against the
annotated evaluation set.
"""

# --- Fusion: matching ---
IOU_MATCH_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.5

# --- Fusion: confirmed confidence formula ---
CONFIRMED_BASE = 0.85
CONFIRMED_MODEL_BOOST = 0.10
CONFIRMED_AGREEMENT_BOOST = 0.05
CONFIRMED_BBOX_MODEL_CONF = 0.8
CONFIRMED_BBOX_IOU = 0.6

# --- Fusion: disputed ---
DISPUTED_MULTIPLIER = 0.85

# --- Fusion: unilateral scaling ---
UNILATERAL_MODEL_SCALE = 0.75  # unvalidated
UNILATERAL_GEOMETRY_SCALE = 0.60  # unvalidated

CONFIDENCE_BOUNDS = {
    "unilateral_model": {"floor": 0.40, "ceiling": 0.85},  # unvalidated
    "unilateral_geometry": {"floor": 0.25, "ceiling": 0.65},  # unvalidated
}

# --- Reading order ---
COLUMN_GAP_RATIO = 0.15         # legacy centre-gap column split (XY_CUT = False)
CAPTION_MAX_DISTANCE = 100.0
# Recursive XY-cut: split a region at real whitespace bands and recurse, instead of
# forcing every page into one or two centre-defined columns. It is the more
# principled algorithm and it handles layouts the legacy split provably gets wrong
# (a full-width title or table across a two-column body — see tests), but on this
# corpus it measured *worse*: MRR 0.7356 vs 0.7457, Hit@1 0.6275 vs 0.6409, with
# recall identical. Raising XY_CUT_MIN_ROW_GAP 4x changed nothing, so the delta is
# entirely in column detection, not band detection. Off by default until there is a
# corpus where it wins; the implementation and its tests stay. See notes.md Stage 6.
XY_CUT = False
XY_CUT_MIN_COLUMN_GAP_RATIO = 0.03  # gutter must span this fraction of page width
XY_CUT_MIN_ROW_GAP = 3.0            # points of clear horizontal whitespace to cut on

# --- Text extraction ---
# pdfplumber inserts a space when the gap between two characters exceeds a
# tolerance. Its default is a flat 3pt, which is wider than the real inter-word gap
# in small type, so author lines, footnotes and table cells come out with the words
# run together ("IreneAmerini1,ElenaBalashova2"). Scaling the tolerance by font size
# instead fixes small text without over-splitting large headings.
TEXT_X_TOLERANCE_RATIO = 0.15

# --- Geometry detector ---
LINE_Y_TOLERANCE = 3.0          # words within this vertical gap share a line
PARAGRAPH_GAP_FACTOR = 1.6      # line gap > factor * line_height breaks a block
HEADER_SIZE_RATIO = 1.15        # font size > ratio * body median => header
HEADER_MAX_LINES = 3            # headers are short
HEADER_MAX_WORDS = 12           # a short all-bold line also counts as a header
# Font-weight markers that signal emphasis (headers), beyond plain "bold".
BOLD_FONT_MARKERS = ("bold", "black", "heavy", "semibold", "medium", "-medi")
HEADER_BOLD_BONUS = 0.05        # confidence bump for bold headers
GEOMETRY_CONFIDENCE_CEIL = 0.95
FIGURE_MIN_AREA_RATIO = 0.03    # graphic cluster must cover >= this of page area
FIGURE_CLUSTER_GAP = 10.0       # merge graphic primitives within this gap
FIGURE_MAX_TEXT_OVERLAP = 0.10  # graphic cluster must be mostly text-free
TABLE_MIN_ROWS = 2
# extract_tables() is ruled-line based: on a partly-ruled table it returns only the
# ruled fragment. If the rendered grid holds less than this fraction of the words in
# the region, fall back to raw region text so no rows are silently dropped.
TABLE_GRID_MIN_COVERAGE = 0.85
# Universal caption markers in prose documents (not a layout assumption).
CAPTION_PREFIX_PATTERN = r"^\s*(figure|fig\.?|table|tab\.?|scheme|algorithm)\s*\.?\s*\d+"

GEOMETRY_CONFIDENCE = {
    "text": 0.70,
    "header": 0.65,
    "table": 0.80,
    "figure": 0.60,
    "caption": 0.70,
}

# --- Chunking ---
# MIN/MAX were chosen by sweeping both against the retrieval benchmark. Raw MRR
# climbs monotonically with chunk size all the way to MIN=600 (0.7584) — a
# containment metric always rewards handing the retriever more text — but that is
# fixed-window chunking wearing a hat, and it costs 59% more retrieved context than
# this setting for +0.027 MRR. 200/500 sits on the Pareto front: it beats every
# larger setting except 600/800 on MRR while staying cheaper to feed to an LLM.
# See notes.md "Stage 4" for the full grid.
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 75  # tail words carried into next chunk on token-limit flush
HEADER_LEVELS = 3
# Let an explicit section number ("3.2.1 Ablations") set the header's depth instead
# of its font-size rank. Font size is a proxy for depth; a section number *is* the
# depth, so where one exists there is nothing left to infer. Fixes documents that
# set every heading at one size, or distinguish levels by weight rather than size —
# a documented weakness of the font-rank-only approach. Deterministic, no model.
HEADER_NUMBERING_LEVELS = True
# A structural boundary (header/table/caption) only ends the running text chunk once
# it holds at least this many words. Below the floor the boundary is crossed and the
# buffer keeps accumulating, so a page of prose broken up by figures stays one chunk
# instead of becoming several stubs. 0 restores the pre-0.3 flush-on-every-boundary
# behaviour.
MIN_CHUNK_TOKENS = 200
# Tables and captions always emit their own chunk. When False they no longer *also*
# split the prose around them.
BREAK_TEXT_ON_TABLE = False
BREAK_TEXT_ON_CAPTION = False
# Write the header line into the body of the chunk it introduces, not only into
# section-path metadata. Without this, text that is laid out as a heading (titles,
# author lines, run-in headers) exists in no chunk and can never be retrieved.
INLINE_HEADER_TEXT = True
# Carry CHUNK_OVERLAP_TOKENS across structural boundaries too, not only across
# token-limit flushes.
OVERLAP_ON_BOUNDARY = False

# --- Model detector (optional, YOLOv8 / DocLayNet) ---
MODEL_WEIGHTS = None            # path or ultralytics-resolvable name; set to enable
MODEL_DPI = 150                 # page render resolution for inference
MODEL_CONF_THRESHOLD = 0.25
# DocLayNet (11 classes) -> the 5 DocStruct labels. Keyed by lowercased class name.
DOCLAYNET_LABEL_MAP = {
    "caption": "caption",
    "footnote": "text",
    "formula": "text",
    "list-item": "text",
    "page-footer": "text",
    "page-header": "text",
    "section-header": "header",
    "title": "header",
    "table": "table",
    "text": "text",
    "picture": "figure",
}

# --- Indexing / retrieval ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "docstruct"
RETRIEVAL_TOP_K = 5

# --- LLM (eval-only: Q&A generation + optional judge; never in the pipeline) ---
LLM_MODEL = "gpt-oss:120b"      # Ollama cloud, open-weights; override via DOCSTRUCT_LLM_MODEL
LLM_TIMEOUT = 120.0

# --- Benchmark / eval ---
QA_PER_DOC = 5                  # questions generated per document
# Characters of source text per gold-generation request. This is a *rate-limit*
# budget, not a context-window one: hosted providers meter tokens per minute far
# below the model's advertised context, and reject anything over the per-minute
# allowance outright. Longer documents are split into consecutive segments and the
# question budget spread across them.
#
# Measured in characters rather than words on purpose. Word count is a terrible
# token proxy for scientific PDFs: a 3,641-word segment of an equation-heavy paper
# tokenised to 12,228 tokens — 3.4 tokens/word, or under 2 characters per token —
# where ordinary prose runs nearer 1.3. Budgeting by words meant every dense
# document silently failed its whole request and lost its questions. At ~2 chars per
# token worst case this leaves headroom under a 12k/min limit.
QA_MAX_CHARS_PER_REQUEST = 14000
# Shortest gold answer span to accept, in words. A one- or two-word span ("DanceOPD")
# is contained by almost any chunk that mentions the topic, so it scores every tool
# alike and destroys the benchmark's ability to discriminate between them. Weaker
# generators drift toward exactly those spans regardless of the prompt, so the floor
# is enforced at validation rather than trusted to instruction-following.
QA_MIN_SPAN_WORDS = 6
# Seconds to wait before each gold-generation request. One segment of a paper is
# most of a free-tier minute's token allowance, so consecutive requests collide by
# construction; pacing up front is cheaper than a rejected request plus the
# provider's Retry-After, and it stops long runs from exhausting the retry ceiling
# in the middle of a document. Set to 0 for a provider with headroom.
QA_REQUEST_PACING_SECONDS = 20.0
BENCHMARK_TOP_K = 5
RELEVANCE_MIN_OVERLAP = 0.6     # token-overlap fallback when answer span isn't an exact substring
# Hybrid retrieval (BM25 lexical + dense vector, fused by Reciprocal Rank Fusion)
RRF_K = 60                      # standard RRF constant
BM25_CANDIDATES = 20            # candidate pool per retriever before fusion
