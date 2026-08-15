"""Central configuration: every numeric threshold the pipeline uses.

No magic numbers live in detector or fusion code. Values marked ``# unvalidated``
are inherited from the v0 prototype and have not yet been tuned against the
annotated evaluation set.

Per-parse overrides go through :func:`override` (used by ``parse(config=...)``): it
temporarily sets module globals under a lock and restores them afterwards, so a
caller's overrides never permanently mutate process-global state and two threads
cannot interleave different configs. This is deliberately lighter than threading a
frozen config object through every function — overridden parses serialize on the
lock rather than running fully in parallel, which is the accepted ceiling.
"""

from __future__ import annotations

import contextlib
import sys
import threading

_override_lock = threading.Lock()


@contextlib.contextmanager
def override(**values):
    """Temporarily set config values for the duration of the block (thread-safe)."""
    module = sys.modules[__name__]
    unknown = [k for k in values if not hasattr(module, k)]
    if unknown:
        raise AttributeError(f"unknown config keys: {unknown}")
    with _override_lock:
        saved = {k: getattr(module, k) for k in values}
        for k, v in values.items():
            setattr(module, k, v)
        try:
            yield
        finally:
            for k, v in saved.items():
                setattr(module, k, v)

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

# --- Fusion: label-aware containment (§5.2) ---
# Suppress a text block only when >= CONTAINMENT_MIN_RATIO of its area is inside a
# table block whose serialized text already covers its words. The one case where
# content provably exists twice; naive containment suppression lost 28% of content
# (decisions.md), so every other nested case is left alone. Targets the benchmark's
# 2.06x duplication. [MEASURE with duplication as primary readout, MRR as guard]
LABEL_AWARE_CONTAINMENT = False
CONTAINMENT_MIN_RATIO = 0.9

# --- Fusion: unilateral scaling ---
UNILATERAL_MODEL_SCALE = 0.75  # unvalidated
UNILATERAL_GEOMETRY_SCALE = 0.60  # unvalidated

CONFIDENCE_BOUNDS = {
    "unilateral_model": {"floor": 0.40, "ceiling": 0.85},  # unvalidated
    "unilateral_geometry": {"floor": 0.25, "ceiling": 0.65},  # unvalidated
}

# --- Reading order ---
COLUMN_GAP_RATIO = 0.15         # legacy centre-gap column split (XY_CUT = False)
# Split on *every* centre gap wider than COLUMN_GAP_RATIO x page_width, yielding k
# columns instead of exactly 1 or 2 (the single largest gap). Removes the structural
# ceiling for 3-column layouts. Expected byte-identical on 1/2-column pages — the
# arXiv corpus — so a change there is a bug. [MEASURE: assert no-op on current corpus]
MULTI_COLUMN = False
# Before column-splitting, cut the page into horizontal bands at full-width blocks (a
# block wider than FULL_WIDTH_RATIO x page_width acts as a separator), then run the
# existing column split within each band. Targets the one case the centre-gap
# splitter provably gets wrong — a full-width title/table across a 2-column body —
# without changing column detection globally the way XY-cut did (its measured loss).
# [MEASURE — highest-value reading-order experiment given decisions.md]
BAND_SPLIT = False
FULL_WIDTH_RATIO = 0.7
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
# Collapse the duplicated offset glyphs that faux-bold rendering produces
# ("Trannsfer hhave mmeanings") with pdfplumber's page.dedupe_chars before
# extraction. Fixes the doubled-glyph bug on doc1. [MEASURE on doc1 + full ablation]
DEDUPE_CHARS = False
# Apply Unicode NFKC normalization and strip soft hyphens (U+00AD) so ligatures
# ("fi"/"fl") and invisible hyphens stop breaking exact substring matching in
# retrieval and in the benchmark's containment scoring. Gold is generated from raw
# text, so normalize both sides or neither. [MEASURE]
NORMALIZE_TEXT = False
# Rejoin words split by a hard line-break hyphen ("trans-\nfer" -> "transfer"), the
# same class of failure the x-tolerance fix addressed for spacing. [MEASURE]
DEHYPHENATE = False

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
# Fixed-point graphic clustering is O(n^2) in primitives; a pathological page (1M
# vector primitives) would hang. Above this count, skip figure clustering on the page
# with a warning rather than stall. Well above any real document's per-page count.
FIGURE_CLUSTER_MAX_PRIMITIVES = 5000
FIGURE_MAX_TEXT_OVERLAP = 0.10  # graphic cluster must be mostly text-free
# Measure a figure's text-freeness by the fraction of the *figure's area* covered by
# overlapping text lines, instead of (overlapping line count / all lines on the
# page) — whose threshold drifts with page density (a sparse page fails a real
# figure on one stray line; a dense page passes a figure that swallows many). Off
# until the annotated detection set re-tunes FIGURE_MAX_TEXT_OVERLAP for the new,
# density-independent semantics. [MEASURE via detection metrics before enabling]
FIGURE_OVERLAP_BY_AREA = False
TABLE_MIN_ROWS = 2
# extract_tables() is ruled-line based: on a partly-ruled table it returns only the
# ruled fragment. If the rendered grid holds less than this fraction of the words in
# the region, fall back to raw region text so no rows are silently dropped.
TABLE_GRID_MIN_COVERAGE = 0.85
# When ruled-line extraction finds no grid, retry with pdfplumber's text strategy so
# borderless tables (the model detector finds them; ruled extraction can't structure
# them) get a grid. The TABLE_GRID_MIN_COVERAGE guard still protects against a bad
# grid replacing good raw text. [MEASURE + eyeball financial-domain PDFs]
TABLE_TEXT_STRATEGY_FALLBACK = False
# Extra settings forwarded to pdfplumber find_tables()/extract_tables() (snap/join
# tolerance, etc.). Empty = pdfplumber defaults. A home for tuning the corpus reaches.
TABLE_SETTINGS: dict = {}
# Table chunk serialization: "plaintext" (space-joined rows, verbatim-substring
# friendly, benchmark default) or "keyvalue" (header: cell pairs per row, answerable
# for "what was X in Q3" lookups). [MEASURE — keyvalue likely needs table-targeted
# gold; the substring benchmark penalizes breaking cell adjacency]
TABLE_SERIALIZATION = "plaintext"
# Split an oversized table into multiple row-segment chunks (repeating the header
# row) instead of emitting one chunk many times MAX_CHUNK_TOKENS. [MEASURE — ~neutral
# on arXiv, protective on financial docs]
TABLE_SPLIT_ROWS = False
# Merge a table continued across a page break into one chunk: the last table on page
# N and the first on page N+1, when their column counts match and their x-extents
# align within MULTIPAGE_TABLE_X_TOLERANCE x page_width, are joined (a repeated header
# row dropped). [MEASURE — protective on reports/forms, ~neutral on arXiv]
MERGE_MULTIPAGE_TABLES = False
MULTIPAGE_TABLE_X_TOLERANCE = 0.1
# Universal caption markers in prose documents (not a layout assumption).
CAPTION_PREFIX_PATTERN = r"^\s*(figure|fig\.?|table|tab\.?|scheme|algorithm)\s*\.?\s*\d+"

GEOMETRY_CONFIDENCE = {
    "text": 0.70,
    "header": 0.65,
    "table": 0.80,
    "figure": 0.60,
    "caption": 0.70,
}

# Drop running headers/footers and page numbers by cross-page repetition: a page's
# top-most or bottom-most line whose digit-normalized text repeats at nearly the same
# y on at least FURNITURE_MIN_PAGES pages is furniture, not content. Deterministic,
# document-global, no model. Catches furniture whatever detector labelled it, so it
# needs no DocLayNet page-header/footer remap. [MEASURE — noise removal, plausible
# retrieval gain and a definite doc.text/markdown quality gain]
STRIP_PAGE_FURNITURE = False
FURNITURE_MIN_PAGES = 3
FURNITURE_Y_TOLERANCE = 10.0    # points; lines within this share a repetition band

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
# Rank header levels by (font size, bold) instead of font size alone, so a bold and a
# regular heading at the same size become different levels (bold above). Weight is a
# real depth signal the font-rank-only approach throws away. [MEASURE — changes only
# section-path metadata, not scored content, but off until measured]
HEADER_RANK_BY_WEIGHT = False
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
# Emit reference/bibliography sections as chunk_type "references" instead of dropping
# them (the default). Excluded from retrieval indexing by default; useful for
# citation-analysis users. Makes the dead "references" enum value real. [MEASURE that
# the False default stays best for retrieval]
KEEP_REFERENCES = False
# Carry CHUNK_OVERLAP_TOKENS across structural boundaries too, not only across
# token-limit flushes. Measured (reports/ablations/08_overlap_on_boundary.json,
# 48 docs / 298 questions): MRR 0.7432 vs 0.7457, NDCG 0.7658 vs 0.7708, Recall
# 0.8826 vs 0.8859, Hit@1 identical — slightly worse on every metric that moved,
# at 86 more chunks and marginally more retrieved context. The intuition (a section
# opening loses the sentence before it) is real, but with MIN_CHUNK_TOKENS in place
# most boundaries are already crossed rather than cut, so the overlap mostly adds
# duplicated text that competes with its own source chunk. Off. See notes.md Stage 7.
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
# Completion budget per gold-generation request. Sized for a *reasoning* model:
# those spend most of the budget on hidden reasoning before emitting any JSON, so a
# budget sized for the answer alone comes back truncated mid-object — measured at
# 1500 tokens, gpt-oss:120b returned `finish_reason: length` and unparseable JSON
# every time, while 8000 completed cleanly.
#
# Lower it on a token-metered provider. Providers charge the *reserved* budget
# against the per-minute limit rather than the tokens actually produced, so a large
# reservation on a small allowance makes a request that can never succeed however
# long it waits.
QA_MAX_COMPLETION_TOKENS = 8000
# Seconds to wait before each gold-generation request. One segment of a paper is
# most of a free-tier minute's token allowance, so consecutive requests collide by
# construction; pacing up front is cheaper than a rejected request plus the
# provider's Retry-After, and it stops long runs from exhausting the retry ceiling
# in the middle of a document. Set to 0 for a provider with headroom.
QA_REQUEST_PACING_SECONDS = 20.0
BENCHMARK_TOP_K = 5
RELEVANCE_MIN_OVERLAP = 0.6     # token-overlap fallback when answer span isn't an exact substring
# Threshold for `region` relevance, used with page-region gold (FinanceBench).
# Higher than the span fallback on purpose: a 1.2k-character evidence block shares
# a long tail of common terms with any chunk from the same filing, so a lenient
# threshold marks half the document relevant and flattens the tools together.
# Swept 2026-08-16 on OHR-Bench, not FinanceBench: the sweep needs a corpus whose
# evidence is actually retrievable, and FinanceBench's is not (notes.md Stage 19).
# 3,558 questions, 7 tools, thresholds 0.1-1.0, re-scored offline from one run's
# dumped overlaps (reports/ohr_region_threshold_sweep.json), so chunking is identical
# at every point and the threshold is the only variable.
# A DocStruct variant places 1st at all ten thresholds, and {docstruct, docstruct_geo}
# hold the top two places at all ten. Margin over the best external tool is +0.045 to
# +0.062 MRR across 0.4-1.0. So the ranking does not depend on this number.
# Kept at 0.7, but read the value honestly: it is where our margin happens to peak
# (+0.0619). Below 0.4 every tool converges toward the 1.0 that 0.0 gives by
# definition, which is why the low end is uninformative rather than favourable.
RELEVANCE_REGION_MIN_OVERLAP = 0.7
# Hybrid retrieval (BM25 lexical + dense vector, fused by Reciprocal Rank Fusion)
RRF_K = 60                      # standard RRF constant
BM25_CANDIDATES = 20            # candidate pool per retriever before fusion
