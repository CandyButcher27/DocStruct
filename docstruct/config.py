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
COLUMN_GAP_RATIO = 0.15
CAPTION_MAX_DISTANCE = 100.0

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
MAX_CHUNK_TOKENS = 400
HEADER_LEVELS = 3

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
BENCHMARK_TOP_K = 5
RELEVANCE_MIN_OVERLAP = 0.6     # token-overlap fallback when answer span isn't an exact substring
