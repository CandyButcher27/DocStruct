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
FIGURE_MIN_AREA_RATIO = 0.03    # graphic cluster must cover >= this of page area
TABLE_MIN_ROWS = 2

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

# --- Indexing / retrieval ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "docstruct"
RETRIEVAL_TOP_K = 5
