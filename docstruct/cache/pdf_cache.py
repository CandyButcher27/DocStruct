"""Disk cache for geometry proposals, keyed by PDF content hash.

Detection is deterministic, so caching by file hash lets tests and re-runs skip
re-detection entirely. Proposals serialize to JSON via dataclass dicts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import List, Optional

from docstruct import config
from docstruct.schema import BoundingBox, Proposal

# Every config value that can change block output (detection, fusion, reading order,
# text/table extraction, furniture, containment). Chunking keys are deliberately
# absent — they are what the block cache exists to let us vary cheaply.
_LAYOUT_CONFIG_KEYS = (
    "IOU_MATCH_THRESHOLD",
    "NMS_IOU_THRESHOLD",
    "CONFIRMED_BASE",
    "CONFIRMED_MODEL_BOOST",
    "CONFIRMED_AGREEMENT_BOOST",
    "CONFIRMED_BBOX_MODEL_CONF",
    "CONFIRMED_BBOX_IOU",
    "DISPUTED_MULTIPLIER",
    "UNILATERAL_MODEL_SCALE",
    "UNILATERAL_GEOMETRY_SCALE",
    "CONFIDENCE_BOUNDS",
    "COLUMN_GAP_RATIO",
    "CAPTION_MAX_DISTANCE",
    "XY_CUT",
    "XY_CUT_MIN_COLUMN_GAP_RATIO",
    "XY_CUT_MIN_ROW_GAP",
    "MULTI_COLUMN",
    "BAND_SPLIT",
    "FULL_WIDTH_RATIO",
    "LINE_Y_TOLERANCE",
    "PARAGRAPH_GAP_FACTOR",
    "HEADER_SIZE_RATIO",
    "HEADER_MAX_LINES",
    "HEADER_MAX_WORDS",
    "BOLD_FONT_MARKERS",
    "HEADER_BOLD_BONUS",
    "HEADER_RANK_BY_WEIGHT",
    "GEOMETRY_CONFIDENCE_CEIL",
    "GEOMETRY_CONFIDENCE",
    "FIGURE_MIN_AREA_RATIO",
    "FIGURE_CLUSTER_GAP",
    "FIGURE_CLUSTER_MAX_PRIMITIVES",
    "FIGURE_MAX_TEXT_OVERLAP",
    "FIGURE_OVERLAP_BY_AREA",
    "TABLE_MIN_ROWS",
    "TABLE_GRID_MIN_COVERAGE",
    "TABLE_TEXT_STRATEGY_FALLBACK",
    "TABLE_SETTINGS",
    "TABLE_SERIALIZATION",
    "MERGE_MULTIPAGE_TABLES",
    "MULTIPAGE_TABLE_X_TOLERANCE",
    "TEXT_X_TOLERANCE_RATIO",
    "DEDUPE_CHARS",
    "NORMALIZE_TEXT",
    "DEHYPHENATE",
    "STRIP_PAGE_FURNITURE",
    "FURNITURE_MIN_PAGES",
    "FURNITURE_Y_TOLERANCE",
    "LABEL_AWARE_CONTAINMENT",
    "CONTAINMENT_MIN_RATIO",
    "CAPTION_PREFIX_PATTERN",
    "MODEL_DPI",
    "MODEL_CONF_THRESHOLD",
    "DOCLAYNET_LABEL_MAP",
)


def layout_config_fingerprint() -> str:
    """Short hash of every config value that can change block output."""
    payload = json.dumps(
        {k: getattr(config, k, None) for k in _LAYOUT_CONFIG_KEYS},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def file_hash(path: str) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def proposals_to_json(proposals: List[Proposal]) -> str:
    return json.dumps([asdict(p) for p in proposals])


def proposals_from_json(payload: str) -> List[Proposal]:
    data = json.loads(payload)
    return [
        Proposal(
            label=d["label"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"]),
            source=d["source"],
            page_num=d["page_num"],
            proposal_id=d["proposal_id"],
        )
        for d in data
    ]


class ProposalCache:
    """JSON-on-disk cache of ``List[Proposal]`` keyed by file hash + namespace."""

    namespace = "geometry"
    # Geometry detection reads layout config (columns, headers, figures, tables), so
    # its cache key must track it or a config change silently serves stale proposals.
    config_aware = True

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, pdf_path: str) -> str:
        if self.config_aware:
            return f"{file_hash(pdf_path)}.{self.namespace}.{layout_config_fingerprint()}.json"
        return f"{file_hash(pdf_path)}.{self.namespace}.json"

    def _path(self, pdf_path: str) -> str:
        return os.path.join(self.cache_dir, self._key(pdf_path))

    def get(self, pdf_path: str) -> Optional[List[Proposal]]:
        path = self._path(pdf_path)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return proposals_from_json(fh.read())

    def set(self, pdf_path: str, proposals: List[Proposal]) -> None:
        with open(self._path(pdf_path), "w", encoding="utf-8") as fh:
            fh.write(proposals_to_json(proposals))
