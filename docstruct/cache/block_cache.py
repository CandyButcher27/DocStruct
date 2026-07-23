"""Disk cache for fused, reading-ordered, text-populated blocks.

Everything upstream of chunking — detection, fusion, reading order, pdfplumber text
and table extraction — is deterministic in the PDF plus the detector/layout config,
and is by far the slowest part of the pipeline. Caching at the block boundary lets a
chunking change be re-benchmarked without redoing any of it.

The key covers the PDF bytes, the weights identity, and every config value that can
change the blocks, so a layout-config change invalidates the entry instead of
silently serving stale blocks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Optional

from docstruct.cache.pdf_cache import file_hash, layout_config_fingerprint
from docstruct.schema import Block, BoundingBox, ConfidenceBreakdown, Source


def blocks_to_json(blocks: List[Block]) -> str:
    return json.dumps([asdict(b) for b in blocks])


def blocks_from_json(payload) -> List[Block]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    return [
        Block(
            bbox=BoundingBox(**d["bbox"]),
            label=d["label"],
            confidence=ConfidenceBreakdown(**d["confidence"]),
            source=Source(d["source"]),
            page_num=d["page_num"],
            block_id=d["block_id"],
            reading_order=d["reading_order"],
            caption_target_id=d["caption_target_id"],
            text=d["text"],
            table_data=d["table_data"],
            font_size=d["font_size"],
        )
        for d in data
    ]


class BlockCache:
    """JSON-on-disk cache of populated ``List[Block]`` plus its fusion diagnostics."""

    def __init__(self, cache_dir: str, weights: Optional[str] = None,
                 variant: Optional[str] = None) -> None:
        self.cache_dir = cache_dir
        self.mode = os.path.basename(weights) if weights else "geometry-only"
        # A single-detector ablation run with the same weights produces different
        # blocks from the hybrid run; without the variant in the key they collide
        # and one silently serves the other's cached output.
        if variant:
            self.mode = f"{self.mode}.{variant}"
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, pdf_path: str) -> str:
        name = f"{file_hash(pdf_path)}.blocks.{self.mode}.{layout_config_fingerprint()}.json"
        return os.path.join(self.cache_dir, name)

    def get(self, pdf_path: str):
        """Return ``(blocks, diagnostics)`` or ``None`` on a miss."""
        path = self._path(pdf_path)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return blocks_from_json(payload["blocks"]), payload["diagnostics"]
        except (ValueError, KeyError, TypeError):
            return None  # stale or truncated entry: re-derive

    def set(self, pdf_path: str, blocks: List[Block], diagnostics: dict) -> None:
        path = self._path(pdf_path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"blocks": [asdict(b) for b in blocks], "diagnostics": diagnostics}, fh)
        os.replace(tmp, path)
