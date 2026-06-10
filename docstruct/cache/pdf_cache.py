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

from docstruct.schema import BoundingBox, Proposal


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

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, pdf_path: str) -> str:
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
