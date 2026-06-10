"""Disk cache for model proposals, namespaced by the weights identifier.

Model inference is the slow stage; caching by PDF hash plus a weights tag avoids
re-running YOLO on unchanged documents across runs.
"""

from __future__ import annotations

import os

from docstruct.cache.pdf_cache import ProposalCache


def _safe_tag(weights: str) -> str:
    return os.path.splitext(os.path.basename(weights))[0] if weights else "model"


class ModelProposalCache(ProposalCache):
    """Proposal cache whose namespace encodes the model weights used."""

    def __init__(self, cache_dir: str, weights: str) -> None:
        super().__init__(cache_dir)
        self.namespace = f"model.{_safe_tag(weights)}"
