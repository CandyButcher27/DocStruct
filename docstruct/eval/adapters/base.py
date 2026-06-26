"""Adapter interface for cross-tool chunking benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalChunk:
    """A retrievable unit produced by some tool's chunker."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class ChunkAdapter:
    """Base adapter: turn a PDF into comparable chunks."""

    name = "base"

    def available(self) -> bool:
        """Whether this adapter's dependencies are importable."""
        return True

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        raise NotImplementedError
