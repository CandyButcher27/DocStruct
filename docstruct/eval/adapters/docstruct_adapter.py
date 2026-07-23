"""DocStruct's own structure-aware chunks as benchmark units."""

from __future__ import annotations

from typing import List, Optional

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk


def _section(chunk) -> str:
    sp = chunk.section_path
    return " > ".join(p for p in (sp.h1, sp.h2, sp.h3) if p)


class DocStructAdapter(ChunkAdapter):
    name = "docstruct"

    def __init__(
        self,
        weights: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.weights = weights
        self.cache_dir = cache_dir
        self.pipeline_mode = pipeline_mode
        if name:
            self.name = name

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        from docstruct.pipeline import run_pipeline

        result = run_pipeline(
            pdf_path,
            weights=self.weights,
            cache_dir=self.cache_dir,
            pipeline_mode=self.pipeline_mode,
        )
        return [
            EvalChunk(
                id=c.chunk_id,
                text=c.content,
                metadata={"section": _section(c), "page": c.page_num, "type": c.chunk_type},
            )
            for c in result.chunks
        ]
