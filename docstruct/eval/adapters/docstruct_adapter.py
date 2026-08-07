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
        # Chunk.page_num is the page a chunk *starts* on, but our units are
        # structural: a section body routinely flows across a page break, and
        # declaring only the first page makes the chunk unscoreable under
        # `--relevance page` whenever the evidence sits after the break. Measured
        # on OHR academic__2305.02437v3: 15 of 28 chunks carried text from a page
        # they never declared, the best-matching page a median of +1 from the one
        # claimed. Recover the full set from the blocks the chunk was built from,
        # which is what every other adapter reports.
        block_page = {b.block_id: b.page_num for b in result.blocks}
        out: List[EvalChunk] = []
        for c in result.chunks:
            pages = sorted({block_page[b] for b in c.source_block_ids if b in block_page})
            if not pages:
                pages = [c.page_num]
            out.append(
                EvalChunk(
                    id=c.chunk_id,
                    text=c.content,
                    metadata={
                        "section": _section(c),
                        "page": c.page_num,
                        "pages": pages,
                        "type": c.chunk_type,
                    },
                )
            )
        return out
