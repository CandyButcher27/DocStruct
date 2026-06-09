"""Query-time retrieval returning chunks with section-path citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from docstruct import config
from docstruct.indexing.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    chunk_type: str
    page_num: int
    section_path: str
    score: float

    def citation(self) -> str:
        location = self.section_path or "(root)"
        return f"[{location}] (page {self.page_num}, score {self.score:.2f})"


def _section_label(metadata: dict) -> str:
    parts = [metadata.get(level) for level in ("h1", "h2", "h3")]
    return " > ".join(p for p in parts if p)


class Retriever:
    """Thin wrapper over :class:`VectorStore` producing cited results."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(
        self, query: str, top_k: int = config.RETRIEVAL_TOP_K, where: Optional[dict] = None
    ) -> List[RetrievalResult]:
        response = self.store.query(query, top_k=top_k, where=where)
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: List[RetrievalResult] = []
        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
            meta = meta or {}
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    content=doc,
                    chunk_type=meta.get("chunk_type", "text"),
                    page_num=int(meta.get("page_num", -1)),
                    section_path=_section_label(meta),
                    score=round(1.0 - float(dist), 4),  # cosine distance -> similarity
                )
            )
        return results
