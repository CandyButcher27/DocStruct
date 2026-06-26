"""Query-time retrieval returning chunks with section-path citations.

Two modes:
- **dense** (default) — cosine vector search; supports section filtering (`where`).
- **hybrid** — dense + BM25 lexical, fused by Reciprocal Rank Fusion (the
  ``RAG_Fundamentals`` two-indexes-plus-RRF recipe). Lexical recall catches exact
  terms, symbols and citations embeddings miss. Hybrid ignores ``where`` (BM25
  runs over the whole collection); use dense mode for section-filtered queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _rrf(rank_lists: List[List[str]], k: int) -> Tuple[List[str], Dict[str, float]]:
    score: Dict[str, float] = {}
    for lst in rank_lists:
        for pos, cid in enumerate(lst):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + pos + 1)
    ordered = sorted(score, key=lambda c: score[c], reverse=True)
    return ordered, score


class Retriever:
    """Wrapper over :class:`VectorStore` producing cited results (dense or hybrid)."""

    def __init__(
        self, store: VectorStore, hybrid: bool = False, rrf_k: int = config.RRF_K
    ) -> None:
        self.store = store
        self.hybrid = hybrid
        self.rrf_k = rrf_k
        self._corpus = None  # (ids, docs, metas, bm25)

    def retrieve(
        self, query: str, top_k: int = config.RETRIEVAL_TOP_K, where: Optional[dict] = None
    ) -> List[RetrievalResult]:
        if self.hybrid and where is None:
            return self._hybrid(query, top_k)
        return self._dense(query, top_k, where)

    def _dense(self, query, top_k, where) -> List[RetrievalResult]:
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

    def _ensure_corpus(self):
        if self._corpus is None:
            from rank_bm25 import BM25Okapi

            got = self.store.collection.get(include=["documents", "metadatas"])
            ids = got["ids"]
            docs = got["documents"] or []
            metas = got.get("metadatas") or [{} for _ in ids]
            bm25 = BM25Okapi([(d or "").lower().split() for d in docs])
            self._corpus = (ids, docs, metas, bm25)
        return self._corpus

    def _hybrid(self, query, top_k) -> List[RetrievalResult]:
        ids, docs, metas, bm25 = self._ensure_corpus()
        if not ids:
            return []
        pool = max(top_k, config.BM25_CANDIDATES)

        resp = self.store.query(query, top_k=min(pool, len(ids)))
        dense_ids = resp.get("ids", [[]])[0]

        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:pool]
        bm_ids = [ids[i] for i in order]

        fused, rrf_score = _rrf([dense_ids, bm_ids], self.rrf_k)
        by_id = {cid: (doc, meta) for cid, doc, meta in zip(ids, docs, metas)}

        results: List[RetrievalResult] = []
        for cid in fused[:top_k]:
            doc, meta = by_id.get(cid, ("", {}))
            meta = meta or {}
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    content=doc,
                    chunk_type=meta.get("chunk_type", "text"),
                    page_num=int(meta.get("page_num", -1)),
                    section_path=_section_label(meta),
                    score=round(rrf_score[cid], 4),
                )
            )
        return results
