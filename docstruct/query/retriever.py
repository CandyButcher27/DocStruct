"""Query-time retrieval returning chunks with section-path citations.

Two modes:
- **dense** (default) — cosine vector search.
- **hybrid** — dense + BM25 lexical, fused by Reciprocal Rank Fusion (the
  ``RAG_Fundamentals`` two-indexes-plus-RRF recipe). Lexical recall catches exact
  terms, symbols and citations embeddings miss.

Both modes honour ``where``: in hybrid mode the BM25 index is built over the
*filtered* subset, so a section-scoped hybrid query ranks against that section only
rather than silently against the whole collection.

Optionally a cross-encoder reranks the fused candidate pool before truncation to
``top_k``. That is a second model at query time — off by default, since it breaks
the fully-local-no-model default the pipeline itself keeps.
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
        self,
        store: VectorStore,
        hybrid: bool = False,
        rrf_k: int = config.RRF_K,
        rerank_model: Optional[str] = None,
    ) -> None:
        self.store = store
        self.hybrid = hybrid
        self.rrf_k = rrf_k
        self.rerank_model = rerank_model
        self._reranker = None
        self._corpus = None  # (ids, docs, metas, bm25)
        self._corpus_key = "__unset__"

    def retrieve(
        self, query: str, top_k: int = config.RETRIEVAL_TOP_K, where: Optional[dict] = None
    ) -> List[RetrievalResult]:
        if self.hybrid:
            return self._hybrid(query, top_k, where)
        return self._dense(query, top_k, where)

    # --- reranking -------------------------------------------------------

    def _ensure_reranker(self):
        if self._reranker is None and self.rerank_model:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.rerank_model)
        return self._reranker

    def _rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        model = self._ensure_reranker()
        if model is None or len(results) < 2:
            return results
        scores = model.predict([(query, r.content) for r in results], show_progress_bar=False)
        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
        out = []
        for result, score in ranked:
            result.score = round(float(score), 4)
            out.append(result)
        return out

    def _pool_size(self, top_k: int) -> int:
        """Candidates to gather before reranking truncates to top_k."""
        if self.rerank_model:
            return max(top_k * 4, config.BM25_CANDIDATES)
        return top_k

    # --- dense -----------------------------------------------------------

    def _dense(self, query, top_k, where) -> List[RetrievalResult]:
        pool = self._pool_size(top_k)
        response = self.store.query(query, top_k=pool, where=where)
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
        return self._rerank(query, results)[:top_k]

    # --- hybrid ----------------------------------------------------------

    def _ensure_corpus(self, where: Optional[dict]):
        """BM25 over the documents matching ``where`` (cached per filter)."""
        key = tuple(sorted(where.items())) if where else None
        if self._corpus is None or self._corpus_key != key:
            from rank_bm25 import BM25Okapi

            kwargs = {"include": ["documents", "metadatas"]}
            if where:
                kwargs["where"] = where
            got = self.store.collection.get(**kwargs)
            ids = got["ids"]
            docs = got["documents"] or []
            metas = got.get("metadatas") or [{} for _ in ids]
            bm25 = BM25Okapi([(d or "").lower().split() for d in docs]) if ids else None
            self._corpus = (ids, docs, metas, bm25)
            self._corpus_key = key
        return self._corpus

    def _hybrid(self, query, top_k, where: Optional[dict] = None) -> List[RetrievalResult]:
        ids, docs, metas, bm25 = self._ensure_corpus(where)
        if not ids:
            return []
        pool = max(self._pool_size(top_k), config.BM25_CANDIDATES)

        resp = self.store.query(query, top_k=min(pool, len(ids)), where=where)
        dense_ids = resp.get("ids", [[]])[0]

        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:pool]
        bm_ids = [ids[i] for i in order]

        fused, rrf_score = _rrf([dense_ids, bm_ids], self.rrf_k)
        by_id = {cid: (doc, meta) for cid, doc, meta in zip(ids, docs, metas)}

        results: List[RetrievalResult] = []
        for cid in fused[:pool]:
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
        return self._rerank(query, results)[:top_k]
