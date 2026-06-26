"""Fair cross-tool retrieval benchmark.

Holds the embedder and retriever constant and varies only the chunker. For each
tool and each document: chunk the PDF, index those chunks (shared embedder), run
that document's questions, and score by answer-span containment. Per-document
indexing isolates chunking quality from cross-document confusion.

Two retrievers are scored side by side, identical for every tool:
- **vector** — dense cosine search (sentence-transformers).
- **hybrid** — dense + BM25 lexical, fused by Reciprocal Rank Fusion (RRF), the
  ``RAG_Fundamentals`` "two indexes + merge" recipe. Lexical recall catches exact
  terms, symbols and citations that embeddings miss.

Metrics per tool (averaged over all questions): MRR, NDCG@k, Recall@k, Hit@1.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

from docstruct import config
from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk
from docstruct.eval.qa_generator import QAItem
from docstruct.eval.relevance import is_relevant

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    name: str
    # primary = hybrid retriever
    mrr: float = 0.0
    ndcg: float = 0.0
    recall: float = 0.0
    hit1: float = 0.0
    # vector-only retriever (for the hybrid-lift comparison)
    vec_mrr: float = 0.0
    vec_ndcg: float = 0.0
    vec_recall: float = 0.0
    vec_hit1: float = 0.0
    n_questions: int = 0
    n_chunks: int = 0
    mean_chunk_words: float = 0.0
    chunk_seconds: float = 0.0
    eval_seconds: float = 0.0
    errors: int = 0
    per_question: List[dict] = field(default_factory=list)


def _score(retrieved_texts: List[str], answer_span: str, k: int):
    flags = [is_relevant(t, answer_span) for t in retrieved_texts[:k]]
    rr = next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    hit1 = 1.0 if flags and flags[0] else 0.0
    recall = 1.0 if any(flags) else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags) if f)
    n_rel = sum(flags)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel)) if n_rel else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return rr, hit1, recall, ndcg


def _rrf(rank_lists: List[List[int]], k: int = config.RRF_K) -> List[int]:
    """Reciprocal Rank Fusion of several ranked index lists."""
    score: Dict[int, float] = {}
    for lst in rank_lists:
        for pos, idx in enumerate(lst):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: score[i], reverse=True)


def _qa_by_doc(qa: List[QAItem]) -> Dict[str, List[QAItem]]:
    out: Dict[str, List[QAItem]] = {}
    for item in qa:
        out.setdefault(item.source_doc, []).append(item)
    return out


def benchmark_tool(
    adapter: ChunkAdapter,
    pdf_paths: List[str],
    qa: List[QAItem],
    embedder,
    top_k: int = config.BENCHMARK_TOP_K,
) -> ToolResult:
    """Benchmark one tool across all documents that have questions."""
    from rank_bm25 import BM25Okapi

    from docstruct.indexing.vector_store import VectorStore

    by_doc = _qa_by_doc(qa)
    result = ToolResult(name=adapter.name)
    v = [0.0, 0.0, 0.0, 0.0]  # rr, hit, recall, ndcg (vector)
    h = [0.0, 0.0, 0.0, 0.0]  # rr, hit, recall, ndcg (hybrid)
    total_words = 0
    candidates = max(top_k * 4, config.BM25_CANDIDATES)

    for doc_idx, pdf in enumerate(pdf_paths):
        doc_id = os.path.basename(pdf)
        cases = by_doc.get(doc_id, [])
        if not cases:
            continue

        try:
            t0 = time.perf_counter()
            chunks: List[EvalChunk] = adapter.chunk(pdf)
            result.chunk_seconds += time.perf_counter() - t0
        except Exception as err:  # noqa: BLE001 - a tool may fail on some PDFs
            logger.warning("%s failed on %s: %s", adapter.name, doc_id, err)
            result.errors += 1
            continue

        chunks = [c for c in chunks if c.text.strip()]
        if not chunks:
            result.errors += 1
            continue
        texts = [c.text for c in chunks]
        result.n_chunks += len(chunks)
        total_words += sum(len(t.split()) for t in texts)

        store = VectorStore(collection_name=f"bench_{adapter.name}_{doc_idx}", embedder=embedder)
        store.collection.add(
            ids=[str(i) for i in range(len(texts))],
            documents=texts,
            embeddings=embedder.encode(texts, show_progress_bar=False).tolist(),
        )
        bm25 = BM25Okapi([t.lower().split() for t in texts])

        t1 = time.perf_counter()
        for case in cases:
            qv = embedder.encode([case.question], show_progress_bar=False).tolist()
            res = store.collection.query(query_embeddings=qv, n_results=min(candidates, len(texts)))
            vec_order = [int(i) for i in res.get("ids", [[]])[0]]
            scores = bm25.get_scores(case.question.lower().split())
            bm_order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:candidates]
            hyb_order = _rrf([vec_order, bm_order])[:top_k]

            vr = _score([texts[i] for i in vec_order[:top_k]], case.answer_span, top_k)
            hr = _score([texts[i] for i in hyb_order], case.answer_span, top_k)
            for i in range(4):
                v[i] += vr[i]; h[i] += hr[i]
            result.n_questions += 1
            result.per_question.append(
                {"doc": doc_id, "question": case.question,
                 "vec_rr": round(vr[0], 4), "hyb_rr": round(hr[0], 4)}
            )
        result.eval_seconds += time.perf_counter() - t1

    n = max(result.n_questions, 1)
    result.mrr, result.hit1, result.recall, result.ndcg = (round(h[0] / n, 4), round(h[1] / n, 4), round(h[2] / n, 4), round(h[3] / n, 4))
    result.vec_mrr, result.vec_hit1, result.vec_recall, result.vec_ndcg = (round(v[0] / n, 4), round(v[1] / n, 4), round(v[2] / n, 4), round(v[3] / n, 4))
    result.mean_chunk_words = round(total_words / max(result.n_chunks, 1), 1)
    result.chunk_seconds = round(result.chunk_seconds, 2)
    result.eval_seconds = round(result.eval_seconds, 2)
    return result


def run_benchmark(
    adapters: Dict[str, ChunkAdapter],
    pdf_paths: List[str],
    qa: List[QAItem],
    top_k: int = config.BENCHMARK_TOP_K,
) -> List[ToolResult]:
    """Benchmark every adapter, ranked by hybrid MRR. Embedder loaded once."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    results = []
    for name, adapter in adapters.items():
        logger.info("benchmarking %s ...", name)
        results.append(benchmark_tool(adapter, pdf_paths, qa, embedder, top_k))
    results.sort(key=lambda r: r.mrr, reverse=True)
    return results
