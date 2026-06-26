"""Fair cross-tool retrieval benchmark.

Holds the embedder and retriever constant and varies only the chunker. For each
tool and each document: chunk the PDF, index those chunks (shared embedder), run
that document's questions, and score by answer-span containment. Per-document
indexing isolates chunking quality from cross-document confusion.

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
    mrr: float = 0.0
    ndcg: float = 0.0
    recall: float = 0.0
    hit1: float = 0.0
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
    from docstruct.indexing.vector_store import VectorStore

    by_doc = _qa_by_doc(qa)
    result = ToolResult(name=adapter.name)
    rr_sum = hit_sum = rec_sum = ndcg_sum = 0.0
    total_words = 0

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
        result.n_chunks += len(chunks)
        total_words += sum(len(c.text.split()) for c in chunks)

        store = VectorStore(collection_name=f"bench_{adapter.name}_{doc_idx}", embedder=embedder)
        store.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embedder.encode([c.text for c in chunks], show_progress_bar=False).tolist(),
        )

        t1 = time.perf_counter()
        for case in cases:
            res = store.collection.query(
                query_embeddings=embedder.encode([case.question], show_progress_bar=False).tolist(),
                n_results=min(top_k, len(chunks)),
            )
            texts = res.get("documents", [[]])[0]
            rr, hit1, recall, ndcg = _score(texts, case.answer_span, top_k)
            rr_sum += rr; hit_sum += hit1; rec_sum += recall; ndcg_sum += ndcg
            result.n_questions += 1
            result.per_question.append(
                {"doc": doc_id, "question": case.question, "rr": round(rr, 4),
                 "hit1": hit1, "recall": recall, "ndcg": round(ndcg, 4)}
            )
        result.eval_seconds += time.perf_counter() - t1

    n = max(result.n_questions, 1)
    result.mrr = round(rr_sum / n, 4)
    result.hit1 = round(hit_sum / n, 4)
    result.recall = round(rec_sum / n, 4)
    result.ndcg = round(ndcg_sum / n, 4)
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
    """Benchmark every adapter, ranked by MRR. Embedder loaded once and shared."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    results = []
    for name, adapter in adapters.items():
        logger.info("benchmarking %s ...", name)
        results.append(benchmark_tool(adapter, pdf_paths, qa, embedder, top_k))
    results.sort(key=lambda r: r.mrr, reverse=True)
    return results
