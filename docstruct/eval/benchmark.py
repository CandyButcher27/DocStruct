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

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from docstruct import config
from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk
from docstruct.eval.coverage import raw_document_text, text_coverage
from docstruct.eval.qa_generator import QAItem
from docstruct.eval.relevance import get_relevance
from docstruct.eval.stats import align_per_question, bootstrap_ci, paired_bootstrap

logger = logging.getLogger(__name__)

# Reported metric -> the per-question field it averages. Everything statistical
# derives from these; adding a metric here is all it takes to get it a CI and a
# paired test.
_METRIC_KEYS = {
    "mrr": "hyb_rr",
    "ndcg": "hyb_ndcg",
    "recall": "hyb_recall",
    "hit1": "hyb_hit1",
    "vec_mrr": "vec_rr",
    "context_words": "context_words",
}


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
    # Words handed to the generator per query (sum over the top-k retrieved chunks).
    # A tool can buy MRR with bigger chunks; this is what that costs downstream.
    context_words: float = 0.0
    mrr_per_kword: float = 0.0
    # Extraction fidelity, measured against raw pdfplumber text — no gold, no LLM.
    # The only quality signal here that is about extraction rather than retrieval.
    coverage: float = 0.0
    duplication: float = 0.0
    chunk_seconds: float = 0.0
    eval_seconds: float = 0.0
    errors: int = 0
    per_question: List[dict] = field(default_factory=list)
    per_doc: List[dict] = field(default_factory=list)
    # 95% bootstrap CI per metric, {"mrr": [lo, hi], ...}. A point estimate over a
    # few hundred questions invites a comparison it cannot support on its own.
    ci: Dict[str, List[float]] = field(default_factory=dict)
    # Paired bootstrap of this tool against the reference tool, per metric.
    # Empty on the reference tool itself.
    vs_reference: Dict[str, dict] = field(default_factory=dict)


def _score(retrieved_texts: List[str], answer_span: str, k: int, relevant=None):
    relevant = relevant or get_relevance("span")
    flags = [relevant(t, answer_span) for t in retrieved_texts[:k]]
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


def _reference_text(pdf_path: str, cache: Dict[str, str]) -> str:
    """Raw document text, extracted once and shared by every tool in the run."""
    if pdf_path not in cache:
        cache[pdf_path] = raw_document_text(pdf_path)
    return cache[pdf_path]


# Shared across `benchmark_tool` calls: the reference text is a property of the
# PDF, not of the tool, so re-extracting it per tool is six times the work for
# identical output.
_REFERENCE_CACHE: Dict[str, str] = {}


def _qa_by_doc(qa: List[QAItem]) -> Dict[str, List[QAItem]]:
    out: Dict[str, List[QAItem]] = {}
    for item in qa:
        out.setdefault(item.source_doc, []).append(item)
    return out


def _ckpt_path(cache_dir: Optional[str], tool_name: str) -> Optional[str]:
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"bench_ckpt_{tool_name}.json")


def _load_ckpt(path: Optional[str]):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _save_ckpt(path: Optional[str], data: dict) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)


def benchmark_tool(
    adapter: ChunkAdapter,
    pdf_paths: List[str],
    qa: List[QAItem],
    embedder,
    top_k: int = config.BENCHMARK_TOP_K,
    cache_dir: Optional[str] = None,
    rrf_k: int = config.RRF_K,
    reranker=None,
    relevance: str = "span",
) -> ToolResult:
    """Benchmark one tool across all documents that have questions."""
    from rank_bm25 import BM25Okapi

    from docstruct.indexing.vector_store import VectorStore

    relevant = get_relevance(relevance)
    by_doc = _qa_by_doc(qa)
    docs_with_qa = [p for p in pdf_paths if os.path.basename(p) in by_doc]
    n_total = len(docs_with_qa)

    ckpt_path = _ckpt_path(cache_dir, adapter.name)
    ckpt = _load_ckpt(ckpt_path)

    result = ToolResult(name=adapter.name)
    v = [0.0, 0.0, 0.0, 0.0]
    h = [0.0, 0.0, 0.0, 0.0]
    total_words = 0
    total_context_words = 0
    done_docs: set = set()

    if ckpt:
        v = ckpt["v"]; h = ckpt["h"]
        total_words = ckpt["total_words"]
        total_context_words = ckpt.get("total_context_words", 0)
        result.n_chunks = ckpt["n_chunks"]
        result.errors = ckpt["errors"]
        result.chunk_seconds = ckpt["chunk_seconds"]
        result.eval_seconds = ckpt["eval_seconds"]
        result.per_question = ckpt["per_question"]
        result.per_doc = ckpt.get("per_doc", [])
        result.n_questions = len(result.per_question)
        done_docs = set(ckpt["done_docs"])
        print(f"  [{adapter.name}] resuming: {len(done_docs)}/{n_total} docs already done", flush=True)

    candidates = max(top_k * 4, config.BM25_CANDIDATES)

    for doc_idx, pdf in enumerate(pdf_paths):
        doc_id = os.path.basename(pdf)
        cases = by_doc.get(doc_id, [])
        if not cases:
            continue
        if doc_id in done_docs:
            print(f"  [{adapter.name}] {doc_id}: skip (cached)", flush=True)
            continue

        doc_num = len(done_docs) + 1
        print(f"  [{adapter.name}] {doc_id} ({doc_num}/{n_total}) chunking...", flush=True)

        try:
            t0 = time.perf_counter()
            chunks: List[EvalChunk] = adapter.chunk(pdf)
            chunk_t = time.perf_counter() - t0
            result.chunk_seconds += chunk_t
        except Exception as err:  # noqa: BLE001
            print(f"  [{adapter.name}] {doc_id}: ERROR — {err}", flush=True)
            logger.warning("%s failed on %s: %s", adapter.name, doc_id, err)
            result.errors += 1
            done_docs.add(doc_id)
            continue

        chunks = [c for c in chunks if c.text.strip()]
        if not chunks:
            print(f"  [{adapter.name}] {doc_id}: ERROR — 0 chunks produced", flush=True)
            result.errors += 1
            done_docs.add(doc_id)
            continue

        texts = [c.text for c in chunks]
        index_texts = texts
        result.n_chunks += len(chunks)
        total_words += sum(len(t.split()) for t in texts)
        print(f"  [{adapter.name}] {doc_id}: {len(chunks)} chunks ({chunk_t:.1f}s), embedding...", flush=True)

        store = VectorStore(collection_name=f"bench_{adapter.name}_{doc_idx}", embedder=embedder)
        store.collection.add(
            ids=[str(i) for i in range(len(index_texts))],
            documents=index_texts,
            embeddings=embedder.encode(index_texts, show_progress_bar=False).tolist(),
        )
        bm25 = BM25Okapi([t.lower().split() for t in index_texts])

        t1 = time.perf_counter()
        doc_hits = 0
        doc_rr = 0.0; doc_recall = 0.0; doc_hit1 = 0.0
        for case in cases:
            qv = embedder.encode([case.question], show_progress_bar=False).tolist()
            res = store.collection.query(query_embeddings=qv, n_results=min(candidates, len(texts)))
            vec_order = [int(i) for i in res.get("ids", [[]])[0]]
            scores = bm25.get_scores(case.question.lower().split())
            bm_order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:candidates]
            hyb_order = _rrf([vec_order, bm_order], k=rrf_k)[:top_k * 4]
            if reranker is not None and hyb_order:
                pairs = [(case.question, texts[i]) for i in hyb_order]
                ce_scores = reranker.predict(pairs, show_progress_bar=False)
                hyb_order = [hyb_order[i] for i in sorted(range(len(hyb_order)), key=lambda x: ce_scores[x], reverse=True)]
            hyb_order = hyb_order[:top_k]

            retrieved = [texts[i] for i in hyb_order]
            total_context_words += sum(len(t.split()) for t in retrieved)
            vr = _score([texts[i] for i in vec_order[:top_k]], case.answer_span, top_k, relevant)
            hr = _score(retrieved, case.answer_span, top_k, relevant)
            for i in range(4):
                v[i] += vr[i]; h[i] += hr[i]
            result.n_questions += 1
            doc_hits += int(hr[2] > 0)
            doc_rr += hr[0]; doc_recall += hr[2]; doc_hit1 += hr[1]
            # Every metric is kept per question, not just RR: bootstrap CIs and
            # the paired test need the raw per-question vector, and it cannot be
            # recovered from an average after the fact.
            result.per_question.append(
                {"doc": doc_id, "question": case.question,
                 "vec_rr": round(vr[0], 4), "hyb_rr": round(hr[0], 4),
                 "hyb_hit1": hr[1], "hyb_recall": hr[2], "hyb_ndcg": round(hr[3], 4),
                 "context_words": sum(len(t.split()) for t in retrieved)}
            )
        eval_t = time.perf_counter() - t1
        result.eval_seconds += eval_t
        done_docs.add(doc_id)

        n_q = max(len(cases), 1)
        doc_avg_words = round(sum(len(t.split()) for t in texts) / max(len(chunks), 1), 1)
        cov = text_coverage(texts, _reference_text(pdf, _REFERENCE_CACHE))
        doc_stat = {
            "doc": doc_id,
            "n_questions": len(cases),
            "n_chunks": len(chunks),
            "avg_words_per_chunk": doc_avg_words,
            "coverage": cov["coverage"],
            "duplication": cov["duplication"],
            "mrr": round(doc_rr / n_q, 4),
            "recall": round(doc_recall / n_q, 4),
            "hit1": round(doc_hit1 / n_q, 4),
            "hits": doc_hits,
        }
        result.per_doc.append(doc_stat)

        running_mrr = round(h[0] / max(result.n_questions, 1), 4)
        print(
            f"  [{adapter.name}] {doc_id}: {doc_hits}/{len(cases)} hits  "
            f"MRR={doc_stat['mrr']}  running_MRR={running_mrr}  ({eval_t:.1f}s)",
            flush=True,
        )

        _save_ckpt(ckpt_path, {
            "v": v, "h": h, "total_words": total_words,
            "total_context_words": total_context_words,
            "n_chunks": result.n_chunks, "errors": result.errors,
            "chunk_seconds": result.chunk_seconds, "eval_seconds": result.eval_seconds,
            "per_question": result.per_question, "per_doc": result.per_doc,
            "done_docs": list(done_docs),
        })

    n = max(result.n_questions, 1)
    result.mrr, result.hit1, result.recall, result.ndcg = (round(h[0] / n, 4), round(h[1] / n, 4), round(h[2] / n, 4), round(h[3] / n, 4))
    result.vec_mrr, result.vec_hit1, result.vec_recall, result.vec_ndcg = (round(v[0] / n, 4), round(v[1] / n, 4), round(v[2] / n, 4), round(v[3] / n, 4))
    result.mean_chunk_words = round(total_words / max(result.n_chunks, 1), 1)
    result.context_words = round(total_context_words / n, 1)
    # MRR bought per 1000 words of retrieved context: how efficiently a tool spends
    # the generator's context window, not just whether it can win by spending more.
    result.mrr_per_kword = round(result.mrr / (result.context_words / 1000.0), 4) if result.context_words else 0.0
    result.chunk_seconds = round(result.chunk_seconds, 2)
    result.eval_seconds = round(result.eval_seconds, 2)
    # Plain mean over documents, not weighted by length: the question is "how much
    # of a document does this tool keep", and a 60-page manual should not be able to
    # hide a tool dropping half of a 4-page one.
    covered = [d["coverage"] for d in result.per_doc if "coverage" in d]
    duped = [d["duplication"] for d in result.per_doc if "duplication" in d]
    result.coverage = round(sum(covered) / len(covered), 4) if covered else 0.0
    result.duplication = round(sum(duped) / len(duped), 4) if duped else 0.0
    result.ci = {
        metric: list(bootstrap_ci([q[key] for q in result.per_question if key in q]))
        for metric, key in _METRIC_KEYS.items()
    }

    return result


def compare_to_reference(results: List[ToolResult], reference: str) -> None:
    """Attach a paired bootstrap of every tool against ``reference``, in place.

    Paired because all tools answer the same questions. Comparing two marginal
    CIs instead would routinely call a consistent per-question difference
    insignificant purely because questions vary far more than tools do.
    """
    ref = next((r for r in results if r.name == reference), None)
    if ref is None or not ref.per_question:
        return
    for result in results:
        if result is ref or not result.per_question:
            continue
        stats: Dict[str, dict] = {}
        for metric, key in _METRIC_KEYS.items():
            ref_scores, other_scores = align_per_question(ref.per_question, result.per_question, key)
            if not ref_scores:
                continue
            # Sign convention: positive means the reference tool is ahead.
            stats[metric] = paired_bootstrap(ref_scores, other_scores)
        result.vs_reference = stats


def run_benchmark(
    adapters: Dict[str, ChunkAdapter],
    pdf_paths: List[str],
    qa: List[QAItem],
    top_k: int = config.BENCHMARK_TOP_K,
    cache_dir: Optional[str] = None,
    rrf_k: int = config.RRF_K,
    reranker_model: Optional[str] = None,
    reference: str = "docstruct",
    relevance: str = "span",
) -> List[ToolResult]:
    """Benchmark every adapter, ranked by hybrid MRR. Embedder loaded once."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    reranker = None
    if reranker_model:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(reranker_model)
        print(f"reranker: {reranker_model}", flush=True)

    results = []
    for name, adapter in adapters.items():
        print(f"\n=== {name} ===", flush=True)
        results.append(benchmark_tool(adapter, pdf_paths, qa, embedder, top_k, cache_dir=cache_dir, rrf_k=rrf_k, reranker=reranker, relevance=relevance))
        r = results[-1]
        lo, hi = r.ci.get("mrr", (0.0, 0.0))
        print(f"  => MRR={r.mrr} [{lo}, {hi}]  NDCG={r.ndcg}  Recall={r.recall}  "
              f"Hit@1={r.hit1}  ({r.n_questions} questions)", flush=True)
    compare_to_reference(results, reference)
    results.sort(key=lambda r: r.mrr, reverse=True)
    return results
