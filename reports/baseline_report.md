# DocStruct retrieval baseline report

_Generated 2026-06-26 22:15 UTC_

## Setup

- **Documents:** 48 born-digital PDFs
- **Questions:** 298 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: pymupdf4llm, docstruct, langchain, unstructured, docling.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pymupdf4llm | **0.6913** | 0.7126 | 0.8289 | 0.6107 | 0.5811 | +0.1102 | 1460 | 457.5 | 1277.17 | 0 |
| 2 | docstruct **(ours)** | **0.6677** | 0.7 | 0.8289 | 0.5705 | 0.5127 | +0.155 | 4252 | 164.0 | 2894.03 | 0 |
| 3 | langchain | **0.6493** | 0.6884 | 0.8221 | 0.5336 | 0.5236 | +0.1257 | 5407 | 102.1 | 1507.55 | 0 |
| 4 | unstructured | **0.6483** | 0.6748 | 0.7886 | 0.5604 | 0.4932 | +0.1551 | 7935 | 85.2 | 1597.64 | 0 |
| 5 | docling | **0.448** | 0.4678 | 0.557 | 0.3826 | 0.3675 | +0.0805 | 937 | 118.6 | 1471.06 | 0 |

## How to read this

- **MRR** — average reciprocal rank of the first chunk that contains the answer (higher = answers surface earlier).
- **Recall@k** — fraction of questions where the answer appears in the top-k.
- **Hit@1** — fraction where the very first chunk already contains the answer.
- **Avg words/chunk** — chunk granularity; huge chunks can inflate recall while hurting precision/precision-of-context.

## DocStruct's unique axis (not in any competitor)

Every chunk carries a section path, enabling **filtered retrieval** (`where={"h1": "4. Experiments"}`). No competitor here exposes section-hierarchy metadata, so this capability is qualitative, not in the table.

## Honest caveats

- Questions are **LLM-generated**: this measures retrieval consistency and the DocStruct-vs-tools delta well, but is weaker than human-judged relevance as an absolute claim.
- Containment relevance can miss paraphrased answers — but it is applied **identically** to every tool, so the ranking stays fair.
- Dataset is arXiv-heavy (born-digital prose); broader domains (legal/financial/manuals) are future work.
- This is a **signal/baseline**, not the Phase-2 public benchmark (50 PDFs, 200 human-checked Q&A).
