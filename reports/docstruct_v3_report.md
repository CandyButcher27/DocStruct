# DocStruct retrieval baseline report

_Generated 2026-06-28 10:48 UTC_

## Setup

- **Documents:** 48 born-digital PDFs
- **Questions:** 298 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: docstruct.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | docstruct **(ours)** | **0.6573** | 0.6928 | 0.8423 | 0.5403 | 0.5041 | +0.1532 | 3773 | 185.1 | 741.48 | 0 |

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


### Per-doc breakdown: docstruct (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc44.pdf | 8 | 36 | 122.6 | 0.2437 | 0.625 | 0.0 | 5/8 |
| doc19.pdf | 2 | 24 | 102.4 | 0.25 | 0.5 | 0.0 | 1/2 |
| doc10.pdf | 9 | 69 | 235.7 | 0.3333 | 0.4444 | 0.2222 | 4/9 |
| doc17.pdf | 8 | 76 | 111.9 | 0.3854 | 0.625 | 0.25 | 5/8 |
| doc24.pdf | 3 | 100 | 56.3 | 0.4167 | 0.6667 | 0.3333 | 2/3 |
| doc32.pdf | 9 | 63 | 65.5 | 0.4204 | 0.7778 | 0.2222 | 7/9 |
| doc6.pdf | 9 | 8 | 151.0 | 0.4444 | 0.8889 | 0.1111 | 8/9 |
| doc45.pdf | 8 | 141 | 170.9 | 0.4792 | 0.75 | 0.375 | 6/8 |
| doc30.pdf | 1 | 247 | 81.2 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc37.pdf | 1 | 90 | 208.7 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc38.pdf | 1 | 140 | 387.0 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc9.pdf | 9 | 35 | 213.9 | 0.5037 | 0.7778 | 0.3333 | 7/9 |
| doc1.pdf | 6 | 41 | 162.6 | 0.5139 | 0.8333 | 0.3333 | 5/6 |
| doc34.pdf | 5 | 217 | 53.0 | 0.5167 | 0.8 | 0.4 | 4/5 |
| doc41.pdf | 9 | 107 | 127.2 | 0.5167 | 0.7778 | 0.4444 | 7/9 |
| doc22.pdf | 8 | 39 | 125.8 | 0.5312 | 0.625 | 0.5 | 5/8 |
| doc7.pdf | 6 | 126 | 263.2 | 0.5333 | 0.6667 | 0.5 | 4/6 |
| doc5.pdf | 7 | 140 | 235.5 | 0.5762 | 1.0 | 0.2857 | 7/7 |
| doc47.pdf | 6 | 51 | 364.7 | 0.5889 | 0.8333 | 0.5 | 5/6 |
| doc49.pdf | 5 | 11 | 317.4 | 0.6167 | 1.0 | 0.4 | 5/5 |
| doc20.pdf | 9 | 164 | 110.8 | 0.6204 | 0.8889 | 0.4444 | 8/9 |
| doc42.pdf | 10 | 26 | 68.2 | 0.6417 | 0.9 | 0.5 | 9/10 |
| doc25.pdf | 9 | 157 | 147.8 | 0.6481 | 0.7778 | 0.5556 | 7/9 |
| doc35.pdf | 7 | 79 | 203.5 | 0.6548 | 0.8571 | 0.5714 | 6/7 |
| doc3.pdf | 3 | 79 | 157.1 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc33.pdf | 3 | 95 | 113.7 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc29.pdf | 10 | 46 | 118.4 | 0.7 | 0.7 | 0.7 | 7/10 |
| doc16.pdf | 7 | 16 | 244.4 | 0.7143 | 0.8571 | 0.5714 | 6/7 |
| doc39.pdf | 7 | 43 | 339.9 | 0.75 | 1.0 | 0.5714 | 7/7 |
| doc2.pdf | 10 | 58 | 371.5 | 0.7533 | 1.0 | 0.6 | 10/10 |
| doc26.pdf | 9 | 73 | 82.1 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| doc50.pdf | 9 | 46 | 162.2 | 0.7778 | 0.7778 | 0.7778 | 7/9 |
| doc8.pdf | 10 | 77 | 364.0 | 0.8 | 1.0 | 0.7 | 10/10 |
| doc46.pdf | 8 | 136 | 150.3 | 0.8125 | 0.875 | 0.75 | 7/8 |
| doc23.pdf | 9 | 9 | 288.1 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| doc28.pdf | 6 | 38 | 120.1 | 0.8333 | 0.8333 | 0.8333 | 5/6 |
| doc48.pdf | 9 | 176 | 177.8 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| doc21.pdf | 9 | 58 | 371.5 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| doc4.pdf | 8 | 114 | 511.6 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc40.pdf | 5 | 47 | 361.1 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc11.pdf | 1 | 16 | 173.9 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc12.pdf | 1 | 92 | 168.9 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc13.pdf | 5 | 65 | 233.0 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc14.pdf | 1 | 149 | 205.4 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 7 | 280.0 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 20 | 86.2 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc27.pdf | 2 | 91 | 100.2 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc43.pdf | 3 | 35 | 123.3 | 1.0 | 1.0 | 1.0 | 3/3 |
