# DocStruct retrieval baseline report

_Generated 2026-06-28 09:06 UTC_

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
| 1 | docstruct **(ours)** | **0.6845** | 0.7151 | 0.8389 | 0.5872 | 0.5132 | +0.1713 | 3679 | 183.9 | 1247.63 | 0 |

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
| doc44.pdf ⚠ | 8 | 33 | 122.0 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc41.pdf | 9 | 107 | 126.3 | 0.3889 | 0.4444 | 0.3333 | 4/9 |
| doc34.pdf | 5 | 215 | 53.3 | 0.39 | 0.8 | 0.2 | 4/5 |
| doc17.pdf | 8 | 75 | 111.8 | 0.4062 | 0.5 | 0.375 | 4/8 |
| doc24.pdf | 3 | 113 | 51.3 | 0.4167 | 0.6667 | 0.3333 | 2/3 |
| doc3.pdf | 3 | 79 | 157.2 | 0.4167 | 0.6667 | 0.3333 | 2/3 |
| doc1.pdf | 6 | 42 | 171.9 | 0.4722 | 0.8333 | 0.1667 | 5/6 |
| doc12.pdf | 1 | 88 | 172.2 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc19.pdf | 2 | 24 | 102.4 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc22.pdf | 8 | 43 | 115.4 | 0.5 | 0.5 | 0.5 | 4/8 |
| doc30.pdf | 1 | 210 | 44.2 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc33.pdf | 3 | 87 | 119.0 | 0.5 | 0.6667 | 0.3333 | 2/3 |
| doc38.pdf | 1 | 143 | 375.5 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc9.pdf | 9 | 60 | 183.1 | 0.5037 | 0.6667 | 0.4444 | 6/9 |
| doc10.pdf | 9 | 62 | 255.4 | 0.5093 | 0.6667 | 0.4444 | 6/9 |
| doc5.pdf | 7 | 140 | 235.5 | 0.5357 | 1.0 | 0.2857 | 7/7 |
| doc6.pdf | 9 | 8 | 151.0 | 0.5778 | 0.8889 | 0.3333 | 8/9 |
| doc49.pdf | 5 | 11 | 317.4 | 0.6 | 0.8 | 0.4 | 4/5 |
| doc47.pdf | 6 | 50 | 365.5 | 0.6167 | 0.8333 | 0.5 | 5/6 |
| doc45.pdf | 8 | 125 | 175.8 | 0.625 | 0.75 | 0.5 | 6/8 |
| doc25.pdf | 9 | 149 | 140.0 | 0.6667 | 0.8889 | 0.5556 | 8/9 |
| doc7.pdf | 6 | 125 | 261.9 | 0.6667 | 0.6667 | 0.6667 | 4/6 |
| doc35.pdf | 7 | 65 | 232.4 | 0.6714 | 0.8571 | 0.5714 | 6/7 |
| doc29.pdf | 10 | 43 | 106.9 | 0.675 | 0.8 | 0.6 | 8/10 |
| doc42.pdf | 10 | 25 | 70.0 | 0.74 | 1.0 | 0.6 | 10/10 |
| doc16.pdf | 7 | 16 | 244.4 | 0.75 | 1.0 | 0.5714 | 7/7 |
| doc20.pdf | 9 | 166 | 109.5 | 0.7722 | 1.0 | 0.6667 | 9/9 |
| doc2.pdf | 10 | 56 | 380.3 | 0.775 | 1.0 | 0.6 | 10/10 |
| doc28.pdf | 6 | 35 | 124.9 | 0.7917 | 1.0 | 0.6667 | 6/6 |
| doc32.pdf | 9 | 65 | 77.2 | 0.8 | 0.8889 | 0.7778 | 8/9 |
| doc50.pdf | 9 | 42 | 168.1 | 0.8 | 0.8889 | 0.7778 | 8/9 |
| doc39.pdf | 7 | 39 | 365.4 | 0.8143 | 1.0 | 0.7143 | 7/7 |
| doc23.pdf | 9 | 8 | 319.6 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc43.pdf | 3 | 34 | 125.4 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc26.pdf | 9 | 79 | 76.0 | 0.8556 | 1.0 | 0.7778 | 9/9 |
| doc40.pdf | 5 | 47 | 361.2 | 0.8667 | 1.0 | 0.8 | 5/5 |
| doc48.pdf | 9 | 176 | 177.8 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| doc4.pdf | 8 | 112 | 511.2 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc46.pdf | 8 | 133 | 151.2 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc8.pdf | 10 | 75 | 370.5 | 0.8833 | 1.0 | 0.8 | 10/10 |
| doc21.pdf | 9 | 56 | 380.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| doc11.pdf | 1 | 12 | 225.8 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc13.pdf | 5 | 54 | 268.8 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc14.pdf | 1 | 151 | 187.8 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 7 | 280.0 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 20 | 86.2 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc27.pdf | 2 | 89 | 98.3 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc37.pdf | 1 | 85 | 218.1 | 1.0 | 1.0 | 1.0 | 1/1 |
