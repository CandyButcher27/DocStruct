# DocStruct retrieval baseline report

_Generated 2026-07-23 11:53 UTC_

## Setup

- **Documents:** 55 born-digital PDFs
- **Questions:** 322 LLM-generated (model `llama-3.3-70b-versatile + gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: docstruct, docstruct_geo, pymupdf4llm, unstructured, langchain, docling.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | docstruct **(ours)** | **0.7254** | [0.6848, 0.7659] | 0.7505 | 0.8634 | 0.6211 | 0.5955 | +0.1299 | 3622 | 335.2 | 2296.9 | 0.3158 | 31.25 | 0 |
| 2 | docstruct_geo | **0.7161** | [0.6762, 0.7563] | 0.745 | 0.882 | 0.6025 | 0.5589 | +0.1572 | 2336 | 336.0 | 2488.4 | 0.2878 | 774.48 | 0 |
| 3 | pymupdf4llm | **0.6752** | [0.6294, 0.7205] | 0.6972 | 0.8137 | 0.5932 | 0.5619 | +0.1133 | 1651 | 447.3 | 2536.4 | 0.2662 | 1846.73 | 0 |
| 4 | unstructured | **0.6329** | [0.5855, 0.679] | 0.6589 | 0.7702 | 0.5466 | 0.4746 | +0.1583 | 8961 | 85.0 | 543.9 | 1.1636 | 1221.07 | 0 |
| 5 | langchain | **0.6295** | [0.5848, 0.6747] | 0.6663 | 0.7919 | 0.5217 | 0.5048 | +0.1247 | 6120 | 100.8 | 521.1 | 1.208 | 711.62 | 0 |
| 6 | docling | **0.5592** | [0.5094, 0.6097] | 0.5777 | 0.6584 | 0.4907 | 0.4568 | +0.1024 | 1912 | 114.8 | 680.7 | 0.8215 | 4321.8 | 0 |

## Is the gap real? Paired bootstrap vs `docstruct`

Every tool answers the **same** questions, so the comparison is paired: one resample of question indices is applied to both tools, which cancels the between-question variance and isolates the difference between the chunkers. Positive Δ means `docstruct` is ahead. 10,000 resamples, seeded.

Reading two overlapping marginal CIs as "not significant" is the standard way to miss a consistent per-question difference — that is what this table exists to prevent, in both directions.

| vs | Metric | Δ (docstruct − tool) | 95% CI of Δ | p | n paired | Verdict |
|---|---|---|---|---|---|---|
| docstruct_geo | MRR | +0.0092 | [-0.0294, 0.0475] | 0.6393 | 322 | not significant |
| docstruct_geo | NDCG | +0.0054 | [-0.0254, 0.0361] | 0.7378 | 322 | not significant |
| docstruct_geo | RECALL | -0.0186 | [-0.0528, 0.0124] | 0.3013 | 322 | not significant |
| docstruct_geo | HIT1 | +0.0186 | [-0.0404, 0.0776] | 0.5633 | 322 | not significant |
| pymupdf4llm | MRR | +0.0502 | [0.0122, 0.0894] | 0.0115 | 322 | **significant** |
| pymupdf4llm | NDCG | +0.0532 | [0.0201, 0.0881] | 0.0018 | 322 | **significant** |
| pymupdf4llm | RECALL | +0.0497 | [0.0124, 0.087] | 0.0137 | 322 | **significant** |
| pymupdf4llm | HIT1 | +0.028 | [-0.0248, 0.0807] | 0.3228 | 322 | not significant |
| unstructured | MRR | +0.0925 | [0.0493, 0.1377] | 0.0001 | 322 | **significant** |
| unstructured | NDCG | +0.0916 | [0.0521, 0.1325] | 0.0001 | 322 | **significant** |
| unstructured | RECALL | +0.0932 | [0.0466, 0.1398] | 0.0001 | 322 | **significant** |
| unstructured | HIT1 | +0.0745 | [0.0155, 0.1335] | 0.0152 | 322 | **significant** |
| langchain | MRR | +0.0959 | [0.0514, 0.1402] | 0.0001 | 322 | **significant** |
| langchain | NDCG | +0.0842 | [0.045, 0.1225] | 0.0001 | 322 | **significant** |
| langchain | RECALL | +0.0714 | [0.0311, 0.1118] | 0.0005 | 322 | **significant** |
| langchain | HIT1 | +0.0994 | [0.0373, 0.1615] | 0.0023 | 322 | **significant** |
| docling | MRR | +0.1662 | [0.1154, 0.2171] | 0.0001 | 322 | **significant** |
| docling | NDCG | +0.1728 | [0.1249, 0.2208] | 0.0001 | 322 | **significant** |
| docling | RECALL | +0.205 | [0.1522, 0.2578] | 0.0001 | 322 | **significant** |
| docling | HIT1 | +0.1304 | [0.0683, 0.1925] | 0.0001 | 322 | **significant** |

## How to read this

- **MRR** — average reciprocal rank of the first chunk that contains the answer (higher = answers surface earlier).
- **MRR 95% CI** — percentile bootstrap over the per-question scores (10,000 resamples, seeded). It is the uncertainty on *this tool's* number in isolation; for comparing two tools use the paired table above, not CI overlap.
- **Recall@k** — fraction of questions where the answer appears in the top-k.
- **Hit@1** — fraction where the very first chunk already contains the answer.
- **Avg words/chunk** — chunk granularity; huge chunks can inflate recall while hurting precision/precision-of-context.
- **Context words** — words actually handed to the generator per query (summed over the top-5 retrieved chunks). MRR can always be bought by making chunks bigger; this is what that costs downstream in context window, latency and token spend.
- **MRR/1k words** — MRR per 1000 words of retrieved context. Rewards finding the answer *and* being cheap to feed to an LLM, which raw MRR alone does not.

## Run configuration

Chunking settings active for this run (full snapshot in the JSON sidecar under `meta.config`):

| Setting | Value |
|---|---|
| `MAX_CHUNK_TOKENS` | `500` |
| `MIN_CHUNK_TOKENS` | `200` |
| `CHUNK_OVERLAP_TOKENS` | `75` |
| `OVERLAP_ON_BOUNDARY` | `False` |
| `BREAK_TEXT_ON_TABLE` | `False` |
| `BREAK_TEXT_ON_CAPTION` | `False` |
| `INLINE_HEADER_TEXT` | `True` |
| `HEADER_LEVELS` | `3` |

## DocStruct's unique axis (not in any competitor)

Every chunk carries a section path, enabling **filtered retrieval** (`where={"h1": "4. Experiments"}`). No competitor here exposes section-hierarchy metadata, so this capability is qualitative, not in the table.

## Honest caveats

- Questions are **LLM-generated**: this measures retrieval consistency and the DocStruct-vs-tools delta well, but is weaker than human-judged relevance as an absolute claim.
- Containment relevance can miss paraphrased answers — but it is applied **identically** to every tool, so the ranking stays fair.
- Dataset is arXiv-heavy (born-digital prose); broader domains (legal/financial/manuals) are future work.
- **Chunk s is not a fair speed comparison when `--cache-dir` is set.** Only the DocStruct adapter uses that cache (detector proposals and populated blocks, keyed by PDF hash), so on a warm cache its column reports cache-hit time while every other tool is measured cold. Compare wall-clock only from a run with no `--cache-dir`, or against `meta.docstruct_cold_chunk_seconds` if present.
- **MRR/1k words is a tradeoff axis, not a ranking.** It necessarily favours tools that emit very small chunks and therefore retrieve very little text, regardless of whether they rank well. Read it next to MRR, not instead of it.
- This is a **signal/baseline**, not the Phase-2 public benchmark (50 PDFs, 200 human-checked Q&A).


### Per-doc breakdown: docstruct (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc56.pdf | 8 | 200 | 139.3 | 0.0625 | 0.125 | 0.0 | 1/8 |
| doc55.pdf | 3 | 41 | 307.5 | 0.1111 | 0.3333 | 0.0 | 1/3 |
| doc44.pdf | 8 | 34 | 394.3 | 0.25 | 0.25 | 0.25 | 2/8 |
| doc17.pdf | 8 | 70 | 276.9 | 0.3375 | 0.625 | 0.125 | 5/8 |
| doc41.pdf | 9 | 87 | 422.0 | 0.4722 | 0.6667 | 0.3333 | 6/9 |
| doc10.pdf | 9 | 61 | 391.7 | 0.5 | 0.6667 | 0.3333 | 6/9 |
| doc30.pdf | 1 | 200 | 139.3 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc9.pdf | 9 | 48 | 317.4 | 0.55 | 0.7778 | 0.4444 | 7/9 |
| doc5.pdf | 7 | 104 | 385.4 | 0.5714 | 0.8571 | 0.2857 | 6/7 |
| doc52.pdf | 4 | 56 | 236.8 | 0.625 | 0.75 | 0.5 | 3/4 |
| doc45.pdf | 8 | 111 | 367.3 | 0.65 | 0.875 | 0.5 | 7/8 |
| doc19.pdf | 2 | 21 | 366.4 | 0.6667 | 1.0 | 0.5 | 2/2 |
| doc24.pdf | 3 | 67 | 215.7 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc32.pdf | 9 | 47 | 314.6 | 0.6667 | 0.7778 | 0.5556 | 7/9 |
| doc33.pdf | 3 | 74 | 346.6 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc43.pdf | 3 | 31 | 335.5 | 0.6667 | 1.0 | 0.3333 | 3/3 |
| doc7.pdf | 6 | 105 | 452.6 | 0.6667 | 0.8333 | 0.5 | 5/6 |
| doc34.pdf | 5 | 164 | 158.5 | 0.7 | 1.0 | 0.4 | 5/5 |
| doc20.pdf | 9 | 82 | 277.3 | 0.7037 | 1.0 | 0.4444 | 9/9 |
| doc2.pdf | 10 | 54 | 450.4 | 0.7167 | 0.9 | 0.6 | 9/10 |
| doc4.pdf | 8 | 132 | 524.7 | 0.7188 | 0.875 | 0.625 | 7/8 |
| doc1.pdf | 6 | 31 | 244.2 | 0.7222 | 1.0 | 0.5 | 6/6 |
| doc22.pdf | 8 | 34 | 386.5 | 0.7292 | 0.875 | 0.625 | 7/8 |
| doc50.pdf | 9 | 32 | 294.0 | 0.7315 | 0.8889 | 0.6667 | 8/9 |
| doc29.pdf | 10 | 41 | 307.5 | 0.7333 | 0.9 | 0.6 | 9/10 |
| doc25.pdf | 9 | 118 | 230.7 | 0.7444 | 0.8889 | 0.6667 | 8/9 |
| doc54.pdf | 2 | 28 | 355.3 | 0.75 | 1.0 | 0.5 | 2/2 |
| doc6.pdf | 9 | 14 | 602.4 | 0.7815 | 1.0 | 0.6667 | 9/9 |
| doc39.pdf | 7 | 40 | 403.1 | 0.7857 | 1.0 | 0.5714 | 7/7 |
| doc49.pdf | 5 | 10 | 418.1 | 0.8 | 1.0 | 0.6 | 5/5 |
| doc42.pdf | 10 | 19 | 387.4 | 0.82 | 1.0 | 0.7 | 10/10 |
| doc21.pdf | 9 | 54 | 450.4 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc48.pdf | 9 | 87 | 452.7 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc26.pdf | 9 | 56 | 236.8 | 0.8426 | 1.0 | 0.7778 | 9/9 |
| doc16.pdf | 7 | 11 | 398.2 | 0.8571 | 0.8571 | 0.8571 | 6/7 |
| doc47.pdf | 6 | 52 | 396.2 | 0.8667 | 1.0 | 0.8333 | 6/6 |
| doc46.pdf | 8 | 90 | 342.2 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc51.pdf | 4 | 118 | 230.7 | 0.875 | 1.0 | 0.75 | 4/4 |
| doc23.pdf | 9 | 9 | 373.8 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| doc35.pdf | 7 | 64 | 299.2 | 0.8929 | 1.0 | 0.8571 | 7/7 |
| doc40.pdf | 5 | 38 | 487.3 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc11.pdf | 1 | 12 | 331.1 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc12.pdf | 1 | 60 | 369.6 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc13.pdf | 5 | 51 | 382.4 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc14.pdf | 1 | 116 | 356.0 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 7 | 478.0 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 17 | 369.2 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc27.pdf | 2 | 66 | 276.8 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc28.pdf | 6 | 28 | 355.3 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc3.pdf | 3 | 75 | 393.2 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc37.pdf | 1 | 118 | 492.9 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc38.pdf | 1 | 164 | 529.2 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc53.pdf | 2 | 66 | 276.8 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc57.pdf | 1 | 32 | 307.7 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc8.pdf | 10 | 75 | 480.0 | 1.0 | 1.0 | 1.0 | 10/10 |


### Per-doc breakdown: docstruct_geo (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc56.pdf ⚠ | 8 | 128 | 166.2 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc44.pdf | 8 | 19 | 423.4 | 0.15 | 0.625 | 0.0 | 5/8 |
| doc24.pdf | 3 | 53 | 234.6 | 0.1667 | 0.3333 | 0.0 | 1/3 |
| doc53.pdf | 2 | 44 | 239.2 | 0.1667 | 0.5 | 0.0 | 1/2 |
| doc55.pdf | 3 | 32 | 282.2 | 0.2778 | 0.6667 | 0.0 | 2/3 |
| doc19.pdf | 2 | 13 | 392.5 | 0.375 | 1.0 | 0.0 | 2/2 |
| doc27.pdf | 2 | 44 | 239.2 | 0.4167 | 1.0 | 0.0 | 2/2 |
| doc12.pdf | 1 | 27 | 519.4 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc30.pdf | 1 | 128 | 166.2 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc17.pdf | 8 | 59 | 226.6 | 0.5417 | 0.625 | 0.5 | 5/8 |
| doc32.pdf | 9 | 41 | 236.8 | 0.5778 | 0.7778 | 0.4444 | 7/9 |
| doc51.pdf | 4 | 55 | 315.6 | 0.5833 | 0.75 | 0.5 | 3/4 |
| doc9.pdf | 9 | 48 | 274.8 | 0.5926 | 0.7778 | 0.4444 | 7/9 |
| doc42.pdf | 10 | 15 | 280.6 | 0.6167 | 0.9 | 0.4 | 9/10 |
| doc1.pdf | 6 | 33 | 223.0 | 0.625 | 1.0 | 0.3333 | 6/6 |
| doc52.pdf | 4 | 56 | 210.5 | 0.625 | 0.75 | 0.5 | 3/4 |
| doc7.pdf | 6 | 58 | 494.0 | 0.6389 | 0.8333 | 0.5 | 5/6 |
| doc45.pdf | 8 | 55 | 427.6 | 0.6458 | 0.875 | 0.5 | 7/8 |
| doc41.pdf | 9 | 50 | 455.6 | 0.6667 | 0.7778 | 0.5556 | 7/9 |
| doc50.pdf | 9 | 21 | 372.6 | 0.6852 | 0.8889 | 0.5556 | 8/9 |
| doc25.pdf | 9 | 55 | 315.6 | 0.7037 | 0.7778 | 0.6667 | 7/9 |
| doc47.pdf | 6 | 34 | 415.9 | 0.7222 | 0.8333 | 0.6667 | 5/6 |
| doc5.pdf | 7 | 59 | 394.7 | 0.7262 | 1.0 | 0.5714 | 7/7 |
| doc29.pdf | 10 | 32 | 282.2 | 0.7333 | 0.8 | 0.7 | 8/10 |
| doc2.pdf | 10 | 37 | 499.7 | 0.75 | 1.0 | 0.6 | 10/10 |
| doc10.pdf | 9 | 39 | 399.6 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| doc26.pdf | 9 | 56 | 210.5 | 0.7593 | 1.0 | 0.5556 | 9/9 |
| doc4.pdf | 8 | 74 | 504.2 | 0.7604 | 1.0 | 0.625 | 8/8 |
| doc49.pdf | 5 | 7 | 436.1 | 0.7667 | 1.0 | 0.6 | 5/5 |
| doc3.pdf | 3 | 38 | 463.6 | 0.7778 | 1.0 | 0.6667 | 3/3 |
| doc48.pdf | 9 | 57 | 416.1 | 0.7778 | 1.0 | 0.6667 | 9/9 |
| doc16.pdf | 7 | 12 | 378.4 | 0.7857 | 0.8571 | 0.7143 | 6/7 |
| doc39.pdf | 7 | 25 | 368.6 | 0.7857 | 1.0 | 0.5714 | 7/7 |
| doc23.pdf | 9 | 6 | 346.8 | 0.8056 | 1.0 | 0.6667 | 9/9 |
| doc22.pdf | 8 | 27 | 320.5 | 0.8125 | 0.875 | 0.75 | 7/8 |
| doc6.pdf | 9 | 8 | 641.9 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc35.pdf | 7 | 40 | 290.2 | 0.8214 | 1.0 | 0.7143 | 7/7 |
| doc33.pdf | 3 | 44 | 430.5 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc43.pdf | 3 | 12 | 458.0 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc46.pdf | 8 | 70 | 316.2 | 0.8542 | 1.0 | 0.75 | 8/8 |
| doc20.pdf | 9 | 61 | 253.4 | 0.8556 | 1.0 | 0.7778 | 9/9 |
| doc40.pdf | 5 | 23 | 460.8 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc8.pdf | 10 | 57 | 426.1 | 0.9 | 1.0 | 0.8 | 10/10 |
| doc11.pdf | 1 | 4 | 470.2 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc13.pdf | 5 | 21 | 494.9 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc14.pdf | 1 | 61 | 501.1 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 3 | 558.7 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 9 | 436.8 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc21.pdf | 9 | 37 | 499.7 | 1.0 | 1.0 | 1.0 | 9/9 |
| doc28.pdf | 6 | 16 | 331.9 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc34.pdf | 5 | 132 | 156.9 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc37.pdf | 1 | 75 | 456.3 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc38.pdf | 1 | 90 | 501.3 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc54.pdf | 2 | 16 | 331.9 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc57.pdf | 1 | 20 | 283.2 | 1.0 | 1.0 | 1.0 | 1/1 |


### Per-doc breakdown: pymupdf4llm (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc30.pdf ⚠ | 1 | 49 | 189.3 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc56.pdf ⚠ | 8 | 49 | 189.3 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc44.pdf | 8 | 16 | 522.6 | 0.025 | 0.125 | 0.0 | 1/8 |
| doc43.pdf | 3 | 9 | 642.6 | 0.1944 | 0.6667 | 0.0 | 2/3 |
| doc55.pdf | 3 | 26 | 382.8 | 0.1944 | 0.6667 | 0.0 | 2/3 |
| doc19.pdf | 2 | 8 | 639.8 | 0.2917 | 1.0 | 0.0 | 2/2 |
| doc17.pdf | 8 | 22 | 604.0 | 0.375 | 0.5 | 0.25 | 4/8 |
| doc41.pdf | 9 | 44 | 574.0 | 0.4444 | 0.4444 | 0.4444 | 4/9 |
| doc32.pdf | 9 | 21 | 500.5 | 0.4537 | 0.6667 | 0.3333 | 6/9 |
| doc48.pdf | 9 | 40 | 391.9 | 0.5 | 0.5556 | 0.4444 | 5/9 |
| doc49.pdf | 5 | 10 | 246.1 | 0.5 | 0.6 | 0.4 | 3/5 |
| doc52.pdf | 4 | 34 | 378.6 | 0.5 | 0.5 | 0.5 | 2/4 |
| doc22.pdf | 8 | 16 | 559.1 | 0.5312 | 0.625 | 0.5 | 5/8 |
| doc4.pdf | 8 | 78 | 568.2 | 0.5312 | 0.625 | 0.5 | 5/8 |
| doc42.pdf | 10 | 12 | 472.8 | 0.5483 | 0.9 | 0.4 | 9/10 |
| doc10.pdf | 9 | 39 | 449.5 | 0.5593 | 0.8889 | 0.3333 | 8/9 |
| doc9.pdf | 9 | 21 | 329.6 | 0.5648 | 0.8889 | 0.3333 | 8/9 |
| doc24.pdf | 3 | 12 | 824.1 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| doc47.pdf | 6 | 46 | 291.1 | 0.6167 | 0.8333 | 0.5 | 5/6 |
| doc7.pdf | 6 | 65 | 456.0 | 0.625 | 0.8333 | 0.5 | 5/6 |
| doc1.pdf | 6 | 22 | 318.1 | 0.6306 | 1.0 | 0.5 | 6/6 |
| doc50.pdf | 9 | 23 | 338.0 | 0.6481 | 0.7778 | 0.5556 | 7/9 |
| doc45.pdf | 8 | 90 | 215.1 | 0.7229 | 1.0 | 0.625 | 8/8 |
| doc33.pdf | 3 | 31 | 571.0 | 0.7333 | 1.0 | 0.6667 | 3/3 |
| doc5.pdf | 7 | 39 | 476.3 | 0.7429 | 0.8571 | 0.7143 | 6/7 |
| doc34.pdf | 5 | 38 | 503.2 | 0.75 | 1.0 | 0.6 | 5/5 |
| doc51.pdf | 4 | 39 | 481.1 | 0.75 | 1.0 | 0.5 | 4/4 |
| doc23.pdf | 9 | 6 | 368.0 | 0.7593 | 1.0 | 0.5556 | 9/9 |
| doc6.pdf | 9 | 8 | 714.9 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| doc29.pdf | 10 | 26 | 382.8 | 0.7833 | 0.9 | 0.7 | 9/10 |
| doc25.pdf | 9 | 39 | 481.1 | 0.8056 | 0.8889 | 0.7778 | 8/9 |
| doc3.pdf | 3 | 27 | 533.8 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc46.pdf | 8 | 56 | 248.3 | 0.8438 | 1.0 | 0.75 | 8/8 |
| doc2.pdf | 10 | 43 | 419.7 | 0.85 | 1.0 | 0.7 | 10/10 |
| doc8.pdf | 10 | 48 | 623.4 | 0.85 | 0.9 | 0.8 | 9/10 |
| doc16.pdf | 7 | 10 | 431.0 | 0.8571 | 0.8571 | 0.8571 | 6/7 |
| doc35.pdf | 7 | 26 | 467.3 | 0.8857 | 1.0 | 0.8571 | 7/7 |
| doc20.pdf | 9 | 31 | 460.3 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| doc21.pdf | 9 | 43 | 419.7 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| doc26.pdf | 9 | 34 | 378.6 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| doc39.pdf | 7 | 26 | 343.2 | 0.8929 | 1.0 | 0.8571 | 7/7 |
| doc40.pdf | 5 | 28 | 464.9 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc28.pdf | 6 | 11 | 510.5 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc11.pdf | 1 | 4 | 561.2 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc12.pdf | 1 | 15 | 885.9 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc13.pdf | 5 | 24 | 516.3 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc14.pdf | 1 | 32 | 769.1 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 3 | 600.0 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 9 | 517.1 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc27.pdf | 2 | 19 | 583.5 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc37.pdf | 1 | 51 | 679.6 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc38.pdf | 1 | 90 | 401.4 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc53.pdf | 2 | 19 | 583.5 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc54.pdf | 2 | 11 | 510.5 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc57.pdf | 1 | 13 | 483.6 | 1.0 | 1.0 | 1.0 | 1/1 |


### Per-doc breakdown: unstructured (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc44.pdf ⚠ | 8 | 95 | 85.7 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc56.pdf ⚠ | 8 | 260 | 77.2 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc55.pdf | 3 | 131 | 82.0 | 0.1667 | 0.3333 | 0.0 | 1/3 |
| doc17.pdf | 8 | 149 | 82.9 | 0.2292 | 0.375 | 0.125 | 3/8 |
| doc41.pdf | 9 | 258 | 93.2 | 0.287 | 0.5556 | 0.1111 | 5/9 |
| doc30.pdf | 1 | 260 | 77.2 | 0.3333 | 1.0 | 0.0 | 1/1 |
| doc45.pdf | 8 | 293 | 65.5 | 0.4313 | 0.75 | 0.25 | 6/8 |
| doc9.pdf | 9 | 106 | 76.0 | 0.4481 | 0.7778 | 0.3333 | 7/9 |
| doc47.pdf | 6 | 164 | 82.7 | 0.4583 | 0.6667 | 0.3333 | 4/6 |
| doc25.pdf | 9 | 217 | 88.8 | 0.4944 | 0.6667 | 0.4444 | 6/9 |
| doc19.pdf | 2 | 43 | 117.5 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc37.pdf | 1 | 359 | 79.7 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc38.pdf | 1 | 451 | 88.3 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc53.pdf | 2 | 131 | 86.9 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc33.pdf | 3 | 203 | 79.7 | 0.5278 | 1.0 | 0.3333 | 3/3 |
| doc7.pdf | 6 | 306 | 95.0 | 0.5417 | 0.6667 | 0.5 | 4/6 |
| doc20.pdf | 9 | 170 | 79.7 | 0.5463 | 0.7778 | 0.4444 | 7/9 |
| doc16.pdf | 7 | 43 | 96.6 | 0.5476 | 0.7143 | 0.4286 | 5/7 |
| doc13.pdf | 5 | 140 | 91.1 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc34.pdf | 5 | 196 | 71.3 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc39.pdf | 7 | 109 | 81.6 | 0.5714 | 0.5714 | 0.5714 | 4/7 |
| doc49.pdf | 5 | 32 | 98.2 | 0.6 | 0.8 | 0.4 | 4/5 |
| doc10.pdf | 9 | 254 | 74.1 | 0.6111 | 0.6667 | 0.5556 | 6/9 |
| doc52.pdf | 4 | 159 | 73.9 | 0.625 | 0.75 | 0.5 | 3/4 |
| doc32.pdf | 9 | 126 | 81.8 | 0.6389 | 0.7778 | 0.5556 | 7/9 |
| doc1.pdf | 6 | 82 | 86.4 | 0.6667 | 0.8333 | 0.5 | 5/6 |
| doc3.pdf | 3 | 188 | 80.4 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc40.pdf | 5 | 153 | 83.6 | 0.6667 | 1.0 | 0.4 | 5/5 |
| doc54.pdf | 2 | 63 | 93.4 | 0.6667 | 1.0 | 0.5 | 2/2 |
| doc6.pdf | 9 | 57 | 99.9 | 0.6667 | 0.6667 | 0.6667 | 6/9 |
| doc35.pdf | 7 | 145 | 82.1 | 0.6786 | 0.8571 | 0.5714 | 6/7 |
| doc42.pdf | 10 | 70 | 80.5 | 0.6833 | 0.8 | 0.6 | 8/10 |
| doc2.pdf | 10 | 182 | 97.7 | 0.7033 | 0.9 | 0.6 | 9/10 |
| doc24.pdf | 3 | 105 | 99.9 | 0.7333 | 1.0 | 0.6667 | 3/3 |
| doc29.pdf | 10 | 131 | 82.0 | 0.7333 | 0.8 | 0.7 | 8/10 |
| doc46.pdf | 8 | 277 | 67.2 | 0.75 | 1.0 | 0.5 | 8/8 |
| doc50.pdf | 9 | 94 | 81.5 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| doc23.pdf | 9 | 25 | 91.8 | 0.8 | 0.8889 | 0.7778 | 8/9 |
| doc4.pdf | 8 | 424 | 100.4 | 0.8125 | 0.875 | 0.75 | 7/8 |
| doc51.pdf | 4 | 217 | 88.8 | 0.8125 | 1.0 | 0.75 | 4/4 |
| doc26.pdf | 9 | 159 | 73.9 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc43.pdf | 3 | 60 | 88.2 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc48.pdf | 9 | 206 | 87.8 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc8.pdf | 10 | 320 | 91.2 | 0.85 | 0.9 | 0.8 | 9/10 |
| doc21.pdf | 9 | 182 | 97.7 | 0.8519 | 1.0 | 0.7778 | 9/9 |
| doc22.pdf | 8 | 122 | 75.6 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc28.pdf | 6 | 63 | 93.4 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| doc5.pdf | 7 | 229 | 85.5 | 0.8929 | 1.0 | 0.8571 | 7/7 |
| doc11.pdf | 1 | 28 | 84.5 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc12.pdf | 1 | 139 | 96.1 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc14.pdf | 1 | 323 | 89.0 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 18 | 103.5 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 48 | 92.1 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc27.pdf | 2 | 131 | 86.9 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc57.pdf | 1 | 65 | 96.0 | 1.0 | 1.0 | 1.0 | 1/1 |


### Per-doc breakdown: langchain (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc14.pdf ⚠ | 1 | 237 | 107.8 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc53.pdf ⚠ | 2 | 86 | 98.0 | 0.0 | 0.0 | 0.0 | 0/2 |
| doc56.pdf ⚠ | 8 | 165 | 107.4 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc24.pdf | 3 | 75 | 70.2 | 0.0667 | 0.3333 | 0.0 | 1/3 |
| doc44.pdf | 8 | 62 | 54.2 | 0.2333 | 0.625 | 0.0 | 5/8 |
| doc47.pdf | 6 | 112 | 132.7 | 0.2639 | 0.6667 | 0.0 | 4/6 |
| doc9.pdf | 9 | 67 | 122.5 | 0.287 | 0.5556 | 0.1111 | 5/9 |
| doc55.pdf | 3 | 89 | 57.7 | 0.3333 | 0.3333 | 0.3333 | 1/3 |
| doc3.pdf | 3 | 108 | 75.1 | 0.4 | 0.6667 | 0.3333 | 2/3 |
| doc17.pdf | 8 | 86 | 81.6 | 0.4167 | 0.625 | 0.25 | 5/8 |
| doc25.pdf | 9 | 161 | 128.3 | 0.4389 | 0.6667 | 0.3333 | 6/9 |
| doc7.pdf | 6 | 234 | 117.6 | 0.4444 | 0.6667 | 0.3333 | 4/6 |
| doc49.pdf | 5 | 23 | 147.9 | 0.4667 | 0.8 | 0.2 | 4/5 |
| doc20.pdf | 9 | 111 | 136.6 | 0.4815 | 0.6667 | 0.3333 | 6/9 |
| doc11.pdf | 1 | 18 | 104.2 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc19.pdf | 2 | 36 | 54.2 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc38.pdf | 1 | 278 | 117.0 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc22.pdf | 8 | 70 | 60.5 | 0.5417 | 0.625 | 0.5 | 5/8 |
| doc52.pdf | 4 | 112 | 66.5 | 0.55 | 0.75 | 0.5 | 3/4 |
| doc5.pdf | 7 | 135 | 159.1 | 0.5833 | 1.0 | 0.2857 | 7/7 |
| doc48.pdf | 9 | 137 | 145.6 | 0.5926 | 0.7778 | 0.4444 | 7/9 |
| doc41.pdf | 9 | 191 | 61.0 | 0.6111 | 0.6667 | 0.5556 | 6/9 |
| doc45.pdf | 8 | 169 | 94.2 | 0.6292 | 0.875 | 0.5 | 7/8 |
| doc34.pdf | 5 | 129 | 67.7 | 0.6333 | 1.0 | 0.4 | 5/5 |
| doc10.pdf | 9 | 155 | 89.7 | 0.6519 | 0.8889 | 0.5556 | 8/9 |
| doc43.pdf | 3 | 44 | 60.6 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc29.pdf | 10 | 89 | 57.7 | 0.6833 | 0.8 | 0.6 | 8/10 |
| doc42.pdf | 10 | 47 | 33.3 | 0.6833 | 0.8 | 0.6 | 8/10 |
| doc16.pdf | 7 | 35 | 125.8 | 0.6905 | 0.8571 | 0.5714 | 6/7 |
| doc23.pdf | 9 | 19 | 107.8 | 0.6944 | 0.8889 | 0.5556 | 8/9 |
| doc46.pdf | 8 | 153 | 110.8 | 0.6979 | 0.875 | 0.625 | 7/8 |
| doc50.pdf | 9 | 70 | 99.5 | 0.7037 | 0.8889 | 0.5556 | 8/9 |
| doc2.pdf | 10 | 142 | 129.0 | 0.72 | 0.9 | 0.6 | 9/10 |
| doc32.pdf | 9 | 81 | 55.5 | 0.7222 | 0.7778 | 0.6667 | 7/9 |
| doc4.pdf | 8 | 340 | 127.1 | 0.7292 | 0.875 | 0.625 | 7/8 |
| doc35.pdf | 7 | 99 | 122.1 | 0.75 | 0.8571 | 0.7143 | 6/7 |
| doc51.pdf | 4 | 161 | 128.3 | 0.75 | 0.75 | 0.75 | 3/4 |
| doc39.pdf | 7 | 80 | 122.9 | 0.7619 | 1.0 | 0.5714 | 7/7 |
| doc26.pdf | 9 | 112 | 66.5 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| doc33.pdf | 3 | 127 | 79.0 | 0.7778 | 1.0 | 0.6667 | 3/3 |
| doc40.pdf | 5 | 115 | 119.8 | 0.8 | 1.0 | 0.6 | 5/5 |
| doc8.pdf | 10 | 238 | 114.3 | 0.8 | 0.9 | 0.7 | 9/10 |
| doc21.pdf | 9 | 142 | 129.0 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc1.pdf | 6 | 57 | 129.1 | 0.8333 | 0.8333 | 0.8333 | 5/6 |
| doc13.pdf | 5 | 103 | 111.6 | 0.8667 | 1.0 | 0.8 | 5/5 |
| doc6.pdf | 9 | 42 | 25.5 | 0.9259 | 1.0 | 0.8889 | 9/9 |
| doc12.pdf | 1 | 106 | 104.8 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 13 | 91.5 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 31 | 49.8 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc27.pdf | 2 | 86 | 98.0 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc28.pdf | 6 | 47 | 71.3 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc30.pdf | 1 | 165 | 107.4 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc37.pdf | 1 | 230 | 55.8 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc54.pdf | 2 | 47 | 71.3 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc57.pdf | 1 | 53 | 40.9 | 1.0 | 1.0 | 1.0 | 1/1 |


### Per-doc breakdown: docling (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc37.pdf ⚠ | 1 | 42 | 127.2 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc44.pdf ⚠ | 8 | 48 | 101.3 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc55.pdf ⚠ | 3 | 21 | 108.0 | 0.0 | 0.0 | 0.0 | 0/3 |
| doc56.pdf ⚠ | 8 | 43 | 110.1 | 0.0 | 0.0 | 0.0 | 0/8 |
| doc53.pdf | 2 | 60 | 85.9 | 0.125 | 0.5 | 0.0 | 1/2 |
| doc50.pdf | 9 | 20 | 127.0 | 0.1481 | 0.2222 | 0.1111 | 2/9 |
| doc17.pdf | 8 | 34 | 130.1 | 0.1875 | 0.25 | 0.125 | 2/8 |
| doc30.pdf | 1 | 43 | 110.1 | 0.2 | 1.0 | 0.0 | 1/1 |
| doc41.pdf | 9 | 37 | 125.8 | 0.2222 | 0.2222 | 0.2222 | 2/9 |
| doc4.pdf | 8 | 31 | 152.7 | 0.25 | 0.25 | 0.25 | 2/8 |
| doc7.pdf | 6 | 20 | 121.8 | 0.3056 | 0.5 | 0.1667 | 3/6 |
| doc3.pdf | 3 | 39 | 127.0 | 0.3333 | 0.3333 | 0.3333 | 1/3 |
| doc9.pdf | 9 | 20 | 123.8 | 0.3519 | 0.5556 | 0.2222 | 5/9 |
| doc51.pdf | 4 | 26 | 135.7 | 0.375 | 0.5 | 0.25 | 2/4 |
| doc49.pdf | 5 | 17 | 105.1 | 0.4667 | 0.6 | 0.4 | 3/5 |
| doc2.pdf | 10 | 27 | 126.9 | 0.4833 | 0.6 | 0.4 | 6/10 |
| doc19.pdf | 2 | 40 | 108.0 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc27.pdf | 2 | 78 | 85.8 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc38.pdf | 1 | 35 | 132.5 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc52.pdf | 4 | 31 | 103.0 | 0.5 | 0.5 | 0.5 | 2/4 |
| doc25.pdf | 9 | 31 | 125.8 | 0.537 | 0.6667 | 0.4444 | 6/9 |
| doc29.pdf | 10 | 25 | 112.2 | 0.545 | 0.7 | 0.5 | 7/10 |
| doc39.pdf | 7 | 19 | 118.4 | 0.5476 | 0.7143 | 0.4286 | 5/7 |
| doc32.pdf | 9 | 54 | 101.0 | 0.5556 | 0.5556 | 0.5556 | 5/9 |
| doc21.pdf | 9 | 27 | 126.9 | 0.5926 | 0.6667 | 0.5556 | 6/9 |
| doc13.pdf | 5 | 53 | 108.0 | 0.6 | 0.6 | 0.6 | 3/5 |
| doc45.pdf | 8 | 16 | 117.1 | 0.6042 | 0.75 | 0.5 | 6/8 |
| doc26.pdf | 9 | 31 | 103.0 | 0.6111 | 0.6667 | 0.5556 | 6/9 |
| doc22.pdf | 8 | 46 | 123.8 | 0.625 | 0.75 | 0.5 | 6/8 |
| doc46.pdf | 8 | 21 | 122.2 | 0.625 | 0.75 | 0.5 | 6/8 |
| doc16.pdf | 7 | 30 | 120.2 | 0.6429 | 0.7143 | 0.5714 | 5/7 |
| doc24.pdf | 3 | 82 | 94.1 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc33.pdf | 3 | 63 | 116.1 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc6.pdf | 9 | 30 | 135.5 | 0.6667 | 0.6667 | 0.6667 | 6/9 |
| doc8.pdf | 10 | 25 | 140.6 | 0.67 | 0.8 | 0.6 | 8/10 |
| doc34.pdf | 5 | 29 | 123.1 | 0.7 | 1.0 | 0.4 | 5/5 |
| doc10.pdf | 9 | 29 | 111.3 | 0.7037 | 0.8889 | 0.5556 | 8/9 |
| doc1.pdf | 6 | 28 | 134.0 | 0.7083 | 1.0 | 0.5 | 6/6 |
| doc48.pdf | 9 | 30 | 129.2 | 0.7222 | 0.7778 | 0.6667 | 7/9 |
| doc20.pdf | 9 | 31 | 137.7 | 0.7259 | 0.8889 | 0.6667 | 8/9 |
| doc42.pdf | 10 | 24 | 131.3 | 0.7333 | 0.8 | 0.7 | 8/10 |
| doc5.pdf | 7 | 28 | 131.8 | 0.75 | 1.0 | 0.5714 | 7/7 |
| doc35.pdf | 7 | 42 | 106.0 | 0.7976 | 1.0 | 0.7143 | 7/7 |
| doc28.pdf | 6 | 48 | 95.8 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc43.pdf | 3 | 44 | 110.5 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc23.pdf | 9 | 17 | 130.3 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| doc47.pdf | 6 | 16 | 116.4 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| doc40.pdf | 5 | 26 | 131.6 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc11.pdf | 1 | 22 | 110.6 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc12.pdf | 1 | 53 | 118.7 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc14.pdf | 1 | 52 | 112.9 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc15.pdf | 5 | 16 | 111.3 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc18.pdf | 3 | 32 | 131.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc54.pdf | 2 | 48 | 95.8 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc57.pdf | 1 | 32 | 108.6 | 1.0 | 1.0 | 1.0 | 1/1 |
