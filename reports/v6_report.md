# DocStruct retrieval baseline report

_Generated 2026-07-23 18:21 UTC_

## Setup

- **Documents:** 92 born-digital PDFs
- **Questions:** 558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: docstruct, docstruct_geo, pymupdf4llm, langchain, unstructured.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | docstruct **(ours)** | **0.8203** | [0.7936, 0.8464] | 0.832 | 0.9427 | 0.7401 | 0.6482 | +0.1721 | 7115 | 339.0 | 2404.4 | 0.3412 | 6.58 | 0 |
| 2 | docstruct_geo | **0.776** | [0.7469, 0.8039] | 0.7988 | 0.9283 | 0.6756 | 0.5863 | +0.1897 | 4530 | 335.0 | 2569.6 | 0.302 | 445.04 | 0 |
| 3 | pymupdf4llm | **0.7646** | [0.7356, 0.7932] | 0.7897 | 0.9194 | 0.6577 | 0.6165 | +0.1481 | 3144 | 443.1 | 2661.5 | 0.2873 | 12327.83 | 0 |
| 4 | langchain | **0.7009** | [0.6685, 0.7338] | 0.7284 | 0.8477 | 0.5986 | 0.5558 | +0.1451 | 11645 | 106.3 | 505.4 | 1.3868 | 1319.54 | 0 |
| 5 | unstructured | **0.6948** | [0.662, 0.7274] | 0.7271 | 0.8561 | 0.592 | 0.5151 | +0.1797 | 16905 | 84.5 | 549.0 | 1.2656 | 1885.99 | 1 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| langchain | 1.0 | 1.1018 |
| unstructured | 0.8334 | 1.3763 |
| docstruct_geo **(ours)** | 0.8222 | 1.327 |
| docstruct **(ours)** | 0.8169 | 2.0603 |
| pymupdf4llm | 0.7675 | 1.3615 |

- **Coverage** — fraction of the document's word *instances* that appear in some chunk. This is where silent loss shows up and nowhere else: dropped table rows, headings that end up in no chunk, skipped figures. Counted as a multiset, so dropping every repeat of a term is not scored as covered.
- **Duplication** — chunk words divided by document words. Above 1.0 means content is emitted more than once, inflating the index and letting two chunks split the evidence for one query. Overlap raises it deliberately, so read it as a cost next to coverage, not as a defect.

## Is the gap real? Paired bootstrap vs `docstruct`

Every tool answers the **same** questions, so the comparison is paired: one resample of question indices is applied to both tools, which cancels the between-question variance and isolates the difference between the chunkers. Positive Δ means `docstruct` is ahead. 10,000 resamples, seeded.

Reading two overlapping marginal CIs as "not significant" is the standard way to miss a consistent per-question difference — that is what this table exists to prevent, in both directions.

| vs | Metric | Δ (docstruct − tool) | 95% CI of Δ | p | n paired | Verdict |
|---|---|---|---|---|---|---|
| docstruct_geo | MRR | +0.0443 | [0.0159, 0.0731] | 0.0026 | 558 | **significant** |
| docstruct_geo | NDCG | +0.0332 | [0.0104, 0.0567] | 0.0043 | 558 | **significant** |
| docstruct_geo | RECALL | +0.0143 | [-0.009, 0.0376] | 0.2689 | 558 | not significant |
| docstruct_geo | HIT1 | +0.0645 | [0.0233, 0.1057] | 0.0032 | 558 | **significant** |
| pymupdf4llm | MRR | +0.0556 | [0.0256, 0.0862] | 0.0001 | 558 | **significant** |
| pymupdf4llm | NDCG | +0.0423 | [0.0175, 0.067] | 0.0008 | 558 | **significant** |
| pymupdf4llm | RECALL | +0.0233 | [-0.0018, 0.0484] | 0.0863 | 558 | not significant |
| pymupdf4llm | HIT1 | +0.0824 | [0.0394, 0.1272] | 0.0001 | 558 | **significant** |
| langchain | MRR | +0.1193 | [0.0797, 0.1578] | 0.0001 | 558 | **significant** |
| langchain | NDCG | +0.1036 | [0.0693, 0.1374] | 0.0001 | 558 | **significant** |
| langchain | RECALL | +0.095 | [0.0627, 0.1272] | 0.0001 | 558 | **significant** |
| langchain | HIT1 | +0.1416 | [0.0878, 0.1935] | 0.0001 | 558 | **significant** |
| unstructured | MRR | +0.1269 | [0.0912, 0.1622] | 0.0001 | 549 | **significant** |
| unstructured | NDCG | +0.1058 | [0.075, 0.1365] | 0.0001 | 549 | **significant** |
| unstructured | RECALL | +0.0856 | [0.0565, 0.1166] | 0.0001 | 549 | **significant** |
| unstructured | HIT1 | +0.1512 | [0.102, 0.1985] | 0.0001 | 549 | **significant** |

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
- **The gold reference reads two-column pages column-wise.** `page.extract_text()` welds the two columns of every line together, which makes the reference text unquotable and was silently rejecting correct gold. Splitting at a detected gutter fixes that, and it has a consequence worth stating: a chunker that interleaves columns can no longer contain a gold span contiguously. That is a real quality difference — interleaved text is wrong text — but the gutter heuristic is the same *kind* of heuristic DocStruct uses, so it favours column-aware tools generally, ours included. Runs before this change are not comparable with runs after it.
- **Chunk s is not a fair speed comparison when `--cache-dir` is set.** Only the DocStruct adapter uses that cache (detector proposals and populated blocks, keyed by PDF hash), so on a warm cache its column reports cache-hit time while every other tool is measured cold. Compare wall-clock only from a run with no `--cache-dir`, or against `meta.docstruct_cold_chunk_seconds` if present.
- **MRR/1k words is a tradeoff axis, not a ranking.** It necessarily favours tools that emit very small chunks and therefore retrieve very little text, regardless of whether they rank well. Read it next to MRR, not instead of it.
- This is a **signal/baseline**, not the Phase-2 public benchmark (50 PDFs, 200 human-checked Q&A).


### Per-doc breakdown: docstruct (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc27.pdf ⚠ | 1 | 66 | 276.8 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc66.pdf | 9 | 70 | 455.1 | 0.2963 | 0.6667 | 0.1111 | 6/9 |
| doc53.pdf | 3 | 66 | 276.8 | 0.3333 | 0.3333 | 0.3333 | 1/3 |
| doc88.pdf | 5 | 32 | 294.0 | 0.4 | 0.8 | 0.2 | 4/5 |
| doc56.pdf | 5 | 200 | 139.3 | 0.44 | 0.8 | 0.2 | 4/5 |
| doc42.pdf | 7 | 19 | 387.4 | 0.4571 | 0.7143 | 0.2857 | 5/7 |
| doc50.pdf | 5 | 32 | 294.0 | 0.5 | 0.8 | 0.4 | 4/5 |
| doc64.pdf | 6 | 64 | 236.1 | 0.5 | 0.5 | 0.5 | 3/6 |
| doc85.pdf | 4 | 23 | 430.8 | 0.5625 | 0.75 | 0.5 | 3/4 |
| doc67.pdf | 5 | 59 | 310.9 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc73.pdf | 6 | 87 | 422.0 | 0.5833 | 0.8333 | 0.3333 | 5/6 |
| doc61.pdf | 5 | 164 | 158.5 | 0.6667 | 1.0 | 0.4 | 5/5 |
| doc65.pdf | 3 | 38 | 294.9 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc9.pdf | 9 | 48 | 317.4 | 0.6759 | 0.8889 | 0.5556 | 8/9 |
| doc33.pdf | 7 | 74 | 346.6 | 0.6786 | 0.8571 | 0.5714 | 6/7 |
| doc84.pdf | 6 | 11 | 475.3 | 0.6806 | 1.0 | 0.5 | 6/6 |
| doc7.pdf | 6 | 105 | 452.6 | 0.7083 | 1.0 | 0.5 | 6/6 |
| doc72.pdf | 5 | 38 | 487.3 | 0.7167 | 1.0 | 0.6 | 5/5 |
| doc86.pdf | 7 | 87 | 452.7 | 0.719 | 1.0 | 0.5714 | 7/7 |
| doc77.pdf | 8 | 34 | 394.3 | 0.7292 | 0.875 | 0.625 | 7/8 |
| doc90.pdf | 9 | 67 | 271.6 | 0.7315 | 1.0 | 0.5556 | 9/9 |
| doc81.pdf | 7 | 87 | 401.5 | 0.7429 | 0.8571 | 0.7143 | 6/7 |
| doc30.pdf | 2 | 200 | 139.3 | 0.75 | 1.0 | 0.5 | 2/2 |
| doc47.pdf | 4 | 52 | 396.2 | 0.75 | 1.0 | 0.5 | 4/4 |
| doc63.pdf | 4 | 19 | 468.0 | 0.75 | 1.0 | 0.5 | 4/4 |
| doc71.pdf | 3 | 40 | 403.1 | 0.75 | 1.0 | 0.6667 | 3/3 |
| doc78.pdf | 6 | 111 | 367.3 | 0.75 | 1.0 | 0.5 | 6/6 |
| doc91.pdf | 4 | 40 | 291.8 | 0.75 | 1.0 | 0.5 | 4/4 |
| doc4.pdf | 7 | 132 | 524.7 | 0.7619 | 0.8571 | 0.7143 | 6/7 |
| doc21.pdf | 8 | 54 | 450.4 | 0.7708 | 1.0 | 0.625 | 8/8 |
| doc41.pdf | 8 | 87 | 422.0 | 0.775 | 1.0 | 0.625 | 8/8 |
| doc23.pdf | 9 | 9 | 373.8 | 0.7778 | 1.0 | 0.5556 | 9/9 |
| doc10.pdf | 8 | 61 | 391.7 | 0.7812 | 0.875 | 0.75 | 7/8 |
| doc79.pdf | 8 | 90 | 342.2 | 0.7812 | 0.875 | 0.75 | 7/8 |
| doc5.pdf | 6 | 104 | 385.4 | 0.7917 | 1.0 | 0.6667 | 6/6 |
| doc82.pdf | 8 | 20 | 410.0 | 0.7917 | 0.875 | 0.75 | 7/8 |
| doc68.pdf | 5 | 98 | 138.3 | 0.8 | 0.8 | 0.8 | 4/5 |
| doc34.pdf | 7 | 164 | 158.5 | 0.8095 | 1.0 | 0.7143 | 7/7 |
| doc17.pdf | 9 | 70 | 276.9 | 0.8148 | 0.8889 | 0.7778 | 8/9 |
| doc35.pdf | 9 | 64 | 299.2 | 0.8148 | 0.8889 | 0.7778 | 8/9 |
| doc74.pdf | 9 | 19 | 387.4 | 0.8148 | 0.8889 | 0.7778 | 8/9 |
| doc16.pdf | 10 | 11 | 398.2 | 0.82 | 0.9 | 0.8 | 9/10 |
| doc38.pdf | 3 | 164 | 529.2 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc45.pdf | 9 | 111 | 367.3 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| doc62.pdf | 8 | 64 | 299.2 | 0.8333 | 1.0 | 0.75 | 8/8 |
| doc89.pdf | 9 | 72 | 314.2 | 0.8333 | 1.0 | 0.7778 | 9/9 |
| doc24.pdf | 9 | 67 | 215.7 | 0.8426 | 1.0 | 0.7778 | 9/9 |
| doc48.pdf | 5 | 87 | 452.7 | 0.85 | 1.0 | 0.8 | 5/5 |
| doc22.pdf | 8 | 34 | 386.5 | 0.8542 | 1.0 | 0.75 | 8/8 |
| doc29.pdf | 8 | 41 | 307.5 | 0.8542 | 1.0 | 0.75 | 8/8 |
| doc6.pdf | 10 | 14 | 602.4 | 0.8583 | 1.0 | 0.8 | 10/10 |
| doc51.pdf | 8 | 118 | 230.7 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc55.pdf | 4 | 41 | 307.5 | 0.875 | 1.0 | 0.75 | 4/4 |
| doc25.pdf | 9 | 118 | 230.7 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| doc2.pdf | 6 | 54 | 450.4 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc3.pdf | 8 | 75 | 393.2 | 0.9167 | 1.0 | 0.875 | 8/8 |
| doc32.pdf | 6 | 47 | 314.6 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc43.pdf | 8 | 31 | 335.5 | 0.9167 | 1.0 | 0.875 | 8/8 |
| doc54.pdf | 6 | 28 | 355.3 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc59.pdf | 6 | 47 | 314.6 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc60.pdf | 6 | 74 | 346.6 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc11.pdf | 7 | 12 | 331.1 | 0.9286 | 1.0 | 0.8571 | 7/7 |
| doc8.pdf | 7 | 75 | 480.0 | 0.9286 | 1.0 | 0.8571 | 7/7 |
| doc92.pdf | 7 | 59 | 551.9 | 0.9286 | 1.0 | 0.8571 | 7/7 |
| doc57.pdf | 8 | 32 | 307.7 | 0.9375 | 1.0 | 0.875 | 8/8 |
| doc18.pdf | 9 | 17 | 369.2 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| doc1.pdf | 8 | 31 | 244.2 | 1.0 | 1.0 | 1.0 | 8/8 |
| doc12.pdf | 4 | 60 | 369.6 | 1.0 | 1.0 | 1.0 | 4/4 |
| doc13.pdf | 5 | 51 | 382.4 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc14.pdf | 6 | 116 | 356.0 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc15.pdf | 2 | 7 | 478.0 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc19.pdf | 9 | 21 | 366.4 | 1.0 | 1.0 | 1.0 | 9/9 |
| doc20.pdf | 9 | 82 | 277.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| doc26.pdf | 2 | 56 | 236.8 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc28.pdf | 3 | 28 | 355.3 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc37.pdf | 2 | 118 | 492.9 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc39.pdf | 3 | 40 | 403.1 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc40.pdf | 6 | 38 | 487.3 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc44.pdf | 6 | 34 | 394.3 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc46.pdf | 1 | 90 | 342.2 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc49.pdf | 7 | 10 | 418.1 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc52.pdf | 1 | 56 | 236.8 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc70.pdf | 2 | 164 | 529.2 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc75.pdf | 1 | 31 | 335.5 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc76.pdf | 2 | 69 | 439.9 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc80.pdf | 7 | 52 | 396.2 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc83.pdf | 8 | 128 | 496.3 | 1.0 | 1.0 | 1.0 | 8/8 |
| doc87.pdf | 7 | 10 | 418.1 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc93.pdf | 6 | 14 | 691.8 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc94.pdf | 5 | 66 | 529.3 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc95.pdf | 5 | 18 | 482.9 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc96.pdf | 5 | 1327 | 292.4 | 1.0 | 1.0 | 1.0 | 5/5 |


### Per-doc breakdown: docstruct_geo (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc27.pdf ⚠ | 1 | 44 | 239.2 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc30.pdf ⚠ | 2 | 128 | 166.2 | 0.0 | 0.0 | 0.0 | 0/2 |
| doc53.pdf | 3 | 44 | 239.2 | 0.0833 | 0.3333 | 0.0 | 1/3 |
| doc46.pdf | 1 | 70 | 316.2 | 0.25 | 1.0 | 0.0 | 1/1 |
| doc55.pdf | 4 | 32 | 282.2 | 0.3125 | 0.5 | 0.25 | 2/4 |
| doc67.pdf | 5 | 37 | 341.5 | 0.4667 | 1.0 | 0.2 | 5/5 |
| doc42.pdf | 7 | 15 | 280.6 | 0.4762 | 0.5714 | 0.4286 | 4/7 |
| doc65.pdf | 3 | 22 | 322.9 | 0.5 | 0.6667 | 0.3333 | 2/3 |
| doc66.pdf | 9 | 51 | 514.2 | 0.5 | 0.6667 | 0.3333 | 6/9 |
| doc86.pdf | 7 | 57 | 416.1 | 0.5357 | 0.7143 | 0.4286 | 5/7 |
| doc64.pdf | 6 | 50 | 225.1 | 0.5556 | 0.8333 | 0.3333 | 5/6 |
| doc71.pdf | 3 | 25 | 368.6 | 0.5556 | 1.0 | 0.3333 | 3/3 |
| doc50.pdf | 5 | 21 | 372.6 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc61.pdf | 5 | 132 | 156.9 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc48.pdf | 5 | 57 | 416.1 | 0.59 | 1.0 | 0.4 | 5/5 |
| doc35.pdf | 9 | 40 | 290.2 | 0.5926 | 0.8889 | 0.3333 | 8/9 |
| doc33.pdf | 7 | 44 | 430.5 | 0.6071 | 0.7143 | 0.5714 | 5/7 |
| doc24.pdf | 9 | 53 | 234.6 | 0.6111 | 0.8889 | 0.4444 | 8/9 |
| doc56.pdf | 5 | 128 | 166.2 | 0.6167 | 1.0 | 0.4 | 5/5 |
| doc91.pdf | 4 | 18 | 334.7 | 0.625 | 0.75 | 0.5 | 3/4 |
| doc14.pdf | 6 | 61 | 501.1 | 0.6389 | 1.0 | 0.3333 | 6/6 |
| doc68.pdf | 5 | 97 | 107.6 | 0.64 | 1.0 | 0.4 | 5/5 |
| doc73.pdf | 6 | 50 | 455.6 | 0.6667 | 1.0 | 0.5 | 6/6 |
| doc78.pdf | 6 | 55 | 427.6 | 0.6667 | 0.8333 | 0.5 | 5/6 |
| doc88.pdf | 5 | 21 | 372.6 | 0.6667 | 0.8 | 0.6 | 4/5 |
| doc45.pdf | 9 | 55 | 427.6 | 0.6944 | 1.0 | 0.5556 | 9/9 |
| doc3.pdf | 8 | 38 | 463.6 | 0.6979 | 0.875 | 0.625 | 7/8 |
| doc51.pdf | 8 | 55 | 315.6 | 0.6979 | 1.0 | 0.5 | 8/8 |
| doc94.pdf | 5 | 32 | 591.9 | 0.7 | 0.8 | 0.6 | 4/5 |
| doc84.pdf | 6 | 13 | 370.2 | 0.7083 | 0.8333 | 0.6667 | 5/6 |
| doc29.pdf | 8 | 32 | 282.2 | 0.7125 | 0.875 | 0.625 | 7/8 |
| doc34.pdf | 7 | 132 | 156.9 | 0.7143 | 0.8571 | 0.5714 | 6/7 |
| doc81.pdf | 7 | 60 | 398.1 | 0.7143 | 0.7143 | 0.7143 | 5/7 |
| doc80.pdf | 7 | 34 | 415.9 | 0.719 | 1.0 | 0.5714 | 7/7 |
| doc74.pdf | 9 | 15 | 280.6 | 0.7222 | 0.8889 | 0.6667 | 8/9 |
| doc89.pdf | 9 | 36 | 412.8 | 0.7222 | 0.8889 | 0.6667 | 8/9 |
| doc13.pdf | 5 | 21 | 494.9 | 0.75 | 1.0 | 0.6 | 5/5 |
| doc47.pdf | 4 | 34 | 415.9 | 0.75 | 1.0 | 0.5 | 4/4 |
| doc7.pdf | 6 | 58 | 494.0 | 0.75 | 1.0 | 0.5 | 6/6 |
| doc79.pdf | 8 | 70 | 316.2 | 0.75 | 0.75 | 0.75 | 6/8 |
| doc43.pdf | 8 | 12 | 458.0 | 0.7542 | 1.0 | 0.625 | 8/8 |
| doc2.pdf | 6 | 37 | 499.7 | 0.7639 | 1.0 | 0.6667 | 6/6 |
| doc72.pdf | 5 | 23 | 460.8 | 0.7667 | 1.0 | 0.6 | 5/5 |
| doc59.pdf | 6 | 41 | 236.8 | 0.7833 | 1.0 | 0.6667 | 6/6 |
| doc10.pdf | 8 | 39 | 399.6 | 0.7917 | 1.0 | 0.625 | 8/8 |
| doc21.pdf | 8 | 37 | 499.7 | 0.7917 | 0.875 | 0.75 | 7/8 |
| doc32.pdf | 6 | 41 | 236.8 | 0.7917 | 1.0 | 0.6667 | 6/6 |
| doc60.pdf | 6 | 44 | 430.5 | 0.7917 | 1.0 | 0.6667 | 6/6 |
| doc82.pdf | 8 | 20 | 391.4 | 0.7917 | 1.0 | 0.625 | 8/8 |
| doc4.pdf | 7 | 74 | 504.2 | 0.7976 | 1.0 | 0.7143 | 7/7 |
| doc90.pdf | 9 | 53 | 241.1 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc17.pdf | 9 | 59 | 226.6 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc23.pdf | 9 | 6 | 346.8 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| doc38.pdf | 3 | 90 | 501.3 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc22.pdf | 8 | 27 | 320.5 | 0.8375 | 1.0 | 0.75 | 8/8 |
| doc9.pdf | 9 | 48 | 274.8 | 0.8426 | 1.0 | 0.7778 | 9/9 |
| doc62.pdf | 8 | 40 | 290.2 | 0.8438 | 1.0 | 0.75 | 8/8 |
| doc6.pdf | 10 | 8 | 641.9 | 0.85 | 1.0 | 0.7 | 10/10 |
| doc41.pdf | 8 | 50 | 455.6 | 0.8542 | 1.0 | 0.75 | 8/8 |
| doc8.pdf | 7 | 57 | 426.1 | 0.8571 | 0.8571 | 0.8571 | 6/7 |
| doc18.pdf | 9 | 9 | 436.8 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| doc1.pdf | 8 | 33 | 223.0 | 0.875 | 1.0 | 0.75 | 8/8 |
| doc12.pdf | 4 | 27 | 519.4 | 0.875 | 1.0 | 0.75 | 4/4 |
| doc57.pdf | 8 | 20 | 283.2 | 0.875 | 1.0 | 0.75 | 8/8 |
| doc19.pdf | 9 | 13 | 392.5 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| doc5.pdf | 6 | 59 | 394.7 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| doc54.pdf | 6 | 16 | 331.9 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| doc95.pdf | 5 | 14 | 452.1 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc20.pdf | 9 | 61 | 253.4 | 0.9167 | 1.0 | 0.8889 | 9/9 |
| doc11.pdf | 7 | 4 | 470.2 | 0.9286 | 1.0 | 0.8571 | 7/7 |
| doc92.pdf | 7 | 31 | 578.5 | 0.9286 | 1.0 | 0.8571 | 7/7 |
| doc16.pdf | 10 | 12 | 378.4 | 0.9333 | 1.0 | 0.9 | 10/10 |
| doc77.pdf | 8 | 19 | 423.4 | 0.9375 | 1.0 | 0.875 | 8/8 |
| doc15.pdf | 2 | 3 | 558.7 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc25.pdf | 9 | 55 | 315.6 | 1.0 | 1.0 | 1.0 | 9/9 |
| doc26.pdf | 2 | 56 | 210.5 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc28.pdf | 3 | 16 | 331.9 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc37.pdf | 2 | 75 | 456.3 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc39.pdf | 3 | 25 | 368.6 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc40.pdf | 6 | 23 | 460.8 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc44.pdf | 6 | 19 | 423.4 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc49.pdf | 7 | 7 | 436.1 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc52.pdf | 1 | 56 | 210.5 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc63.pdf | 4 | 10 | 476.8 | 1.0 | 1.0 | 1.0 | 4/4 |
| doc70.pdf | 2 | 90 | 501.3 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc75.pdf | 1 | 12 | 458.0 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc76.pdf | 2 | 43 | 433.3 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc83.pdf | 8 | 57 | 640.2 | 1.0 | 1.0 | 1.0 | 8/8 |
| doc85.pdf | 4 | 14 | 407.9 | 1.0 | 1.0 | 1.0 | 4/4 |
| doc87.pdf | 7 | 7 | 436.1 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc93.pdf | 6 | 6 | 821.7 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc96.pdf | 5 | 795 | 276.9 | 1.0 | 1.0 | 1.0 | 5/5 |


### Per-doc breakdown: pymupdf4llm (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc59.pdf | 6 | 21 | 500.5 | 0.3056 | 0.5 | 0.1667 | 3/6 |
| doc27.pdf | 1 | 19 | 583.5 | 0.3333 | 1.0 | 0.0 | 1/1 |
| doc42.pdf | 7 | 12 | 472.8 | 0.3381 | 0.8571 | 0.1429 | 6/7 |
| doc66.pdf | 9 | 28 | 508.9 | 0.3426 | 0.5556 | 0.2222 | 5/9 |
| doc34.pdf | 7 | 38 | 503.2 | 0.4095 | 0.7143 | 0.2857 | 5/7 |
| doc53.pdf | 3 | 19 | 583.5 | 0.4167 | 0.6667 | 0.3333 | 2/3 |
| doc94.pdf | 5 | 27 | 613.1 | 0.45 | 0.6 | 0.4 | 3/5 |
| doc81.pdf | 7 | 61 | 298.6 | 0.4762 | 0.8571 | 0.2857 | 6/7 |
| doc30.pdf | 2 | 49 | 189.3 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc65.pdf | 3 | 21 | 438.4 | 0.5 | 0.6667 | 0.3333 | 2/3 |
| doc71.pdf | 3 | 26 | 343.2 | 0.5 | 1.0 | 0.0 | 3/3 |
| doc43.pdf | 8 | 9 | 642.6 | 0.5104 | 0.875 | 0.25 | 7/8 |
| doc4.pdf | 7 | 78 | 568.2 | 0.5357 | 0.8571 | 0.2857 | 6/7 |
| doc50.pdf | 5 | 23 | 338.0 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc64.pdf | 6 | 25 | 445.2 | 0.5667 | 1.0 | 0.3333 | 6/6 |
| doc88.pdf | 5 | 23 | 338.0 | 0.5667 | 0.8 | 0.4 | 4/5 |
| doc44.pdf | 6 | 16 | 522.6 | 0.5694 | 1.0 | 0.3333 | 6/6 |
| doc78.pdf | 6 | 90 | 215.1 | 0.5833 | 0.6667 | 0.5 | 4/6 |
| doc39.pdf | 3 | 26 | 343.2 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| doc67.pdf | 5 | 38 | 395.3 | 0.6167 | 1.0 | 0.4 | 5/5 |
| doc33.pdf | 7 | 31 | 571.0 | 0.619 | 0.7143 | 0.5714 | 5/7 |
| doc9.pdf | 9 | 21 | 329.6 | 0.6333 | 0.7778 | 0.5556 | 7/9 |
| doc45.pdf | 9 | 90 | 215.1 | 0.6389 | 0.8889 | 0.4444 | 8/9 |
| doc61.pdf | 5 | 38 | 503.2 | 0.64 | 0.8 | 0.6 | 4/5 |
| doc95.pdf | 5 | 25 | 252.3 | 0.64 | 1.0 | 0.4 | 5/5 |
| doc90.pdf | 9 | 30 | 491.7 | 0.6667 | 0.8889 | 0.4444 | 8/9 |
| doc84.pdf | 6 | 15 | 422.9 | 0.6806 | 1.0 | 0.5 | 6/6 |
| doc74.pdf | 9 | 12 | 472.8 | 0.6833 | 1.0 | 0.5556 | 9/9 |
| doc77.pdf | 8 | 16 | 522.6 | 0.6875 | 1.0 | 0.375 | 8/8 |
| doc68.pdf | 5 | 34 | 326.9 | 0.7 | 0.8 | 0.6 | 4/5 |
| doc32.pdf | 6 | 21 | 500.5 | 0.7083 | 0.8333 | 0.6667 | 5/6 |
| doc10.pdf | 8 | 39 | 449.5 | 0.7125 | 1.0 | 0.5 | 8/8 |
| doc1.pdf | 8 | 22 | 318.1 | 0.75 | 1.0 | 0.5 | 8/8 |
| doc13.pdf | 5 | 24 | 516.3 | 0.75 | 1.0 | 0.6 | 5/5 |
| doc82.pdf | 8 | 17 | 470.0 | 0.75 | 0.75 | 0.75 | 6/8 |
| doc35.pdf | 9 | 26 | 467.3 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| doc56.pdf | 5 | 49 | 189.3 | 0.7667 | 1.0 | 0.6 | 5/5 |
| doc18.pdf | 9 | 9 | 517.1 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| doc86.pdf | 7 | 40 | 391.9 | 0.7857 | 0.8571 | 0.7143 | 6/7 |
| doc92.pdf | 7 | 46 | 375.0 | 0.7857 | 0.8571 | 0.7143 | 6/7 |
| doc29.pdf | 8 | 26 | 382.8 | 0.7917 | 0.875 | 0.75 | 7/8 |
| doc48.pdf | 5 | 40 | 391.9 | 0.8 | 1.0 | 0.6 | 5/5 |
| doc96.pdf | 5 | 364 | 517.6 | 0.8 | 0.8 | 0.8 | 4/5 |
| doc60.pdf | 6 | 31 | 571.0 | 0.8056 | 1.0 | 0.6667 | 6/6 |
| doc6.pdf | 10 | 8 | 714.9 | 0.8083 | 1.0 | 0.7 | 10/10 |
| doc79.pdf | 8 | 56 | 248.3 | 0.8125 | 1.0 | 0.75 | 8/8 |
| doc19.pdf | 9 | 8 | 639.8 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc24.pdf | 9 | 12 | 824.1 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc80.pdf | 7 | 46 | 291.1 | 0.8214 | 1.0 | 0.7143 | 7/7 |
| doc16.pdf | 10 | 10 | 431.0 | 0.825 | 1.0 | 0.7 | 10/10 |
| doc17.pdf | 9 | 22 | 604.0 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc23.pdf | 9 | 6 | 368.0 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| doc54.pdf | 6 | 11 | 510.5 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc63.pdf | 4 | 6 | 782.2 | 0.8333 | 1.0 | 0.75 | 4/4 |
| doc73.pdf | 6 | 44 | 574.0 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc93.pdf | 6 | 6 | 755.0 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc62.pdf | 8 | 26 | 467.3 | 0.8542 | 1.0 | 0.75 | 8/8 |
| doc8.pdf | 7 | 48 | 623.4 | 0.8571 | 0.8571 | 0.8571 | 6/7 |
| doc87.pdf | 7 | 10 | 246.1 | 0.8571 | 0.8571 | 0.8571 | 6/7 |
| doc72.pdf | 5 | 28 | 464.9 | 0.8667 | 1.0 | 0.8 | 5/5 |
| doc89.pdf | 9 | 30 | 432.9 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| doc12.pdf | 4 | 15 | 885.9 | 0.875 | 1.0 | 0.75 | 4/4 |
| doc22.pdf | 8 | 16 | 559.1 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc3.pdf | 8 | 27 | 533.8 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc41.pdf | 8 | 44 | 574.0 | 0.875 | 1.0 | 0.75 | 8/8 |
| doc51.pdf | 8 | 39 | 481.1 | 0.875 | 0.875 | 0.875 | 7/8 |
| doc55.pdf | 4 | 26 | 382.8 | 0.875 | 1.0 | 0.75 | 4/4 |
| doc91.pdf | 4 | 12 | 637.5 | 0.875 | 1.0 | 0.75 | 4/4 |
| doc7.pdf | 6 | 65 | 456.0 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| doc14.pdf | 6 | 32 | 769.1 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc2.pdf | 6 | 43 | 419.7 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc25.pdf | 9 | 39 | 481.1 | 0.9167 | 1.0 | 0.8889 | 9/9 |
| doc57.pdf | 8 | 13 | 483.6 | 0.9375 | 1.0 | 0.875 | 8/8 |
| doc83.pdf | 8 | 49 | 619.8 | 0.9375 | 1.0 | 0.875 | 8/8 |
| doc11.pdf | 7 | 4 | 561.2 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc15.pdf | 2 | 3 | 600.0 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc20.pdf | 9 | 31 | 460.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| doc21.pdf | 8 | 43 | 419.7 | 1.0 | 1.0 | 1.0 | 8/8 |
| doc26.pdf | 2 | 34 | 378.6 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc28.pdf | 3 | 11 | 510.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc37.pdf | 2 | 51 | 679.6 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc38.pdf | 3 | 90 | 401.4 | 1.0 | 1.0 | 1.0 | 3/3 |
| doc40.pdf | 6 | 28 | 464.9 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc46.pdf | 1 | 56 | 248.3 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc47.pdf | 4 | 46 | 291.1 | 1.0 | 1.0 | 1.0 | 4/4 |
| doc49.pdf | 7 | 10 | 246.1 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc5.pdf | 6 | 39 | 476.3 | 1.0 | 1.0 | 1.0 | 6/6 |
| doc52.pdf | 1 | 34 | 378.6 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc70.pdf | 2 | 90 | 401.4 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc75.pdf | 1 | 9 | 642.6 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc76.pdf | 2 | 46 | 379.1 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc85.pdf | 4 | 7 | 747.3 | 1.0 | 1.0 | 1.0 | 4/4 |


### Per-doc breakdown: langchain (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc27.pdf ⚠ | 1 | 86 | 98.0 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc71.pdf | 3 | 80 | 122.9 | 0.0833 | 0.3333 | 0.0 | 1/3 |
| doc73.pdf | 6 | 191 | 61.0 | 0.3333 | 0.3333 | 0.3333 | 2/6 |
| doc3.pdf | 8 | 108 | 75.1 | 0.4 | 0.5 | 0.375 | 4/8 |
| doc67.pdf | 5 | 119 | 108.4 | 0.4 | 0.8 | 0.2 | 4/5 |
| doc33.pdf | 7 | 127 | 79.0 | 0.4048 | 0.7143 | 0.1429 | 5/7 |
| doc47.pdf | 4 | 112 | 132.7 | 0.4125 | 1.0 | 0.25 | 4/4 |
| doc39.pdf | 3 | 80 | 122.9 | 0.4444 | 0.6667 | 0.3333 | 2/3 |
| doc8.pdf | 7 | 238 | 114.3 | 0.4643 | 0.7143 | 0.2857 | 5/7 |
| doc35.pdf | 9 | 99 | 122.1 | 0.4722 | 0.6667 | 0.3333 | 6/9 |
| doc80.pdf | 7 | 112 | 132.7 | 0.4762 | 0.8571 | 0.2857 | 6/7 |
| doc17.pdf | 9 | 86 | 81.6 | 0.5 | 0.6667 | 0.3333 | 6/9 |
| doc37.pdf | 2 | 230 | 55.8 | 0.5 | 0.5 | 0.5 | 1/2 |
| doc81.pdf | 7 | 178 | 140.8 | 0.5 | 0.5714 | 0.4286 | 4/7 |
| doc66.pdf | 9 | 145 | 128.2 | 0.5037 | 0.6667 | 0.4444 | 6/9 |
| doc32.pdf | 6 | 81 | 55.5 | 0.5417 | 0.6667 | 0.5 | 4/6 |
| doc50.pdf | 5 | 70 | 99.5 | 0.55 | 0.8 | 0.4 | 4/5 |
| doc14.pdf | 6 | 237 | 107.8 | 0.5556 | 0.6667 | 0.5 | 4/6 |
| doc86.pdf | 7 | 137 | 145.6 | 0.5714 | 0.7143 | 0.4286 | 5/7 |
| doc88.pdf | 5 | 70 | 99.5 | 0.59 | 1.0 | 0.4 | 5/5 |
| doc72.pdf | 5 | 115 | 119.8 | 0.6 | 0.6 | 0.6 | 3/5 |
| doc20.pdf | 9 | 111 | 136.6 | 0.6111 | 0.6667 | 0.5556 | 6/9 |
| doc24.pdf | 9 | 75 | 70.2 | 0.6111 | 0.6667 | 0.5556 | 6/9 |
| doc12.pdf | 4 | 106 | 104.8 | 0.625 | 1.0 | 0.25 | 4/4 |
| doc63.pdf | 4 | 37 | 39.0 | 0.625 | 0.75 | 0.5 | 3/4 |
| doc54.pdf | 6 | 47 | 71.3 | 0.6389 | 0.8333 | 0.5 | 5/6 |
| doc56.pdf | 5 | 165 | 107.4 | 0.65 | 0.8 | 0.6 | 4/5 |
| doc2.pdf | 6 | 142 | 129.0 | 0.6667 | 1.0 | 0.3333 | 6/6 |
| doc40.pdf | 6 | 115 | 119.8 | 0.6667 | 0.6667 | 0.6667 | 4/6 |
| doc53.pdf | 3 | 86 | 98.0 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc64.pdf | 6 | 105 | 55.8 | 0.6667 | 0.8333 | 0.5 | 5/6 |
| doc7.pdf | 6 | 234 | 117.6 | 0.6667 | 0.6667 | 0.6667 | 4/6 |
| doc18.pdf | 9 | 31 | 49.8 | 0.6704 | 0.8889 | 0.5556 | 8/9 |
| doc4.pdf | 7 | 340 | 127.1 | 0.6786 | 0.8571 | 0.5714 | 6/7 |
| doc10.pdf | 8 | 155 | 89.7 | 0.6875 | 0.75 | 0.625 | 6/8 |
| doc21.pdf | 8 | 142 | 129.0 | 0.6875 | 0.75 | 0.625 | 6/8 |
| doc51.pdf | 8 | 161 | 128.3 | 0.6875 | 0.75 | 0.625 | 6/8 |
| doc77.pdf | 8 | 62 | 54.2 | 0.6875 | 0.75 | 0.625 | 6/8 |
| doc45.pdf | 9 | 169 | 94.2 | 0.6889 | 0.8889 | 0.5556 | 8/9 |
| doc57.pdf | 8 | 53 | 40.9 | 0.6979 | 0.875 | 0.625 | 7/8 |
| doc16.pdf | 10 | 35 | 125.8 | 0.7083 | 0.9 | 0.6 | 9/10 |
| doc44.pdf | 6 | 62 | 54.2 | 0.7083 | 1.0 | 0.5 | 6/6 |
| doc79.pdf | 8 | 153 | 110.8 | 0.7083 | 0.875 | 0.625 | 7/8 |
| doc25.pdf | 9 | 161 | 128.3 | 0.7222 | 0.7778 | 0.6667 | 7/9 |
| doc78.pdf | 6 | 169 | 94.2 | 0.7222 | 0.8333 | 0.6667 | 5/6 |
| doc93.pdf | 6 | 37 | 120.5 | 0.7222 | 1.0 | 0.5 | 6/6 |
| doc1.pdf | 8 | 57 | 129.1 | 0.7292 | 1.0 | 0.5 | 8/8 |
| doc89.pdf | 9 | 103 | 133.3 | 0.7407 | 0.8889 | 0.6667 | 8/9 |
| doc42.pdf | 7 | 47 | 33.3 | 0.7429 | 0.8571 | 0.7143 | 6/7 |
| doc15.pdf | 2 | 13 | 91.5 | 0.75 | 1.0 | 0.5 | 2/2 |
| doc30.pdf | 2 | 165 | 107.4 | 0.75 | 1.0 | 0.5 | 2/2 |
| doc38.pdf | 3 | 278 | 117.0 | 0.75 | 1.0 | 0.6667 | 3/3 |
| doc55.pdf | 4 | 89 | 57.7 | 0.75 | 1.0 | 0.5 | 4/4 |
| doc60.pdf | 6 | 127 | 79.0 | 0.75 | 1.0 | 0.5 | 6/6 |
| doc91.pdf | 4 | 68 | 113.2 | 0.75 | 0.75 | 0.75 | 3/4 |
| doc41.pdf | 8 | 191 | 61.0 | 0.7542 | 1.0 | 0.625 | 8/8 |
| doc82.pdf | 8 | 69 | 125.0 | 0.7542 | 1.0 | 0.625 | 8/8 |
| doc74.pdf | 9 | 47 | 33.3 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| doc90.pdf | 9 | 111 | 84.5 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| doc19.pdf | 9 | 36 | 54.2 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| doc43.pdf | 8 | 44 | 60.6 | 0.7812 | 0.875 | 0.75 | 7/8 |
| doc83.pdf | 8 | 208 | 72.8 | 0.7812 | 0.875 | 0.75 | 7/8 |
| doc11.pdf | 7 | 18 | 104.2 | 0.7857 | 1.0 | 0.5714 | 7/7 |
| doc22.pdf | 8 | 70 | 60.5 | 0.7917 | 1.0 | 0.625 | 8/8 |
| doc62.pdf | 8 | 99 | 122.1 | 0.7917 | 1.0 | 0.625 | 8/8 |
| doc61.pdf | 5 | 129 | 67.7 | 0.8 | 1.0 | 0.6 | 5/5 |
| doc68.pdf | 5 | 94 | 99.3 | 0.8 | 0.8 | 0.8 | 4/5 |
| doc95.pdf | 5 | 51 | 132.2 | 0.8 | 0.8 | 0.8 | 4/5 |
| doc96.pdf | 5 | 1706 | 134.0 | 0.8 | 0.8 | 0.8 | 4/5 |
| doc34.pdf | 7 | 129 | 67.7 | 0.8095 | 1.0 | 0.7143 | 7/7 |
| doc23.pdf | 9 | 19 | 107.8 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc28.pdf | 3 | 47 | 71.3 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc59.pdf | 6 | 81 | 55.5 | 0.8333 | 0.8333 | 0.8333 | 5/6 |
| doc65.pdf | 3 | 71 | 79.4 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| doc84.pdf | 6 | 51 | 122.6 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc9.pdf | 9 | 67 | 122.5 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| doc92.pdf | 7 | 141 | 143.2 | 0.8333 | 1.0 | 0.7143 | 7/7 |
| doc87.pdf | 7 | 23 | 147.9 | 0.8571 | 1.0 | 0.7143 | 7/7 |
| doc5.pdf | 6 | 135 | 159.1 | 0.8667 | 1.0 | 0.8333 | 6/6 |
| doc13.pdf | 5 | 103 | 111.6 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc94.pdf | 5 | 128 | 137.9 | 0.9 | 1.0 | 0.8 | 5/5 |
| doc49.pdf | 7 | 23 | 147.9 | 0.9286 | 1.0 | 0.8571 | 7/7 |
| doc6.pdf | 10 | 42 | 25.5 | 0.95 | 1.0 | 0.9 | 10/10 |
| doc26.pdf | 2 | 112 | 66.5 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc29.pdf | 8 | 89 | 57.7 | 1.0 | 1.0 | 1.0 | 8/8 |
| doc46.pdf | 1 | 153 | 110.8 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc48.pdf | 5 | 137 | 145.6 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc52.pdf | 1 | 112 | 66.5 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc70.pdf | 2 | 278 | 117.0 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc75.pdf | 1 | 44 | 60.6 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc76.pdf | 2 | 147 | 79.1 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc85.pdf | 4 | 39 | 95.9 | 1.0 | 1.0 | 1.0 | 4/4 |


### Per-doc breakdown: unstructured (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| doc27.pdf ⚠ | 1 | 131 | 86.9 | 0.0 | 0.0 | 0.0 | 0/1 |
| doc71.pdf ⚠ | 3 | 109 | 81.6 | 0.0 | 0.0 | 0.0 | 0/3 |
| doc67.pdf | 5 | 161 | 91.8 | 0.14 | 0.4 | 0.0 | 2/5 |
| doc30.pdf | 2 | 260 | 77.2 | 0.1667 | 0.5 | 0.0 | 1/2 |
| doc33.pdf | 7 | 203 | 79.7 | 0.2976 | 0.5714 | 0.1429 | 4/7 |
| doc39.pdf | 3 | 109 | 81.6 | 0.3333 | 0.3333 | 0.3333 | 1/3 |
| doc66.pdf | 9 | 174 | 88.9 | 0.4259 | 0.5556 | 0.3333 | 5/9 |
| doc81.pdf | 7 | 289 | 75.3 | 0.4333 | 0.8571 | 0.1429 | 6/7 |
| doc17.pdf | 9 | 149 | 82.9 | 0.4444 | 0.6667 | 0.3333 | 6/9 |
| doc84.pdf | 6 | 67 | 93.0 | 0.4444 | 0.6667 | 0.3333 | 4/6 |
| doc73.pdf | 6 | 258 | 93.2 | 0.4583 | 0.6667 | 0.3333 | 4/6 |
| doc56.pdf | 5 | 260 | 77.2 | 0.4667 | 0.8 | 0.2 | 4/5 |
| doc46.pdf | 1 | 277 | 67.2 | 0.5 | 1.0 | 0.0 | 1/1 |
| doc65.pdf | 3 | 103 | 87.7 | 0.5 | 0.6667 | 0.3333 | 2/3 |
| doc7.pdf | 6 | 306 | 95.0 | 0.5 | 0.5 | 0.5 | 3/6 |
| doc10.pdf | 8 | 254 | 74.1 | 0.5208 | 0.75 | 0.375 | 6/8 |
| doc20.pdf | 9 | 170 | 79.7 | 0.5278 | 0.6667 | 0.4444 | 6/9 |
| doc28.pdf | 3 | 63 | 93.4 | 0.5278 | 1.0 | 0.3333 | 3/3 |
| doc4.pdf | 7 | 424 | 100.4 | 0.5405 | 0.8571 | 0.4286 | 6/7 |
| doc8.pdf | 7 | 320 | 91.2 | 0.5714 | 0.5714 | 0.5714 | 4/7 |
| doc12.pdf | 4 | 139 | 96.1 | 0.5833 | 1.0 | 0.25 | 4/4 |
| doc51.pdf | 8 | 217 | 88.8 | 0.5833 | 0.75 | 0.5 | 6/8 |
| doc88.pdf | 5 | 94 | 81.5 | 0.6 | 0.6 | 0.6 | 3/5 |
| doc92.pdf | 7 | 202 | 87.5 | 0.6071 | 0.7143 | 0.5714 | 5/7 |
| doc40.pdf | 6 | 153 | 83.6 | 0.6111 | 0.8333 | 0.5 | 5/6 |
| doc54.pdf | 6 | 63 | 93.4 | 0.6111 | 0.8333 | 0.5 | 5/6 |
| doc64.pdf | 6 | 149 | 83.3 | 0.6167 | 0.8333 | 0.5 | 5/6 |
| doc63.pdf | 4 | 55 | 87.7 | 0.625 | 0.75 | 0.5 | 3/4 |
| doc72.pdf | 5 | 153 | 83.6 | 0.64 | 0.8 | 0.6 | 4/5 |
| doc89.pdf | 9 | 157 | 87.3 | 0.6426 | 1.0 | 0.4444 | 9/9 |
| doc3.pdf | 8 | 188 | 80.4 | 0.6458 | 0.875 | 0.5 | 7/8 |
| doc74.pdf | 9 | 70 | 80.5 | 0.6481 | 0.8889 | 0.4444 | 8/9 |
| doc61.pdf | 5 | 196 | 71.3 | 0.65 | 1.0 | 0.4 | 5/5 |
| doc23.pdf | 9 | 25 | 91.8 | 0.6519 | 1.0 | 0.4444 | 9/9 |
| doc13.pdf | 5 | 140 | 91.1 | 0.6667 | 0.8 | 0.6 | 4/5 |
| doc50.pdf | 5 | 94 | 81.5 | 0.6667 | 0.8 | 0.6 | 4/5 |
| doc53.pdf | 3 | 131 | 86.9 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| doc55.pdf | 4 | 131 | 82.0 | 0.6667 | 1.0 | 0.5 | 4/4 |
| doc47.pdf | 4 | 164 | 82.7 | 0.675 | 1.0 | 0.5 | 4/4 |
| doc21.pdf | 8 | 182 | 97.7 | 0.6875 | 0.75 | 0.625 | 6/8 |
| doc35.pdf | 9 | 145 | 82.1 | 0.6944 | 0.7778 | 0.6667 | 7/9 |
| doc62.pdf | 8 | 145 | 82.1 | 0.6979 | 0.875 | 0.625 | 7/8 |
| doc16.pdf | 10 | 43 | 96.6 | 0.7 | 0.8 | 0.6 | 8/10 |
| doc9.pdf | 9 | 106 | 76.0 | 0.7037 | 0.8889 | 0.5556 | 8/9 |
| doc34.pdf | 7 | 196 | 71.3 | 0.7143 | 0.8571 | 0.5714 | 6/7 |
| doc42.pdf | 7 | 70 | 80.5 | 0.7143 | 0.8571 | 0.5714 | 6/7 |
| doc80.pdf | 7 | 164 | 82.7 | 0.7143 | 0.8571 | 0.5714 | 6/7 |
| doc77.pdf | 8 | 95 | 85.7 | 0.7188 | 0.875 | 0.625 | 7/8 |
| doc45.pdf | 9 | 293 | 65.5 | 0.7222 | 0.7778 | 0.6667 | 7/9 |
| doc57.pdf | 8 | 65 | 96.0 | 0.7292 | 0.875 | 0.625 | 7/8 |
| doc22.pdf | 8 | 122 | 75.6 | 0.7438 | 1.0 | 0.625 | 8/8 |
| doc26.pdf | 2 | 159 | 73.9 | 0.75 | 1.0 | 0.5 | 2/2 |
| doc37.pdf | 2 | 359 | 79.7 | 0.75 | 1.0 | 0.5 | 2/2 |
| doc38.pdf | 3 | 451 | 88.3 | 0.75 | 1.0 | 0.6667 | 3/3 |
| doc41.pdf | 8 | 258 | 93.2 | 0.75 | 0.875 | 0.625 | 7/8 |
| doc85.pdf | 4 | 57 | 88.2 | 0.75 | 0.75 | 0.75 | 3/4 |
| doc95.pdf | 5 | 59 | 105.9 | 0.75 | 1.0 | 0.6 | 5/5 |
| doc59.pdf | 6 | 126 | 81.8 | 0.7639 | 1.0 | 0.6667 | 6/6 |
| doc83.pdf | 8 | 353 | 87.7 | 0.7708 | 1.0 | 0.625 | 8/8 |
| doc79.pdf | 8 | 277 | 67.2 | 0.775 | 0.875 | 0.75 | 7/8 |
| doc43.pdf | 8 | 60 | 88.2 | 0.7812 | 1.0 | 0.625 | 8/8 |
| doc82.pdf | 8 | 80 | 96.4 | 0.7812 | 1.0 | 0.625 | 8/8 |
| doc25.pdf | 9 | 217 | 88.8 | 0.7815 | 1.0 | 0.6667 | 9/9 |
| doc86.pdf | 7 | 206 | 87.8 | 0.7857 | 0.8571 | 0.7143 | 6/7 |
| doc78.pdf | 6 | 293 | 65.5 | 0.7917 | 1.0 | 0.6667 | 6/6 |
| doc68.pdf | 5 | 124 | 86.2 | 0.8 | 0.8 | 0.8 | 4/5 |
| doc14.pdf | 6 | 323 | 89.0 | 0.8056 | 1.0 | 0.6667 | 6/6 |
| doc1.pdf | 8 | 82 | 86.4 | 0.8125 | 0.875 | 0.75 | 7/8 |
| doc29.pdf | 8 | 131 | 82.0 | 0.8125 | 0.875 | 0.75 | 7/8 |
| doc91.pdf | 4 | 91 | 84.4 | 0.8125 | 1.0 | 0.75 | 4/4 |
| doc24.pdf | 9 | 105 | 99.9 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| doc32.pdf | 6 | 126 | 81.8 | 0.8333 | 0.8333 | 0.8333 | 5/6 |
| doc44.pdf | 6 | 95 | 85.7 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc93.pdf | 6 | 47 | 97.8 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| doc48.pdf | 5 | 206 | 87.8 | 0.84 | 1.0 | 0.8 | 5/5 |
| doc6.pdf | 10 | 57 | 99.9 | 0.85 | 0.9 | 0.8 | 9/10 |
| doc87.pdf | 7 | 32 | 98.2 | 0.8571 | 1.0 | 0.7143 | 7/7 |
| doc5.pdf | 6 | 229 | 85.5 | 0.8667 | 1.0 | 0.8333 | 6/6 |
| doc60.pdf | 6 | 203 | 79.7 | 0.875 | 1.0 | 0.8333 | 6/6 |
| doc49.pdf | 7 | 32 | 98.2 | 0.9048 | 1.0 | 0.8571 | 7/7 |
| doc19.pdf | 9 | 43 | 117.5 | 0.9111 | 1.0 | 0.8889 | 9/9 |
| doc2.pdf | 6 | 182 | 97.7 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| doc18.pdf | 9 | 48 | 92.1 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| doc11.pdf | 7 | 28 | 84.5 | 1.0 | 1.0 | 1.0 | 7/7 |
| doc15.pdf | 2 | 18 | 103.5 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc52.pdf | 1 | 159 | 73.9 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc70.pdf | 2 | 451 | 88.3 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc75.pdf | 1 | 60 | 88.2 | 1.0 | 1.0 | 1.0 | 1/1 |
| doc76.pdf | 2 | 223 | 75.6 | 1.0 | 1.0 | 1.0 | 2/2 |
| doc94.pdf | 5 | 185 | 93.6 | 1.0 | 1.0 | 1.0 | 5/5 |
| doc96.pdf | 5 | 2436 | 85.0 | 1.0 | 1.0 | 1.0 | 5/5 |
