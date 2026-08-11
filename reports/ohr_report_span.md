# DocStruct retrieval baseline report

_Generated 2026-08-11 21:15 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: docstruct, docstruct_geo, pymupdf4llm, llamaindex_semantic, unstructured, llamaindex, langchain.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | docstruct **(ours)** | **0.7059** | [0.6923, 0.7194] | 0.7126 | 0.7867 | 0.6501 | 0.6224 | +0.0835 | 9080 | 316.8 | 2194.2 | 0.3217 | 4.33 | 0 |
| 2 | docstruct_geo | **0.7047** | [0.6909, 0.7185] | 0.7133 | 0.7889 | 0.6442 | 0.6188 | +0.0859 | 5810 | 306.0 | 2328.4 | 0.3027 | 2.21 | 0 |
| 3 | pymupdf4llm | **0.6992** | [0.685, 0.7131] | 0.7049 | 0.7718 | 0.6495 | 0.6234 | +0.0758 | 3756 | 424.6 | 2424.5 | 0.2884 | 1956.75 | 0 |
| 4 | llamaindex_semantic | **0.654** | [0.6399, 0.6679] | 0.6727 | 0.776 | 0.5725 | 0.5349 | +0.1191 | 3366 | 482.0 | 4697.7 | 0.1392 | 1105.96 | 0 |
| 5 | unstructured | **0.6539** | [0.6393, 0.6682] | 0.6705 | 0.7517 | 0.5852 | 0.5944 | +0.0595 | 18424 | 87.2 | 560.6 | 1.1664 | 1210.89 | 3 |
| 6 | llamaindex | **0.6483** | [0.6337, 0.6628] | 0.6571 | 0.7305 | 0.5936 | 0.5891 | +0.0592 | 5794 | 295.2 | 1430.1 | 0.4533 | 733.92 | 0 |
| 7 | langchain | **0.6406** | [0.6257, 0.6548] | 0.651 | 0.7195 | 0.5854 | 0.5884 | +0.0522 | 13877 | 128.5 | 637.6 | 1.0047 | 683.61 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| llamaindex_semantic | 1.0 | 1.0 |
| llamaindex | 1.0 | 1.0474 |
| langchain | 1.0 | 1.1005 |
| pymupdf4llm | 0.9674 | 1.0979 |
| docstruct_geo **(ours)** | 0.9638 | 1.132 |
| docstruct **(ours)** | 0.9632 | 1.8322 |
| unstructured | 0.9201 | 1.0567 |

- **Coverage** — fraction of the document's word *instances* that appear in some chunk. This is where silent loss shows up and nowhere else: dropped table rows, headings that end up in no chunk, skipped figures. Counted as a multiset, so dropping every repeat of a term is not scored as covered.
- **Duplication** — chunk words divided by document words. Above 1.0 means content is emitted more than once, inflating the index and letting two chunks split the evidence for one query. Overlap raises it deliberately, so read it as a cost next to coverage, not as a defect.

## Is the gap real? Paired bootstrap vs `docstruct`

Every tool answers the **same** questions, so the comparison is paired: one resample of question indices is applied to both tools, which cancels the between-question variance and isolates the difference between the chunkers. Positive Δ means `docstruct` is ahead. 10,000 resamples, seeded.

Reading two overlapping marginal CIs as "not significant" is the standard way to miss a consistent per-question difference — that is what this table exists to prevent, in both directions.

| vs | Metric | Δ (docstruct − tool) | 95% CI of Δ | p | n paired | Verdict |
|---|---|---|---|---|---|---|
| docstruct_geo | MRR | +0.0012 | [-0.0079, 0.0104] | 0.8031 | 3558 | not significant |
| docstruct_geo | NDCG | -0.0007 | [-0.0082, 0.007] | 0.8564 | 3558 | not significant |
| docstruct_geo | RECALL | -0.0022 | [-0.0096, 0.0051] | 0.5728 | 3558 | not significant |
| docstruct_geo | HIT1 | +0.0059 | [-0.0076, 0.0197] | 0.4107 | 3558 | not significant |
| pymupdf4llm | MRR | +0.0067 | [-0.0043, 0.0178] | 0.2321 | 3558 | not significant |
| pymupdf4llm | NDCG | +0.0077 | [-0.0023, 0.0177] | 0.1286 | 3558 | not significant |
| pymupdf4llm | RECALL | +0.0149 | [0.0042, 0.025] | 0.0063 | 3558 | **significant** |
| pymupdf4llm | HIT1 | +0.0006 | [-0.0138, 0.0149] | 0.9595 | 3558 | not significant |
| llamaindex_semantic | MRR | +0.0518 | [0.039, 0.0648] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | NDCG | +0.0399 | [0.0286, 0.0514] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | RECALL | +0.0107 | [-0.0011, 0.0222] | 0.0803 | 3558 | not significant |
| llamaindex_semantic | HIT1 | +0.0776 | [0.0604, 0.095] | 0.0001 | 3558 | **significant** |
| unstructured | MRR | +0.0491 | [0.0367, 0.0616] | 0.0001 | 3423 | **significant** |
| unstructured | NDCG | +0.039 | [0.028, 0.0503] | 0.0001 | 3423 | **significant** |
| unstructured | RECALL | +0.0316 | [0.0204, 0.0429] | 0.0001 | 3423 | **significant** |
| unstructured | HIT1 | +0.0622 | [0.0459, 0.0789] | 0.0001 | 3423 | **significant** |
| llamaindex | MRR | +0.0575 | [0.0462, 0.0694] | 0.0001 | 3558 | **significant** |
| llamaindex | NDCG | +0.0555 | [0.0452, 0.0663] | 0.0001 | 3558 | **significant** |
| llamaindex | RECALL | +0.0562 | [0.0453, 0.0677] | 0.0001 | 3558 | **significant** |
| llamaindex | HIT1 | +0.0565 | [0.0413, 0.0717] | 0.0001 | 3558 | **significant** |
| langchain | MRR | +0.0652 | [0.0535, 0.0774] | 0.0001 | 3558 | **significant** |
| langchain | NDCG | +0.0615 | [0.0509, 0.0725] | 0.0001 | 3558 | **significant** |
| langchain | RECALL | +0.0672 | [0.0562, 0.0787] | 0.0001 | 3558 | **significant** |
| langchain | HIT1 | +0.0644 | [0.0492, 0.0801] | 0.0001 | 3558 | **significant** |

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
| academic__2305.02437v3.pdf | 66 | 42 | 218.1 | 0.1894 | 0.197 | 0.1818 | 13/66 |
| manual__DSA-278777.pdf | 45 | 52 | 187.0 | 0.3278 | 0.3556 | 0.3111 | 16/45 |
| academic__2403.20330v2.pdf | 69 | 45 | 275.2 | 0.3536 | 0.4348 | 0.3043 | 30/69 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 869 | 238.9 | 0.3744 | 0.4638 | 0.3333 | 32/69 |
| academic__2405.14458v1.pdf | 64 | 41 | 317.5 | 0.3776 | 0.4688 | 0.3281 | 30/64 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 771 | 235.4 | 0.4185 | 0.5309 | 0.3457 | 43/81 |
| academic__2404.10198v2.pdf | 51 | 16 | 356.4 | 0.4235 | 0.5686 | 0.3529 | 29/51 |
| academic__2305.14160v4.pdf | 72 | 50 | 267.8 | 0.431 | 0.5 | 0.3889 | 36/72 |
| academic__2402.03216v4.pdf | 78 | 61 | 324.5 | 0.4459 | 0.5256 | 0.3846 | 41/78 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 728 | 231.2 | 0.4471 | 0.525 | 0.4 | 42/80 |
| finance__AMAZON_2017_10K.pdf | 75 | 192 | 368.0 | 0.4649 | 0.56 | 0.4 | 42/75 |
| finance__AMAZON_2019_10K.pdf | 81 | 188 | 355.3 | 0.4901 | 0.5062 | 0.4815 | 41/81 |
| academic__2409.01704v1.pdf | 60 | 33 | 298.7 | 0.5228 | 0.6 | 0.4667 | 36/60 |
| finance__AES_2022_10K.pdf | 78 | 582 | 400.5 | 0.5417 | 0.5897 | 0.5128 | 46/78 |
| finance__VERIZON_2021_10K.pdf | 84 | 378 | 304.5 | 0.555 | 0.6548 | 0.5 | 55/84 |
| finance__AMD_2022_10K.pdf | 62 | 247 | 379.9 | 0.575 | 0.6935 | 0.4839 | 43/62 |
| academic__2405.14831v1.pdf | 63 | 72 | 220.1 | 0.5751 | 0.7143 | 0.4762 | 45/63 |
| academic__2409.16145v1.pdf | 51 | 36 | 299.0 | 0.6029 | 0.7059 | 0.5098 | 36/51 |
| finance__3M_2023Q2_10Q.pdf | 63 | 234 | 377.3 | 0.6098 | 0.6984 | 0.5556 | 44/63 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 37 | 344.4 | 0.6259 | 0.6667 | 0.6 | 30/45 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 11 | 403.8 | 0.6736 | 0.8333 | 0.5833 | 10/12 |
| academic__2310.11511v1.pdf | 57 | 62 | 357.1 | 0.6804 | 0.7544 | 0.6316 | 43/57 |
| manual__Guide-for-international-students-web.pdf | 45 | 146 | 107.2 | 0.7007 | 0.8667 | 0.6 | 39/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 36 | 399.0 | 0.713 | 0.8222 | 0.6222 | 37/45 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 1309 | 331.3 | 0.7201 | 0.8571 | 0.6508 | 54/63 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 5 | 346.2 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| manual__obs-productdesc-en.pdf | 45 | 101 | 179.5 | 0.7333 | 0.8444 | 0.6444 | 38/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 42 | 293.5 | 0.7488 | 0.8333 | 0.6905 | 35/42 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 32 | 424.5 | 0.75 | 0.8222 | 0.6889 | 37/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 18 | 578.6 | 0.7528 | 0.9667 | 0.6333 | 29/30 |
| manual__dgx_a100.pdf | 45 | 164 | 200.2 | 0.7537 | 0.8667 | 0.6667 | 39/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 129 | 224.1 | 0.7667 | 0.8667 | 0.6889 | 39/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 38 | 462.7 | 0.787 | 0.8444 | 0.7556 | 38/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 97 | 466.9 | 0.7944 | 0.8667 | 0.7556 | 39/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 81 | 382.4 | 0.8074 | 0.8444 | 0.7778 | 38/45 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 76 | 159.0 | 0.8096 | 0.9333 | 0.7333 | 42/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 63 | 536.2 | 0.8119 | 0.9111 | 0.7556 | 41/45 |
| manual__Macbook_air.pdf | 45 | 51 | 247.6 | 0.813 | 0.8667 | 0.7778 | 39/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 170 | 261.6 | 0.8259 | 0.8889 | 0.7778 | 40/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 48 | 602.9 | 0.8278 | 0.9778 | 0.7333 | 44/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 86 | 315.0 | 0.8287 | 0.9167 | 0.75 | 33/36 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 12 | 532.5 | 0.8333 | 0.9444 | 0.7778 | 17/18 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 6 | 337.5 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 4 | 406.8 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 3 | 297.0 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 94 | 437.2 | 0.8341 | 0.9778 | 0.7556 | 44/45 |
| manual__mi_phone.pdf | 45 | 35 | 233.1 | 0.8426 | 0.9556 | 0.7556 | 43/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 31 | 379.7 | 0.8481 | 0.9333 | 0.7778 | 42/45 |
| manual__owners-manual-2170416.pdf | 45 | 52 | 246.2 | 0.8481 | 0.9111 | 0.8 | 41/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 17 | 492.4 | 0.851 | 0.9697 | 0.7576 | 32/33 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 58 | 387.7 | 0.8607 | 0.9333 | 0.8222 | 42/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 15 | 386.4 | 0.8611 | 1.0 | 0.75 | 12/12 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 8 | 482.9 | 0.8611 | 1.0 | 0.75 | 12/12 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 7 | 489.3 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 11 | 339.5 | 0.8743 | 1.0 | 0.8333 | 24/24 |
| manual__honor_watch_gs_pro.pdf | 45 | 68 | 298.8 | 0.8852 | 0.9556 | 0.8222 | 43/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 11 | 544.2 | 0.8889 | 0.9333 | 0.8667 | 14/15 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 7 | 469.1 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 7 | 374.6 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| manual__nova_y70.pdf | 46 | 77 | 239.0 | 0.8913 | 0.9348 | 0.8478 | 43/46 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 15 | 535.3 | 0.8929 | 1.0 | 0.8095 | 21/21 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 42 | 532.3 | 0.8963 | 0.9556 | 0.8444 | 43/45 |
| manual__t480_ug_en.pdf | 45 | 244 | 364.8 | 0.8963 | 0.9778 | 0.8444 | 44/45 |
| manual__watch_d.pdf | 45 | 44 | 248.7 | 0.8963 | 0.9556 | 0.8444 | 43/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 23 | 332.7 | 0.8989 | 0.9778 | 0.8444 | 44/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 122 | 76.6 | 0.9056 | 0.9778 | 0.8444 | 44/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 11 | 560.4 | 0.9167 | 1.0 | 0.8333 | 12/12 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 12 | 476.4 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 11 | 514.1 | 0.9167 | 1.0 | 0.8333 | 24/24 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 3 | 397.3 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 6 | 546.7 | 0.9167 | 1.0 | 0.8889 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 5 | 583.0 | 0.9167 | 1.0 | 0.8333 | 12/12 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 62 | 507.4 | 0.9185 | 0.9778 | 0.8667 | 44/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 109 | 550.9 | 0.9185 | 0.9778 | 0.8667 | 44/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 24 | 505.8 | 0.9198 | 1.0 | 0.8519 | 27/27 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 88 | 473.8 | 0.9296 | 0.9556 | 0.9111 | 43/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 48 | 344.1 | 0.9316 | 0.9744 | 0.8974 | 38/39 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 47 | 636.1 | 0.9389 | 1.0 | 0.9111 | 45/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 7 | 558.1 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 7 | 511.6 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 6 | 509.8 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 38 | 583.5 | 0.9556 | 0.9778 | 0.9333 | 44/45 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 29 | 606.2 | 0.9556 | 1.0 | 0.9111 | 45/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 22 | 496.4 | 0.9611 | 1.0 | 0.9333 | 30/30 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 15 | 550.3 | 0.963 | 1.0 | 0.9444 | 18/18 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 21 | 619.6 | 0.9646 | 1.0 | 0.9394 | 33/33 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 7 | 350.0 | 0.9667 | 1.0 | 0.9333 | 15/15 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 8 | 586.9 | 0.9722 | 1.0 | 0.9444 | 18/18 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 6 | 545.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 6 | 564.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 13 | 430.4 | 1.0 | 1.0 | 1.0 | 15/15 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 574.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 8 | 634.4 | 1.0 | 1.0 | 1.0 | 21/21 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 427.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 5 | 270.0 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: docstruct_geo (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2305.02437v3.pdf | 66 | 28 | 220.8 | 0.1591 | 0.197 | 0.1364 | 13/66 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 546 | 217.8 | 0.285 | 0.3768 | 0.2319 | 26/69 |
| academic__2403.20330v2.pdf | 69 | 48 | 231.4 | 0.3391 | 0.4638 | 0.2464 | 32/69 |
| manual__DSA-278777.pdf | 45 | 23 | 287.2 | 0.363 | 0.4444 | 0.3111 | 20/45 |
| academic__2402.03216v4.pdf | 78 | 30 | 377.1 | 0.3876 | 0.5128 | 0.3077 | 40/78 |
| academic__2405.14458v1.pdf | 64 | 26 | 302.6 | 0.3906 | 0.4688 | 0.3281 | 30/64 |
| finance__AMAZON_2017_10K.pdf | 75 | 81 | 556.1 | 0.4044 | 0.48 | 0.3467 | 36/75 |
| finance__AMAZON_2019_10K.pdf | 81 | 82 | 516.4 | 0.4486 | 0.4938 | 0.4074 | 40/81 |
| academic__2305.14160v4.pdf | 72 | 26 | 361.5 | 0.4502 | 0.5556 | 0.375 | 40/72 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 440 | 219.2 | 0.4633 | 0.6 | 0.3875 | 48/80 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 490 | 212.4 | 0.471 | 0.5802 | 0.4074 | 47/81 |
| academic__2409.01704v1.pdf | 60 | 26 | 268.2 | 0.5097 | 0.5667 | 0.4667 | 34/60 |
| finance__AES_2022_10K.pdf | 78 | 418 | 351.0 | 0.5137 | 0.6154 | 0.4615 | 48/78 |
| academic__2404.10198v2.pdf | 51 | 15 | 293.3 | 0.5556 | 0.5882 | 0.5294 | 30/51 |
| finance__VERIZON_2021_10K.pdf | 84 | 302 | 258.2 | 0.5599 | 0.6429 | 0.5 | 54/84 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 8 | 331.5 | 0.5833 | 0.8333 | 0.4167 | 10/12 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 18 | 463.1 | 0.5841 | 0.6667 | 0.5333 | 30/45 |
| academic__2405.14831v1.pdf | 63 | 53 | 220.2 | 0.5884 | 0.7143 | 0.5079 | 45/63 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 22 | 427.5 | 0.5963 | 0.6889 | 0.5111 | 31/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 155 | 369.5 | 0.6037 | 0.6508 | 0.5714 | 41/63 |
| finance__AMD_2022_10K.pdf | 62 | 182 | 354.2 | 0.629 | 0.7097 | 0.5806 | 44/62 |
| academic__2409.16145v1.pdf | 51 | 22 | 367.1 | 0.6536 | 0.7059 | 0.6078 | 36/51 |
| academic__2310.11511v1.pdf | 57 | 42 | 389.7 | 0.6629 | 0.7544 | 0.5965 | 43/57 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 874 | 287.9 | 0.6839 | 0.8413 | 0.5556 | 53/63 |
| manual__obs-productdesc-en.pdf | 45 | 97 | 145.9 | 0.713 | 0.8 | 0.6444 | 36/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 28 | 356.0 | 0.7467 | 0.8444 | 0.6889 | 38/45 |
| manual__dgx_a100.pdf | 45 | 155 | 141.6 | 0.7656 | 0.9111 | 0.6667 | 41/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 22 | 405.2 | 0.7674 | 0.8444 | 0.7111 | 38/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 29 | 669.2 | 0.7711 | 0.8444 | 0.7111 | 38/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 13 | 508.8 | 0.7747 | 0.963 | 0.6667 | 26/27 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 20 | 363.8 | 0.7778 | 0.881 | 0.6905 | 37/42 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 3 | 475.7 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| manual__Guide-for-international-students-web.pdf | 45 | 131 | 83.5 | 0.7852 | 0.8667 | 0.7111 | 39/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 65 | 266.2 | 0.7907 | 0.8444 | 0.7556 | 38/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 90 | 482.8 | 0.7933 | 0.8667 | 0.7556 | 39/45 |
| manual__Macbook_air.pdf | 45 | 44 | 263.2 | 0.8037 | 0.8667 | 0.7556 | 39/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 60 | 375.9 | 0.8081 | 0.8667 | 0.7778 | 39/45 |
| manual__owners-manual-2170416.pdf | 45 | 28 | 278.3 | 0.81 | 0.9333 | 0.7111 | 42/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 109 | 237.7 | 0.8193 | 0.9111 | 0.7556 | 41/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 6 | 551.8 | 0.8222 | 1.0 | 0.6667 | 15/15 |
| manual__t480_ug_en.pdf | 45 | 123 | 381.2 | 0.8293 | 0.9333 | 0.7778 | 42/45 |
| manual__mi_phone.pdf | 45 | 22 | 261.8 | 0.8304 | 0.9333 | 0.7556 | 42/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 5 | 568.4 | 0.8333 | 1.0 | 0.6667 | 18/18 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 300.7 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 3 | 656.7 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 2 | 486.5 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 40 | 445.9 | 0.8333 | 0.8889 | 0.8056 | 32/36 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 3 | 598.7 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 8 | 299.2 | 0.8403 | 1.0 | 0.7083 | 24/24 |
| manual__nova_y70.pdf | 46 | 46 | 245.8 | 0.8551 | 0.9348 | 0.7826 | 43/46 |
| manual__2021-Apple-Catalog.pdf | 45 | 26 | 234.4 | 0.8619 | 0.9556 | 0.8 | 43/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 5 | 690.0 | 0.8667 | 0.9333 | 0.8 | 14/15 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 3 | 412.0 | 0.8667 | 1.0 | 0.7333 | 15/15 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 20 | 771.3 | 0.8667 | 0.9778 | 0.7778 | 44/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 24 | 505.9 | 0.8722 | 0.9778 | 0.8 | 44/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 2 | 402.5 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 3 | 620.0 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 67 | 126.6 | 0.8926 | 0.9333 | 0.8667 | 42/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 28 | 367.4 | 0.8963 | 1.0 | 0.8222 | 45/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 26 | 702.5 | 0.897 | 1.0 | 0.8222 | 45/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 8 | 455.1 | 0.9028 | 1.0 | 0.8333 | 12/12 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 47 | 531.3 | 0.9056 | 1.0 | 0.8444 | 45/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 10 | 417.6 | 0.9056 | 1.0 | 0.8444 | 45/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 49 | 690.1 | 0.9074 | 0.9778 | 0.8667 | 44/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 39 | 311.7 | 0.91 | 0.9778 | 0.8667 | 44/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 7 | 572.7 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 320.5 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 359.2 | 0.9167 | 1.0 | 0.8333 | 12/12 |
| manual__watch_d.pdf | 45 | 35 | 201.8 | 0.9185 | 1.0 | 0.8444 | 45/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 45 | 567.9 | 0.92 | 0.9556 | 0.9111 | 43/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 7 | 648.4 | 0.9206 | 1.0 | 0.8571 | 21/21 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 9 | 521.8 | 0.9242 | 0.9697 | 0.8788 | 32/33 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 28 | 550.3 | 0.9259 | 0.9778 | 0.8889 | 44/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 9 | 617.6 | 0.9278 | 1.0 | 0.8667 | 30/30 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 20 | 572.5 | 0.9296 | 0.9778 | 0.8889 | 44/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 6 | 631.7 | 0.9306 | 1.0 | 0.875 | 24/24 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 35 | 332.6 | 0.9423 | 1.0 | 0.8974 | 39/39 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 4 | 477.8 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 36 | 618.4 | 0.9444 | 1.0 | 0.9111 | 45/45 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 3 | 572.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 5 | 578.4 | 0.9583 | 1.0 | 0.9167 | 12/12 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 12 | 614.2 | 0.9596 | 1.0 | 0.9394 | 33/33 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 15 | 641.1 | 0.9667 | 1.0 | 0.9333 | 45/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 4 | 619.5 | 0.9722 | 1.0 | 0.9444 | 18/18 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 6 | 738.2 | 0.9722 | 1.0 | 0.9444 | 18/18 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 5 | 564.6 | 0.9762 | 1.0 | 0.9524 | 21/21 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 17 | 476.3 | 0.9778 | 1.0 | 0.9667 | 30/30 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 4 | 462.5 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 1 | 820.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 108.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 4 | 264.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 711.2 | 1.0 | 1.0 | 1.0 | 12/12 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 5 | 408.8 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 2 | 253.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 353.4 | 1.0 | 1.0 | 1.0 | 9/9 |


### Per-doc breakdown: pymupdf4llm (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| manual__DSA-278777.pdf | 45 | 21 | 259.4 | 0.3044 | 0.3333 | 0.2889 | 15/45 |
| academic__2403.20330v2.pdf | 69 | 20 | 441.1 | 0.3237 | 0.4638 | 0.2319 | 32/69 |
| finance__AMAZON_2019_10K.pdf | 81 | 83 | 489.0 | 0.3344 | 0.3704 | 0.3086 | 30/81 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 217 | 428.8 | 0.3536 | 0.4783 | 0.2899 | 33/69 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 179 | 435.3 | 0.365 | 0.5125 | 0.2875 | 41/80 |
| finance__AMAZON_2017_10K.pdf | 75 | 85 | 506.4 | 0.3809 | 0.4533 | 0.3467 | 34/75 |
| academic__2305.02437v3.pdf | 66 | 20 | 438.9 | 0.403 | 0.4697 | 0.3636 | 31/66 |
| academic__2402.03216v4.pdf | 78 | 18 | 517.4 | 0.4316 | 0.4872 | 0.3846 | 38/78 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 197 | 408.6 | 0.4387 | 0.4938 | 0.4074 | 40/81 |
| academic__2305.14160v4.pdf | 72 | 16 | 527.4 | 0.4692 | 0.5556 | 0.4167 | 40/72 |
| finance__AES_2022_10K.pdf | 78 | 255 | 515.4 | 0.4791 | 0.5513 | 0.4231 | 43/78 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 25 | 346.4 | 0.4963 | 0.5778 | 0.4444 | 26/45 |
| academic__2405.14458v1.pdf | 64 | 18 | 535.7 | 0.5096 | 0.6094 | 0.4531 | 39/64 |
| finance__AMD_2022_10K.pdf | 62 | 121 | 507.0 | 0.5685 | 0.5968 | 0.5484 | 37/62 |
| academic__2404.10198v2.pdf | 51 | 13 | 414.2 | 0.5719 | 0.6471 | 0.5098 | 33/51 |
| academic__2409.01704v1.pdf | 60 | 19 | 443.9 | 0.575 | 0.7 | 0.5 | 42/60 |
| manual__obs-productdesc-en.pdf | 45 | 65 | 171.8 | 0.5778 | 0.6444 | 0.5333 | 29/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 92 | 552.5 | 0.5926 | 0.619 | 0.5714 | 39/63 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 40 | 186.2 | 0.6 | 0.6 | 0.6 | 27/45 |
| finance__VERIZON_2021_10K.pdf | 84 | 120 | 582.6 | 0.602 | 0.6786 | 0.5714 | 57/84 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 382 | 569.4 | 0.6151 | 0.7143 | 0.5397 | 45/63 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 4 | 654.5 | 0.625 | 0.6667 | 0.5833 | 8/12 |
| academic__2310.11511v1.pdf | 57 | 30 | 540.0 | 0.6348 | 0.7719 | 0.5789 | 44/57 |
| academic__2409.16145v1.pdf | 51 | 22 | 428.4 | 0.6379 | 0.7255 | 0.5882 | 37/51 |
| academic__2405.14831v1.pdf | 63 | 28 | 476.8 | 0.6534 | 0.7302 | 0.5873 | 46/63 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 25 | 353.7 | 0.7063 | 0.8222 | 0.6222 | 37/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 75 | 125.9 | 0.7248 | 0.9111 | 0.5778 | 41/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 39 | 387.5 | 0.7407 | 0.8 | 0.6889 | 36/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 108 | 231.8 | 0.7519 | 0.8 | 0.7111 | 36/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 40 | 289.1 | 0.7556 | 0.8222 | 0.7111 | 37/45 |
| manual__Macbook_air.pdf | 45 | 71 | 165.8 | 0.76 | 0.8444 | 0.6889 | 38/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 16 | 437.4 | 0.7619 | 0.8095 | 0.7143 | 34/42 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 31 | 622.5 | 0.7833 | 0.8667 | 0.7333 | 39/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 9 | 738.8 | 0.7852 | 1.0 | 0.6667 | 27/27 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 91 | 468.9 | 0.7896 | 0.8444 | 0.7556 | 38/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 19 | 488.4 | 0.7907 | 0.8667 | 0.7556 | 39/45 |
| manual__t480_ug_en.pdf | 45 | 168 | 275.9 | 0.7915 | 0.9111 | 0.7111 | 41/45 |
| manual__owners-manual-2170416.pdf | 45 | 32 | 234.6 | 0.8037 | 0.8667 | 0.7556 | 39/45 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 17 | 358.9 | 0.8074 | 0.8667 | 0.7556 | 39/45 |
| manual__nova_y70.pdf | 46 | 45 | 261.9 | 0.8261 | 0.8696 | 0.7826 | 40/46 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 7 | 406.7 | 0.8287 | 1.0 | 0.7222 | 18/18 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 305.7 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 3 | 456.0 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 14 | 342.4 | 0.8359 | 0.9394 | 0.7576 | 31/33 |
| manual__dgx_a100.pdf | 45 | 120 | 175.7 | 0.85 | 0.9111 | 0.8 | 41/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 17 | 245.8 | 0.8519 | 0.9111 | 0.8 | 41/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 53 | 459.2 | 0.8556 | 0.9778 | 0.7778 | 44/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 49 | 362.1 | 0.8634 | 0.9167 | 0.8333 | 33/36 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 30 | 346.8 | 0.8711 | 0.9778 | 0.7778 | 44/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 21 | 735.1 | 0.8785 | 0.9778 | 0.8 | 44/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 29 | 420.7 | 0.8833 | 0.9556 | 0.8444 | 43/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 18 | 323.2 | 0.8846 | 0.9231 | 0.8462 | 36/39 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 3 | 656.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 268.7 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 3 | 549.0 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 3 | 551.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| manual__mi_phone.pdf | 45 | 37 | 149.2 | 0.8889 | 0.9333 | 0.8667 | 42/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 34 | 452.0 | 0.8896 | 0.9778 | 0.8222 | 44/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 10 | 551.4 | 0.8917 | 0.9667 | 0.8333 | 29/30 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 15 | 762.9 | 0.8944 | 0.9111 | 0.8889 | 41/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 5 | 694.2 | 0.9 | 0.9333 | 0.8667 | 14/15 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 5 | 716.2 | 0.9028 | 1.0 | 0.8333 | 12/12 |
| manual__2021-Apple-Catalog.pdf | 45 | 55 | 114.8 | 0.9037 | 0.9556 | 0.8667 | 43/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 7 | 647.3 | 0.9048 | 1.0 | 0.8095 | 21/21 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 23 | 795.3 | 0.9093 | 0.9778 | 0.8667 | 44/45 |
| manual__watch_d.pdf | 45 | 27 | 256.2 | 0.913 | 0.9778 | 0.8667 | 44/45 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 2 | 487.0 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 3 | 217.0 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 47 | 535.9 | 0.9185 | 0.9556 | 0.8889 | 43/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 9 | 273.8 | 0.9271 | 1.0 | 0.875 | 24/24 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 45 | 750.3 | 0.9296 | 0.9556 | 0.9111 | 43/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 5 | 655.8 | 0.9333 | 1.0 | 0.8667 | 15/15 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 14 | 523.6 | 0.9343 | 1.0 | 0.8788 | 33/33 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 16 | 607.6 | 0.9352 | 1.0 | 0.8889 | 45/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 16 | 711.9 | 0.9389 | 0.9778 | 0.9111 | 44/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 6 | 419.7 | 0.9444 | 1.0 | 0.8889 | 18/18 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 6 | 667.5 | 0.9444 | 1.0 | 0.8889 | 18/18 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 3 | 574.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 712.2 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 3 | 682.7 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 3 | 614.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 367.0 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| manual__honor_watch_gs_pro.pdf | 45 | 42 | 295.5 | 0.9444 | 0.9556 | 0.9333 | 43/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 11 | 407.5 | 0.95 | 1.0 | 0.9 | 30/30 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 6 | 741.2 | 0.963 | 1.0 | 0.9444 | 18/18 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 8 | 343.4 | 0.9683 | 1.0 | 0.9524 | 21/21 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 3 | 640.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 4 | 465.8 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 5 | 579.2 | 1.0 | 1.0 | 1.0 | 12/12 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 1 | 824.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 8 | 473.4 | 1.0 | 1.0 | 1.0 | 24/24 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 217.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 264.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 5 | 253.2 | 1.0 | 1.0 | 1.0 | 15/15 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 4 | 132.8 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: llamaindex_semantic (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2305.02437v3.pdf | 66 | 29 | 123.7 | 0.1391 | 0.2424 | 0.0909 | 16/66 |
| academic__2405.14458v1.pdf | 64 | 30 | 101.4 | 0.175 | 0.2188 | 0.1562 | 14/64 |
| academic__2402.03216v4.pdf | 78 | 27 | 230.8 | 0.1795 | 0.2051 | 0.1667 | 16/78 |
| academic__2305.14160v4.pdf | 72 | 21 | 206.9 | 0.2069 | 0.3056 | 0.1528 | 22/72 |
| academic__2403.20330v2.pdf | 69 | 25 | 265.0 | 0.2111 | 0.3188 | 0.1304 | 22/69 |
| academic__2404.10198v2.pdf | 51 | 15 | 86.2 | 0.2647 | 0.3922 | 0.1569 | 20/51 |
| manual__DSA-278777.pdf | 45 | 10 | 425.0 | 0.2852 | 0.3556 | 0.2222 | 16/45 |
| academic__2409.01704v1.pdf | 60 | 18 | 174.3 | 0.3067 | 0.6 | 0.1333 | 36/60 |
| manual__t480_ug_en.pdf | 45 | 245 | 187.5 | 0.3204 | 0.6667 | 0.1778 | 30/45 |
| academic__2405.14831v1.pdf | 63 | 38 | 145.5 | 0.3471 | 0.4444 | 0.2698 | 28/63 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 14 | 591.7 | 0.3696 | 0.5778 | 0.2667 | 26/45 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 141 | 691.1 | 0.4261 | 0.5802 | 0.3333 | 47/81 |
| finance__AES_2022_10K.pdf | 78 | 231 | 607.5 | 0.4752 | 0.5897 | 0.4103 | 46/78 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 164 | 678.8 | 0.4829 | 0.6377 | 0.3768 | 44/69 |
| finance__AMAZON_2019_10K.pdf | 81 | 73 | 578.0 | 0.4877 | 0.6296 | 0.4198 | 51/81 |
| academic__2310.11511v1.pdf | 57 | 56 | 92.5 | 0.4944 | 0.614 | 0.4211 | 35/57 |
| finance__VERIZON_2021_10K.pdf | 84 | 123 | 592.8 | 0.5252 | 0.6429 | 0.4405 | 54/84 |
| finance__AMAZON_2017_10K.pdf | 75 | 77 | 581.5 | 0.5298 | 0.6933 | 0.4267 | 52/75 |
| finance__AMD_2022_10K.pdf | 62 | 115 | 545.3 | 0.5874 | 0.6613 | 0.5323 | 41/62 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 6 | 437.0 | 0.5972 | 0.8333 | 0.4167 | 10/12 |
| manual__obs-productdesc-en.pdf | 45 | 36 | 365.5 | 0.617 | 0.7778 | 0.5333 | 35/45 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 138 | 655.0 | 0.6185 | 0.7875 | 0.5125 | 63/80 |
| finance__3M_2023Q2_10Q.pdf | 63 | 85 | 632.3 | 0.6254 | 0.6825 | 0.5873 | 43/63 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 436 | 535.7 | 0.641 | 0.8254 | 0.5397 | 52/63 |
| academic__2409.16145v1.pdf | 51 | 28 | 287.5 | 0.6755 | 0.8431 | 0.5882 | 43/51 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 5 | 835.4 | 0.6759 | 1.0 | 0.4222 | 45/45 |
| manual__nova_y70.pdf | 46 | 32 | 353.3 | 0.7036 | 0.8043 | 0.6304 | 37/46 |
| manual__Macbook_air.pdf | 45 | 29 | 396.3 | 0.71 | 0.8444 | 0.6444 | 38/45 |
| manual__dgx_a100.pdf | 45 | 43 | 477.9 | 0.7185 | 0.8667 | 0.6222 | 39/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 4 | 225.0 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 8 | 567.4 | 0.7317 | 1.0 | 0.5714 | 21/21 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 14 | 660.2 | 0.747 | 0.8889 | 0.6444 | 40/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 21 | 427.2 | 0.7474 | 0.9111 | 0.6444 | 41/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 11 | 814.9 | 0.7589 | 0.9778 | 0.6 | 44/45 |
| manual__owners-manual-2170416.pdf | 45 | 39 | 195.6 | 0.7633 | 0.8667 | 0.7333 | 39/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 183 | 118.2 | 0.7663 | 0.8889 | 0.6889 | 40/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 5 | 382.2 | 0.7778 | 1.0 | 0.5556 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 4 | 449.0 | 0.7778 | 1.0 | 0.5556 | 9/9 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 28 | 226.9 | 0.7815 | 0.9111 | 0.6667 | 41/45 |
| manual__mi_phone.pdf | 45 | 19 | 289.8 | 0.7833 | 0.9111 | 0.6889 | 41/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 8 | 355.2 | 0.787 | 1.0 | 0.6111 | 18/18 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 5 | 394.0 | 0.787 | 1.0 | 0.6667 | 9/9 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 8 | 431.2 | 0.7889 | 0.9333 | 0.6667 | 14/15 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 10 | 700.8 | 0.7917 | 0.881 | 0.7381 | 37/42 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 12 | 747.2 | 0.7944 | 0.9111 | 0.6889 | 41/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 33 | 469.3 | 0.797 | 0.9111 | 0.7333 | 41/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 22 | 805.8 | 0.8009 | 0.8889 | 0.75 | 32/36 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 36 | 427.8 | 0.8074 | 0.9333 | 0.7111 | 42/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 28 | 690.8 | 0.81 | 0.9333 | 0.7111 | 42/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 7 | 670.4 | 0.8258 | 1.0 | 0.697 | 33/33 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 7 | 413.1 | 0.8333 | 1.0 | 0.6667 | 12/12 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 273.3 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 4 | 341.8 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 326.6 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| manual__2021-Apple-Catalog.pdf | 45 | 6 | 969.3 | 0.8333 | 0.9556 | 0.7333 | 43/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 15 | 440.9 | 0.8346 | 0.963 | 0.7407 | 26/27 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 34 | 721.4 | 0.8359 | 0.9778 | 0.7333 | 44/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 8 | 553.6 | 0.838 | 0.9444 | 0.7778 | 17/18 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 16 | 755.8 | 0.8385 | 1.0 | 0.7333 | 45/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 17 | 665.5 | 0.8426 | 0.8889 | 0.8222 | 40/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 5 | 652.0 | 0.8444 | 1.0 | 0.7333 | 15/15 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 359.2 | 0.8542 | 1.0 | 0.75 | 12/12 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 6 | 668.2 | 0.8611 | 1.0 | 0.7222 | 18/18 |
| manual__watch_d.pdf | 45 | 20 | 344.1 | 0.8648 | 0.9333 | 0.8222 | 42/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 9 | 387.2 | 0.875 | 1.0 | 0.75 | 12/12 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 711.2 | 0.875 | 1.0 | 0.75 | 12/12 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 10 | 555.8 | 0.8806 | 0.9667 | 0.8333 | 29/30 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 39 | 643.7 | 0.8833 | 0.9556 | 0.8444 | 43/45 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 5 | 343.2 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 4 | 243.2 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 268.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 3 | 412.0 | 0.8889 | 1.0 | 0.8 | 15/15 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 5 | 756.8 | 0.8958 | 1.0 | 0.7917 | 24/24 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 27 | 676.5 | 0.8981 | 0.9778 | 0.8444 | 44/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 11 | 528.7 | 0.9017 | 0.9744 | 0.8462 | 38/39 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 22 | 513.5 | 0.9037 | 1.0 | 0.8222 | 45/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 22 | 516.7 | 0.9056 | 1.0 | 0.8222 | 45/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 49 | 872.3 | 0.9119 | 0.9556 | 0.8889 | 43/45 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 9 | 812.0 | 0.9192 | 1.0 | 0.8485 | 33/33 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 67 | 503.1 | 0.9222 | 0.9778 | 0.8667 | 44/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 36 | 334.1 | 0.9241 | 0.9778 | 0.8889 | 44/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 20 | 761.4 | 0.9296 | 0.9778 | 0.8889 | 44/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 6 | 399.0 | 0.9306 | 1.0 | 0.875 | 24/24 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 13 | 738.3 | 0.9315 | 1.0 | 0.8889 | 45/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 18 | 567.4 | 0.9352 | 1.0 | 0.8889 | 45/45 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 5 | 370.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 5 | 408.8 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 4 | 462.5 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 7 | 634.4 | 0.95 | 1.0 | 0.9 | 30/30 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 4 | 619.5 | 0.9722 | 1.0 | 0.9444 | 18/18 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 5 | 553.0 | 1.0 | 1.0 | 1.0 | 21/21 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 108.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 315.0 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 263.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 3 | 168.7 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: unstructured (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 1451 | 76.5 | 0.3072 | 0.3623 | 0.2754 | 25/69 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 1204 | 74.8 | 0.3171 | 0.4375 | 0.2375 | 35/80 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 1308 | 74.3 | 0.3819 | 0.4815 | 0.3333 | 39/81 |
| academic__2305.02437v3.pdf | 66 | 120 | 74.1 | 0.399 | 0.4545 | 0.3485 | 30/66 |
| finance__AMAZON_2019_10K.pdf | 81 | 537 | 78.4 | 0.4016 | 0.4444 | 0.3704 | 36/81 |
| manual__DSA-278777.pdf | 45 | 93 | 65.9 | 0.4111 | 0.4444 | 0.3778 | 20/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 555 | 80.6 | 0.436 | 0.48 | 0.4 | 36/75 |
| finance__AES_2022_10K.pdf | 78 | 1481 | 94.5 | 0.4417 | 0.5385 | 0.3718 | 42/78 |
| academic__2305.14160v4.pdf | 72 | 98 | 87.2 | 0.4419 | 0.5278 | 0.3889 | 38/72 |
| academic__2402.03216v4.pdf | 78 | 129 | 82.5 | 0.453 | 0.4744 | 0.4359 | 37/78 |
| academic__2403.20330v2.pdf | 69 | 124 | 81.7 | 0.4824 | 0.5507 | 0.4348 | 38/69 |
| academic__2409.01704v1.pdf | 60 | 91 | 95.0 | 0.5006 | 0.6333 | 0.4167 | 38/60 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 2624 | 88.6 | 0.5222 | 0.619 | 0.4603 | 39/63 |
| finance__VERIZON_2021_10K.pdf | 84 | 819 | 88.2 | 0.5351 | 0.631 | 0.4762 | 53/84 |
| academic__2409.16145v1.pdf | 51 | 112 | 85.0 | 0.5441 | 0.6471 | 0.4706 | 33/51 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 114 | 55.4 | 0.5626 | 0.6889 | 0.4889 | 31/45 |
| academic__2404.10198v2.pdf | 51 | 67 | 80.3 | 0.5719 | 0.6471 | 0.5098 | 33/51 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 98 | 65.9 | 0.5852 | 0.6 | 0.5778 | 27/45 |
| academic__2405.14831v1.pdf | 63 | 157 | 86.2 | 0.5865 | 0.7937 | 0.4286 | 50/63 |
| finance__3M_2023Q2_10Q.pdf | 63 | 675 | 79.6 | 0.5918 | 0.6667 | 0.5397 | 42/63 |
| finance__AMD_2022_10K.pdf | 62 | 700 | 89.6 | 0.5968 | 0.6452 | 0.5645 | 40/62 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 25 | 101.4 | 0.5972 | 0.8333 | 0.4167 | 10/12 |
| academic__2405.14458v1.pdf | 64 | 107 | 90.7 | 0.6052 | 0.6875 | 0.5469 | 44/64 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 9 | 85.9 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 5 | 92.4 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 310 | 49.5 | 0.6219 | 0.8222 | 0.4889 | 37/45 |
| academic__2310.11511v1.pdf | 57 | 179 | 89.8 | 0.6254 | 0.7193 | 0.5614 | 41/57 |
| manual__obs-productdesc-en.pdf | 45 | 216 | 60.6 | 0.6526 | 0.7556 | 0.5778 | 34/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 54 | 114.7 | 0.6643 | 0.7143 | 0.6429 | 30/42 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 81 | 109.7 | 0.6785 | 0.8 | 0.6 | 36/45 |
| manual__owners-manual-2170416.pdf | 45 | 140 | 54.6 | 0.6841 | 0.8222 | 0.6 | 37/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 211 | 111.4 | 0.687 | 0.8444 | 0.5556 | 38/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 247 | 128.2 | 0.6893 | 0.8444 | 0.6 | 38/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 127 | 49.2 | 0.6963 | 0.8444 | 0.5778 | 38/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 145 | 124.2 | 0.6996 | 0.8667 | 0.5778 | 39/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 19 | 148.7 | 0.7042 | 0.9167 | 0.5833 | 11/12 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 70 | 116.7 | 0.7111 | 0.7556 | 0.6667 | 34/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 30 | 89.7 | 0.713 | 0.8333 | 0.6111 | 15/18 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 100.9 | 0.7222 | 0.7778 | 0.6667 | 7/9 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 35 | 106.3 | 0.723 | 0.8667 | 0.6444 | 39/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 30 | 107.2 | 0.7361 | 1.0 | 0.5833 | 12/12 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 12 | 141.1 | 0.7407 | 1.0 | 0.5556 | 9/9 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 108 | 104.4 | 0.7407 | 0.8667 | 0.6444 | 39/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 86 | 95.7 | 0.7463 | 0.8222 | 0.6889 | 37/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 20 | 133.6 | 0.7528 | 0.9167 | 0.6667 | 11/12 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 27 | 137.4 | 0.7657 | 0.9444 | 0.6667 | 17/18 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 179 | 93.7 | 0.7662 | 0.9167 | 0.6667 | 33/36 |
| manual__dgx_a100.pdf | 45 | 324 | 64.1 | 0.7674 | 0.8667 | 0.7111 | 39/45 |
| manual__mi_phone.pdf | 45 | 53 | 93.8 | 0.7741 | 0.8222 | 0.7333 | 37/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 58 | 103.3 | 0.779 | 0.8889 | 0.7037 | 24/27 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 66 | 87.7 | 0.7885 | 0.8974 | 0.7179 | 35/39 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 105 | 90.6 | 0.7907 | 0.8889 | 0.7111 | 40/45 |
| manual__watch_d.pdf | 45 | 92 | 73.7 | 0.7981 | 0.9556 | 0.6667 | 43/45 |
| manual__nova_y70.pdf | 46 | 167 | 67.0 | 0.8022 | 0.9348 | 0.6957 | 43/46 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 47 | 94.4 | 0.8056 | 0.9394 | 0.7273 | 31/33 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 34 | 91.0 | 0.8056 | 1.0 | 0.6667 | 15/15 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 15 | 128.7 | 0.8056 | 0.8889 | 0.7778 | 8/9 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 107 | 131.9 | 0.8056 | 0.8889 | 0.7556 | 40/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 109 | 104.7 | 0.8074 | 0.9111 | 0.7111 | 41/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 384 | 105.3 | 0.8185 | 0.8889 | 0.7778 | 40/45 |
| manual__t480_ug_en.pdf | 45 | 463 | 97.9 | 0.8193 | 0.9556 | 0.7111 | 43/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 228 | 104.1 | 0.8241 | 0.9111 | 0.7556 | 41/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 40 | 111.0 | 0.825 | 0.9667 | 0.7 | 29/30 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 126 | 134.1 | 0.8259 | 0.8889 | 0.7778 | 40/45 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 10 | 132.9 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 13 | 121.9 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 79 | 130.4 | 0.8341 | 0.8889 | 0.8 | 40/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 35 | 102.2 | 0.8403 | 1.0 | 0.7083 | 24/24 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 91 | 99.5 | 0.8463 | 0.9556 | 0.7556 | 43/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 40 | 137.7 | 0.8511 | 0.9333 | 0.8 | 28/30 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 19 | 92.5 | 0.8611 | 1.0 | 0.7778 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 14 | 124.9 | 0.8611 | 1.0 | 0.7778 | 9/9 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 92 | 124.4 | 0.8674 | 0.9556 | 0.8 | 43/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 9 | 84.2 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 31 | 133.5 | 0.875 | 1.0 | 0.7778 | 18/18 |
| manual__honor_watch_gs_pro.pdf | 45 | 146 | 81.8 | 0.8759 | 0.9556 | 0.8222 | 43/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 24 | 142.2 | 0.8889 | 0.9333 | 0.8667 | 14/15 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 53 | 131.2 | 0.8949 | 1.0 | 0.8182 | 33/33 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 11 | 107.5 | 0.9056 | 1.0 | 0.8667 | 15/15 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 31 | 136.9 | 0.9143 | 1.0 | 0.8571 | 21/21 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 17 | 136.3 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 7 | 128.6 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 13 | 144.5 | 0.9259 | 1.0 | 0.8889 | 9/9 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 130 | 111.0 | 0.9407 | 0.9778 | 0.9111 | 44/45 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 19 | 101.6 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 14 | 129.1 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 13 | 109.5 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 20 | 108.3 | 0.9583 | 1.0 | 0.9167 | 24/24 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 24 | 109.9 | 0.9619 | 1.0 | 0.9524 | 21/21 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 92.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 5 | 116.2 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 7 | 75.1 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: llamaindex (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2402.03216v4.pdf | 78 | 52 | 128.6 | 0.0667 | 0.1026 | 0.0513 | 8/78 |
| academic__2404.10198v2.pdf | 51 | 22 | 63.8 | 0.0758 | 0.1373 | 0.0392 | 7/51 |
| academic__2405.14458v1.pdf | 64 | 46 | 70.8 | 0.106 | 0.1719 | 0.0781 | 11/64 |
| academic__2305.02437v3.pdf | 66 | 41 | 92.3 | 0.1596 | 0.197 | 0.1364 | 13/66 |
| academic__2409.01704v1.pdf | 60 | 37 | 88.2 | 0.1867 | 0.2833 | 0.1333 | 17/60 |
| academic__2403.20330v2.pdf | 69 | 54 | 131.2 | 0.1908 | 0.2899 | 0.1304 | 20/69 |
| academic__2305.14160v4.pdf | 72 | 35 | 128.6 | 0.1926 | 0.2361 | 0.1667 | 17/72 |
| academic__2310.11511v1.pdf | 57 | 64 | 87.0 | 0.2392 | 0.3509 | 0.1754 | 20/57 |
| manual__DSA-278777.pdf | 45 | 27 | 164.0 | 0.3333 | 0.3333 | 0.3333 | 15/45 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 429 | 275.4 | 0.358 | 0.4638 | 0.2899 | 32/69 |
| academic__2405.14831v1.pdf | 63 | 62 | 94.3 | 0.3675 | 0.4603 | 0.3175 | 29/63 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 338 | 282.7 | 0.4479 | 0.55 | 0.3875 | 44/80 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 374 | 276.0 | 0.4535 | 0.5802 | 0.3827 | 47/81 |
| finance__AMAZON_2019_10K.pdf | 81 | 139 | 321.8 | 0.4702 | 0.5185 | 0.4444 | 42/81 |
| finance__AMAZON_2017_10K.pdf | 75 | 148 | 321.0 | 0.4747 | 0.56 | 0.4267 | 42/75 |
| academic__2409.16145v1.pdf | 51 | 37 | 230.4 | 0.5056 | 0.6275 | 0.4118 | 32/51 |
| finance__AES_2022_10K.pdf | 78 | 453 | 327.5 | 0.5686 | 0.6538 | 0.5256 | 51/78 |
| finance__VERIZON_2021_10K.pdf | 84 | 237 | 326.6 | 0.5702 | 0.6429 | 0.5357 | 54/84 |
| finance__3M_2023Q2_10Q.pdf | 63 | 178 | 321.3 | 0.5886 | 0.6508 | 0.5397 | 41/63 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 35 | 245.9 | 0.5944 | 0.6444 | 0.5556 | 29/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 32 | 300.2 | 0.6137 | 0.7556 | 0.5111 | 34/45 |
| finance__AMD_2022_10K.pdf | 62 | 197 | 338.1 | 0.6282 | 0.6774 | 0.5968 | 42/62 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 8 | 337.9 | 0.6389 | 0.8333 | 0.5 | 10/12 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 791 | 310.1 | 0.6889 | 0.8095 | 0.619 | 51/63 |
| manual__obs-productdesc-en.pdf | 45 | 43 | 327.3 | 0.6915 | 0.8 | 0.6444 | 36/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 33 | 355.4 | 0.7063 | 0.8222 | 0.6222 | 37/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 27 | 343.5 | 0.7204 | 0.8222 | 0.6444 | 37/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 67 | 299.3 | 0.7241 | 0.8 | 0.6667 | 36/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 57 | 285.5 | 0.7285 | 0.8889 | 0.6222 | 40/45 |
| manual__dgx_a100.pdf | 45 | 86 | 253.8 | 0.7296 | 0.7778 | 0.6889 | 35/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 21 | 346.5 | 0.7321 | 0.881 | 0.6429 | 37/42 |
| manual__Macbook_air.pdf | 45 | 35 | 348.5 | 0.7463 | 0.8667 | 0.6667 | 39/45 |
| manual__mi_phone.pdf | 45 | 17 | 347.3 | 0.7667 | 0.8889 | 0.6667 | 40/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 50 | 325.2 | 0.7704 | 0.9111 | 0.6889 | 41/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 320.3 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| manual__2021-Apple-Catalog.pdf | 45 | 28 | 218.5 | 0.7796 | 0.8889 | 0.7111 | 40/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 32 | 296.1 | 0.7859 | 0.9111 | 0.6889 | 41/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 14 | 295.6 | 0.7889 | 1.0 | 0.6667 | 18/18 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 28 | 328.6 | 0.7981 | 0.8667 | 0.7556 | 39/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 134 | 328.1 | 0.8 | 0.8889 | 0.7333 | 40/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 18 | 322.3 | 0.8 | 0.9667 | 0.7 | 29/30 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 58 | 317.2 | 0.8056 | 0.8611 | 0.75 | 31/36 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 78 | 326.9 | 0.8156 | 0.9111 | 0.7556 | 41/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 61 | 311.3 | 0.8219 | 0.9778 | 0.7333 | 44/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 34 | 311.8 | 0.8304 | 0.9111 | 0.7778 | 41/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 12 | 298.0 | 0.8333 | 0.9167 | 0.75 | 11/12 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 5 | 300.2 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 77 | 300.4 | 0.8333 | 0.9111 | 0.7778 | 41/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 35 | 338.9 | 0.8352 | 0.9333 | 0.7556 | 42/45 |
| manual__t480_ug_en.pdf | 45 | 137 | 363.1 | 0.8352 | 0.8889 | 0.8 | 40/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 35 | 360.7 | 0.8359 | 0.9111 | 0.8 | 41/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 11 | 324.8 | 0.8444 | 0.9333 | 0.8 | 14/15 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 23 | 304.3 | 0.8469 | 1.0 | 0.7407 | 27/27 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 51 | 307.6 | 0.8481 | 0.9111 | 0.8 | 41/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 112 | 314.1 | 0.85 | 0.9333 | 0.8 | 42/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 6 | 341.8 | 0.8519 | 1.0 | 0.7778 | 9/9 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 23 | 296.5 | 0.8556 | 0.9556 | 0.7778 | 43/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 9 | 330.3 | 0.8583 | 1.0 | 0.7778 | 18/18 |
| manual__owners-manual-2170416.pdf | 45 | 25 | 331.4 | 0.8611 | 0.9111 | 0.8222 | 41/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 19 | 314.8 | 0.8726 | 0.9744 | 0.7949 | 38/39 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 370.2 | 0.875 | 1.0 | 0.75 | 12/12 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 9 | 279.4 | 0.8796 | 1.0 | 0.7778 | 18/18 |
| manual__nova_y70.pdf | 46 | 31 | 394.4 | 0.8804 | 0.8913 | 0.8696 | 41/46 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 6 | 327.3 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 4 | 323.0 | 0.8889 | 1.0 | 0.8 | 15/15 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 22 | 341.6 | 0.8939 | 0.9697 | 0.8182 | 32/33 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 10 | 288.1 | 0.8958 | 1.0 | 0.8333 | 12/12 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 37 | 321.0 | 0.9019 | 0.9556 | 0.8667 | 43/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 9 | 336.7 | 0.9028 | 1.0 | 0.8333 | 12/12 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 74 | 356.6 | 0.9074 | 0.9556 | 0.8667 | 43/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 15 | 312.5 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 4 | 255.2 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 15 | 316.3 | 0.9192 | 0.9394 | 0.9091 | 31/33 |
| manual__watch_d.pdf | 45 | 19 | 394.5 | 0.9222 | 1.0 | 0.8444 | 45/45 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 29 | 340.3 | 0.9296 | 0.9778 | 0.8889 | 44/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 10 | 334.2 | 0.9333 | 1.0 | 0.8667 | 15/15 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 7 | 360.9 | 0.9375 | 1.0 | 0.875 | 24/24 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 15 | 295.2 | 0.9407 | 0.9778 | 0.9111 | 44/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 6 | 334.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 6 | 304.3 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 269.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 6 | 329.3 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 337.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 6 | 308.7 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 14 | 333.9 | 0.9524 | 1.0 | 0.9048 | 21/21 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 13 | 359.5 | 0.9667 | 1.0 | 0.9333 | 30/30 |
| manual__honor_watch_gs_pro.pdf | 45 | 36 | 356.3 | 0.9778 | 1.0 | 0.9556 | 45/45 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 280.7 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 10 | 286.5 | 1.0 | 1.0 | 1.0 | 21/21 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 12 | 322.0 | 1.0 | 1.0 | 1.0 | 24/24 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 216.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 319.5 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 268.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 7 | 311.1 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 2 | 254.0 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: langchain (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2305.02437v3.pdf | 66 | 68 | 57.7 | 0.0712 | 0.1061 | 0.0455 | 7/66 |
| academic__2402.03216v4.pdf | 78 | 81 | 86.0 | 0.0859 | 0.1154 | 0.0769 | 9/78 |
| academic__2404.10198v2.pdf | 51 | 42 | 33.7 | 0.1085 | 0.1765 | 0.0784 | 9/51 |
| academic__2305.14160v4.pdf | 72 | 64 | 74.5 | 0.1435 | 0.1667 | 0.125 | 12/72 |
| academic__2403.20330v2.pdf | 69 | 81 | 90.5 | 0.1763 | 0.2609 | 0.1304 | 18/69 |
| academic__2409.01704v1.pdf | 60 | 67 | 52.0 | 0.1853 | 0.2833 | 0.1333 | 17/60 |
| academic__2405.14458v1.pdf | 64 | 74 | 44.5 | 0.1927 | 0.2188 | 0.1719 | 14/64 |
| academic__2310.11511v1.pdf | 57 | 119 | 47.0 | 0.2351 | 0.3684 | 0.1754 | 21/57 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 907 | 135.8 | 0.265 | 0.3768 | 0.1884 | 26/69 |
| manual__DSA-278777.pdf | 45 | 47 | 99.9 | 0.2833 | 0.3333 | 0.2444 | 15/45 |
| academic__2405.14831v1.pdf | 63 | 105 | 57.8 | 0.3294 | 0.3968 | 0.2698 | 25/63 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 744 | 134.5 | 0.4167 | 0.475 | 0.375 | 38/80 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 787 | 136.9 | 0.4253 | 0.5062 | 0.3827 | 41/81 |
| finance__AMAZON_2019_10K.pdf | 81 | 344 | 127.5 | 0.451 | 0.5185 | 0.3951 | 42/81 |
| academic__2409.16145v1.pdf | 51 | 78 | 112.6 | 0.4908 | 0.6078 | 0.4118 | 31/51 |
| finance__AMAZON_2017_10K.pdf | 75 | 363 | 128.1 | 0.4933 | 0.52 | 0.48 | 39/75 |
| finance__VERIZON_2021_10K.pdf | 84 | 627 | 131.6 | 0.525 | 0.619 | 0.4762 | 52/84 |
| finance__AES_2022_10K.pdf | 78 | 1167 | 134.4 | 0.5335 | 0.5769 | 0.5128 | 45/78 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 77 | 118.3 | 0.6111 | 0.6667 | 0.5556 | 30/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 429 | 129.4 | 0.6222 | 0.6984 | 0.5714 | 44/63 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 1921 | 135.5 | 0.6243 | 0.7302 | 0.5556 | 46/63 |
| finance__AMD_2022_10K.pdf | 62 | 513 | 128.9 | 0.6277 | 0.6774 | 0.5968 | 42/62 |
| manual__obs-productdesc-en.pdf | 45 | 125 | 113.5 | 0.65 | 0.7556 | 0.5556 | 34/45 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 13 | 138.9 | 0.6667 | 1.0 | 0.4444 | 9/9 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 21 | 141.0 | 0.7083 | 0.8333 | 0.5833 | 10/12 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 130 | 131.9 | 0.71 | 0.8889 | 0.5778 | 40/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 28 | 138.5 | 0.7292 | 0.8333 | 0.6667 | 10/12 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 204 | 133.8 | 0.7352 | 0.8667 | 0.6444 | 39/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 78 | 130.5 | 0.7441 | 0.8222 | 0.7111 | 37/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 22 | 146.7 | 0.75 | 0.8333 | 0.6667 | 10/12 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 14 | 136.9 | 0.7593 | 1.0 | 0.5556 | 9/9 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 151 | 128.2 | 0.7602 | 0.8611 | 0.6944 | 31/36 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 162 | 132.2 | 0.767 | 0.9111 | 0.6889 | 41/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 75 | 132.9 | 0.7674 | 0.8667 | 0.6889 | 39/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 90 | 125.6 | 0.7693 | 0.9111 | 0.6667 | 41/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 89 | 106.2 | 0.7722 | 0.8889 | 0.6889 | 40/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 61 | 128.8 | 0.7778 | 0.8571 | 0.7143 | 36/42 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 23 | 134.4 | 0.7778 | 0.8889 | 0.6667 | 16/18 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 125.2 | 0.7778 | 0.7778 | 0.7778 | 7/9 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 269 | 139.2 | 0.787 | 0.9111 | 0.6889 | 41/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 23 | 138.2 | 0.7917 | 0.8333 | 0.75 | 10/12 |
| manual__Macbook_air.pdf | 45 | 104 | 116.9 | 0.7926 | 0.9111 | 0.7111 | 41/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 91 | 139.0 | 0.7944 | 0.8667 | 0.7333 | 39/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 141 | 142.7 | 0.7952 | 0.9556 | 0.6889 | 43/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 42 | 123.3 | 0.7965 | 0.9091 | 0.7273 | 30/33 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 28 | 139.9 | 0.8 | 0.8667 | 0.7333 | 13/15 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 75 | 130.8 | 0.8 | 0.8222 | 0.7778 | 37/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 212 | 133.6 | 0.8007 | 0.9111 | 0.7333 | 41/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 125 | 140.4 | 0.8056 | 0.8889 | 0.7556 | 40/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 56 | 130.5 | 0.8056 | 0.963 | 0.7037 | 26/27 |
| manual__t480_ug_en.pdf | 45 | 386 | 130.1 | 0.8081 | 0.9333 | 0.7556 | 42/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 26 | 139.2 | 0.8111 | 0.9333 | 0.7333 | 14/15 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 49 | 142.3 | 0.8185 | 0.9111 | 0.7333 | 41/45 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 15 | 135.7 | 0.8278 | 1.0 | 0.7778 | 9/9 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 30 | 146.6 | 0.8306 | 0.9444 | 0.7778 | 17/18 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 7 | 131.9 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 17 | 132.5 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 5 | 116.4 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 5 | 110.6 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 179 | 128.9 | 0.8352 | 0.9111 | 0.7778 | 41/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 356 | 134.1 | 0.8356 | 0.9778 | 0.7556 | 44/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 126 | 128.6 | 0.8519 | 0.9111 | 0.8 | 41/45 |
| manual__owners-manual-2170416.pdf | 45 | 62 | 133.1 | 0.8519 | 0.9333 | 0.8 | 42/45 |
| manual__nova_y70.pdf | 46 | 91 | 134.9 | 0.8533 | 0.9348 | 0.7826 | 43/46 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 33 | 129.3 | 0.8611 | 1.0 | 0.75 | 24/24 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 12 | 132.2 | 0.8611 | 0.9167 | 0.8333 | 11/12 |
| manual__mi_phone.pdf | 45 | 51 | 114.4 | 0.863 | 0.9333 | 0.8 | 42/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 50 | 129.3 | 0.8675 | 0.9231 | 0.8205 | 36/39 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 95 | 134.1 | 0.8704 | 0.9778 | 0.7778 | 44/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 34 | 147.3 | 0.873 | 0.9524 | 0.8095 | 20/21 |
| manual__dgx_a100.pdf | 45 | 204 | 108.5 | 0.8778 | 0.9111 | 0.8444 | 41/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 60 | 101.0 | 0.8852 | 0.9333 | 0.8444 | 42/45 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 12 | 128.8 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 15 | 140.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 45 | 138.5 | 0.8944 | 0.9667 | 0.8333 | 29/30 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 102 | 131.3 | 0.9 | 0.9333 | 0.8667 | 42/45 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 81 | 133.6 | 0.9074 | 0.9556 | 0.8667 | 43/45 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 59 | 138.8 | 0.9116 | 1.0 | 0.8485 | 33/33 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 5 | 137.4 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| manual__honor_watch_gs_pro.pdf | 45 | 103 | 126.6 | 0.9185 | 0.9778 | 0.8667 | 44/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 8 | 111.0 | 0.9259 | 1.0 | 0.8889 | 9/9 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 37 | 132.6 | 0.9333 | 1.0 | 0.8667 | 30/30 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 36 | 125.9 | 0.9333 | 0.9556 | 0.9111 | 43/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 21 | 121.2 | 0.9375 | 1.0 | 0.875 | 24/24 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 90 | 139.3 | 0.9407 | 0.9778 | 0.9111 | 44/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 33 | 146.7 | 0.9444 | 1.0 | 0.8889 | 18/18 |
| manual__watch_d.pdf | 45 | 57 | 131.2 | 0.9519 | 0.9778 | 0.9333 | 44/45 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 11 | 119.1 | 0.9556 | 1.0 | 0.9333 | 15/15 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 21 | 132.7 | 0.9722 | 1.0 | 0.9444 | 18/18 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 15 | 145.5 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 15 | 132.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 25 | 122.1 | 1.0 | 1.0 | 1.0 | 21/21 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 111.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 9 | 121.7 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 17 | 136.7 | 1.0 | 1.0 | 1.0 | 9/9 |
