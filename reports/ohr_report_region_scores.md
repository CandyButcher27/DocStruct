# DocStruct retrieval baseline report

_Generated 2026-08-15 19:59 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: docstruct, docstruct_geo, pymupdf4llm, langchain, unstructured, llamaindex, llamaindex_semantic.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | docstruct **(ours)** | **0.6659** | [0.652, 0.68] | 0.6833 | 0.7782 | 0.5911 | 0.5691 | +0.0968 | 9080 | 316.8 | 2194.3 | 0.3035 | 56.25 | 0 |
| 2 | docstruct_geo | **0.6567** | [0.6426, 0.6706] | 0.683 | 0.7841 | 0.5705 | 0.548 | +0.1087 | 5810 | 306.0 | 2328.6 | 0.282 | 51.8 | 0 |
| 3 | pymupdf4llm | **0.604** | [0.5892, 0.6186] | 0.6241 | 0.7083 | 0.5354 | 0.5112 | +0.0928 | 3756 | 424.6 | 2424.5 | 0.2491 | 2036.59 | 0 |
| 4 | langchain | **0.6031** | [0.5881, 0.6178] | 0.6243 | 0.704 | 0.5326 | 0.5415 | +0.0616 | 13877 | 128.5 | 637.6 | 0.9459 | 645.46 | 0 |
| 5 | unstructured | **0.6008** | [0.5857, 0.6156] | 0.6248 | 0.709 | 0.5259 | 0.5414 | +0.0594 | 18424 | 87.2 | 560.5 | 1.0719 | 1117.19 | 3 |
| 6 | llamaindex | **0.5885** | [0.5737, 0.6031] | 0.6102 | 0.6979 | 0.5135 | 0.5121 | +0.0764 | 5794 | 295.2 | 1430.1 | 0.4115 | 644.33 | 0 |
| 7 | llamaindex_semantic | **0.5747** | [0.56, 0.5886] | 0.6064 | 0.7293 | 0.4744 | 0.4345 | +0.1402 | 3366 | 482.0 | 4697.0 | 0.1224 | 964.07 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| langchain | 1.0 | 1.1005 |
| llamaindex | 1.0 | 1.0474 |
| llamaindex_semantic | 1.0 | 1.0 |
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
| docstruct_geo | MRR | +0.0092 | [-0.002, 0.0208] | 0.113 | 3558 | not significant |
| docstruct_geo | NDCG | +0.0002 | [-0.0097, 0.0104] | 0.9602 | 3558 | not significant |
| docstruct_geo | RECALL | -0.0059 | [-0.016, 0.0039] | 0.2597 | 3558 | not significant |
| docstruct_geo | HIT1 | +0.0205 | [0.0048, 0.0365] | 0.0115 | 3558 | **significant** |
| pymupdf4llm | MRR | +0.0619 | [0.049, 0.0751] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | NDCG | +0.0592 | [0.0471, 0.0716] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | RECALL | +0.07 | [0.0568, 0.0835] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | HIT1 | +0.0556 | [0.0393, 0.0722] | 0.0001 | 3558 | **significant** |
| langchain | MRR | +0.0627 | [0.0491, 0.0761] | 0.0001 | 3558 | **significant** |
| langchain | NDCG | +0.0589 | [0.0465, 0.0713] | 0.0001 | 3558 | **significant** |
| langchain | RECALL | +0.0742 | [0.0607, 0.0874] | 0.0001 | 3558 | **significant** |
| langchain | HIT1 | +0.0582 | [0.041, 0.0756] | 0.0001 | 3558 | **significant** |
| unstructured | MRR | +0.0627 | [0.0477, 0.0775] | 0.0001 | 3423 | **significant** |
| unstructured | NDCG | +0.0559 | [0.042, 0.0697] | 0.0001 | 3423 | **significant** |
| unstructured | RECALL | +0.0663 | [0.0523, 0.0806] | 0.0001 | 3423 | **significant** |
| unstructured | HIT1 | +0.0625 | [0.0438, 0.0815] | 0.0001 | 3423 | **significant** |
| llamaindex | MRR | +0.0773 | [0.0636, 0.0909] | 0.0001 | 3558 | **significant** |
| llamaindex | NDCG | +0.073 | [0.0603, 0.0855] | 0.0001 | 3558 | **significant** |
| llamaindex | RECALL | +0.0804 | [0.0672, 0.0936] | 0.0001 | 3558 | **significant** |
| llamaindex | HIT1 | +0.0773 | [0.0599, 0.0947] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | MRR | +0.0912 | [0.076, 0.1063] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | NDCG | +0.0768 | [0.0629, 0.0905] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | RECALL | +0.0489 | [0.0346, 0.0627] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | HIT1 | +0.1166 | [0.0975, 0.136] | 0.0001 | 3558 | **significant** |

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
| academic__2305.02437v3.pdf | 66 | 42 | 218.1 | 0.202 | 0.2273 | 0.1818 | 15/66 |
| academic__2405.14458v1.pdf | 64 | 41 | 317.5 | 0.3047 | 0.4219 | 0.2344 | 27/64 |
| academic__2403.20330v2.pdf | 69 | 45 | 275.2 | 0.3198 | 0.4203 | 0.2609 | 29/69 |
| academic__2305.14160v4.pdf | 72 | 50 | 267.8 | 0.3481 | 0.4167 | 0.3056 | 30/72 |
| academic__2404.10198v2.pdf | 51 | 16 | 356.4 | 0.3549 | 0.5098 | 0.2745 | 26/51 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 869 | 238.9 | 0.4256 | 0.5507 | 0.3623 | 38/69 |
| manual__DSA-278777.pdf | 45 | 52 | 187.0 | 0.4278 | 0.5111 | 0.3778 | 23/45 |
| academic__2402.03216v4.pdf | 78 | 61 | 324.5 | 0.4314 | 0.5769 | 0.3462 | 45/78 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 771 | 235.4 | 0.4387 | 0.5556 | 0.3704 | 45/81 |
| academic__2409.01704v1.pdf | 60 | 33 | 298.7 | 0.4394 | 0.5667 | 0.35 | 34/60 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 728 | 231.2 | 0.4565 | 0.6125 | 0.35 | 49/80 |
| academic__2409.16145v1.pdf | 51 | 36 | 299.0 | 0.5229 | 0.6667 | 0.4118 | 34/51 |
| finance__AMAZON_2017_10K.pdf | 75 | 192 | 368.0 | 0.5731 | 0.7067 | 0.48 | 53/75 |
| finance__AES_2022_10K.pdf | 78 | 582 | 400.5 | 0.5756 | 0.6282 | 0.5513 | 49/78 |
| finance__AMAZON_2019_10K.pdf | 81 | 188 | 355.3 | 0.58 | 0.6543 | 0.5309 | 53/81 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 97 | 466.9 | 0.587 | 0.7111 | 0.4889 | 32/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 36 | 399.0 | 0.587 | 0.7111 | 0.4889 | 32/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 234 | 377.3 | 0.5926 | 0.7302 | 0.5079 | 46/63 |
| finance__AMD_2022_10K.pdf | 62 | 247 | 379.9 | 0.6008 | 0.7258 | 0.5 | 45/62 |
| finance__VERIZON_2021_10K.pdf | 84 | 378 | 304.5 | 0.6036 | 0.7381 | 0.5238 | 62/84 |
| manual__obs-productdesc-en.pdf | 45 | 101 | 179.5 | 0.6185 | 0.8 | 0.4889 | 36/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 37 | 344.4 | 0.6259 | 0.6889 | 0.5778 | 31/45 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 11 | 403.8 | 0.6319 | 0.8333 | 0.5 | 10/12 |
| academic__2405.14831v1.pdf | 63 | 72 | 220.1 | 0.6386 | 0.8095 | 0.5238 | 51/63 |
| manual__Guide-for-international-students-web.pdf | 45 | 146 | 107.2 | 0.6541 | 0.8444 | 0.5556 | 38/45 |
| academic__2310.11511v1.pdf | 57 | 62 | 357.1 | 0.6547 | 0.7719 | 0.5789 | 44/57 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 5 | 346.2 | 0.6667 | 0.7778 | 0.5556 | 7/9 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 1309 | 331.3 | 0.682 | 0.8413 | 0.6032 | 53/63 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 48 | 602.9 | 0.687 | 0.8889 | 0.5556 | 40/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 32 | 424.5 | 0.6915 | 0.8222 | 0.6 | 37/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 42 | 293.5 | 0.6952 | 0.7857 | 0.6429 | 33/42 |
| manual__Macbook_air.pdf | 45 | 51 | 247.6 | 0.6978 | 0.8 | 0.6444 | 36/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 18 | 578.6 | 0.7078 | 0.9333 | 0.6 | 28/30 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 31 | 379.7 | 0.7093 | 0.8222 | 0.6444 | 37/45 |
| manual__dgx_a100.pdf | 45 | 164 | 200.2 | 0.7111 | 0.8444 | 0.6222 | 38/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 4 | 406.8 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 38 | 462.7 | 0.7241 | 0.8 | 0.6667 | 36/45 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 76 | 159.0 | 0.7274 | 0.8889 | 0.6 | 40/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 81 | 382.4 | 0.7296 | 0.7556 | 0.7111 | 34/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 129 | 224.1 | 0.7315 | 0.8444 | 0.6444 | 38/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 63 | 536.2 | 0.7396 | 0.8889 | 0.6444 | 40/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 17 | 492.4 | 0.751 | 0.8788 | 0.6667 | 29/33 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 12 | 532.5 | 0.7519 | 0.9444 | 0.6667 | 17/18 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 11 | 514.1 | 0.7535 | 0.9167 | 0.625 | 22/24 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 58 | 387.7 | 0.7667 | 0.8667 | 0.6889 | 39/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 94 | 437.2 | 0.7822 | 0.9556 | 0.6889 | 43/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 42 | 532.3 | 0.7833 | 0.8889 | 0.6889 | 40/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 170 | 261.6 | 0.787 | 0.8667 | 0.7333 | 39/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 11 | 339.5 | 0.791 | 0.9583 | 0.7083 | 23/24 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 86 | 315.0 | 0.7917 | 0.8611 | 0.7222 | 31/36 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 5 | 583.0 | 0.7917 | 1.0 | 0.5833 | 12/12 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 7 | 489.3 | 0.7963 | 1.0 | 0.6667 | 9/9 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 15 | 386.4 | 0.8056 | 1.0 | 0.6667 | 12/12 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 109 | 550.9 | 0.8093 | 0.8889 | 0.7556 | 40/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 8 | 482.9 | 0.8125 | 1.0 | 0.6667 | 12/12 |
| manual__mi_phone.pdf | 45 | 35 | 233.1 | 0.8167 | 0.9556 | 0.7111 | 43/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 88 | 473.8 | 0.8174 | 0.8889 | 0.7778 | 40/45 |
| manual__owners-manual-2170416.pdf | 45 | 52 | 246.2 | 0.8204 | 0.8889 | 0.7778 | 40/45 |
| manual__t480_ug_en.pdf | 45 | 244 | 364.8 | 0.8222 | 0.9556 | 0.7333 | 43/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 68 | 298.8 | 0.8259 | 0.9333 | 0.7333 | 42/45 |
| manual__nova_y70.pdf | 46 | 77 | 239.0 | 0.8261 | 0.913 | 0.7391 | 42/46 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 47 | 636.1 | 0.8293 | 0.9333 | 0.7778 | 42/45 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 6 | 337.5 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 6 | 509.8 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 3 | 297.0 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| manual__watch_d.pdf | 45 | 44 | 248.7 | 0.8341 | 0.8889 | 0.8 | 40/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 62 | 507.4 | 0.837 | 0.9333 | 0.7556 | 42/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 122 | 76.6 | 0.8433 | 0.9111 | 0.8 | 41/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 15 | 550.3 | 0.8519 | 0.9444 | 0.7778 | 17/18 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 21 | 619.6 | 0.8535 | 0.9697 | 0.7879 | 32/33 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 29 | 606.2 | 0.8563 | 0.9778 | 0.7778 | 44/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 11 | 560.4 | 0.8611 | 1.0 | 0.75 | 12/12 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 11 | 544.2 | 0.8611 | 1.0 | 0.8 | 15/15 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 6 | 546.7 | 0.8611 | 1.0 | 0.7778 | 9/9 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 23 | 332.7 | 0.8656 | 0.9556 | 0.8 | 43/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 15 | 535.3 | 0.8849 | 1.0 | 0.8095 | 21/21 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 7 | 469.1 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 7 | 374.6 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 48 | 344.1 | 0.8974 | 0.9744 | 0.8462 | 38/39 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 38 | 583.5 | 0.9111 | 0.9778 | 0.8667 | 44/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 12 | 476.4 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 3 | 397.3 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 24 | 505.8 | 0.9198 | 1.0 | 0.8519 | 27/27 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 8 | 586.9 | 0.9352 | 1.0 | 0.8889 | 18/18 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 22 | 496.4 | 0.9389 | 1.0 | 0.9 | 30/30 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 6 | 545.3 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 7 | 558.1 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 7 | 511.6 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 13 | 430.4 | 0.9667 | 1.0 | 0.9333 | 15/15 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 7 | 350.0 | 0.9667 | 1.0 | 0.9333 | 15/15 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 6 | 564.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 574.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 8 | 634.4 | 1.0 | 1.0 | 1.0 | 21/21 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 427.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 5 | 270.0 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: docstruct_geo (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2305.02437v3.pdf | 66 | 28 | 220.8 | 0.2058 | 0.2424 | 0.1818 | 16/66 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 546 | 217.8 | 0.2807 | 0.3913 | 0.2319 | 27/69 |
| academic__2403.20330v2.pdf | 69 | 48 | 231.4 | 0.3039 | 0.4348 | 0.2174 | 30/69 |
| academic__2405.14458v1.pdf | 64 | 26 | 302.6 | 0.343 | 0.4844 | 0.25 | 31/64 |
| academic__2305.14160v4.pdf | 72 | 26 | 361.5 | 0.3789 | 0.5 | 0.3056 | 36/72 |
| academic__2402.03216v4.pdf | 78 | 30 | 377.1 | 0.4053 | 0.5769 | 0.3077 | 45/78 |
| manual__DSA-278777.pdf | 45 | 23 | 287.2 | 0.4396 | 0.5556 | 0.3778 | 25/45 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 440 | 219.2 | 0.4606 | 0.65 | 0.35 | 52/80 |
| academic__2404.10198v2.pdf | 51 | 15 | 293.3 | 0.4755 | 0.5098 | 0.451 | 26/51 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 490 | 212.4 | 0.479 | 0.5679 | 0.4321 | 46/81 |
| finance__3M_2023Q2_10Q.pdf | 63 | 155 | 369.5 | 0.5183 | 0.6508 | 0.4444 | 41/63 |
| finance__AES_2022_10K.pdf | 78 | 418 | 351.0 | 0.5291 | 0.641 | 0.4615 | 50/78 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 8 | 331.5 | 0.5417 | 0.8333 | 0.3333 | 10/12 |
| academic__2409.01704v1.pdf | 60 | 26 | 268.2 | 0.5444 | 0.6167 | 0.5 | 37/60 |
| academic__2409.16145v1.pdf | 51 | 22 | 367.1 | 0.5627 | 0.7059 | 0.4706 | 36/51 |
| finance__VERIZON_2021_10K.pdf | 84 | 302 | 258.2 | 0.5738 | 0.6786 | 0.5 | 57/84 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 22 | 427.5 | 0.5815 | 0.7111 | 0.4667 | 32/45 |
| finance__AMAZON_2019_10K.pdf | 81 | 82 | 516.4 | 0.5887 | 0.7037 | 0.5062 | 57/81 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 90 | 482.8 | 0.5889 | 0.7778 | 0.4444 | 35/45 |
| finance__AMD_2022_10K.pdf | 62 | 182 | 354.2 | 0.6035 | 0.6935 | 0.5484 | 43/62 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 874 | 287.9 | 0.6209 | 0.8095 | 0.4921 | 51/63 |
| manual__obs-productdesc-en.pdf | 45 | 97 | 145.9 | 0.6278 | 0.7778 | 0.5111 | 35/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 18 | 463.1 | 0.6322 | 0.7778 | 0.5333 | 35/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 28 | 356.0 | 0.6393 | 0.8222 | 0.5111 | 37/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 81 | 556.1 | 0.6431 | 0.7467 | 0.5867 | 56/75 |
| academic__2405.14831v1.pdf | 63 | 53 | 220.2 | 0.6466 | 0.8095 | 0.5397 | 51/63 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 22 | 405.2 | 0.6737 | 0.8222 | 0.5778 | 37/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 29 | 669.2 | 0.6774 | 0.8 | 0.6 | 36/45 |
| manual__dgx_a100.pdf | 45 | 155 | 141.6 | 0.6933 | 0.8444 | 0.6 | 38/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 26 | 702.5 | 0.6941 | 0.8889 | 0.5778 | 40/45 |
| academic__2310.11511v1.pdf | 57 | 42 | 389.7 | 0.695 | 0.807 | 0.614 | 46/57 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 20 | 771.3 | 0.7015 | 0.8889 | 0.5778 | 40/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 60 | 375.9 | 0.7059 | 0.8 | 0.6667 | 36/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 20 | 363.8 | 0.7123 | 0.8571 | 0.619 | 36/42 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 28 | 550.3 | 0.7137 | 0.8444 | 0.6444 | 38/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 13 | 508.8 | 0.7173 | 0.963 | 0.5926 | 26/27 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 300.7 | 0.7222 | 0.7778 | 0.6667 | 7/9 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 3 | 475.7 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 2 | 402.5 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 24 | 505.9 | 0.7256 | 0.9556 | 0.5778 | 43/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 8 | 299.2 | 0.7257 | 0.9583 | 0.5417 | 23/24 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 6 | 551.8 | 0.7333 | 1.0 | 0.5333 | 15/15 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 109 | 237.7 | 0.7333 | 0.8444 | 0.6444 | 38/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 131 | 83.5 | 0.737 | 0.8444 | 0.6444 | 38/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 49 | 690.1 | 0.7415 | 0.8667 | 0.6667 | 39/45 |
| manual__Macbook_air.pdf | 45 | 44 | 263.2 | 0.7426 | 0.8222 | 0.6889 | 37/45 |
| manual__t480_ug_en.pdf | 45 | 123 | 381.2 | 0.7433 | 0.8667 | 0.6889 | 39/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 40 | 445.9 | 0.7546 | 0.8889 | 0.6667 | 32/36 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 3 | 598.7 | 0.7593 | 1.0 | 0.5556 | 9/9 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 9 | 617.6 | 0.7639 | 0.9667 | 0.6333 | 29/30 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 9 | 521.8 | 0.7652 | 0.9091 | 0.6667 | 30/33 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 5 | 568.4 | 0.7685 | 1.0 | 0.5556 | 18/18 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 28 | 367.4 | 0.7741 | 0.8667 | 0.7111 | 39/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 47 | 531.3 | 0.7767 | 0.9111 | 0.6889 | 41/45 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 3 | 620.0 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 45 | 567.9 | 0.78 | 0.8889 | 0.7111 | 40/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 65 | 266.2 | 0.787 | 0.8444 | 0.7556 | 38/45 |
| manual__owners-manual-2170416.pdf | 45 | 28 | 278.3 | 0.7952 | 0.9333 | 0.6889 | 42/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 8 | 455.1 | 0.8056 | 1.0 | 0.6667 | 12/12 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 359.2 | 0.8056 | 1.0 | 0.6667 | 12/12 |
| manual__mi_phone.pdf | 45 | 22 | 261.8 | 0.8081 | 0.9333 | 0.7111 | 42/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 20 | 572.5 | 0.8093 | 0.9333 | 0.7111 | 42/45 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 67 | 126.6 | 0.8148 | 0.9111 | 0.7556 | 41/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 26 | 234.4 | 0.8185 | 0.8889 | 0.7556 | 40/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 5 | 690.0 | 0.8222 | 1.0 | 0.6667 | 15/15 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 36 | 618.4 | 0.8267 | 0.9111 | 0.7778 | 41/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 4 | 477.8 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 3 | 656.7 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 2 | 486.5 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 3 | 412.0 | 0.8333 | 1.0 | 0.6667 | 15/15 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 5 | 578.4 | 0.8403 | 1.0 | 0.75 | 12/12 |
| manual__nova_y70.pdf | 46 | 46 | 245.8 | 0.8406 | 0.9348 | 0.7609 | 43/46 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 15 | 641.1 | 0.8407 | 0.9778 | 0.7333 | 44/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 39 | 311.7 | 0.8444 | 0.9333 | 0.7778 | 42/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 6 | 738.2 | 0.8472 | 0.9444 | 0.7778 | 17/18 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 6 | 631.7 | 0.8472 | 1.0 | 0.7083 | 24/24 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 711.2 | 0.8472 | 1.0 | 0.75 | 12/12 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 12 | 614.2 | 0.8611 | 1.0 | 0.7576 | 33/33 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 10 | 417.6 | 0.8648 | 1.0 | 0.7778 | 45/45 |
| manual__watch_d.pdf | 45 | 35 | 201.8 | 0.8667 | 0.9556 | 0.8 | 43/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 7 | 572.7 | 0.8722 | 1.0 | 0.7778 | 18/18 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 7 | 648.4 | 0.873 | 1.0 | 0.7619 | 21/21 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 4 | 619.5 | 0.8796 | 1.0 | 0.7778 | 18/18 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 4 | 462.5 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 353.4 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 35 | 332.6 | 0.8996 | 0.9744 | 0.8462 | 38/39 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 320.5 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 5 | 564.6 | 0.9286 | 1.0 | 0.8571 | 21/21 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 17 | 476.3 | 0.9289 | 1.0 | 0.9 | 30/30 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 3 | 572.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 5 | 408.8 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 1 | 820.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 108.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 4 | 264.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 2 | 253.0 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: pymupdf4llm (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 179 | 435.3 | 0.1977 | 0.275 | 0.15 | 22/80 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 217 | 428.8 | 0.2159 | 0.2754 | 0.1884 | 19/69 |
| academic__2403.20330v2.pdf | 69 | 20 | 441.1 | 0.2432 | 0.4058 | 0.1449 | 28/69 |
| finance__AMAZON_2017_10K.pdf | 75 | 85 | 506.4 | 0.2758 | 0.3467 | 0.24 | 26/75 |
| finance__AMAZON_2019_10K.pdf | 81 | 83 | 489.0 | 0.2819 | 0.321 | 0.2593 | 26/81 |
| manual__DSA-278777.pdf | 45 | 21 | 259.4 | 0.2822 | 0.3111 | 0.2667 | 14/45 |
| academic__2305.14160v4.pdf | 72 | 16 | 527.4 | 0.3532 | 0.4861 | 0.2778 | 35/72 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 197 | 408.6 | 0.3539 | 0.3827 | 0.3333 | 31/81 |
| academic__2402.03216v4.pdf | 78 | 18 | 517.4 | 0.3658 | 0.4487 | 0.3077 | 35/78 |
| academic__2305.02437v3.pdf | 66 | 20 | 438.9 | 0.3758 | 0.4697 | 0.3182 | 31/66 |
| finance__AES_2022_10K.pdf | 78 | 255 | 515.4 | 0.4026 | 0.5 | 0.3333 | 39/78 |
| academic__2404.10198v2.pdf | 51 | 13 | 414.2 | 0.449 | 0.5686 | 0.3725 | 29/51 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 382 | 569.4 | 0.4503 | 0.5556 | 0.381 | 35/63 |
| academic__2405.14458v1.pdf | 64 | 18 | 535.7 | 0.4516 | 0.5781 | 0.3906 | 37/64 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 25 | 346.4 | 0.4685 | 0.5778 | 0.4 | 26/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 92 | 552.5 | 0.4788 | 0.5238 | 0.4444 | 33/63 |
| academic__2409.16145v1.pdf | 51 | 22 | 428.4 | 0.4801 | 0.6471 | 0.4118 | 33/51 |
| academic__2409.01704v1.pdf | 60 | 19 | 443.9 | 0.4844 | 0.6667 | 0.3833 | 40/60 |
| finance__VERIZON_2021_10K.pdf | 84 | 120 | 582.6 | 0.4921 | 0.5714 | 0.4405 | 48/84 |
| manual__obs-productdesc-en.pdf | 45 | 65 | 171.8 | 0.5074 | 0.5556 | 0.4667 | 25/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 40 | 186.2 | 0.5093 | 0.5556 | 0.4889 | 25/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 16 | 437.4 | 0.5167 | 0.6667 | 0.4286 | 28/42 |
| finance__AMD_2022_10K.pdf | 62 | 121 | 507.0 | 0.5247 | 0.5806 | 0.5 | 36/62 |
| academic__2310.11511v1.pdf | 57 | 30 | 540.0 | 0.5895 | 0.7544 | 0.5088 | 43/57 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 53 | 459.2 | 0.6044 | 0.7778 | 0.4889 | 35/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 91 | 468.9 | 0.6107 | 0.7556 | 0.5333 | 34/45 |
| academic__2405.14831v1.pdf | 63 | 28 | 476.8 | 0.6111 | 0.6825 | 0.5556 | 43/63 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 305.7 | 0.6111 | 0.6667 | 0.5556 | 6/9 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 4 | 654.5 | 0.625 | 0.6667 | 0.5833 | 8/12 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 25 | 353.7 | 0.6396 | 0.7556 | 0.5556 | 34/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 40 | 289.1 | 0.6526 | 0.7556 | 0.6 | 34/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 34 | 452.0 | 0.6663 | 0.8222 | 0.5556 | 37/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 31 | 622.5 | 0.6759 | 0.8222 | 0.6 | 37/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 75 | 125.9 | 0.6767 | 0.8667 | 0.5333 | 39/45 |
| manual__owners-manual-2170416.pdf | 45 | 32 | 234.6 | 0.6859 | 0.7778 | 0.6222 | 35/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 39 | 387.5 | 0.6926 | 0.7778 | 0.6222 | 35/45 |
| manual__Macbook_air.pdf | 45 | 71 | 165.8 | 0.6933 | 0.7556 | 0.6444 | 34/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 19 | 488.4 | 0.6963 | 0.8222 | 0.6222 | 37/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 108 | 231.8 | 0.7 | 0.7333 | 0.6667 | 33/45 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 17 | 358.9 | 0.7037 | 0.8 | 0.6222 | 36/45 |
| manual__nova_y70.pdf | 46 | 45 | 261.9 | 0.7072 | 0.8043 | 0.6304 | 37/46 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 21 | 735.1 | 0.7181 | 0.9111 | 0.5778 | 41/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 45 | 750.3 | 0.7204 | 0.8444 | 0.6222 | 38/45 |
| manual__t480_ug_en.pdf | 45 | 168 | 275.9 | 0.7219 | 0.8444 | 0.6444 | 38/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 17 | 245.8 | 0.7296 | 0.8444 | 0.6444 | 38/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 14 | 342.4 | 0.7298 | 0.8485 | 0.6364 | 28/33 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 9 | 738.8 | 0.7358 | 1.0 | 0.5926 | 27/27 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 47 | 535.9 | 0.7459 | 0.8667 | 0.6667 | 39/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 49 | 362.1 | 0.7556 | 0.8889 | 0.6944 | 32/36 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 30 | 346.8 | 0.7626 | 0.8889 | 0.6889 | 40/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 5 | 716.2 | 0.7639 | 1.0 | 0.5833 | 12/12 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 10 | 551.4 | 0.7639 | 0.9 | 0.6667 | 27/30 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 268.7 | 0.7778 | 0.7778 | 0.7778 | 7/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 3 | 614.0 | 0.7778 | 0.8889 | 0.6667 | 8/9 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 29 | 420.7 | 0.7796 | 0.9111 | 0.7111 | 41/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 15 | 762.9 | 0.7852 | 0.8889 | 0.7111 | 40/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 5 | 694.2 | 0.7889 | 1.0 | 0.6 | 15/15 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 14 | 523.6 | 0.8005 | 0.9697 | 0.6667 | 32/33 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 712.2 | 0.8056 | 1.0 | 0.6667 | 12/12 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 6 | 741.2 | 0.8102 | 0.9444 | 0.7222 | 17/18 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 3 | 551.3 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 9 | 273.8 | 0.816 | 0.9167 | 0.75 | 22/24 |
| manual__watch_d.pdf | 45 | 27 | 256.2 | 0.8204 | 0.8889 | 0.7778 | 40/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 7 | 406.7 | 0.8241 | 1.0 | 0.7222 | 18/18 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 18 | 323.2 | 0.8291 | 0.9231 | 0.7436 | 36/39 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 16 | 607.6 | 0.8296 | 0.9556 | 0.7556 | 43/45 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 2 | 487.0 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 3 | 456.0 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 3 | 549.0 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 7 | 647.3 | 0.8333 | 1.0 | 0.6667 | 21/21 |
| manual__dgx_a100.pdf | 45 | 120 | 175.7 | 0.8333 | 0.8667 | 0.8 | 39/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 16 | 711.9 | 0.837 | 0.9556 | 0.7556 | 43/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 8 | 473.4 | 0.8382 | 1.0 | 0.75 | 24/24 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 23 | 795.3 | 0.8396 | 0.9556 | 0.7778 | 43/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 55 | 114.8 | 0.8537 | 0.9111 | 0.8222 | 41/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 11 | 407.5 | 0.8556 | 0.9667 | 0.7667 | 29/30 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 5 | 655.8 | 0.8667 | 1.0 | 0.7333 | 15/15 |
| manual__mi_phone.pdf | 45 | 37 | 149.2 | 0.8778 | 0.9333 | 0.8444 | 42/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 6 | 419.7 | 0.8796 | 1.0 | 0.7778 | 18/18 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 6 | 667.5 | 0.8796 | 1.0 | 0.7778 | 18/18 |
| manual__honor_watch_gs_pro.pdf | 45 | 42 | 295.5 | 0.8852 | 0.9556 | 0.8222 | 43/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 3 | 640.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 3 | 656.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 3 | 682.7 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 5 | 579.2 | 0.9167 | 1.0 | 0.8333 | 12/12 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 3 | 217.0 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 4 | 465.8 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 3 | 574.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 367.0 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 8 | 343.4 | 0.9683 | 1.0 | 0.9524 | 21/21 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 1 | 824.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 217.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 264.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 5 | 253.2 | 1.0 | 1.0 | 1.0 | 15/15 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 4 | 132.8 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: langchain (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2404.10198v2.pdf ⚠ | 51 | 42 | 33.7 | 0.0 | 0.0 | 0.0 | 0/51 |
| academic__2409.01704v1.pdf | 60 | 67 | 52.0 | 0.0417 | 0.05 | 0.0333 | 3/60 |
| academic__2305.14160v4.pdf | 72 | 64 | 74.5 | 0.0444 | 0.0694 | 0.0278 | 5/72 |
| academic__2305.02437v3.pdf | 66 | 68 | 57.7 | 0.0455 | 0.0455 | 0.0455 | 3/66 |
| academic__2405.14458v1.pdf | 64 | 74 | 44.5 | 0.0508 | 0.0625 | 0.0469 | 4/64 |
| academic__2310.11511v1.pdf | 57 | 119 | 47.0 | 0.0526 | 0.0526 | 0.0526 | 3/57 |
| academic__2402.03216v4.pdf | 78 | 81 | 86.0 | 0.0699 | 0.0897 | 0.0641 | 7/78 |
| academic__2403.20330v2.pdf | 69 | 81 | 90.5 | 0.099 | 0.1159 | 0.087 | 8/69 |
| manual__DSA-278777.pdf | 45 | 47 | 99.9 | 0.2 | 0.2222 | 0.1778 | 10/45 |
| academic__2405.14831v1.pdf | 63 | 105 | 57.8 | 0.2995 | 0.4127 | 0.2222 | 26/63 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 907 | 135.8 | 0.329 | 0.4783 | 0.2319 | 33/69 |
| academic__2409.16145v1.pdf | 51 | 78 | 112.6 | 0.3758 | 0.4902 | 0.2941 | 25/51 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 787 | 136.9 | 0.4206 | 0.5309 | 0.358 | 43/81 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 744 | 134.5 | 0.4262 | 0.5625 | 0.3375 | 45/80 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 1921 | 135.5 | 0.5169 | 0.6825 | 0.4286 | 43/63 |
| manual__obs-productdesc-en.pdf | 45 | 125 | 113.5 | 0.5667 | 0.7333 | 0.4222 | 33/45 |
| finance__AES_2022_10K.pdf | 78 | 1167 | 134.4 | 0.5688 | 0.6538 | 0.5128 | 51/78 |
| finance__AMAZON_2019_10K.pdf | 81 | 344 | 127.5 | 0.5897 | 0.7531 | 0.4815 | 61/81 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 77 | 118.3 | 0.6019 | 0.7111 | 0.5111 | 32/45 |
| finance__VERIZON_2021_10K.pdf | 84 | 627 | 131.6 | 0.6163 | 0.7857 | 0.5119 | 66/84 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 22 | 146.7 | 0.625 | 0.8333 | 0.4167 | 10/12 |
| finance__3M_2023Q2_10Q.pdf | 63 | 429 | 129.4 | 0.6267 | 0.746 | 0.5397 | 47/63 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 162 | 132.2 | 0.643 | 0.8 | 0.5556 | 36/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 23 | 138.2 | 0.6458 | 0.75 | 0.5833 | 9/12 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 130 | 131.9 | 0.6515 | 0.8222 | 0.5333 | 37/45 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 21 | 141.0 | 0.6667 | 0.8333 | 0.5 | 10/12 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 15 | 140.3 | 0.6667 | 0.7778 | 0.5556 | 7/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 13 | 138.9 | 0.6667 | 1.0 | 0.4444 | 9/9 |
| finance__AMD_2022_10K.pdf | 62 | 513 | 128.9 | 0.6685 | 0.7419 | 0.629 | 46/62 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 28 | 138.5 | 0.6833 | 0.8333 | 0.5833 | 10/12 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 75 | 132.9 | 0.6867 | 0.8444 | 0.5778 | 38/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 78 | 130.5 | 0.6952 | 0.7778 | 0.6444 | 35/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 363 | 128.1 | 0.6956 | 0.84 | 0.6 | 63/75 |
| manual__Macbook_air.pdf | 45 | 104 | 116.9 | 0.6978 | 0.8222 | 0.6222 | 37/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 204 | 133.8 | 0.6981 | 0.8667 | 0.5778 | 39/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 28 | 139.9 | 0.7 | 0.8667 | 0.5333 | 13/15 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 14 | 136.9 | 0.7037 | 1.0 | 0.4444 | 9/9 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 269 | 139.2 | 0.7093 | 0.8667 | 0.6 | 39/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 90 | 125.6 | 0.7204 | 0.8444 | 0.6222 | 38/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 151 | 128.2 | 0.7324 | 0.8611 | 0.6389 | 31/36 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 26 | 139.2 | 0.7333 | 0.9333 | 0.6 | 14/15 |
| manual__Guide-for-international-students-web.pdf | 45 | 89 | 106.2 | 0.7333 | 0.8889 | 0.6222 | 40/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 356 | 134.1 | 0.7356 | 0.9556 | 0.6222 | 43/45 |
| manual__t480_ug_en.pdf | 45 | 386 | 130.1 | 0.7378 | 0.8889 | 0.6667 | 40/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 23 | 134.4 | 0.7407 | 0.8889 | 0.6111 | 16/18 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 8 | 111.0 | 0.7407 | 0.8889 | 0.6667 | 8/9 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 42 | 123.3 | 0.747 | 0.8788 | 0.6667 | 29/33 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 75 | 130.8 | 0.7489 | 0.8222 | 0.6889 | 37/45 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 7 | 131.9 | 0.75 | 1.0 | 0.6667 | 3/3 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 91 | 139.0 | 0.7556 | 0.8222 | 0.6889 | 37/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 141 | 142.7 | 0.7574 | 0.9333 | 0.6222 | 42/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 212 | 133.6 | 0.7581 | 0.8667 | 0.6889 | 39/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 126 | 128.6 | 0.7648 | 0.8667 | 0.6889 | 39/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 33 | 129.3 | 0.7667 | 0.9583 | 0.6667 | 23/24 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 125 | 140.4 | 0.7667 | 0.8667 | 0.6889 | 39/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 61 | 128.8 | 0.7718 | 0.881 | 0.6905 | 37/42 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 49 | 142.3 | 0.7741 | 0.8667 | 0.6889 | 39/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 125.2 | 0.7778 | 0.7778 | 0.7778 | 7/9 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 33 | 146.7 | 0.7778 | 0.8333 | 0.7222 | 15/18 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 5 | 137.4 | 0.8056 | 1.0 | 0.6667 | 6/6 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 56 | 130.5 | 0.8056 | 0.963 | 0.7037 | 26/27 |
| manual__nova_y70.pdf | 46 | 91 | 134.9 | 0.8062 | 0.9348 | 0.6957 | 43/46 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 50 | 129.3 | 0.812 | 0.9231 | 0.7179 | 36/39 |
| manual__owners-manual-2170416.pdf | 45 | 62 | 133.1 | 0.8185 | 0.9111 | 0.7556 | 41/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 30 | 146.6 | 0.8194 | 0.8889 | 0.7778 | 16/18 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 34 | 147.3 | 0.8254 | 0.9048 | 0.7619 | 19/21 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 15 | 135.7 | 0.8278 | 1.0 | 0.7778 | 9/9 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 81 | 133.6 | 0.8285 | 0.9111 | 0.7778 | 41/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 17 | 132.5 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 111.5 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 5 | 116.4 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 5 | 110.6 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| manual__2021-Apple-Catalog.pdf | 45 | 60 | 101.0 | 0.837 | 0.8889 | 0.8 | 40/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 95 | 134.1 | 0.8407 | 0.9556 | 0.7556 | 43/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 179 | 128.9 | 0.8407 | 0.9333 | 0.7778 | 42/45 |
| manual__dgx_a100.pdf | 45 | 204 | 108.5 | 0.8407 | 0.9111 | 0.7778 | 41/45 |
| manual__mi_phone.pdf | 45 | 51 | 114.4 | 0.8407 | 0.9111 | 0.7778 | 41/45 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 59 | 138.8 | 0.8419 | 0.9697 | 0.7576 | 32/33 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 21 | 121.2 | 0.8507 | 0.9583 | 0.7917 | 23/24 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 11 | 119.1 | 0.8556 | 0.9333 | 0.8 | 14/15 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 45 | 138.5 | 0.8611 | 0.9333 | 0.8 | 28/30 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 12 | 132.2 | 0.8611 | 0.9167 | 0.8333 | 11/12 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 102 | 131.3 | 0.8667 | 0.9333 | 0.8222 | 42/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 90 | 139.3 | 0.8704 | 0.9333 | 0.8222 | 42/45 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 12 | 128.8 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 37 | 132.6 | 0.8944 | 1.0 | 0.8 | 30/30 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 21 | 132.7 | 0.9074 | 1.0 | 0.8333 | 18/18 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 36 | 125.9 | 0.9074 | 0.9556 | 0.8667 | 43/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 103 | 126.6 | 0.9074 | 0.9778 | 0.8444 | 44/45 |
| manual__watch_d.pdf | 45 | 57 | 131.2 | 0.9296 | 0.9778 | 0.8889 | 44/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 15 | 145.5 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 15 | 132.3 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 17 | 136.7 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 25 | 122.1 | 1.0 | 1.0 | 1.0 | 21/21 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 9 | 121.7 | 1.0 | 1.0 | 1.0 | 6/6 |


### Per-doc breakdown: unstructured (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 1451 | 76.5 | 0.2976 | 0.3623 | 0.2609 | 25/69 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 1204 | 74.8 | 0.3012 | 0.4625 | 0.2 | 37/80 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 100.9 | 0.3889 | 0.4444 | 0.3333 | 4/9 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 35 | 106.3 | 0.3889 | 0.4444 | 0.3556 | 20/45 |
| academic__2305.14160v4.pdf | 72 | 98 | 87.2 | 0.3926 | 0.4722 | 0.3472 | 34/72 |
| academic__2305.02437v3.pdf | 66 | 120 | 74.1 | 0.4028 | 0.4697 | 0.3485 | 31/66 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 1308 | 74.3 | 0.4031 | 0.4938 | 0.358 | 40/81 |
| finance__AES_2022_10K.pdf | 78 | 1481 | 94.5 | 0.4233 | 0.5385 | 0.3462 | 42/78 |
| academic__2403.20330v2.pdf | 69 | 124 | 81.7 | 0.4304 | 0.4783 | 0.3913 | 33/69 |
| manual__DSA-278777.pdf | 45 | 93 | 65.9 | 0.4611 | 0.5111 | 0.4222 | 23/45 |
| academic__2409.01704v1.pdf | 60 | 91 | 95.0 | 0.4631 | 0.6 | 0.3833 | 36/60 |
| finance__AMAZON_2019_10K.pdf | 81 | 537 | 78.4 | 0.4681 | 0.5556 | 0.3951 | 45/81 |
| academic__2402.03216v4.pdf | 78 | 129 | 82.5 | 0.478 | 0.5256 | 0.4487 | 41/78 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 2624 | 88.6 | 0.4899 | 0.5873 | 0.4127 | 37/63 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 98 | 65.9 | 0.4963 | 0.5111 | 0.4889 | 23/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 310 | 49.5 | 0.507 | 0.7111 | 0.3778 | 32/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 555 | 80.6 | 0.5127 | 0.5733 | 0.4667 | 43/75 |
| academic__2404.10198v2.pdf | 51 | 67 | 80.3 | 0.5163 | 0.6078 | 0.451 | 31/51 |
| finance__VERIZON_2021_10K.pdf | 84 | 819 | 88.2 | 0.5194 | 0.6667 | 0.4405 | 56/84 |
| academic__2405.14458v1.pdf | 64 | 107 | 90.7 | 0.5206 | 0.625 | 0.4531 | 40/64 |
| academic__2409.16145v1.pdf | 51 | 112 | 85.0 | 0.5343 | 0.6275 | 0.4706 | 32/51 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 114 | 55.4 | 0.5359 | 0.6444 | 0.4667 | 29/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 20 | 133.6 | 0.5361 | 0.8333 | 0.4167 | 10/12 |
| manual__obs-productdesc-en.pdf | 45 | 216 | 60.6 | 0.5481 | 0.6889 | 0.4444 | 31/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 54 | 114.7 | 0.5524 | 0.5952 | 0.5238 | 25/42 |
| finance__AMD_2022_10K.pdf | 62 | 700 | 89.6 | 0.5621 | 0.6613 | 0.5 | 41/62 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 247 | 128.2 | 0.5752 | 0.7333 | 0.4889 | 33/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 58 | 103.3 | 0.5846 | 0.7037 | 0.5185 | 19/27 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 81 | 109.7 | 0.5878 | 0.6667 | 0.5333 | 30/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 145 | 124.2 | 0.5904 | 0.7333 | 0.4889 | 33/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 675 | 79.6 | 0.595 | 0.6825 | 0.5397 | 43/63 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 25 | 101.4 | 0.5972 | 0.8333 | 0.4167 | 10/12 |
| academic__2405.14831v1.pdf | 63 | 157 | 86.2 | 0.5992 | 0.7778 | 0.4603 | 49/63 |
| academic__2310.11511v1.pdf | 57 | 179 | 89.8 | 0.605 | 0.7018 | 0.5439 | 40/57 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 211 | 111.4 | 0.6093 | 0.7778 | 0.4667 | 35/45 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 9 | 85.9 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 5 | 92.4 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 27 | 137.4 | 0.6157 | 0.7222 | 0.5556 | 13/18 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 12 | 141.1 | 0.6296 | 1.0 | 0.3333 | 9/9 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 107 | 131.9 | 0.6341 | 0.7556 | 0.5556 | 34/45 |
| manual__owners-manual-2170416.pdf | 45 | 140 | 54.6 | 0.6341 | 0.7556 | 0.5556 | 34/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 127 | 49.2 | 0.6519 | 0.7778 | 0.5556 | 35/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 30 | 89.7 | 0.6574 | 0.7778 | 0.5556 | 14/18 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 179 | 93.7 | 0.6574 | 0.8333 | 0.5278 | 30/36 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 86 | 95.7 | 0.663 | 0.7556 | 0.5778 | 34/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 108 | 104.4 | 0.6648 | 0.7778 | 0.5778 | 35/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 15 | 128.7 | 0.6667 | 0.6667 | 0.6667 | 6/9 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 92.5 | 0.6667 | 0.6667 | 0.6667 | 2/3 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 384 | 105.3 | 0.6785 | 0.8 | 0.6222 | 36/45 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 91 | 99.5 | 0.6815 | 0.8 | 0.5778 | 36/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 70 | 116.7 | 0.6852 | 0.7333 | 0.6444 | 33/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 105 | 90.6 | 0.6889 | 0.8 | 0.6 | 36/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 9 | 84.2 | 0.6944 | 0.8889 | 0.5556 | 8/9 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 19 | 148.7 | 0.7 | 0.9167 | 0.5833 | 11/12 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 30 | 107.2 | 0.7042 | 0.9167 | 0.5833 | 11/12 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 47 | 94.4 | 0.7146 | 0.8485 | 0.6364 | 28/33 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 13 | 121.9 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 66 | 87.7 | 0.7329 | 0.8462 | 0.6667 | 33/39 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 34 | 91.0 | 0.7389 | 0.9333 | 0.6 | 14/15 |
| manual__mi_phone.pdf | 45 | 53 | 93.8 | 0.7407 | 0.8 | 0.6889 | 36/45 |
| manual__dgx_a100.pdf | 45 | 324 | 64.1 | 0.7452 | 0.8444 | 0.6889 | 38/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 92 | 124.4 | 0.747 | 0.8444 | 0.6889 | 38/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 31 | 133.5 | 0.75 | 0.8889 | 0.6667 | 16/18 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 7 | 128.6 | 0.75 | 0.8333 | 0.6667 | 5/6 |
| manual__watch_d.pdf | 45 | 92 | 73.7 | 0.7537 | 0.9111 | 0.6222 | 41/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 228 | 104.1 | 0.7574 | 0.8667 | 0.6889 | 39/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 126 | 134.1 | 0.763 | 0.8444 | 0.6889 | 38/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 79 | 130.4 | 0.7648 | 0.8444 | 0.7111 | 38/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 40 | 137.7 | 0.7678 | 0.8667 | 0.7 | 26/30 |
| manual__t480_ug_en.pdf | 45 | 463 | 97.9 | 0.7681 | 0.9333 | 0.6444 | 42/45 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 19 | 92.5 | 0.7778 | 1.0 | 0.6667 | 9/9 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 53 | 131.2 | 0.7778 | 0.8788 | 0.697 | 29/33 |
| manual__nova_y70.pdf | 46 | 167 | 67.0 | 0.7804 | 0.913 | 0.6739 | 42/46 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 40 | 111.0 | 0.7917 | 0.9333 | 0.6667 | 28/30 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 109 | 104.7 | 0.7926 | 0.9111 | 0.6889 | 41/45 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 14 | 124.9 | 0.8056 | 1.0 | 0.6667 | 9/9 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 35 | 102.2 | 0.8333 | 1.0 | 0.7083 | 24/24 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 5 | 116.2 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 10 | 132.9 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 31 | 136.9 | 0.8333 | 0.9048 | 0.7619 | 19/21 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 130 | 111.0 | 0.8444 | 0.9333 | 0.7778 | 42/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 146 | 81.8 | 0.8519 | 0.9333 | 0.8 | 42/45 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 24 | 109.9 | 0.8667 | 0.9048 | 0.8571 | 19/21 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 13 | 144.5 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 24 | 142.2 | 0.8722 | 1.0 | 0.8 | 15/15 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 20 | 108.3 | 0.8785 | 0.9583 | 0.8333 | 23/24 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 17 | 136.3 | 0.8796 | 1.0 | 0.7778 | 18/18 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 19 | 101.6 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 14 | 129.1 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 11 | 107.5 | 0.9022 | 1.0 | 0.8667 | 15/15 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 13 | 109.5 | 0.9028 | 1.0 | 0.8333 | 12/12 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 7 | 75.1 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: llamaindex (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2404.10198v2.pdf | 51 | 22 | 63.8 | 0.0196 | 0.0196 | 0.0196 | 1/51 |
| academic__2305.02437v3.pdf | 66 | 41 | 92.3 | 0.0417 | 0.0606 | 0.0303 | 4/66 |
| academic__2305.14160v4.pdf | 72 | 35 | 128.6 | 0.0514 | 0.0694 | 0.0417 | 5/72 |
| academic__2310.11511v1.pdf | 57 | 64 | 87.0 | 0.0526 | 0.0702 | 0.0351 | 4/57 |
| academic__2409.01704v1.pdf | 60 | 37 | 88.2 | 0.0542 | 0.0667 | 0.05 | 4/60 |
| academic__2402.03216v4.pdf | 78 | 52 | 128.6 | 0.0667 | 0.0897 | 0.0513 | 7/78 |
| academic__2405.14458v1.pdf | 64 | 46 | 70.8 | 0.082 | 0.0938 | 0.0781 | 6/64 |
| academic__2403.20330v2.pdf | 69 | 54 | 131.2 | 0.1087 | 0.1449 | 0.087 | 10/69 |
| manual__DSA-278777.pdf | 45 | 27 | 164.0 | 0.2044 | 0.2222 | 0.2 | 10/45 |
| academic__2405.14831v1.pdf | 63 | 62 | 94.3 | 0.3558 | 0.4127 | 0.3175 | 26/63 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 429 | 275.4 | 0.3676 | 0.4928 | 0.2899 | 34/69 |
| academic__2409.16145v1.pdf | 51 | 37 | 230.4 | 0.3935 | 0.5294 | 0.3137 | 27/51 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 374 | 276.0 | 0.4076 | 0.5556 | 0.3086 | 45/81 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 338 | 282.7 | 0.4612 | 0.6 | 0.375 | 48/80 |
| finance__AES_2022_10K.pdf | 78 | 453 | 327.5 | 0.5419 | 0.6667 | 0.4615 | 52/78 |
| manual__obs-productdesc-en.pdf | 45 | 43 | 327.3 | 0.5915 | 0.7556 | 0.4889 | 34/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 35 | 245.9 | 0.5944 | 0.6667 | 0.5333 | 30/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 32 | 300.2 | 0.597 | 0.7778 | 0.4667 | 35/45 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 8 | 337.9 | 0.5972 | 0.8333 | 0.4167 | 10/12 |
| finance__3M_2023Q2_10Q.pdf | 63 | 178 | 321.3 | 0.6032 | 0.6984 | 0.5397 | 44/63 |
| finance__VERIZON_2021_10K.pdf | 84 | 237 | 326.6 | 0.6153 | 0.7857 | 0.5119 | 66/84 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 791 | 310.1 | 0.6156 | 0.7778 | 0.5238 | 49/63 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 67 | 299.3 | 0.6185 | 0.7111 | 0.5333 | 32/45 |
| manual__Macbook_air.pdf | 45 | 35 | 348.5 | 0.6204 | 0.7333 | 0.5333 | 33/45 |
| finance__AMAZON_2019_10K.pdf | 81 | 139 | 321.8 | 0.6315 | 0.7654 | 0.5556 | 62/81 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 35 | 360.7 | 0.6348 | 0.7778 | 0.5778 | 35/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 33 | 355.4 | 0.6507 | 0.7778 | 0.5556 | 35/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 134 | 328.1 | 0.6537 | 0.7556 | 0.5778 | 34/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 148 | 321.0 | 0.6549 | 0.8267 | 0.5333 | 62/75 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 6 | 334.0 | 0.6574 | 1.0 | 0.4444 | 9/9 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 34 | 311.8 | 0.6644 | 0.8 | 0.5778 | 36/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 112 | 314.1 | 0.6663 | 0.8222 | 0.5778 | 37/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 320.3 | 0.6667 | 0.7778 | 0.5556 | 7/9 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 2 | 254.0 | 0.6667 | 1.0 | 0.3333 | 3/3 |
| manual__Guide-for-international-students-web.pdf | 45 | 32 | 296.1 | 0.673 | 0.8667 | 0.5333 | 39/45 |
| manual__dgx_a100.pdf | 45 | 86 | 253.8 | 0.6741 | 0.7556 | 0.6 | 34/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 35 | 338.9 | 0.6804 | 0.8444 | 0.5556 | 38/45 |
| finance__AMD_2022_10K.pdf | 62 | 197 | 338.1 | 0.682 | 0.7742 | 0.629 | 48/62 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 78 | 326.9 | 0.6941 | 0.8889 | 0.5556 | 40/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 21 | 346.5 | 0.696 | 0.881 | 0.5952 | 37/42 |
| manual__mi_phone.pdf | 45 | 17 | 347.3 | 0.6963 | 0.8444 | 0.5778 | 38/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 27 | 343.5 | 0.6981 | 0.8 | 0.6222 | 36/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 58 | 317.2 | 0.7037 | 0.7778 | 0.6389 | 28/36 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 57 | 285.5 | 0.7052 | 0.8667 | 0.6 | 39/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 50 | 325.2 | 0.7056 | 0.8444 | 0.6222 | 38/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 15 | 312.5 | 0.713 | 0.8333 | 0.6111 | 15/18 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 14 | 295.6 | 0.7148 | 0.8889 | 0.6111 | 16/18 |
| manual__t480_ug_en.pdf | 45 | 137 | 363.1 | 0.7185 | 0.7778 | 0.6667 | 35/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 28 | 218.5 | 0.7193 | 0.8444 | 0.6444 | 38/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 269.0 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 61 | 311.3 | 0.7319 | 0.9333 | 0.6222 | 42/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 28 | 328.6 | 0.7426 | 0.8444 | 0.6889 | 38/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 18 | 322.3 | 0.7483 | 0.9 | 0.6667 | 27/30 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 4 | 255.2 | 0.75 | 1.0 | 0.5 | 6/6 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 51 | 307.6 | 0.7519 | 0.8222 | 0.6889 | 37/45 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 6 | 329.3 | 0.7593 | 0.8889 | 0.6667 | 8/9 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 74 | 356.6 | 0.7607 | 0.8889 | 0.6667 | 40/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 77 | 300.4 | 0.7656 | 0.9111 | 0.6667 | 41/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 23 | 304.3 | 0.7667 | 0.963 | 0.6296 | 26/27 |
| manual__owners-manual-2170416.pdf | 45 | 25 | 331.4 | 0.7889 | 0.8667 | 0.7333 | 39/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 9 | 336.7 | 0.7917 | 1.0 | 0.6667 | 12/12 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 6 | 341.8 | 0.7963 | 1.0 | 0.6667 | 9/9 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 10 | 288.1 | 0.7986 | 1.0 | 0.6667 | 12/12 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 23 | 296.5 | 0.8 | 0.8889 | 0.7333 | 40/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 19 | 314.8 | 0.8043 | 0.8974 | 0.7436 | 35/39 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 11 | 324.8 | 0.8056 | 0.9333 | 0.7333 | 14/15 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 9 | 330.3 | 0.8074 | 0.9444 | 0.7222 | 17/18 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 22 | 341.6 | 0.8131 | 0.9394 | 0.697 | 31/33 |
| manual__honor_watch_gs_pro.pdf | 45 | 36 | 356.3 | 0.8148 | 0.9556 | 0.6889 | 43/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 15 | 316.3 | 0.8157 | 0.9394 | 0.7576 | 31/33 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 6 | 327.3 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 12 | 298.0 | 0.8333 | 0.9167 | 0.75 | 11/12 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 5 | 300.2 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 7 | 311.1 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 370.2 | 0.8333 | 1.0 | 0.6667 | 12/12 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 37 | 321.0 | 0.8537 | 0.9333 | 0.8 | 42/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 7 | 360.9 | 0.8542 | 0.9583 | 0.75 | 23/24 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 10 | 334.2 | 0.8556 | 1.0 | 0.7333 | 15/15 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 6 | 308.7 | 0.8611 | 1.0 | 0.7778 | 9/9 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 29 | 340.3 | 0.8648 | 0.9778 | 0.7778 | 44/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 9 | 279.4 | 0.8657 | 1.0 | 0.7778 | 18/18 |
| manual__nova_y70.pdf | 46 | 31 | 394.4 | 0.8659 | 0.8913 | 0.8478 | 41/46 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 15 | 295.2 | 0.8685 | 0.9556 | 0.8 | 43/45 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 4 | 323.0 | 0.8889 | 1.0 | 0.8 | 15/15 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 337.0 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 10 | 286.5 | 0.8905 | 1.0 | 0.8095 | 21/21 |
| manual__watch_d.pdf | 45 | 19 | 394.5 | 0.9 | 1.0 | 0.8 | 45/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 12 | 322.0 | 0.9097 | 1.0 | 0.8333 | 24/24 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 14 | 333.9 | 0.9143 | 1.0 | 0.8571 | 21/21 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 319.5 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 13 | 359.5 | 0.9333 | 1.0 | 0.9 | 30/30 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 6 | 304.3 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 280.7 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 216.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 268.5 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: llamaindex_semantic (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2405.14458v1.pdf | 64 | 30 | 101.4 | 0.0469 | 0.0469 | 0.0469 | 3/64 |
| academic__2404.10198v2.pdf | 51 | 15 | 86.2 | 0.0588 | 0.0784 | 0.0392 | 4/51 |
| academic__2409.01704v1.pdf | 60 | 18 | 174.3 | 0.0667 | 0.1 | 0.05 | 6/60 |
| academic__2310.11511v1.pdf | 57 | 56 | 92.5 | 0.0848 | 0.1228 | 0.0526 | 7/57 |
| academic__2305.14160v4.pdf | 72 | 21 | 206.9 | 0.0975 | 0.1806 | 0.0556 | 13/72 |
| academic__2305.02437v3.pdf | 66 | 29 | 123.7 | 0.1023 | 0.1515 | 0.0758 | 10/66 |
| academic__2403.20330v2.pdf | 69 | 25 | 265.0 | 0.1527 | 0.2319 | 0.1014 | 16/69 |
| academic__2405.14831v1.pdf | 63 | 38 | 145.5 | 0.2434 | 0.3016 | 0.1905 | 19/63 |
| manual__DSA-278777.pdf | 45 | 10 | 425.0 | 0.2833 | 0.3556 | 0.2222 | 16/45 |
| manual__t480_ug_en.pdf | 45 | 245 | 187.5 | 0.3204 | 0.6667 | 0.1778 | 30/45 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 141 | 691.1 | 0.3521 | 0.5802 | 0.2469 | 47/81 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 164 | 678.8 | 0.3601 | 0.5507 | 0.2609 | 38/69 |
| academic__2402.03216v4.pdf | 78 | 27 | 230.8 | 0.3643 | 0.4487 | 0.3205 | 35/78 |
| finance__AES_2022_10K.pdf | 78 | 231 | 607.5 | 0.4656 | 0.6154 | 0.3846 | 48/78 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 14 | 591.7 | 0.5019 | 0.8444 | 0.3111 | 38/45 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 138 | 655.0 | 0.51 | 0.6625 | 0.4125 | 53/80 |
| manual__obs-productdesc-en.pdf | 45 | 36 | 365.5 | 0.5226 | 0.7333 | 0.4 | 33/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 77 | 581.5 | 0.5327 | 0.6667 | 0.44 | 50/75 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 6 | 437.0 | 0.5417 | 0.8333 | 0.3333 | 10/12 |
| manual__Macbook_air.pdf | 45 | 29 | 396.3 | 0.5444 | 0.6889 | 0.4667 | 31/45 |
| finance__AMD_2022_10K.pdf | 62 | 115 | 545.3 | 0.5538 | 0.6935 | 0.4677 | 43/62 |
| finance__VERIZON_2021_10K.pdf | 84 | 123 | 592.8 | 0.5657 | 0.7143 | 0.4643 | 60/84 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 436 | 535.7 | 0.573 | 0.7619 | 0.4762 | 48/63 |
| finance__3M_2023Q2_10Q.pdf | 63 | 85 | 632.3 | 0.5836 | 0.7619 | 0.4762 | 48/63 |
| academic__2409.16145v1.pdf | 51 | 28 | 287.5 | 0.5954 | 0.7255 | 0.5294 | 37/51 |
| finance__AMAZON_2019_10K.pdf | 81 | 73 | 578.0 | 0.6064 | 0.7531 | 0.5309 | 61/81 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 11 | 814.9 | 0.6107 | 0.9111 | 0.4 | 41/45 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 273.3 | 0.6111 | 1.0 | 0.3333 | 3/3 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 14 | 660.2 | 0.6248 | 0.7778 | 0.5111 | 35/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 5 | 835.4 | 0.6259 | 1.0 | 0.3333 | 45/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 21 | 427.2 | 0.6259 | 0.8667 | 0.4667 | 39/45 |
| manual__nova_y70.pdf | 46 | 32 | 353.3 | 0.6264 | 0.7609 | 0.5435 | 35/46 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 10 | 700.8 | 0.6313 | 0.8571 | 0.5238 | 36/42 |
| manual__dgx_a100.pdf | 45 | 43 | 477.9 | 0.6396 | 0.7556 | 0.5556 | 34/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 36 | 427.8 | 0.6415 | 0.7556 | 0.5556 | 34/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 33 | 469.3 | 0.6663 | 0.8667 | 0.5556 | 39/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 4 | 225.0 | 0.6667 | 0.7778 | 0.5556 | 7/9 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 22 | 805.8 | 0.6745 | 0.8611 | 0.5833 | 31/36 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 8 | 567.4 | 0.6762 | 1.0 | 0.4762 | 21/21 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 28 | 690.8 | 0.6785 | 0.8222 | 0.5778 | 37/45 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 4 | 449.0 | 0.6852 | 1.0 | 0.4444 | 9/9 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 49 | 872.3 | 0.6878 | 0.7778 | 0.6222 | 35/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 711.2 | 0.6944 | 1.0 | 0.4167 | 12/12 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 34 | 721.4 | 0.7026 | 0.9778 | 0.5333 | 44/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 7 | 413.1 | 0.7083 | 1.0 | 0.4167 | 12/12 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 39 | 643.7 | 0.7107 | 0.9111 | 0.6222 | 41/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 5 | 394.0 | 0.713 | 1.0 | 0.5556 | 9/9 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 28 | 226.9 | 0.7148 | 0.8667 | 0.5778 | 39/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 5 | 382.2 | 0.7222 | 1.0 | 0.4444 | 9/9 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 268.3 | 0.7222 | 0.8889 | 0.5556 | 8/9 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 8 | 553.6 | 0.7269 | 0.9444 | 0.5556 | 17/18 |
| manual__owners-manual-2170416.pdf | 45 | 39 | 195.6 | 0.7289 | 0.8889 | 0.6667 | 40/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 12 | 747.2 | 0.7296 | 0.8889 | 0.6 | 40/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 17 | 665.5 | 0.73 | 0.8667 | 0.6667 | 39/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 22 | 513.5 | 0.7304 | 0.8889 | 0.6 | 40/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 16 | 755.8 | 0.7533 | 1.0 | 0.6 | 45/45 |
| manual__mi_phone.pdf | 45 | 19 | 289.8 | 0.7533 | 0.9111 | 0.6667 | 41/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 8 | 431.2 | 0.7556 | 0.9333 | 0.6 | 14/15 |
| manual__2021-Apple-Catalog.pdf | 45 | 6 | 969.3 | 0.7556 | 0.9111 | 0.6222 | 41/45 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 22 | 516.7 | 0.757 | 0.9778 | 0.6222 | 44/45 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 5 | 408.8 | 0.7593 | 1.0 | 0.5556 | 9/9 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 6 | 399.0 | 0.7639 | 0.9583 | 0.5833 | 23/24 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 7 | 670.4 | 0.7652 | 1.0 | 0.6061 | 33/33 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 5 | 652.0 | 0.7667 | 1.0 | 0.6 | 15/15 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 183 | 118.2 | 0.7674 | 0.8889 | 0.6889 | 40/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 6 | 668.2 | 0.7778 | 1.0 | 0.5556 | 18/18 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 10 | 555.8 | 0.7806 | 0.9333 | 0.6667 | 28/30 |
| manual__honor_watch_gs_pro.pdf | 45 | 36 | 334.1 | 0.7815 | 0.9333 | 0.6667 | 42/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 8 | 355.2 | 0.787 | 1.0 | 0.6111 | 18/18 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 359.2 | 0.7917 | 1.0 | 0.6667 | 12/12 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 9 | 387.2 | 0.7944 | 1.0 | 0.6667 | 12/12 |
| manual__watch_d.pdf | 45 | 20 | 344.1 | 0.8007 | 0.9111 | 0.7333 | 41/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 27 | 676.5 | 0.8093 | 0.9556 | 0.7111 | 43/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 7 | 634.4 | 0.8111 | 1.0 | 0.6667 | 30/30 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 11 | 528.7 | 0.8128 | 0.9744 | 0.7179 | 38/39 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 18 | 567.4 | 0.8137 | 0.9556 | 0.7333 | 43/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 15 | 440.9 | 0.816 | 0.9259 | 0.7407 | 25/27 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 20 | 761.4 | 0.8193 | 0.9556 | 0.7111 | 43/45 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 9 | 812.0 | 0.8333 | 1.0 | 0.697 | 33/33 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 108.0 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 4 | 341.8 | 0.8333 | 0.8889 | 0.7778 | 8/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 326.6 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 67 | 503.1 | 0.8378 | 0.9333 | 0.7556 | 42/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 5 | 756.8 | 0.8542 | 1.0 | 0.7083 | 24/24 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 5 | 370.0 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 13 | 738.3 | 0.8722 | 1.0 | 0.7778 | 45/45 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 5 | 343.2 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 4 | 243.2 | 0.8889 | 1.0 | 0.8333 | 6/6 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 3 | 412.0 | 0.8889 | 1.0 | 0.8 | 15/15 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 4 | 462.5 | 0.8889 | 0.8889 | 0.8889 | 8/9 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 4 | 619.5 | 0.9444 | 1.0 | 0.8889 | 18/18 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 5 | 553.0 | 0.9683 | 1.0 | 0.9524 | 21/21 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 315.0 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 263.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 3 | 168.7 | 1.0 | 1.0 | 1.0 | 3/3 |
