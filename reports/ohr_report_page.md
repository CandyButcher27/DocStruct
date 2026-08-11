# DocStruct retrieval baseline report

_Generated 2026-08-11 19:07 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: unstructured, langchain, llamaindex, pymupdf4llm, llamaindex_semantic, docstruct, docstruct_geo.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | unstructured | **0.795** | [0.7835, 0.8062] | 0.8062 | 0.9194 | 0.7125 | 0.7484 | +0.0466 | 18424 | 87.2 | 560.5 | 1.4184 | 1065.85 | 3 |
| 2 | langchain | **0.7562** | [0.744, 0.768] | 0.773 | 0.8879 | 0.6692 | 0.7122 | +0.044 | 13877 | 128.5 | 637.5 | 1.1862 | 646.07 | 0 |
| 3 | llamaindex | **0.7294** | [0.717, 0.7418] | 0.7513 | 0.8822 | 0.6287 | 0.6707 | +0.0587 | 5794 | 295.2 | 1430.1 | 0.51 | 641.03 | 0 |
| 4 | pymupdf4llm | **0.6684** | [0.6552, 0.6814] | 0.7116 | 0.8406 | 0.5573 | 0.5692 | +0.0992 | 3756 | 425.8 | 2434.9 | 0.2745 | 2353.95 | 0 |
| 5 | llamaindex_semantic | **0.6515** | [0.6387, 0.6644] | 0.6957 | 0.8617 | 0.518 | 0.5058 | +0.1457 | 3366 | 482.0 | 4697.7 | 0.1387 | 948.83 | 0 |
| 6 | docstruct **(ours)** | **0.6004** | [0.5864, 0.6148] | 0.6284 | 0.753 | 0.5039 | 0.5588 | +0.0416 | 9080 | 316.8 | 2194.3 | 0.2736 | 1843.41 | 0 |
| 7 | docstruct_geo | **0.4703** | [0.4559, 0.4854] | 0.5011 | 0.613 | 0.3811 | 0.4302 | +0.0401 | 5810 | 306.0 | 2328.6 | 0.202 | 853.37 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| langchain | 1.0 | 1.1005 |
| llamaindex | 1.0 | 1.0474 |
| llamaindex_semantic | 1.0 | 1.0 |
| pymupdf4llm | 0.9674 | 1.1077 |
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
| unstructured | MRR | -0.1881 | [-0.2028, -0.1733] | 0.0001 | 3423 | **significant** |
| unstructured | NDCG | -0.1715 | [-0.1853, -0.1579] | 0.0001 | 3423 | **significant** |
| unstructured | RECALL | -0.1598 | [-0.1747, -0.1452] | 0.0001 | 3423 | **significant** |
| unstructured | HIT1 | -0.2022 | [-0.2209, -0.1832] | 0.0001 | 3423 | **significant** |
| langchain | MRR | -0.155 | [-0.1694, -0.1404] | 0.0001 | 3558 | **significant** |
| langchain | NDCG | -0.1441 | [-0.1577, -0.1305] | 0.0001 | 3558 | **significant** |
| langchain | RECALL | -0.1346 | [-0.1498, -0.12] | 0.0001 | 3558 | **significant** |
| langchain | HIT1 | -0.1644 | [-0.1827, -0.1459] | 0.0001 | 3558 | **significant** |
| llamaindex | MRR | -0.1279 | [-0.1431, -0.1126] | 0.0001 | 3558 | **significant** |
| llamaindex | NDCG | -0.122 | [-0.136, -0.1078] | 0.0001 | 3558 | **significant** |
| llamaindex | RECALL | -0.1284 | [-0.1436, -0.1133] | 0.0001 | 3558 | **significant** |
| llamaindex | HIT1 | -0.1237 | [-0.1425, -0.1046] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | MRR | -0.0672 | [-0.0835, -0.0506] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | NDCG | -0.0824 | [-0.0977, -0.0668] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | RECALL | -0.0868 | [-0.1032, -0.0703] | 0.0001 | 3558 | **significant** |
| pymupdf4llm | HIT1 | -0.0528 | [-0.0728, -0.0326] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | MRR | -0.0509 | [-0.0682, -0.0334] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | NDCG | -0.0673 | [-0.0832, -0.0513] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | RECALL | -0.1088 | [-0.1256, -0.0919] | 0.0001 | 3558 | **significant** |
| llamaindex_semantic | HIT1 | -0.0138 | [-0.0354, 0.0079] | 0.214 | 3558 | not significant |
| docstruct_geo | MRR | +0.1305 | [0.115, 0.1461] | 0.0001 | 3558 | **significant** |
| docstruct_geo | NDCG | +0.1279 | [0.1128, 0.143] | 0.0001 | 3558 | **significant** |
| docstruct_geo | RECALL | +0.1411 | [0.1237, 0.158] | 0.0001 | 3558 | **significant** |
| docstruct_geo | HIT1 | +0.1228 | [0.1043, 0.1408] | 0.0001 | 3558 | **significant** |

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


### Per-doc breakdown: unstructured (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| finance__AES_2022_10K.pdf | 78 | 1481 | 94.5 | 0.5607 | 0.7179 | 0.4615 | 56/78 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 98 | 65.9 | 0.5996 | 0.6889 | 0.5556 | 31/45 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 1308 | 74.3 | 0.6401 | 0.8025 | 0.5309 | 65/81 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 1451 | 76.5 | 0.6423 | 0.8261 | 0.5362 | 57/69 |
| academic__2409.01704v1.pdf | 60 | 91 | 95.0 | 0.6533 | 0.8167 | 0.55 | 49/60 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 1204 | 74.8 | 0.6806 | 0.825 | 0.575 | 66/80 |
| finance__AMD_2022_10K.pdf | 62 | 700 | 89.6 | 0.6968 | 0.8871 | 0.5968 | 55/62 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 211 | 111.4 | 0.6981 | 0.8222 | 0.6 | 37/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 247 | 128.2 | 0.7056 | 0.8889 | 0.6222 | 40/45 |
| manual__obs-productdesc-en.pdf | 45 | 216 | 60.6 | 0.7193 | 0.9111 | 0.6 | 41/45 |
| academic__2409.16145v1.pdf | 51 | 112 | 85.0 | 0.7206 | 0.902 | 0.5882 | 46/51 |
| finance__3M_2023Q2_10Q.pdf | 63 | 675 | 79.6 | 0.7222 | 0.8254 | 0.6508 | 52/63 |
| academic__2403.20330v2.pdf | 69 | 124 | 81.7 | 0.7232 | 0.9275 | 0.5797 | 64/69 |
| finance__AMAZON_2017_10K.pdf | 75 | 555 | 80.6 | 0.7256 | 0.9067 | 0.6133 | 68/75 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 70 | 116.7 | 0.7267 | 0.8667 | 0.6444 | 39/45 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 2624 | 88.6 | 0.7291 | 0.9048 | 0.619 | 57/63 |
| finance__VERIZON_2021_10K.pdf | 84 | 819 | 88.2 | 0.7292 | 0.9405 | 0.6071 | 79/84 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 9 | 84.2 | 0.7407 | 1.0 | 0.5556 | 9/9 |
| finance__AMAZON_2019_10K.pdf | 81 | 537 | 78.4 | 0.7481 | 0.9012 | 0.642 | 73/81 |
| academic__2405.14458v1.pdf | 64 | 107 | 90.7 | 0.7497 | 0.9219 | 0.6406 | 59/64 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 114 | 55.4 | 0.7526 | 0.8667 | 0.6889 | 39/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 30 | 89.7 | 0.7593 | 0.8889 | 0.6667 | 16/18 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 105 | 90.6 | 0.7593 | 0.8444 | 0.7111 | 38/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 384 | 105.3 | 0.76 | 0.8889 | 0.6889 | 40/45 |
| academic__2305.02437v3.pdf | 66 | 120 | 74.1 | 0.7644 | 0.9242 | 0.6515 | 61/66 |
| manual__2021-Apple-Catalog.pdf | 45 | 127 | 49.2 | 0.7722 | 0.9556 | 0.6444 | 43/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 179 | 93.7 | 0.7731 | 0.9167 | 0.6667 | 33/36 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 24 | 109.9 | 0.7754 | 0.9048 | 0.7143 | 19/21 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 35 | 106.3 | 0.78 | 0.9333 | 0.7111 | 42/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 108 | 104.4 | 0.7833 | 0.8444 | 0.7333 | 38/45 |
| manual__t480_ug_en.pdf | 45 | 463 | 97.9 | 0.7867 | 0.9556 | 0.6667 | 43/45 |
| academic__2404.10198v2.pdf | 51 | 67 | 80.3 | 0.7882 | 0.8824 | 0.7255 | 45/51 |
| academic__2405.14831v1.pdf | 63 | 157 | 86.2 | 0.7921 | 0.9683 | 0.6667 | 61/63 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 47 | 94.4 | 0.7955 | 0.8788 | 0.7273 | 29/33 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 53 | 131.2 | 0.803 | 0.9394 | 0.697 | 31/33 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 310 | 49.5 | 0.8063 | 0.9778 | 0.7111 | 44/45 |
| academic__2305.14160v4.pdf | 72 | 98 | 87.2 | 0.81 | 0.9722 | 0.6944 | 70/72 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 86 | 95.7 | 0.8111 | 0.8667 | 0.7556 | 39/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 145 | 124.2 | 0.8237 | 0.9556 | 0.7333 | 43/45 |
| academic__2402.03216v4.pdf | 78 | 129 | 82.5 | 0.8301 | 0.9231 | 0.7564 | 72/78 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 17 | 136.3 | 0.8306 | 0.9444 | 0.7778 | 17/18 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 92 | 124.4 | 0.8333 | 0.9778 | 0.7333 | 44/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 15 | 128.7 | 0.8333 | 1.0 | 0.7778 | 9/9 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 5 | 116.2 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 5 | 92.4 | 0.8333 | 1.0 | 0.6667 | 3/3 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 81 | 109.7 | 0.8407 | 0.9778 | 0.7333 | 44/45 |
| manual__mi_phone.pdf | 45 | 53 | 93.8 | 0.8407 | 0.8889 | 0.8 | 40/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 228 | 104.1 | 0.8444 | 0.9111 | 0.8 | 41/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 58 | 103.3 | 0.8519 | 0.9259 | 0.8148 | 25/27 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 66 | 87.7 | 0.8568 | 0.9744 | 0.7692 | 38/39 |
| manual__nova_y70.pdf | 46 | 167 | 67.0 | 0.8601 | 0.9783 | 0.7826 | 45/46 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 126 | 134.1 | 0.8611 | 0.9333 | 0.8222 | 42/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 19 | 148.7 | 0.8611 | 1.0 | 0.75 | 12/12 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 107 | 131.9 | 0.8663 | 0.9778 | 0.8 | 44/45 |
| manual__DSA-278777.pdf | 45 | 93 | 65.9 | 0.8674 | 1.0 | 0.7778 | 45/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 24 | 142.2 | 0.8722 | 1.0 | 0.8 | 15/15 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 54 | 114.7 | 0.8738 | 0.9762 | 0.8095 | 41/42 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 91 | 99.5 | 0.8759 | 0.9556 | 0.8222 | 43/45 |
| academic__2310.11511v1.pdf | 57 | 179 | 89.8 | 0.8865 | 0.9649 | 0.8246 | 55/57 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 35 | 102.2 | 0.8889 | 1.0 | 0.7917 | 24/24 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 14 | 129.1 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 13 | 121.9 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 40 | 137.7 | 0.8944 | 1.0 | 0.8333 | 30/30 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 20 | 133.6 | 0.8958 | 1.0 | 0.8333 | 12/12 |
| manual__watch_d.pdf | 45 | 92 | 73.7 | 0.8981 | 0.9778 | 0.8444 | 44/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 20 | 108.3 | 0.8993 | 1.0 | 0.8333 | 24/24 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 34 | 91.0 | 0.9 | 1.0 | 0.8 | 15/15 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 100.9 | 0.9111 | 1.0 | 0.8889 | 9/9 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 19 | 92.5 | 0.9167 | 1.0 | 0.8889 | 9/9 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 25 | 101.4 | 0.9167 | 1.0 | 0.8333 | 12/12 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 31 | 133.5 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| manual__dgx_a100.pdf | 45 | 324 | 64.1 | 0.9222 | 1.0 | 0.8667 | 45/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 13 | 144.5 | 0.9259 | 1.0 | 0.8889 | 9/9 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 31 | 136.9 | 0.9286 | 1.0 | 0.8571 | 21/21 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 130 | 111.0 | 0.9296 | 0.9778 | 0.8889 | 44/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 40 | 111.0 | 0.9333 | 1.0 | 0.8667 | 30/30 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 27 | 137.4 | 0.9352 | 1.0 | 0.8889 | 18/18 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 79 | 130.4 | 0.9378 | 0.9556 | 0.9333 | 43/45 |
| manual__owners-manual-2170416.pdf | 45 | 140 | 54.6 | 0.9407 | 0.9778 | 0.9111 | 44/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 146 | 81.8 | 0.9426 | 1.0 | 0.9111 | 45/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 30 | 107.2 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 10 | 132.9 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 19 | 101.6 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 109 | 104.7 | 0.9778 | 1.0 | 0.9556 | 45/45 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 9 | 85.9 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 92.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 12 | 141.1 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 7 | 128.6 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 7 | 75.1 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 11 | 107.5 | 1.0 | 1.0 | 1.0 | 15/15 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 14 | 124.9 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 13 | 109.5 | 1.0 | 1.0 | 1.0 | 12/12 |


### Per-doc breakdown: langchain (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2409.01704v1.pdf | 60 | 67 | 52.0 | 0.3664 | 0.4833 | 0.3 | 29/60 |
| academic__2405.14458v1.pdf | 64 | 74 | 44.5 | 0.4365 | 0.6719 | 0.2969 | 43/64 |
| academic__2305.02437v3.pdf | 66 | 68 | 57.7 | 0.4439 | 0.6364 | 0.3333 | 42/66 |
| academic__2402.03216v4.pdf | 78 | 81 | 86.0 | 0.4558 | 0.6795 | 0.3333 | 53/78 |
| academic__2404.10198v2.pdf | 51 | 42 | 33.7 | 0.4614 | 0.7255 | 0.2941 | 37/51 |
| academic__2405.14831v1.pdf | 63 | 105 | 57.8 | 0.5336 | 0.8413 | 0.3333 | 53/63 |
| academic__2403.20330v2.pdf | 69 | 81 | 90.5 | 0.563 | 0.7536 | 0.4493 | 52/69 |
| academic__2310.11511v1.pdf | 57 | 119 | 47.0 | 0.5772 | 0.7193 | 0.4912 | 41/57 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 787 | 136.9 | 0.5924 | 0.8025 | 0.4568 | 65/81 |
| academic__2305.14160v4.pdf | 72 | 64 | 74.5 | 0.609 | 0.8194 | 0.4722 | 59/72 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 907 | 135.8 | 0.6452 | 0.8406 | 0.5217 | 58/69 |
| finance__AES_2022_10K.pdf | 78 | 1167 | 134.4 | 0.6455 | 0.7949 | 0.5641 | 62/78 |
| manual__obs-productdesc-en.pdf | 45 | 125 | 113.5 | 0.6533 | 0.8667 | 0.5111 | 39/45 |
| manual__DSA-278777.pdf | 45 | 47 | 99.9 | 0.6926 | 0.8889 | 0.5778 | 40/45 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 744 | 134.5 | 0.7115 | 0.8125 | 0.6375 | 65/80 |
| finance__AMD_2022_10K.pdf | 62 | 513 | 128.9 | 0.7207 | 0.8387 | 0.6613 | 52/62 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 1921 | 135.5 | 0.7228 | 0.8413 | 0.6508 | 53/63 |
| academic__2409.16145v1.pdf | 51 | 78 | 112.6 | 0.7245 | 0.8824 | 0.6078 | 45/51 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 204 | 133.8 | 0.7296 | 0.8667 | 0.6444 | 39/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 77 | 118.3 | 0.7333 | 0.8222 | 0.6444 | 37/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 269 | 139.2 | 0.7333 | 0.8667 | 0.6444 | 39/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 75 | 130.8 | 0.7359 | 0.8222 | 0.6889 | 37/45 |
| finance__VERIZON_2021_10K.pdf | 84 | 627 | 131.6 | 0.7373 | 0.9405 | 0.5833 | 79/84 |
| manual__Guide-for-international-students-web.pdf | 45 | 89 | 106.2 | 0.7396 | 0.9333 | 0.6222 | 42/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 356 | 134.1 | 0.7485 | 0.9556 | 0.6444 | 43/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 23 | 134.4 | 0.7519 | 0.9444 | 0.6111 | 17/18 |
| finance__3M_2023Q2_10Q.pdf | 63 | 429 | 129.4 | 0.7537 | 0.8889 | 0.6667 | 56/63 |
| manual__t480_ug_en.pdf | 45 | 386 | 130.1 | 0.7563 | 0.9333 | 0.6667 | 42/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 151 | 128.2 | 0.7602 | 0.8889 | 0.6667 | 32/36 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 162 | 132.2 | 0.7719 | 0.9111 | 0.6889 | 41/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 42 | 123.3 | 0.7737 | 0.8788 | 0.697 | 29/33 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 90 | 125.6 | 0.7793 | 0.9333 | 0.6889 | 42/45 |
| finance__AMAZON_2019_10K.pdf | 81 | 344 | 127.5 | 0.7796 | 0.8889 | 0.7037 | 72/81 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 91 | 139.0 | 0.7833 | 0.8222 | 0.7556 | 37/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 363 | 128.1 | 0.7927 | 0.9333 | 0.7067 | 70/75 |
| manual__Macbook_air.pdf | 45 | 104 | 116.9 | 0.7948 | 0.9333 | 0.7111 | 42/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 179 | 128.9 | 0.8037 | 0.9556 | 0.7111 | 43/45 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 5 | 137.4 | 0.8056 | 1.0 | 0.6667 | 6/6 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 59 | 138.8 | 0.8066 | 0.9697 | 0.697 | 32/33 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 13 | 138.9 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 130 | 131.9 | 0.8163 | 0.9333 | 0.7333 | 42/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 212 | 133.6 | 0.8267 | 0.9333 | 0.7556 | 42/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 125 | 140.4 | 0.8267 | 0.9333 | 0.7556 | 42/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 75 | 132.9 | 0.8274 | 0.9333 | 0.7556 | 42/45 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 15 | 140.3 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 49 | 142.3 | 0.8333 | 0.9111 | 0.7556 | 41/45 |
| manual__owners-manual-2170416.pdf | 45 | 62 | 133.1 | 0.8333 | 0.9556 | 0.7556 | 43/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 78 | 130.5 | 0.8341 | 0.9556 | 0.7556 | 43/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 60 | 101.0 | 0.8407 | 0.9778 | 0.7333 | 44/45 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 25 | 122.1 | 0.8413 | 1.0 | 0.7619 | 21/21 |
| manual__nova_y70.pdf | 46 | 91 | 134.9 | 0.8442 | 0.9348 | 0.7609 | 43/46 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 36 | 125.9 | 0.8444 | 0.9556 | 0.7778 | 43/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 21 | 121.2 | 0.8507 | 1.0 | 0.75 | 24/24 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 8 | 111.0 | 0.8519 | 1.0 | 0.7778 | 9/9 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 33 | 146.7 | 0.8611 | 1.0 | 0.7222 | 18/18 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 102 | 131.3 | 0.8611 | 0.9333 | 0.8 | 42/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 33 | 129.3 | 0.8611 | 0.9583 | 0.7917 | 23/24 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 28 | 139.9 | 0.8667 | 1.0 | 0.8 | 15/15 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 50 | 129.3 | 0.8675 | 0.9744 | 0.7692 | 38/39 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 126 | 128.6 | 0.8748 | 0.9556 | 0.8222 | 43/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 22 | 146.7 | 0.875 | 1.0 | 0.75 | 12/12 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 30 | 146.6 | 0.8889 | 1.0 | 0.7778 | 18/18 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 17 | 132.5 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 56 | 130.5 | 0.8951 | 0.963 | 0.8519 | 26/27 |
| manual__mi_phone.pdf | 45 | 51 | 114.4 | 0.9007 | 0.9778 | 0.8444 | 44/45 |
| manual__dgx_a100.pdf | 45 | 204 | 108.5 | 0.9019 | 0.9778 | 0.8444 | 44/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 125.2 | 0.9111 | 1.0 | 0.8889 | 9/9 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 45 | 138.5 | 0.9111 | 1.0 | 0.8333 | 30/30 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 141 | 142.7 | 0.9148 | 1.0 | 0.8667 | 45/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 21 | 132.7 | 0.9167 | 0.9444 | 0.8889 | 17/18 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 21 | 141.0 | 0.9167 | 1.0 | 0.8333 | 12/12 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 37 | 132.6 | 0.9167 | 1.0 | 0.8333 | 30/30 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 61 | 128.8 | 0.9206 | 0.9762 | 0.881 | 41/42 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 90 | 139.3 | 0.9222 | 0.9556 | 0.8889 | 43/45 |
| manual__honor_watch_gs_pro.pdf | 45 | 103 | 126.6 | 0.9267 | 1.0 | 0.8667 | 45/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 26 | 139.2 | 0.9333 | 1.0 | 0.8667 | 15/15 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 15 | 135.7 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 34 | 147.3 | 0.9524 | 1.0 | 0.9048 | 21/21 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 23 | 138.2 | 0.9583 | 1.0 | 0.9167 | 12/12 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 12 | 132.2 | 0.9583 | 1.0 | 0.9167 | 12/12 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 95 | 134.1 | 0.963 | 1.0 | 0.9333 | 45/45 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 81 | 133.6 | 0.963 | 1.0 | 0.9333 | 45/45 |
| manual__watch_d.pdf | 45 | 57 | 131.2 | 0.963 | 0.9778 | 0.9556 | 44/45 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 11 | 119.1 | 0.9667 | 1.0 | 0.9333 | 15/15 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 15 | 145.5 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 15 | 132.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 7 | 131.9 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 28 | 138.5 | 1.0 | 1.0 | 1.0 | 12/12 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 111.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 14 | 136.9 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 9 | 121.7 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 12 | 128.8 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 5 | 116.4 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 17 | 136.7 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 5 | 110.6 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: llamaindex (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2409.01704v1.pdf | 60 | 37 | 88.2 | 0.3517 | 0.5833 | 0.2333 | 35/60 |
| academic__2404.10198v2.pdf | 51 | 22 | 63.8 | 0.4879 | 0.7255 | 0.3529 | 37/51 |
| academic__2310.11511v1.pdf | 57 | 64 | 87.0 | 0.4904 | 0.7018 | 0.3684 | 40/57 |
| academic__2402.03216v4.pdf | 78 | 52 | 128.6 | 0.5077 | 0.7051 | 0.3718 | 55/78 |
| academic__2405.14458v1.pdf | 64 | 46 | 70.8 | 0.5094 | 0.6562 | 0.4375 | 42/64 |
| academic__2403.20330v2.pdf | 69 | 54 | 131.2 | 0.5307 | 0.8116 | 0.3768 | 56/69 |
| academic__2305.02437v3.pdf | 66 | 41 | 92.3 | 0.5551 | 0.8182 | 0.3939 | 54/66 |
| finance__AES_2022_10K.pdf | 78 | 453 | 327.5 | 0.575 | 0.7564 | 0.4615 | 59/78 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 374 | 276.0 | 0.6117 | 0.8025 | 0.4938 | 65/81 |
| academic__2405.14831v1.pdf | 63 | 62 | 94.3 | 0.6238 | 0.8413 | 0.5079 | 53/63 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 429 | 275.4 | 0.6302 | 0.8551 | 0.4783 | 59/69 |
| academic__2409.16145v1.pdf | 51 | 37 | 230.4 | 0.6314 | 0.8431 | 0.5098 | 43/51 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 134 | 328.1 | 0.637 | 0.7778 | 0.5333 | 35/45 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 791 | 310.1 | 0.646 | 0.8413 | 0.5079 | 53/63 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 338 | 282.7 | 0.6475 | 0.8125 | 0.55 | 65/80 |
| academic__2305.14160v4.pdf | 72 | 35 | 128.6 | 0.6528 | 0.8472 | 0.5417 | 61/72 |
| finance__3M_2023Q2_10Q.pdf | 63 | 178 | 321.3 | 0.6563 | 0.7778 | 0.5873 | 49/63 |
| manual__Macbook_air.pdf | 45 | 35 | 348.5 | 0.6778 | 0.8444 | 0.5556 | 38/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 35 | 360.7 | 0.69 | 0.8889 | 0.6 | 40/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 33 | 355.4 | 0.6907 | 0.8 | 0.6 | 36/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 78 | 326.9 | 0.693 | 0.8889 | 0.5556 | 40/45 |
| finance__VERIZON_2021_10K.pdf | 84 | 237 | 326.6 | 0.6938 | 0.9167 | 0.5357 | 77/84 |
| manual__DSA-278777.pdf | 45 | 27 | 164.0 | 0.6989 | 0.8889 | 0.5778 | 40/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 77 | 300.4 | 0.7007 | 0.8667 | 0.6 | 39/45 |
| finance__AMD_2022_10K.pdf | 62 | 197 | 338.1 | 0.7081 | 0.8065 | 0.6452 | 50/62 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 35 | 245.9 | 0.7156 | 0.8222 | 0.6444 | 37/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 112 | 314.1 | 0.7237 | 0.8889 | 0.6222 | 40/45 |
| manual__dgx_a100.pdf | 45 | 86 | 253.8 | 0.7237 | 0.8889 | 0.6 | 40/45 |
| manual__obs-productdesc-en.pdf | 45 | 43 | 327.3 | 0.7259 | 0.9556 | 0.5556 | 43/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 32 | 296.1 | 0.7274 | 0.9333 | 0.6 | 42/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 58 | 317.2 | 0.7338 | 0.8333 | 0.6667 | 30/36 |
| manual__mi_phone.pdf | 45 | 17 | 347.3 | 0.7359 | 0.9111 | 0.6222 | 41/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 15 | 312.5 | 0.7407 | 0.9444 | 0.5556 | 17/18 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 34 | 311.8 | 0.7415 | 0.8222 | 0.6889 | 37/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 27 | 343.5 | 0.7444 | 0.8667 | 0.6444 | 39/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 50 | 325.2 | 0.7452 | 0.8667 | 0.6667 | 39/45 |
| manual__t480_ug_en.pdf | 45 | 137 | 363.1 | 0.7519 | 0.8444 | 0.6889 | 38/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 67 | 299.3 | 0.7537 | 0.8667 | 0.6667 | 39/45 |
| finance__AMAZON_2017_10K.pdf | 75 | 148 | 321.0 | 0.7558 | 0.9467 | 0.6267 | 71/75 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 9 | 279.4 | 0.7593 | 1.0 | 0.6111 | 18/18 |
| manual__2021-Apple-Catalog.pdf | 45 | 28 | 218.5 | 0.7822 | 0.9556 | 0.6889 | 43/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 57 | 285.5 | 0.783 | 0.9333 | 0.6889 | 42/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 14 | 295.6 | 0.787 | 0.9444 | 0.6667 | 17/18 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 51 | 307.6 | 0.7963 | 0.8667 | 0.7333 | 39/45 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 18 | 322.3 | 0.8011 | 0.9667 | 0.7 | 29/30 |
| finance__AMAZON_2019_10K.pdf | 81 | 139 | 321.8 | 0.8091 | 0.9136 | 0.7407 | 74/81 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 15 | 295.2 | 0.81 | 0.9778 | 0.7111 | 44/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 32 | 300.2 | 0.8119 | 0.9778 | 0.6889 | 44/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 28 | 328.6 | 0.8204 | 0.9333 | 0.7556 | 42/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 15 | 316.3 | 0.8232 | 0.9394 | 0.7576 | 31/33 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 9 | 330.3 | 0.8241 | 1.0 | 0.7222 | 18/18 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 11 | 324.8 | 0.8244 | 1.0 | 0.7333 | 15/15 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 23 | 304.3 | 0.8272 | 0.963 | 0.7037 | 26/27 |
| manual__owners-manual-2170416.pdf | 45 | 25 | 331.4 | 0.8278 | 0.9111 | 0.7778 | 41/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 19 | 314.8 | 0.8333 | 0.9487 | 0.7436 | 37/39 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 10 | 286.5 | 0.8333 | 0.9524 | 0.7619 | 20/21 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 269.0 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 370.2 | 0.8333 | 1.0 | 0.6667 | 12/12 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 21 | 346.5 | 0.8361 | 1.0 | 0.7381 | 42/42 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 74 | 356.6 | 0.8378 | 0.9556 | 0.7333 | 43/45 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 22 | 341.6 | 0.8419 | 1.0 | 0.7273 | 33/33 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 10 | 288.1 | 0.8611 | 1.0 | 0.75 | 12/12 |
| manual__nova_y70.pdf | 46 | 31 | 394.4 | 0.8623 | 0.9565 | 0.8043 | 44/46 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 23 | 296.5 | 0.8648 | 0.9778 | 0.7778 | 44/45 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 6 | 341.8 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 37 | 321.0 | 0.8748 | 0.9556 | 0.8222 | 43/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 7 | 360.9 | 0.8785 | 1.0 | 0.7917 | 24/24 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 61 | 311.3 | 0.8841 | 1.0 | 0.8 | 45/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 6 | 334.0 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 320.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 6 | 329.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| manual__honor_watch_gs_pro.pdf | 45 | 36 | 356.3 | 0.8963 | 1.0 | 0.8 | 45/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 10 | 334.2 | 0.9 | 1.0 | 0.8 | 15/15 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 4 | 323.0 | 0.9 | 1.0 | 0.8 | 15/15 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 8 | 337.9 | 0.9028 | 1.0 | 0.8333 | 12/12 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 14 | 333.9 | 0.9143 | 1.0 | 0.8571 | 21/21 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 29 | 340.3 | 0.9148 | 1.0 | 0.8444 | 45/45 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 12 | 322.0 | 0.9167 | 1.0 | 0.8333 | 24/24 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 319.5 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| manual__watch_d.pdf | 45 | 19 | 394.5 | 0.9185 | 1.0 | 0.8444 | 45/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 35 | 338.9 | 0.9222 | 0.9778 | 0.8667 | 44/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 9 | 336.7 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 5 | 300.2 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 7 | 311.1 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 337.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 6 | 308.7 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 13 | 359.5 | 0.9611 | 1.0 | 0.9333 | 30/30 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 6 | 327.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 280.7 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 12 | 298.0 | 1.0 | 1.0 | 1.0 | 12/12 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 216.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 6 | 304.3 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 4 | 255.2 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 268.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 2 | 254.0 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: pymupdf4llm (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 217 | 429.2 | 0.4399 | 0.6232 | 0.3333 | 43/69 |
| academic__2403.20330v2.pdf | 69 | 20 | 442.0 | 0.4599 | 0.7826 | 0.2754 | 54/69 |
| finance__3M_2023Q2_10Q.pdf | 63 | 92 | 552.5 | 0.4706 | 0.6032 | 0.3968 | 38/63 |
| academic__2409.01704v1.pdf | 60 | 19 | 505.3 | 0.4889 | 0.7167 | 0.3667 | 43/60 |
| academic__2404.10198v2.pdf | 51 | 13 | 458.3 | 0.4905 | 0.8627 | 0.2745 | 44/51 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 382 | 570.5 | 0.4907 | 0.6984 | 0.3492 | 44/63 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 53 | 459.2 | 0.5015 | 0.7333 | 0.3556 | 33/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 40 | 186.2 | 0.5093 | 0.5778 | 0.4667 | 26/45 |
| finance__AES_2022_10K.pdf | 78 | 255 | 520.5 | 0.5098 | 0.6154 | 0.4487 | 48/78 |
| finance__AMAZON_2017_10K.pdf | 75 | 85 | 506.4 | 0.5296 | 0.6933 | 0.4267 | 52/75 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 179 | 435.8 | 0.5296 | 0.725 | 0.425 | 58/80 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 197 | 409.0 | 0.5595 | 0.679 | 0.4815 | 55/81 |
| academic__2402.03216v4.pdf | 78 | 18 | 520.2 | 0.5645 | 0.7692 | 0.4487 | 60/78 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 45 | 750.3 | 0.5667 | 0.7111 | 0.4667 | 32/45 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 91 | 468.9 | 0.5681 | 0.7333 | 0.4667 | 33/45 |
| finance__VERIZON_2021_10K.pdf | 84 | 120 | 582.6 | 0.5772 | 0.7381 | 0.4762 | 62/84 |
| academic__2409.16145v1.pdf | 51 | 22 | 428.4 | 0.5879 | 0.8431 | 0.451 | 43/51 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 25 | 353.7 | 0.6022 | 0.8444 | 0.4444 | 38/45 |
| finance__AMD_2022_10K.pdf | 62 | 121 | 507.4 | 0.6032 | 0.7581 | 0.5161 | 47/62 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 21 | 735.1 | 0.607 | 0.8222 | 0.4444 | 37/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 34 | 452.0 | 0.6107 | 0.7778 | 0.4889 | 35/45 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 25 | 346.4 | 0.6222 | 0.8 | 0.5111 | 36/45 |
| academic__2405.14458v1.pdf | 64 | 18 | 537.1 | 0.6286 | 0.875 | 0.4844 | 56/64 |
| manual__obs-productdesc-en.pdf | 45 | 65 | 171.8 | 0.6296 | 0.8222 | 0.5111 | 37/45 |
| academic__2305.02437v3.pdf | 66 | 20 | 438.9 | 0.6364 | 0.8636 | 0.5 | 57/66 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 40 | 289.1 | 0.6415 | 0.7556 | 0.5778 | 34/45 |
| finance__AMAZON_2019_10K.pdf | 81 | 83 | 489.0 | 0.6609 | 0.8272 | 0.5556 | 67/81 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 31 | 622.5 | 0.6619 | 0.8222 | 0.5778 | 37/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 23 | 795.3 | 0.6619 | 0.7778 | 0.6 | 35/45 |
| academic__2305.14160v4.pdf | 72 | 16 | 551.6 | 0.6671 | 0.9028 | 0.5 | 65/72 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 9 | 738.8 | 0.671 | 0.963 | 0.5185 | 26/27 |
| academic__2310.11511v1.pdf | 57 | 30 | 540.0 | 0.6743 | 0.8596 | 0.5789 | 49/57 |
| academic__2405.14831v1.pdf | 63 | 28 | 477.3 | 0.6783 | 0.9048 | 0.5238 | 57/63 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 30 | 346.8 | 0.6811 | 0.8222 | 0.6 | 37/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 6 | 419.7 | 0.6852 | 1.0 | 0.4444 | 18/18 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 14 | 523.6 | 0.6854 | 0.9091 | 0.5152 | 30/33 |
| manual__t480_ug_en.pdf | 45 | 168 | 275.9 | 0.6922 | 0.9333 | 0.5333 | 42/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 75 | 125.9 | 0.7033 | 0.9333 | 0.5333 | 42/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 19 | 488.4 | 0.7037 | 0.8444 | 0.6222 | 38/45 |
| manual__nova_y70.pdf | 46 | 45 | 261.9 | 0.7043 | 0.8478 | 0.587 | 39/46 |
| manual__Macbook_air.pdf | 45 | 71 | 165.8 | 0.7052 | 0.8444 | 0.6222 | 38/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 49 | 362.1 | 0.7102 | 0.8611 | 0.6389 | 31/36 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 47 | 535.9 | 0.7137 | 0.8222 | 0.6444 | 37/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 14 | 342.4 | 0.7146 | 0.8485 | 0.6061 | 28/33 |
| manual__DSA-278777.pdf | 45 | 21 | 259.4 | 0.72 | 0.9111 | 0.6222 | 41/45 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 29 | 420.7 | 0.7222 | 0.9111 | 0.6222 | 41/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 17 | 245.8 | 0.7259 | 0.8667 | 0.6222 | 39/45 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 8 | 343.4 | 0.7294 | 0.9524 | 0.619 | 20/21 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 39 | 387.5 | 0.7389 | 0.8667 | 0.6444 | 39/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 6 | 741.2 | 0.7454 | 1.0 | 0.5556 | 18/18 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 8 | 473.4 | 0.7479 | 1.0 | 0.5833 | 24/24 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 5 | 579.2 | 0.75 | 1.0 | 0.5 | 12/12 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 108 | 231.8 | 0.757 | 0.8444 | 0.7111 | 38/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 16 | 437.4 | 0.7627 | 0.9286 | 0.6667 | 39/42 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 16 | 711.9 | 0.763 | 0.8889 | 0.6667 | 40/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 5 | 716.2 | 0.7639 | 1.0 | 0.5833 | 12/12 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 5 | 694.2 | 0.7778 | 1.0 | 0.6 | 15/15 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 17 | 358.9 | 0.7815 | 0.8889 | 0.6889 | 40/45 |
| manual__dgx_a100.pdf | 45 | 120 | 175.7 | 0.7944 | 0.9111 | 0.6889 | 41/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 7 | 406.7 | 0.7963 | 1.0 | 0.6667 | 18/18 |
| manual__owners-manual-2170416.pdf | 45 | 32 | 234.6 | 0.7989 | 0.9333 | 0.7111 | 42/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 7 | 647.3 | 0.8016 | 1.0 | 0.619 | 21/21 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 10 | 551.4 | 0.8056 | 1.0 | 0.6667 | 30/30 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 712.2 | 0.8056 | 1.0 | 0.6667 | 12/12 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 16 | 607.6 | 0.8074 | 0.9556 | 0.7111 | 43/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 18 | 323.2 | 0.8094 | 0.9744 | 0.6923 | 38/39 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 3 | 549.0 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 3 | 551.3 | 0.8148 | 1.0 | 0.6667 | 9/9 |
| manual__2021-Apple-Catalog.pdf | 45 | 55 | 114.8 | 0.8256 | 0.9778 | 0.7333 | 44/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 5 | 655.8 | 0.8333 | 1.0 | 0.6667 | 15/15 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 2 | 487.0 | 0.8333 | 1.0 | 0.6667 | 6/6 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 3 | 682.7 | 0.8333 | 1.0 | 0.6667 | 9/9 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 11 | 407.5 | 0.8344 | 0.9667 | 0.7333 | 29/30 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 9 | 273.8 | 0.8368 | 0.9583 | 0.75 | 23/24 |
| manual__honor_watch_gs_pro.pdf | 45 | 42 | 295.5 | 0.8444 | 0.9333 | 0.7556 | 42/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 15 | 762.9 | 0.8452 | 1.0 | 0.7333 | 45/45 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 4 | 654.5 | 0.8472 | 1.0 | 0.75 | 12/12 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 6 | 667.5 | 0.8796 | 1.0 | 0.7778 | 18/18 |
| manual__watch_d.pdf | 45 | 27 | 256.2 | 0.8859 | 1.0 | 0.8222 | 45/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 3 | 640.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 3 | 656.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 268.7 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 3 | 614.0 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| manual__mi_phone.pdf | 45 | 37 | 149.2 | 0.9111 | 0.9778 | 0.8667 | 44/45 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 3 | 217.0 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 4 | 465.8 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 305.7 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 3 | 574.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 3 | 456.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 367.0 | 0.9444 | 1.0 | 0.9167 | 12/12 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 1 | 824.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 217.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 264.5 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 5 | 253.2 | 1.0 | 1.0 | 1.0 | 15/15 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 4 | 132.8 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: llamaindex_semantic (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| manual__t480_ug_en.pdf | 45 | 245 | 187.5 | 0.3204 | 0.6667 | 0.1778 | 30/45 |
| finance__AES_2022_10K.pdf | 78 | 231 | 607.5 | 0.3737 | 0.5256 | 0.3077 | 41/78 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 141 | 691.1 | 0.3852 | 0.679 | 0.2469 | 55/81 |
| academic__2403.20330v2.pdf | 69 | 25 | 265.0 | 0.4271 | 0.7826 | 0.2174 | 54/69 |
| academic__2405.14831v1.pdf | 63 | 38 | 145.5 | 0.4402 | 0.6825 | 0.2381 | 43/63 |
| academic__2405.14458v1.pdf | 64 | 30 | 101.4 | 0.4865 | 0.7188 | 0.375 | 46/64 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 49 | 872.3 | 0.487 | 0.6222 | 0.4 | 28/45 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 164 | 678.8 | 0.4949 | 0.7101 | 0.3913 | 49/69 |
| academic__2305.14160v4.pdf | 72 | 21 | 206.9 | 0.5178 | 0.8194 | 0.3333 | 59/72 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 14 | 591.7 | 0.5241 | 0.9111 | 0.2889 | 41/45 |
| manual__Macbook_air.pdf | 45 | 29 | 396.3 | 0.5359 | 0.7111 | 0.4444 | 32/45 |
| finance__AMD_2022_10K.pdf | 62 | 115 | 545.3 | 0.5417 | 0.7419 | 0.4032 | 46/62 |
| finance__VERIZON_2021_10K.pdf | 84 | 123 | 592.8 | 0.5458 | 0.7381 | 0.4167 | 62/84 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 28 | 690.8 | 0.547 | 0.6889 | 0.4667 | 31/45 |
| academic__2409.01704v1.pdf | 60 | 18 | 174.3 | 0.5539 | 0.8833 | 0.3667 | 53/60 |
| manual__Guide-for-international-students-web.pdf | 45 | 21 | 427.2 | 0.5593 | 0.8667 | 0.3778 | 39/45 |
| academic__2305.02437v3.pdf | 66 | 29 | 123.7 | 0.5611 | 0.8182 | 0.3939 | 54/66 |
| manual__dgx_a100.pdf | 45 | 43 | 477.9 | 0.5652 | 0.7556 | 0.4444 | 34/45 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 22 | 805.8 | 0.5722 | 0.8333 | 0.4722 | 30/36 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 436 | 535.7 | 0.5828 | 0.7937 | 0.4762 | 50/63 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 39 | 643.7 | 0.5922 | 0.8667 | 0.4667 | 39/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 36 | 427.8 | 0.5959 | 0.8 | 0.4444 | 36/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 85 | 632.3 | 0.5976 | 0.7937 | 0.4762 | 50/63 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 138 | 655.0 | 0.6023 | 0.775 | 0.4875 | 62/80 |
| finance__AMAZON_2017_10K.pdf | 75 | 77 | 581.5 | 0.6031 | 0.7733 | 0.48 | 58/75 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 34 | 721.4 | 0.6037 | 0.9111 | 0.4444 | 41/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 5 | 835.4 | 0.6148 | 1.0 | 0.3111 | 45/45 |
| academic__2310.11511v1.pdf | 57 | 56 | 92.5 | 0.6175 | 0.8421 | 0.4737 | 48/57 |
| manual__obs-productdesc-en.pdf | 45 | 36 | 365.5 | 0.6396 | 0.8889 | 0.4889 | 40/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 11 | 814.9 | 0.6459 | 0.9333 | 0.4444 | 42/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 67 | 503.1 | 0.6544 | 0.7778 | 0.5778 | 35/45 |
| finance__AMAZON_2019_10K.pdf | 81 | 73 | 578.0 | 0.6551 | 0.7531 | 0.5926 | 61/81 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 17 | 665.5 | 0.6585 | 0.8444 | 0.5778 | 38/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 20 | 761.4 | 0.6637 | 0.9778 | 0.4667 | 44/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 8 | 553.6 | 0.6657 | 0.9444 | 0.5 | 17/18 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 6 | 437.0 | 0.6667 | 1.0 | 0.4167 | 12/12 |
| academic__2404.10198v2.pdf | 51 | 15 | 86.2 | 0.6748 | 1.0 | 0.4314 | 51/51 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 16 | 755.8 | 0.6763 | 0.9778 | 0.5111 | 44/45 |
| academic__2402.03216v4.pdf | 78 | 27 | 230.8 | 0.6786 | 0.9231 | 0.5385 | 72/78 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 183 | 118.2 | 0.6793 | 0.8222 | 0.6 | 37/45 |
| academic__2409.16145v1.pdf | 51 | 28 | 287.5 | 0.6804 | 0.8627 | 0.5882 | 44/51 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 33 | 469.3 | 0.6911 | 0.8889 | 0.5778 | 40/45 |
| manual__nova_y70.pdf | 46 | 32 | 353.3 | 0.7116 | 0.8261 | 0.6522 | 38/46 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 5 | 382.2 | 0.7222 | 1.0 | 0.4444 | 9/9 |
| manual__2021-Apple-Catalog.pdf | 45 | 6 | 969.3 | 0.7222 | 1.0 | 0.4889 | 45/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 11 | 528.7 | 0.7231 | 0.9487 | 0.5897 | 37/39 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 7 | 670.4 | 0.7298 | 1.0 | 0.5455 | 33/33 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 28 | 226.9 | 0.7333 | 0.9333 | 0.5556 | 42/45 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 22 | 513.5 | 0.7437 | 0.9556 | 0.6222 | 43/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 7 | 413.1 | 0.75 | 1.0 | 0.5 | 12/12 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 22 | 516.7 | 0.7526 | 0.9778 | 0.6 | 44/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 6 | 668.2 | 0.7546 | 1.0 | 0.5556 | 18/18 |
| manual__mi_phone.pdf | 45 | 19 | 289.8 | 0.7589 | 0.9333 | 0.6667 | 42/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 8 | 567.4 | 0.7698 | 1.0 | 0.619 | 21/21 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 12 | 747.2 | 0.7741 | 0.9778 | 0.6222 | 44/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 27 | 676.5 | 0.7741 | 0.8889 | 0.6889 | 40/45 |
| manual__watch_d.pdf | 45 | 20 | 344.1 | 0.7756 | 0.9333 | 0.6889 | 42/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 4 | 225.0 | 0.7778 | 1.0 | 0.5556 | 9/9 |
| manual__honor_watch_gs_pro.pdf | 45 | 36 | 334.1 | 0.7785 | 0.9778 | 0.6444 | 44/45 |
| manual__DSA-278777.pdf | 45 | 10 | 425.0 | 0.7804 | 1.0 | 0.6222 | 45/45 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 6 | 399.0 | 0.7847 | 1.0 | 0.5833 | 24/24 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 18 | 567.4 | 0.7859 | 0.9333 | 0.6889 | 42/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 10 | 700.8 | 0.7909 | 0.9762 | 0.6905 | 41/42 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 711.2 | 0.7917 | 1.0 | 0.5833 | 12/12 |
| manual__owners-manual-2170416.pdf | 45 | 39 | 195.6 | 0.8144 | 0.9111 | 0.7556 | 41/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 8 | 431.2 | 0.8222 | 1.0 | 0.6667 | 15/15 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 10 | 555.8 | 0.8222 | 1.0 | 0.6667 | 30/30 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 7 | 634.4 | 0.8278 | 1.0 | 0.7 | 30/30 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 14 | 660.2 | 0.8296 | 1.0 | 0.6889 | 45/45 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 15 | 440.9 | 0.8364 | 1.0 | 0.7407 | 27/27 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 8 | 355.2 | 0.8519 | 1.0 | 0.7222 | 18/18 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 5 | 756.8 | 0.8542 | 1.0 | 0.7083 | 24/24 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 5 | 652.0 | 0.8556 | 1.0 | 0.7333 | 15/15 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 13 | 738.3 | 0.8593 | 1.0 | 0.7556 | 45/45 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 9 | 812.0 | 0.8687 | 1.0 | 0.7576 | 33/33 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 5 | 394.0 | 0.8704 | 1.0 | 0.7778 | 9/9 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 3 | 268.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 5 | 408.8 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 326.6 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 4 | 449.0 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 4 | 619.5 | 0.9167 | 1.0 | 0.8333 | 18/18 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 3 | 412.0 | 0.9333 | 1.0 | 0.8667 | 15/15 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 5 | 370.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 4 | 462.5 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 9 | 387.2 | 0.9583 | 1.0 | 0.9167 | 12/12 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 359.2 | 0.9583 | 1.0 | 0.9167 | 12/12 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 5 | 553.0 | 0.9762 | 1.0 | 0.9524 | 21/21 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 273.3 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 108.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 5 | 343.2 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 4 | 243.2 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 315.0 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 4 | 341.8 | 1.0 | 1.0 | 1.0 | 9/9 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 2 | 263.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 3 | 168.7 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: docstruct (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| finance__AMD_2022_10K.pdf | 62 | 247 | 379.9 | 0.3204 | 0.5 | 0.1935 | 31/62 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 97 | 466.9 | 0.3315 | 0.4444 | 0.2444 | 20/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 122 | 76.6 | 0.3611 | 0.4222 | 0.3333 | 19/45 |
| academic__2409.01704v1.pdf | 60 | 33 | 298.7 | 0.3622 | 0.4667 | 0.3167 | 28/60 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 31 | 379.7 | 0.3711 | 0.5333 | 0.2889 | 24/45 |
| manual__mi_phone.pdf | 45 | 35 | 233.1 | 0.3944 | 0.4889 | 0.3333 | 22/45 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 37 | 344.4 | 0.3963 | 0.4889 | 0.3333 | 22/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 170 | 261.6 | 0.4 | 0.5333 | 0.3111 | 24/45 |
| academic__2305.02437v3.pdf | 66 | 42 | 218.1 | 0.4056 | 0.5303 | 0.3333 | 35/66 |
| academic__2405.14458v1.pdf | 64 | 41 | 317.5 | 0.4185 | 0.6094 | 0.2969 | 39/64 |
| manual__Macbook_air.pdf | 45 | 51 | 247.6 | 0.4341 | 0.5333 | 0.3778 | 24/45 |
| finance__AES_2022_10K.pdf | 78 | 582 | 400.5 | 0.4378 | 0.5513 | 0.3718 | 43/78 |
| academic__2404.10198v2.pdf | 51 | 16 | 356.4 | 0.4402 | 0.7451 | 0.2745 | 38/51 |
| manual__Guide-for-international-students-web.pdf | 45 | 146 | 107.2 | 0.4485 | 0.6 | 0.3556 | 27/45 |
| academic__2403.20330v2.pdf | 69 | 45 | 275.2 | 0.4604 | 0.5942 | 0.3913 | 41/69 |
| manual__obs-productdesc-en.pdf | 45 | 101 | 179.5 | 0.4674 | 0.6667 | 0.3333 | 30/45 |
| finance__3M_2023Q2_10Q.pdf | 63 | 234 | 377.3 | 0.4815 | 0.5556 | 0.4286 | 35/63 |
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 36 | 399.0 | 0.4859 | 0.6222 | 0.3778 | 28/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 12 | 532.5 | 0.4898 | 0.8333 | 0.3333 | 15/18 |
| academic__2402.03216v4.pdf | 78 | 61 | 324.5 | 0.5085 | 0.7179 | 0.3846 | 56/78 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 771 | 235.4 | 0.5113 | 0.679 | 0.4321 | 55/81 |
| finance__AMAZON_2017_10K.pdf | 75 | 192 | 368.0 | 0.536 | 0.72 | 0.4133 | 54/75 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 11 | 403.8 | 0.5625 | 0.75 | 0.4167 | 9/12 |
| finance__VERIZON_2021_10K.pdf | 84 | 378 | 304.5 | 0.5631 | 0.8095 | 0.4286 | 68/84 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 5 | 583.0 | 0.5694 | 0.75 | 0.4167 | 9/12 |
| academic__2409.16145v1.pdf | 51 | 36 | 299.0 | 0.5824 | 0.8039 | 0.4314 | 41/51 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 18 | 578.6 | 0.5867 | 0.8333 | 0.4667 | 25/30 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 23 | 332.7 | 0.5919 | 0.8222 | 0.4667 | 37/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 5 | 346.2 | 0.5926 | 0.6667 | 0.5556 | 6/9 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 11 | 514.1 | 0.5938 | 0.7917 | 0.4583 | 19/24 |
| manual__t480_ug_en.pdf | 45 | 244 | 364.8 | 0.5981 | 0.8 | 0.4444 | 36/45 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 869 | 238.9 | 0.5986 | 0.7826 | 0.5072 | 54/69 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 8 | 634.4 | 0.5992 | 0.7143 | 0.5238 | 15/21 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 94 | 437.2 | 0.6037 | 0.7556 | 0.5111 | 34/45 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 88 | 473.8 | 0.6044 | 0.7778 | 0.4889 | 35/45 |
| academic__2405.14831v1.pdf | 63 | 72 | 220.1 | 0.6069 | 0.8095 | 0.4444 | 51/63 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 8 | 586.9 | 0.6111 | 0.6667 | 0.5556 | 12/18 |
| academic__2310.11511v1.pdf | 57 | 62 | 357.1 | 0.6126 | 0.7544 | 0.5088 | 43/57 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 12 | 476.4 | 0.613 | 0.8333 | 0.4444 | 15/18 |
| manual__owners-manual-2170416.pdf | 45 | 52 | 246.2 | 0.6174 | 0.7333 | 0.5556 | 33/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 42 | 532.3 | 0.6185 | 0.7333 | 0.5333 | 33/45 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 48 | 602.9 | 0.6248 | 0.8 | 0.4889 | 36/45 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 32 | 424.5 | 0.6285 | 0.8444 | 0.4889 | 38/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 63 | 536.2 | 0.6315 | 0.8444 | 0.5111 | 38/45 |
| manual__nova_y70.pdf | 46 | 77 | 239.0 | 0.6377 | 0.8261 | 0.5 | 38/46 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 86 | 315.0 | 0.6389 | 0.6944 | 0.5833 | 25/36 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 728 | 231.2 | 0.6419 | 0.775 | 0.5625 | 62/80 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 58 | 387.7 | 0.6593 | 0.7778 | 0.5778 | 35/45 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 38 | 462.7 | 0.6619 | 0.8 | 0.5778 | 36/45 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 109 | 550.9 | 0.6711 | 0.8222 | 0.5778 | 37/45 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 1309 | 331.3 | 0.6786 | 0.8254 | 0.5873 | 52/63 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 17 | 492.4 | 0.6818 | 0.7879 | 0.6061 | 26/33 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 21 | 619.6 | 0.6838 | 0.9394 | 0.5455 | 31/33 |
| manual__watch_d.pdf | 45 | 44 | 248.7 | 0.6896 | 0.8222 | 0.6 | 37/45 |
| finance__AMAZON_2019_10K.pdf | 81 | 188 | 355.3 | 0.6938 | 0.8519 | 0.5802 | 69/81 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 8 | 482.9 | 0.6944 | 0.75 | 0.6667 | 9/12 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 47 | 636.1 | 0.6952 | 0.8444 | 0.5778 | 38/45 |
| manual__DSA-278777.pdf | 45 | 52 | 187.0 | 0.7007 | 0.8889 | 0.6 | 40/45 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 11 | 560.4 | 0.7083 | 0.9167 | 0.5833 | 11/12 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 11 | 339.5 | 0.7083 | 0.8333 | 0.625 | 20/24 |
| manual__dgx_a100.pdf | 45 | 164 | 200.2 | 0.7107 | 0.8444 | 0.6222 | 38/45 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 81 | 382.4 | 0.7119 | 0.7556 | 0.6889 | 34/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 42 | 293.5 | 0.7298 | 0.8333 | 0.6667 | 35/42 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 4 | 406.8 | 0.7315 | 1.0 | 0.5556 | 9/9 |
| manual__honor_watch_gs_pro.pdf | 45 | 68 | 298.8 | 0.7396 | 0.8444 | 0.6667 | 38/45 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 7 | 489.3 | 0.7407 | 1.0 | 0.5556 | 9/9 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 24 | 505.8 | 0.7438 | 0.8519 | 0.6667 | 23/27 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 7 | 350.0 | 0.7467 | 0.9333 | 0.6667 | 14/15 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 3 | 397.3 | 0.75 | 1.0 | 0.5 | 6/6 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 76 | 159.0 | 0.7515 | 0.9111 | 0.6444 | 41/45 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 29 | 606.2 | 0.757 | 0.9556 | 0.6222 | 43/45 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 129 | 224.1 | 0.7611 | 0.8889 | 0.6667 | 40/45 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 15 | 535.3 | 0.7619 | 0.9048 | 0.6667 | 19/21 |
| academic__2305.14160v4.pdf | 72 | 50 | 267.8 | 0.763 | 0.9167 | 0.6806 | 66/72 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 6 | 564.3 | 0.7778 | 1.0 | 0.6667 | 9/9 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 11 | 544.2 | 0.7778 | 1.0 | 0.6667 | 15/15 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 15 | 550.3 | 0.7917 | 1.0 | 0.6667 | 18/18 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 7 | 558.1 | 0.8 | 0.8889 | 0.7778 | 8/9 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 38 | 583.5 | 0.8296 | 0.9333 | 0.7556 | 42/45 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 48 | 344.1 | 0.8312 | 0.9487 | 0.7436 | 37/39 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 15 | 386.4 | 0.8611 | 1.0 | 0.75 | 12/12 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 7 | 511.6 | 0.8611 | 1.0 | 0.7778 | 9/9 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 6 | 546.7 | 0.8611 | 1.0 | 0.7778 | 9/9 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 62 | 507.4 | 0.8804 | 0.9778 | 0.8222 | 44/45 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 6 | 545.3 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 6 | 509.8 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 22 | 496.4 | 0.9167 | 0.9667 | 0.8667 | 29/30 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 7 | 469.1 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 7 | 374.6 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 13 | 430.4 | 0.9467 | 1.0 | 0.9333 | 15/15 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 3 | 574.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 1 | 427.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 6 | 337.5 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 5 | 270.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 3 | 297.0 | 1.0 | 1.0 | 1.0 | 3/3 |


### Per-doc breakdown: docstruct_geo (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| law__ENERGYXXILTD_05_08_2015-EX-10.13-Transportation_AGREEMENT.pdf | 45 | 22 | 427.5 | 0.1322 | 0.3111 | 0.0 | 14/45 |
| academic__2310.11511v1.pdf | 57 | 42 | 389.7 | 0.1722 | 0.2632 | 0.1228 | 15/57 |
| law__ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual_Property_Agreement.pdf | 45 | 18 | 463.1 | 0.2185 | 0.2667 | 0.1778 | 12/45 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 490 | 212.4 | 0.2204 | 0.3086 | 0.1728 | 25/81 |
| academic__2305.02437v3.pdf | 66 | 28 | 220.8 | 0.2394 | 0.4394 | 0.1515 | 29/66 |
| academic__2405.14458v1.pdf | 64 | 26 | 302.6 | 0.2539 | 0.4531 | 0.1562 | 29/64 |
| law__CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance_Agreement1.pdf | 45 | 45 | 567.9 | 0.3007 | 0.5111 | 0.1778 | 23/45 |
| law__ADMA_BioManufacturing_LLC_-_Amendment_3_to_Manufacturing_Agreement_.pdf | 18 | 4 | 619.5 | 0.3056 | 0.5 | 0.2222 | 9/18 |
| finance__3M_2023Q2_10Q.pdf | 63 | 155 | 369.5 | 0.3077 | 0.5079 | 0.1905 | 32/63 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 546 | 217.8 | 0.3256 | 0.4783 | 0.2464 | 33/69 |
| finance__AMAZON_2017_10K.pdf | 75 | 81 | 556.1 | 0.3327 | 0.4267 | 0.28 | 32/75 |
| finance__AMD_2022_10K.pdf | 62 | 182 | 354.2 | 0.3527 | 0.4677 | 0.2742 | 29/62 |
| law__IDREAMSKYTECHNOLOGYLTD_07_03_2014-EX-10.39-Cooperation_Agreement_on_Mobile_Game_Business.pdf | 45 | 24 | 505.9 | 0.373 | 0.4889 | 0.3111 | 22/45 |
| law__FIBROGENINC_10_01_2014-EX-10.11-COLLABORATION_AGREEMENT.pdf | 45 | 47 | 531.3 | 0.3856 | 0.5333 | 0.3111 | 24/45 |
| academic__2402.03216v4.pdf | 78 | 30 | 377.1 | 0.3876 | 0.5385 | 0.2949 | 42/78 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 2 | 402.5 | 0.3889 | 0.6667 | 0.1111 | 6/9 |
| academic__2409.01704v1.pdf | 60 | 26 | 268.2 | 0.3903 | 0.5167 | 0.3 | 31/60 |
| law__Array_BioPharma_Inc._-_LICENSE_DEVELOPMENT_AND_COMMERCIALIZATION_AGREEMENT.pdf | 45 | 90 | 482.8 | 0.4019 | 0.5111 | 0.3111 | 23/45 |
| manual__obs-productdesc-en.pdf | 45 | 97 | 145.9 | 0.4019 | 0.6 | 0.2444 | 27/45 |
| manual__t480_ug_en.pdf | 45 | 123 | 381.2 | 0.4044 | 0.5333 | 0.3333 | 24/45 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 10 | 417.6 | 0.4056 | 0.6 | 0.2667 | 27/45 |
| law__CcRealEstateIncomeFundadv_20181205_POS_8C_EX-99._H_3__11447739_EX-99._H_3__Marketing_Agreement.pdf | 33 | 9 | 521.8 | 0.4091 | 0.4848 | 0.3636 | 16/33 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 440 | 219.2 | 0.41 | 0.5875 | 0.3 | 47/80 |
| law__RISEEDUCATIONCAYMANLTD_04_17_2020-EX-4.23-SERVICE_AGREEMENT.pdf | 24 | 8 | 299.2 | 0.4201 | 0.5 | 0.375 | 12/24 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 28 | 367.4 | 0.4267 | 0.6 | 0.3333 | 27/45 |
| manual__2021-Apple-Catalog.pdf | 45 | 26 | 234.4 | 0.4274 | 0.5111 | 0.3778 | 23/45 |
| finance__AES_2022_10K.pdf | 78 | 418 | 351.0 | 0.4293 | 0.5513 | 0.359 | 43/78 |
| law__TheglobeComInc_19990503_S-1A_EX-10.20_5416126_EX-10.20_Co-Branding_Agreement.pdf | 27 | 13 | 508.8 | 0.4302 | 0.5926 | 0.3333 | 16/27 |
| academic__2403.20330v2.pdf | 69 | 48 | 231.4 | 0.4304 | 0.6522 | 0.3043 | 45/69 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 5 | 568.4 | 0.4306 | 0.6667 | 0.2778 | 12/18 |
| academic__2404.10198v2.pdf | 51 | 15 | 293.3 | 0.4386 | 0.549 | 0.3529 | 28/51 |
| academic__2409.16145v1.pdf | 51 | 22 | 367.1 | 0.4484 | 0.6078 | 0.3529 | 31/51 |
| academic__2405.14831v1.pdf | 63 | 53 | 220.2 | 0.4497 | 0.619 | 0.3333 | 39/63 |
| manual__honor_watch_gs_pro.pdf | 45 | 39 | 311.7 | 0.4552 | 0.5778 | 0.3778 | 26/45 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 874 | 287.9 | 0.4574 | 0.6032 | 0.3651 | 38/63 |
| finance__AMAZON_2019_10K.pdf | 81 | 82 | 516.4 | 0.4586 | 0.5309 | 0.4074 | 43/81 |
| law__GULFSOUTHMEDICALSUPPLYINC_12_24_1997-EX-4-AFFILIATE_AGREEMENT.pdf | 21 | 5 | 564.6 | 0.4603 | 0.5714 | 0.381 | 12/21 |
| law__TRICITYBANKSHARESCORP_05_15_1998-EX-10-OUTSOURCING_AGREEMENT.pdf | 45 | 20 | 771.3 | 0.4626 | 0.6 | 0.3778 | 27/45 |
| law__OPTIMIZEDTRANSPORTATIONMANAGEMENT_INC_07_26_2000-EX-6.6-DISTRIBUTOR_AGREEMENT.pdf | 9 | 3 | 475.7 | 0.463 | 0.6667 | 0.3333 | 6/9 |
| law__UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate_Agreement.pdf | 9 | 3 | 620.0 | 0.463 | 0.6667 | 0.3333 | 6/9 |
| manual__DSA-278777.pdf | 45 | 23 | 287.2 | 0.4667 | 0.6 | 0.3778 | 27/45 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 109 | 237.7 | 0.4748 | 0.5778 | 0.4 | 26/45 |
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 29 | 669.2 | 0.477 | 0.6889 | 0.3556 | 31/45 |
| manual__Macbook_air.pdf | 45 | 44 | 263.2 | 0.4785 | 0.5778 | 0.4222 | 26/45 |
| manual__dgx_a100.pdf | 45 | 155 | 141.6 | 0.4785 | 0.5778 | 0.4222 | 26/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 4 | 711.2 | 0.4792 | 0.75 | 0.3333 | 9/12 |
| law__MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE_AGREEMENT.pdf | 45 | 49 | 690.1 | 0.4859 | 0.6 | 0.4222 | 27/45 |
| manual__Guide-for-international-students-web.pdf | 45 | 131 | 83.5 | 0.4878 | 0.6444 | 0.4 | 29/45 |
| law__CYBERIANOUTPOSTINC_07_09_1998-EX-10.13-PROMOTION_AGREEMENT.pdf | 15 | 5 | 690.0 | 0.4889 | 0.6 | 0.4 | 9/15 |
| manual__nova_y70.pdf | 46 | 46 | 245.8 | 0.4935 | 0.6957 | 0.3478 | 32/46 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 22 | 405.2 | 0.5078 | 0.7556 | 0.3556 | 34/45 |
| manual__mi_phone.pdf | 45 | 22 | 261.8 | 0.5093 | 0.5778 | 0.4667 | 26/45 |
| law__NEXSTARFINANCEHOLDINGSINC_03_27_2002-EX-10.26-OUTSOURCING_AGREEMENT.pdf | 45 | 26 | 702.5 | 0.5107 | 0.6889 | 0.4222 | 31/45 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 3 | 300.7 | 0.5185 | 0.6667 | 0.4444 | 6/9 |
| law__ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing_Agreement.pdf | 36 | 40 | 445.9 | 0.5231 | 0.5833 | 0.4722 | 21/36 |
| law__BANUESTRAFINANCIALCORP_09_08_2006-EX-10.16-AGENCY_AGREEMENT.pdf | 33 | 12 | 614.2 | 0.5303 | 0.6364 | 0.4242 | 21/33 |
| law__ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship_Agreement.pdf | 12 | 8 | 331.5 | 0.5417 | 0.75 | 0.3333 | 9/12 |
| law__KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing_Agreement.pdf | 45 | 28 | 356.0 | 0.5422 | 0.7111 | 0.4222 | 32/45 |
| law__CnsPharmaceuticalsInc_20200326_8-K_EX-10.1_12079626_EX-10.1_Development_Agreement.pdf | 15 | 6 | 551.8 | 0.55 | 0.8 | 0.4 | 12/15 |
| law__IntegrityFunds_20200121_485BPOS_EX-99.E_UNDR_CONTR_11948727_EX-99.E_UNDR_CONTR_Service_Agreement.pdf | 24 | 6 | 631.7 | 0.5556 | 0.75 | 0.4167 | 18/24 |
| law__XACCT_Technologies_Inc.SUPPORT_AND_MAINTENANCE_AGREEMENT.pdf | 9 | 3 | 598.7 | 0.5556 | 0.6667 | 0.4444 | 6/9 |
| law__N2KINC_10_16_1997-EX-10.16-SPONSORSHIP_AGREEMENT.pdf | 30 | 9 | 617.6 | 0.5583 | 0.7 | 0.4667 | 21/30 |
| law__WOMENSGOLFUNLIMITEDINC_03_29_2000-EX-10.13-ENDORSEMENT_AGREEMENT.pdf | 21 | 7 | 648.4 | 0.5651 | 0.6667 | 0.5238 | 14/21 |
| law__SIBANNAC_INC_12_04_2017-EX-2.1-Strategic_Alliance_Agreement.pdf | 15 | 3 | 412.0 | 0.5667 | 0.6 | 0.5333 | 9/15 |
| finance__VERIZON_2021_10K.pdf | 84 | 302 | 258.2 | 0.5671 | 0.7262 | 0.4762 | 61/84 |
| academic__2305.14160v4.pdf | 72 | 26 | 361.5 | 0.5731 | 0.7361 | 0.4722 | 53/72 |
| manual__guojixueshengshenghuozhinanyingwen9.1.pdf | 45 | 65 | 266.2 | 0.583 | 0.7778 | 0.4667 | 35/45 |
| law__ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation_Agreement.pdf | 12 | 4 | 359.2 | 0.5833 | 0.75 | 0.4167 | 9/12 |
| law__AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE_AGREEMENT.pdf | 45 | 20 | 572.5 | 0.5889 | 0.8 | 0.4889 | 36/45 |
| manual__watch_d.pdf | 45 | 35 | 201.8 | 0.5907 | 0.7556 | 0.4889 | 34/45 |
| law__VAXCYTE_INC_05_22_2020-EX-10.19-SUPPLY_AGREEMENT.pdf | 45 | 28 | 550.3 | 0.6126 | 0.7556 | 0.5333 | 34/45 |
| law__ASIANDRAGONGROUPINC_08_11_2005-EX-10.5-Reseller_Agreement.pdf | 42 | 20 | 363.8 | 0.619 | 0.7143 | 0.5476 | 30/42 |
| manual__owners-manual-2170416.pdf | 45 | 28 | 278.3 | 0.637 | 0.7111 | 0.5778 | 32/45 |
| law__DYNTEKINC_07_30_1999-EX-10-ONLINE_HOSTING_AGREEMENT.pdf | 18 | 7 | 572.7 | 0.6593 | 0.8333 | 0.5556 | 15/18 |
| law__ADUROBIOTECH_INC_06_02_2020-EX-10.7-CONSULTING_AGREEMENT.pdf | 9 | 4 | 462.5 | 0.6667 | 0.6667 | 0.6667 | 6/9 |
| law__KNOWLABS_INC_08_15_2005-EX-10-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 9 | 3 | 656.7 | 0.6667 | 0.6667 | 0.6667 | 6/9 |
| law__AMERICASSHOPPINGMALLINC_12_10_1999-EX-10.2-SITE_DEVELOPMENT_AND_HOSTING_AGREEMENT.pdf | 12 | 5 | 578.4 | 0.6736 | 1.0 | 0.5 | 12/12 |
| law__WHITESMOKE_INC_11_08_2011-EX-10.26-PROMOTION_AND_DISTRIBUTION_AGREEMENT.pdf | 45 | 60 | 375.9 | 0.6881 | 0.8 | 0.6444 | 36/45 |
| law__HALITRON_INC_03_01_2005-EX-10.15-SPONSORSHIP_AND_DEVELOPMENT_AGREEMENT.pdf | 18 | 6 | 738.2 | 0.7083 | 0.8333 | 0.6111 | 15/18 |
| law__ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_EX-10.18_Supply_Agreement.pdf | 45 | 15 | 641.1 | 0.7296 | 0.8667 | 0.6222 | 39/45 |
| law__VnueInc_20150914_8-K_EX-10.1_9259571_EX-10.1_Promotion_Agreement.pdf | 9 | 5 | 353.4 | 0.7481 | 1.0 | 0.6667 | 9/9 |
| law__ScansourceInc_20190822_10-K_EX-10.38_11793958_EX-10.38_Distributor_Agreement2.pdf | 9 | 5 | 408.8 | 0.7593 | 1.0 | 0.5556 | 9/9 |
| manual__8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | 45 | 67 | 126.6 | 0.7759 | 0.8889 | 0.6889 | 40/45 |
| law__HealthcareIntegratedTechnologiesInc_20190812_8-K_EX-10.1_11776966_EX-10.1_Reseller_Agreement.pdf | 12 | 8 | 455.1 | 0.8056 | 1.0 | 0.6667 | 12/12 |
| law__DIVERSINETCORP_03_01_2012-EX-4-RESELLER_AGREEMENT.pdf | 45 | 36 | 618.4 | 0.8107 | 0.9556 | 0.7111 | 43/45 |
| law__CHERRYHILLMORTGAGEINVESTMENTCORP_09_26_2013-EX-10.1-Strategic_Alliance_Agreement.pdf | 30 | 17 | 476.3 | 0.8444 | 0.9333 | 0.7667 | 28/30 |
| law__BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT_AGREEMENT_1_.pdf | 39 | 35 | 332.6 | 0.8568 | 0.9744 | 0.7692 | 38/39 |
| law__ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT_VENTURE_AGREEMENT.pdf | 9 | 4 | 477.8 | 0.8889 | 1.0 | 0.7778 | 9/9 |
| law__OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-AGENCY_AGREEMENT2.pdf | 6 | 2 | 320.5 | 0.9167 | 1.0 | 0.8333 | 6/6 |
| law__MARSHALLHOLDINGSINTERNATIONAL_INC_04_14_2004-EX-10.15-ENDORSEMENT_AGREEMENT.pdf | 9 | 3 | 572.0 | 0.9444 | 1.0 | 0.8889 | 9/9 |
| law__FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV_AGREE-SERVICES_AGREEMENT_SECONDAMENDMENT.pdf | 3 | 1 | 820.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__MACY_S_INC_05_11_2020-EX-99.4-JOINT_FILING_AGREEMENT.pdf | 3 | 2 | 108.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__NETGEAR_INC_04_21_2003-EX-10.16-AMENDMENT_TO_THE_DISTRIBUTOR_AGREEMENT_BETWEEN_INGRAM_MICRO_AND_NETGEAR-.pdf | 6 | 2 | 486.5 | 1.0 | 1.0 | 1.0 | 6/6 |
| law__PlayboyEnterprisesInc_20090220_10-QA_EX-10.2_4091580_EX-10.2_Content_License_Agreement__Marketing_Agreement__Sales-.pdf | 3 | 4 | 264.0 | 1.0 | 1.0 | 1.0 | 3/3 |
| law__VIVINT_SOLAR_INC._-_NON-COMPETITION_AGREEMENT.pdf | 3 | 2 | 253.0 | 1.0 | 1.0 | 1.0 | 3/3 |
