# DocStruct retrieval baseline report

_Generated 2026-11-20 11:58 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: unstructured.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | unstructured | **0.6567** | [0.6424, 0.6707] | 0.6734 | 0.7546 | 0.5877 | 0.5965 | +0.0602 | 19036 | 86.8 | 555.8 | 1.1815 | 629.11 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| unstructured | 0.9224 | 1.0548 |

- **Coverage** — fraction of the document's word *instances* that appear in some chunk. This is where silent loss shows up and nowhere else: dropped table rows, headings that end up in no chunk, skipped figures. Counted as a multiset, so dropping every repeat of a term is not scored as covered.
- **Duplication** — chunk words divided by document words. Above 1.0 means content is emitted more than once, inflating the index and letting two chunks split the evidence for one query. Overlap raises it deliberately, so read it as a cost next to coverage, not as a defect.

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
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 1451 | 76.5 | 0.3072 | 0.3623 | 0.2754 | 25/69 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 1204 | 74.8 | 0.3171 | 0.4375 | 0.2375 | 35/80 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 1309 | 74.2 | 0.3819 | 0.4815 | 0.3333 | 39/81 |
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
| manual__Macbook_air.pdf | 45 | 143 | 80.5 | 0.7026 | 0.8444 | 0.6 | 38/45 |
| law__SEPARATEACCOUNTIIOFAGL_05_02_2011-EX-99._J_4_-UNCONDITIONAL_CAPITAL_MAINTENANCE_AGREEMENT.pdf | 12 | 19 | 148.7 | 0.7042 | 0.9167 | 0.5833 | 11/12 |
| law__MPLXLP_06_17_2015-EX-10.1-TRANSPORTATION_SERVICES_AGREEMENT.pdf | 45 | 70 | 116.7 | 0.7111 | 0.7556 | 0.6667 | 34/45 |
| law__EcoScienceSolutionsInc_20171117_8-K_EX-10.1_10956472_EX-10.1_Endorsement_Agreement.pdf | 18 | 30 | 89.7 | 0.713 | 0.8333 | 0.6111 | 15/18 |
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 306 | 81.1 | 0.713 | 0.8 | 0.6444 | 36/45 |
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
| manual__Guide-for-international-students-web.pdf | 45 | 162 | 55.1 | 0.7685 | 0.8444 | 0.7111 | 38/45 |
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
