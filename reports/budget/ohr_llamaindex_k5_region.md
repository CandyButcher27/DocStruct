# DocStruct retrieval baseline report

_Generated 2026-11-20 11:50 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: llamaindex.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | llamaindex | **0.5887** | [0.5739, 0.6033] | 0.6104 | 0.6979 | 0.5138 | 0.5121 | +0.0766 | 5794 | 295.2 | 1430.1 | 0.4116 | 522.42 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| llamaindex | 1.0 | 1.0474 |

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
| law__FIDELITYNATIONALINFORMATIONSERVICES_INC_08_05_2009-EX-10.3-INTELLECTUAL_PROPERTY_AGREEMENT.pdf | 45 | 67 | 299.3 | 0.6185 | 0.7111 | 0.5333 | 32/45 |
| manual__Macbook_air.pdf | 45 | 35 | 348.5 | 0.6204 | 0.7333 | 0.5333 | 33/45 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 791 | 310.1 | 0.6275 | 0.7778 | 0.5397 | 49/63 |
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
