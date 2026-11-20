# DocStruct retrieval baseline report

_Generated 2026-11-20 11:50 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: docstruct_geo.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | docstruct_geo | **0.6568** | [0.6427, 0.6707] | 0.6831 | 0.7841 | 0.5705 | 0.548 | +0.1088 | 5810 | 306.0 | 2328.2 | 0.2821 | 804.97 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| docstruct_geo **(ours)** | 0.9638 | 1.132 |

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


### Per-doc breakdown: docstruct_geo (worst first)

| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |
|---|---|---|---|---|---|---|---|
| academic__2305.02437v3.pdf | 66 | 28 | 220.8 | 0.2058 | 0.2424 | 0.1818 | 16/66 |
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 546 | 217.8 | 0.2843 | 0.3913 | 0.2319 | 27/69 |
| academic__2403.20330v2.pdf | 69 | 48 | 231.7 | 0.3039 | 0.4348 | 0.2174 | 30/69 |
| academic__2405.14458v1.pdf | 64 | 26 | 302.6 | 0.343 | 0.4844 | 0.25 | 31/64 |
| academic__2305.14160v4.pdf | 72 | 26 | 361.5 | 0.3789 | 0.5 | 0.3056 | 36/72 |
| academic__2402.03216v4.pdf | 78 | 30 | 377.2 | 0.4053 | 0.5769 | 0.3077 | 45/78 |
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
| manual__watch_d.pdf | 45 | 35 | 201.5 | 0.8667 | 0.9556 | 0.8 | 43/45 |
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
