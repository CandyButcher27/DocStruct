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
| 1 | unstructured | **0.6032** | [0.5884, 0.6179] | 0.6274 | 0.7122 | 0.5275 | 0.5435 | +0.0597 | 19036 | 86.8 | 555.8 | 1.0853 | 634.63 | 0 |

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
| finance__JPMORGAN_2023Q2_10Q.pdf | 69 | 1451 | 76.5 | 0.2976 | 0.3623 | 0.2609 | 25/69 |
| finance__JPMORGAN_2021Q1_10Q.pdf | 80 | 1204 | 74.8 | 0.3012 | 0.4625 | 0.2 | 37/80 |
| law__GRIDIRONBIONUTRIENTS_INC_02_05_2020-EX-10.3-SUPPLY_AGREEMENT.pdf | 9 | 8 | 100.9 | 0.3889 | 0.4444 | 0.3333 | 4/9 |
| law__GWG_HOLDINGS_INC._-_ORDERLY_MARKETING_AGREEMENT.pdf | 45 | 35 | 106.3 | 0.3889 | 0.4444 | 0.3556 | 20/45 |
| academic__2305.14160v4.pdf | 72 | 98 | 87.2 | 0.3926 | 0.4722 | 0.3472 | 34/72 |
| finance__JPMORGAN_2022Q2_10Q.pdf | 81 | 1309 | 74.2 | 0.3932 | 0.4938 | 0.3457 | 40/81 |
| academic__2305.02437v3.pdf | 66 | 120 | 74.1 | 0.4028 | 0.4697 | 0.3485 | 31/66 |
| finance__AES_2022_10K.pdf | 78 | 1481 | 94.5 | 0.4233 | 0.5385 | 0.3462 | 42/78 |
| academic__2403.20330v2.pdf | 69 | 124 | 81.7 | 0.4304 | 0.4783 | 0.3913 | 33/69 |
| manual__DSA-278777.pdf | 45 | 93 | 65.9 | 0.4611 | 0.5111 | 0.4222 | 23/45 |
| academic__2409.01704v1.pdf | 60 | 91 | 95.0 | 0.4631 | 0.6 | 0.3833 | 36/60 |
| finance__AMAZON_2019_10K.pdf | 81 | 537 | 78.4 | 0.4681 | 0.5556 | 0.3951 | 45/81 |
| academic__2402.03216v4.pdf | 78 | 129 | 82.5 | 0.478 | 0.5256 | 0.4487 | 41/78 |
| finance__JPMORGAN_2022_10K.pdf | 63 | 2624 | 88.6 | 0.4952 | 0.6032 | 0.4127 | 38/63 |
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
| manual__Macbook_air.pdf | 45 | 143 | 80.5 | 0.6163 | 0.7556 | 0.5111 | 34/45 |
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
| manual__User_Manual_1500S_Classic_EN.pdf | 45 | 306 | 81.1 | 0.687 | 0.8 | 0.6 | 36/45 |
| law__PareteumCorp_20081001_8-K_EX-99.1_2654808_EX-99.1_Hosting_Agreement.pdf | 45 | 105 | 90.6 | 0.6889 | 0.8 | 0.6 | 36/45 |
| law__RgcResourcesInc_20151216_8-K_EX-10.3_9372751_EX-10.3_Franchise_Agreement.pdf | 9 | 9 | 84.2 | 0.6944 | 0.8889 | 0.5556 | 8/9 |
| manual__Guide-for-international-students-web.pdf | 45 | 162 | 55.1 | 0.6981 | 0.8 | 0.6222 | 36/45 |
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
