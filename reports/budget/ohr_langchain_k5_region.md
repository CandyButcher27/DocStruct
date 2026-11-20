# DocStruct retrieval baseline report

_Generated 2026-11-20 11:51 UTC_

## Setup

- **Documents:** 95 born-digital PDFs
- **Questions:** 3558 LLM-generated (model `gpt-oss:120b`), each with a verbatim answer span validated against the source
- **Embedder (constant):** `all-MiniLM-L6-v2`  |  **Retrievers:** dense cosine and hybrid (dense + BM25 fused by RRF, k=60), top-5, per-document index
- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span (normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall
- **Fair-comparison principle:** embedder + retrievers are identical for every tool; **only the chunker varies**, so the table measures chunking quality. The hybrid retriever is the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.

Tools benchmarked: langchain.

## Leaderboard (ranked by MRR)

| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@5 | Recall@5 | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | langchain | **0.6029** | [0.5878, 0.6176] | 0.6242 | 0.704 | 0.5323 | 0.5412 | +0.0617 | 13877 | 128.5 | 637.6 | 0.9456 | 428.2 | 0 |

## Extraction fidelity (no gold, no LLM)

Measured against each PDF's own raw pdfplumber text, so the document is its own ground truth. This is the only cross-tool quality signal in the report that measures **extraction** rather than retrieval, and the only one available for the whole corpus — hand-annotated detection boxes exist for two documents.

| Tool | Coverage | Duplication |
|---|---|---|
| langchain | 1.0 | 1.1005 |

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
| finance__AES_2022_10K.pdf | 78 | 1167 | 134.4 | 0.5603 | 0.6538 | 0.5 | 51/78 |
| manual__obs-productdesc-en.pdf | 45 | 125 | 113.5 | 0.5667 | 0.7333 | 0.4222 | 33/45 |
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
