# Reference verification list — DocStruct paper

`paper/refs.bib`, 35 entries. Generated 2026-08-16.

## How these were obtained (read this first)

Provenance is not uniform, and the difference matters:

- **5 papers were downloaded and read** this session — their PDFs were fetched from
  arXiv and parsed to study layout conventions. Existence is certain.
- **24 entries carry an arXiv ID**, and every one was resolved against the arXiv API
  (`export.arxiv.org/api/query`) on 2026-08-16. All 24 IDs exist and all titles match.
  The author lists in the bib now come from that API response.
- **The remaining entries came from model knowledge**, not from a search. They are
  long-established works whose existence is not in doubt, but their *bibliographic
  detail* (volume, pages, exact venue string) has not been machine-checked.

No web search tool was used at any point. Where something is marked verified below,
it was verified by an HTTP request whose response is quoted, not by recall.

## Group A — arXiv-verified (24). Confirm venue/year only.

Every ID below returned a real record. The *arXiv* metadata is confirmed; what still
needs a human or a search is whether the **published venue** attributed in the bib is
correct (e.g. an entry claiming ICCV/CVPR/ACL acceptance).

| # | Key | arXiv | Title (per arXiv) | First author (per arXiv) | Bib claims venue |
|---|---|---|---|---|---|
| 1 | `chi2019scitsr` | [1908.04729](https://arxiv.org/abs/1908.04729) | Complicated Table Structure Recognition | Zewen Chi +5 | arXiv preprint arXiv:1908.04729 |
| 2 | `zhong2019publaynet` | [1908.07836](https://arxiv.org/abs/1908.07836) | PubLayNet: largest dataset ever for document layout analysis | Xu Zhong +2 | International Conference on Document Ana |
| 3 | `li2020docbank` | [2006.01038](https://arxiv.org/abs/2006.01038) | DocBank: A Benchmark Dataset for Document Layout Analysis | Minghao Li +6 | International Conference on Computationa |
| 4 | `smock2022pubtables` | [2110.00061](https://arxiv.org/abs/2110.00061) | PubTables-1M: Towards comprehensive table extraction from unstructured | Brandon Smock +2 | IEEE/CVF Conference on Computer Vision a |
| 5 | `meuschke2023benchmark` | [2303.09957](https://arxiv.org/abs/2303.09957) | A Benchmark of PDF Information Extraction Tools using a Multi-Task and | Norman Meuschke +4 | arXiv preprint arXiv:2303.09957 |
| 6 | `islam2023financebench` | [2311.11944](https://arxiv.org/abs/2311.11944) | FinanceBench: A New Benchmark for Financial Question Answering | Pranab Islam +5 | arXiv preprint arXiv:2311.11944 |
| 7 | `jimeno2024financialchunking` | [2402.05131](https://arxiv.org/abs/2402.05131) | Financial Report Chunking for Effective Retrieval Augmented Generation | Antonio Jimeno Yepes +4 | arXiv preprint arXiv:2402.05131 |
| 8 | `duarte2024lumberchunker` | [2406.17526](https://arxiv.org/abs/2406.17526) | LumberChunker: Long-Form Narrative Document Segmentation | André V. Duarte +5 | arXiv preprint arXiv:2406.17526 |
| 9 | `ma2024mmlongbenchdoc` | [2407.01523](https://arxiv.org/abs/2407.01523) | MMLongBench-Doc: Benchmarking Long-context Document Understanding with | Yubo Ma +15 | Advances in Neural Information Processin |
| 10 | `auer2024docling` | [2408.09869](https://arxiv.org/abs/2408.09869) | Docling Technical Report | Christoph Auer +18 | arXiv preprint arXiv:2408.09869 |
| 11 | `li2025readoc` | [2409.05137](https://arxiv.org/abs/2409.05137) | READoc: A Unified Benchmark for Realistic Document Structured Extracti | Zichao Li +7 | Findings of the Association for Computat |
| 12 | `pdfparsingcomparative2024` | [2410.09871](https://arxiv.org/abs/2410.09871) | A Comparative Study of PDF Parsing Tools Across Diverse Document Categ | Narayan S. Adhikari +1 | arXiv preprint arXiv:2410.09871 |
| 13 | `qu2024chunkingsurvey` | [2410.13070](https://arxiv.org/abs/2410.13070) | Is Semantic Chunking Worth the Computational Cost? | Renyi Qu +2 | arXiv preprint arXiv:2410.13070 |
| 14 | `zhang2025ocrhindersrag` | [2412.02592](https://arxiv.org/abs/2412.02592) | OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-A | Junyuan Zhang +8 | IEEE/CVF International Conference on Com |
| 15 | `ouyang2025omnidocbench` | [2412.07626](https://arxiv.org/abs/2412.07626) | OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehen | Linke Ouyang +19 | IEEE/CVF Conference on Computer Vision a |
| 16 | `livathinos2025docling` | [2501.17887](https://arxiv.org/abs/2501.17887) | Docling: An Efficient Open-Source Toolkit for AI-driven Document Conve | Nikolaos Livathinos +16 | arXiv preprint arXiv:2501.17887 |
| 17 | `reconstructingcontext2025` | [2504.19754](https://arxiv.org/abs/2504.19754) | Reconstructing Context: Evaluating Advanced Chunking Strategies for Re | Carlo Merola +1 | arXiv preprint arXiv:2504.19754 |
| 18 | `rethinkingchunksize2025` | [2505.21700](https://arxiv.org/abs/2505.21700) | Rethinking Chunk Size For Long-Document Retrieval: A Multi-Dataset Ana | Sinchana Ramakanth Bhat +3 | arXiv preprint arXiv:2505.21700 |
| 19 | `chunkingqa2026` | [2601.14123](https://arxiv.org/abs/2601.14123) | A Systematic Analysis of Chunking Strategies for Reliable Question Ans | Sofia Bennani +1 | arXiv preprint arXiv:2601.14123 |
| 20 | `topochunker2026` | [2603.18409](https://arxiv.org/abs/2603.18409) | TopoChunker: Topology-Aware Agentic Document Chunking Framework | Xiaoyu Liu | arXiv preprint arXiv:2603.18409 |
| 21 | `elbachyr2026empirical` | [2604.12047](https://arxiv.org/abs/2604.12047) | Empirical Evaluation of PDF Parsing and Chunking for Financial Questio | Omar El Bachyr +7 | Proceedings of the IEEE/ACM 48th Interna |
| 22 | `multidocfusion2026` | [2604.12352](https://arxiv.org/abs/2604.12352) | MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enha | Joongmin Shin +4 | arXiv preprint arXiv:2604.12352 |
| 23 | `structuredtabularchunking2026` | [2605.00318](https://arxiv.org/abs/2605.00318) | Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Gener | Pooja Guttal +5 | arXiv preprint arXiv:2605.00318 |
| 24 | `chunkingcost2026` | [2606.00881](https://arxiv.org/abs/2606.00881) | Chunking Methods on Retrieval-Augmented Generation - Effectiveness Eva | Mateusz Śmigielski +7 | arXiv preprint arXiv:2606.00881 |

**Specifically worth a second look:**

- `zhang2025ocrhindersrag` — bib says **ICCV 2025**. Confirm acceptance.
- `ouyang2025omnidocbench` — bib says **CVPR 2025**. Confirm acceptance.
- `li2025readoc` — bib says **ACL Findings 2025**. Confirm.
- `pfitzmann2022doclaynet` — bib says **KDD 2022**, DOI `10.1145/3534678.3539043`. Confirm.
- `elbachyr2026empirical` — bib says **ICSE-SEIP 2026**, DOI `10.1145/3786583.3786911`. Confirm.
- `multidocfusion2026` — arXiv title matched at 0.84 similarity, not 1.0. Check exact wording.

## Group B — DOI present, not arXiv (4). Confirm the DOI resolves to the right work.

| # | Key | DOI | Title | Author |
|---|---|---|---|---|
| 1 | `pfitzmann2022doclaynet` | [10.1145/3534678.3539043](https://doi.org/10.1145/3534678.3539043) | {DocLayNet}: A Large Human-Annotated Dataset for Document-La | Pfitzmann, Birgit and Auer, Christoph and Do |
| 2 | `beeferman1999statistical` | [10.1023/A:1007506220214](https://doi.org/10.1023/A:1007506220214) | Statistical Models for Text Segmentation | Beeferman, Doug and Berger, Adam and Laffert |
| 3 | `pevzner2002critique` | [10.1162/089120102317341756](https://doi.org/10.1162/089120102317341756) | A Critique and Improvement of an Evaluation Metric for Text  | Pevzner, Lev and Hearst, Marti A. |

`beeferman1999statistical` and `pevzner2002critique` were added this session from model
knowledge; both DOIs were confirmed to resolve, but the volume/page numbers were not
independently checked.

## Group C — NOT machine-verified (7). These need the most attention.

No arXiv ID and no DOI in the entry. All are well-known works, so the risk is not that
they are invented — it is wrong year, wrong venue string, wrong page range, or wrong
edition.

| # | Key | Title | Author | Year | Venue as written |
|---|---|---|---|---|---|
| 1 | `lewis2020rag` | Retrieval-Augmented Generation for Knowledge-Intensi | Lewis, Patrick and Perez, Ethan and Pikt | — | Advances in Neural Information Processin |
| 2 | `smith2024chunking` | Evaluating Chunking Strategies for Retrieval | Smith, Brandon and Troynikov, Anton | 2024 | — |
| 3 | `cormack2009rrf` | Reciprocal Rank Fusion Outperforms {Condorcet} and I | Cormack, Gordon V. and Clarke, Charles L | — | Proceedings of the 32nd International AC |
| 4 | `robertson2009bm25` | The Probabilistic Relevance Framework: {BM25} and Be | Robertson, Stephen and Zaragoza, Hugo | — | Foundations and Trends in Information Re |
| 5 | `reimers2019sbert` | Sentence-{BERT}: Sentence Embeddings using {Siamese} | Reimers, Nils and Gurevych, Iryna | — | Proceedings of EMNLP-IJCNLP |
| 6 | `jarvelin2002ndcg` | Cumulated Gain-Based Evaluation of {IR} Techniques | J{\"a}rvelin, Kalervo and Kek{\"a}l{\"a} | — | ACM Transactions on Information Systems |
| 7 | `efron1993bootstrap` | An Introduction to the Bootstrap | Efron, Bradley and Tibshirani, Robert J. | — | Chapman \& Hall |

`smith2024chunking` (Chroma technical report, "Evaluating Chunking Strategies for
Retrieval") is the one to check hardest in this group: it is a company technical report
rather than a peer-reviewed paper, so its citable form is least standardised.

`europepmc` is a website citation, not a paper.

## What to ask the other model

> For each entry below, find the authoritative record (arXiv listing, DOI landing page,
> ACL Anthology, or publisher page). Report: (1) does the work exist; (2) the exact
> title; (3) the complete author list in order; (4) the year of the version cited;
> (5) the published venue, or "preprint only" if never published; (6) DOI if any.
> Flag any entry where the venue claimed here is not supported by the record.
> Do not fill gaps from memory — if you cannot find a source, say so.

That last sentence matters. A verification pass that quietly reconstructs a plausible
author list is worse than no pass, because it launders a guess into a checked fact.
This file exists because exactly that had happened to one entry:
`pdfparsingcomparative2024` was attributed to "Bast, Hannah and others" and is actually
by Adhikari and Agarwal.
