# Related work — who the competitors are and what they measure

Compiled 2026-08-05 from a literature sweep seeded by `pdf_parsing_papers.html`.
Read this before writing any paper section, and before claiming novelty.
Companion files: [`benchmark-datasets.md`](benchmark-datasets.md) (what corpora we
can adopt), [`metrics-justification.md`](metrics-justification.md) (which metrics
we report and why).

**Where DocStruct sits.** Four literatures touch this project. Only one of them
is our actual competition.

```
A. parser fidelity      PDF -> text/markdown, scored against a source-of-truth rendering
B. layout analysis      page image -> boxes, scored as detection (mAP)
C. chunking for RAG     text -> chunks, scored by retrieval
D. parse x chunk x RAG  PDF -> chunks -> retrieval -> answer   <-- DocStruct lives here
```

DocStruct's claim is in **D**: that a *deterministic, structure-aware* chunker
built on a parser beats generic splitters at retrieval, with no LLM in the loop.
The nearest published work in D is El Bachyr et al. (ICSE-SEIP 2026) and
OHR-Bench (ICCV 2025); the nearest in C is the Chroma technical report.

---

## A. Parser / document-conversion fidelity

| Work | Venue / id | Metrics | Datasets | Relevance to us |
|---|---|---|---|---|
| **OmniDocBench** — Ouyang et al. | CVPR 2025, `arXiv:2412.07626` | Normalized Edit Distance (text, formula, reading order), **TEDS** (tables), **CDM** (formulas) | 981 pages, 9 doc types, full layout + reading-order + table annotation | The reference benchmark for "is the parse right". We currently score *nothing* on it. |
| **READoc** — Li et al. | ACL Findings 2025, `arXiv:2409.05137` | Text edit distance, TEDS, **KTDS**, vocabulary F1, reading-order metric | READoc-arXiv / -GitHub / -Zenodo; GT derived from **LaTeX / Markdown source**, no human annotation | Cheapest credible parse-fidelity number for us: arXiv PDFs, and our corpus is already arXiv-heavy. |
| **Docling** — Auer et al. | `arXiv:2408.09869`, `arXiv:2501.17887` | Pages/sec, conversion quality; MIT-licensed toolkit | — | Already a baseline in our benchmark (dropped from default: 10× slower, always last). |
| **MinerU / Marker / olmOCR-bench** | industry + `arXiv:2510.15349` etc. | olmOCR-bench % , pages/sec on GPU | born-digital + scanned | Throughput competitors. Marker 2 ~2.9 pages/s on B200 at 76.0%; MinerU 72.7%; Docling 50.3%. On **born-digital** Marker 2 and MinerU tie ~83.4. |
| **A Comparative Study of PDF Parsing Tools** | `arXiv:2410.09871` | Precision/Recall/F1, BLEU-4, local alignment, IoU/Jaccard for tables | multi-category docs | Tool-comparison framing closest to our leaderboard, but no retrieval stage. |
| **Freiburg text-extraction benchmark** | `ad-publications.cs.uni-freiburg.de/benchmark.pdf` | word/paragraph-level extraction quality | own corpus | Predecessor of the whole "extraction quality" family. |

**Takeaway:** in family A we would *lose* today, or rather we cannot compete at
all — DocStruct is not a Markdown converter and reports no edit-distance number.
Say this in the paper explicitly rather than letting a reviewer discover it.

## B. Layout-analysis datasets and detectors

| Work | id | Metrics | Notes |
|---|---|---|---|
| **PubLayNet** | `arXiv:1908.07836` | mAP@0.5:0.95 | 360k pages, PubMed only — low layout variability |
| **DocBank** | `arXiv:2006.01038` | token-level F1 / mAP | arXiv-sourced, weak supervision from LaTeX |
| **DocLayNet** | KDD 2022, `arXiv:2206.01062` | mAP@0.5:0.95, incl. **inter-annotator agreement as a ceiling** | 80,863 human-annotated pages, 11 classes, 6 domains. **This is what our YOLO weights are trained on** — so its val split is the natural home for our Layer-1 detection metric |
| **PubTables-1M** | `arXiv:2110.00061` | table detection / TSR / functional analysis | |
| **SciTSR / Complicated TSR** | `arXiv:1908.04729` | P/R/F1 macro + micro | |

**Takeaway:** our detection layer (`eval/runner.py`, mAP@0.5) is measured on
**two hand-annotated documents**. DocLayNet's val split replaces that outright and
costs nothing to adopt — the label map already exists (`DOCLAYNET_LABEL_MAP`).

## C. Chunking strategy evaluation (text-level, no PDF)

| Work | id | Metrics | Data | Findings that constrain us |
|---|---|---|---|---|
| **Chroma TR — Evaluating Chunking Strategies for Retrieval** (Smith & Troynikov, 2024) | trychroma.com/research/evaluating-chunking | **token-level IoU, Precision, Recall, Precision_Ω** | 5 corpora (SoTU, Wikitext, Chatlogs, Finance, Pubmed), 472 queries, GPT-4-Turbo-generated queries + verbatim excerpts | The de-facto standard metric set for *chunking specifically*. ClusterSemanticChunker 87.3% recall / 8.0 IoU vs RecursiveCharacterTextSplitter 88.1% / 6.9; LLMChunker 91.9% recall / 3.9 IoU. Their IoU is exactly the "MRR per 1k context" idea we invented independently — **theirs is the citable version**. |
| **Financial Report Chunking for Effective RAG** | `arXiv:2402.05131` | retrieval + answer accuracy | financial reports | Element-type (structure-based) chunking beats fixed size. Closest prior claim to ours. |
| **LumberChunker** | LLM-guided chunk boundaries | retrieval | narrative | Strong but 1.65× full-text-scan overhead + LLM in the loop — the exact cost our contract refuses. |
| **Is Semantic Chunking Worth the Computational Cost?** / **Chunking Methods on RAG** | `arXiv:2606.00881` | quality vs cost | multi-task | Semantic chunking's gains often do not pay for their compute. **Supports our determinism argument.** |
| **A Systematic Analysis of Chunking Strategies for Reliable QA** | `arXiv:2601.14123` | EM / semantic quality; SPLADE + Mistral-8B | Natural Questions | Overlap gives ~no benefit; sentence chunking most cost-effective; **"context cliff" past ~2.5k tokens**. Our 200/500 config lands ~2.4k context words at top-5 — right at that cliff. Cite it, and note our own context-cost sweep found the same shape. |
| **Rethinking Chunk Size for Long-Document Retrieval** | `arXiv:2505.21700` | retrieval, multi-dataset | | Chunk size dominates; matches our monotonic MRR-vs-size finding. |
| **TopoChunker** / **STC (tabular)** / **MultiDocFusion** | `2603.18409`, `2605.00318`, `2604.12352` | retrieval + chunk-count reduction | | 2026 wave of structure-aware chunkers. **STC and MultiDocFusion are the newest direct competitors to our positioning** and both post-date our design. |

## D. Parse × chunk × retrieval, end to end — the actual competition

### D1. El Bachyr et al., *Empirical Evaluation of PDF Parsing and Chunking for Financial Question Answering with RAG* — ICSE-SEIP 2026, `arXiv:2604.12047`

The single closest paper to ours. Grid study:

- **Parsers (6):** PyPDF2, PyMuPDF, pdfminer.six, pdfplumber, pypdfium2, Unstructured.
- **Chunkers (6):** token, sentence, recursive, semantic, SDPM, neural — 512-token cap, varied overlap.
- **Retrievers (4):** BM25, SPLADE-v3, E5-large, ColBERT.
- **Data:** FinanceBench (150 QA / 84 PDFs) + **TableQuest** (116 table QA, newly released, built on the same PDFs).
- **Metrics:** Precision@1, Recall@3, Recall@k, MRR, nDCG; end-to-end **Number Match**.
- **Results:** E5-large MRR 0.700 on FinanceBench; ColBERT 0.844 on TableQuest; neural chunking with **25% overlap** best (0.658 / 0.833); GPT-5 73.28% answer accuracy at 3 retrieved pages.
- **Conclusion:** lightweight parser+chunker combinations already do well; moderate overlap helps.

**Why this matters to us.** Same experimental skeleton (hold retriever fixed, vary
chunker; MRR/nDCG/Recall@k), *public* data, and a released artifact. Our advantage:
they treat parsing and chunking as two off-the-shelf knobs, evaluate no
structure-aware chunker, and retrieve at **page** granularity. Their finding that
25% overlap helps is in direct tension with `2601.14123` (overlap useless) and with
our own `OVERLAP_ON_BOUNDARY` result — a resolvable disagreement worth a paragraph.
Their weakness is our opening: **no layout-aware chunker in the grid**.

### D2. OHR-Bench — *OCR Hinders RAG* — ICCV 2025, `arXiv:2412.02592`

8,561 document pages / 8,498 QA over 7 domains (textbook, law, finance, newspaper,
manual, academic, administration), **human-verified ground-truth structured data
per page**, evidence typed as TXT / TAB / FOR / CHA / **RO (reading order)**.
Metrics are generalized-LCS / F1 at the retrieval stage plus generation scores;
noise is injected in two flavours (semantic, formatting) to trace the cascade.

Relevance: it is the strongest existing evidence that *parse quality propagates to
retrieval quality* — our whole thesis — and it ships per-evidence-type breakdowns
(RO especially) that would let us show the reading-order work pays off. Caveat: it
is OCR-oriented, so much of it is scanned; we need its born-digital subset.

---

## Gaps in the literature that our paper can own

1. **Nobody evaluates a deterministic layout-aware chunker against generic splitters with the embedder and retriever held fixed on public PDFs.** D1 varies parsers but not chunker *structure-awareness*; C works on plain text with no PDF layout at all.
2. **Nobody reports coverage/duplication next to rank quality.** Every chunking paper measures what was retrieved, none measures what was silently *dropped* by the chunker. Our `coverage.py` is, as far as this sweep found, unique — and it is cheap to compute for every baseline.
3. **The overlap question is openly contradictory** across `2604.12047` (+) and `2601.14123` (−). A clean single-variable ablation settles it.
4. **Determinism/reproducibility is asserted, never measured.** An LLM chunker is not byte-reproducible; nobody reports it. We can, trivially.

## Where competitors are ahead of us — state these, do not bury them

| Axis | Who wins | By how much | Our position |
|---|---|---|---|
| Parse fidelity (edit distance / TEDS) | MinerU, Marker 2, Docling, OmniDocBench leaders | We report **no number at all** | Out of scope by contract, but must be acknowledged; a READoc-arXiv run would at least place us. |
| Coverage of document words | LangChain recursive splitter | 1.00 vs our **0.817** | Structural filtering (references dropped, figures skipped) is deliberate; report it, argue it. |
| Table structure quality | PubTables-1M / TSR literature (TEDS) | unscored by us | We emit table chunks and never verify their structure. |
| Scanned / OCR documents | anything OCR-based | total | Explicit non-goal (`likely_scanned` diagnostic exists). |
| Absolute retrieval numbers | D1 reports MRR 0.700–0.844 | not comparable — different corpus, different retrievers, page-level gold | Never put their numbers beside ours without the caveat. |
| Gold-standard quality | FinanceBench (human), OHR-Bench (human-verified) | ours is `gpt-oss:120b`-generated | **Migrated 2026-08-11** — the headline is now OHR-Bench human gold; the internal corpus is for ablations. |
| Neural/LLM chunk baselines | LumberChunker, ClusterSemanticChunker | LumberChunker never run | `llamaindex_semantic` is now in the tool set and loses to us in all three OHR modes; ClusterSemanticChunker still not run. |
| Retrieval under `page` relevance | unstructured (0.795 vs our 0.600), langchain, llamaindex, pymupdf4llm | we are **6th of 7** | Page mode rewards small chunks and unstructured has the smallest in the field. Real, and reported — see `relevance-modes.md`. |
| Context efficiency (MRR / 1k words) | unstructured 1.166 vs our 0.322 | **3.6×** | We retrieve 2,194 words per query against their 561. We buy rank with context. |
| Academic-domain retrieval | unstructured 0.5151, pymupdf4llm 0.5112 vs our 0.4526 (`span`) | 5th of 7 | Our weakest domain in every mode, on a 10-doc slice. The PMC paper corpus exists to settle it. |

**Gap 5 to own, added 2026-08-11:** *nobody reports whether their ranking survives a
change of relevance rule.* D1 reports page-level gold only; D2 reports its own. We
ran three modes over identical chunks and the ranking inverts — first becomes fifth,
sixth becomes first. That is a methodological contribution, and it is also the
reason our own headline has to carry all three numbers.
</content>
</invoke>
