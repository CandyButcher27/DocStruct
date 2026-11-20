# Review 1 — received 2026-11-20

Verbatim text of the review, followed by how each point is handled.

## Review text

The paper presents a well-structured and technically interesting approach to layout-aware PDF chunking. The proposed DOCSTRUCT pipeline demonstrates promising retrieval and structural recovery performance, supported by comparative experiments and reproducibility analysis. However, broader evaluation and further validation of the design parameters would strengthen the generalisability and practical contribution of the work.

Points for Improvement:
1. Broader Evaluation: Extend evaluation to multilingual documents, scanned PDFs, and more complex document layouts.
2. Methodological Validation: Provide stronger validation of detector-fusion parameters and chunk-size settings using independent datasets and ablation studies.
3. Performance Analysis: Further investigate the computational cost, larger retrieval-context requirements, and limited improvement from the vision-based detector.

1. In Section 5.1 and Table 1, the authors acknowledge that DocStruct returns a significantly higher context word count per query (2,194 words) compared to baselines like Unstructured (560 words) and LangChain (638 words). Because retrieval evaluations under span and region definitions inherently favor larger blocks of retrieved text, it remains unclear whether DocStruct's superior MRR stems from high-quality layout preservation or simply from returning roughly 3.5× to 4× more text to the evaluator.
2. The evaluation should control for total returned tokens/words or include an ablation tuning baseline chunkers to comparable context budgets (e.g., larger target chunk sizes or returning more top-$k$ chunks for the smaller baselines). This would isolate the true benefit of layout-aware segmentation from raw context volume.

6. General Comments:
Strengths

Evaluation Bias Insight: Demonstrates chunk-size bias by showing how DOCSTRUCT ranks first under span rules but drops to sixth under page rules.

High Reproducibility: The baseline pipeline achieves total byte-identical reproducibility across separate processes while removing costly GPU dependencies.

Areas for Improvement

Missing Baseline: Lacks a direct comparison against semantic splitters like LumberChunker in the final evaluation table to fully prove layout-aware superiority.

Financial Failure Mode: Struggles heavily on FinanceBench with a zero MRR score, requiring a clearer plan to handle complex numeric table cells.

The system is real and the reporting is unusual as it admits the vision component is nothing. The "ranks first" claim, though, is complicated by the chunk size: DocStruct is returning approximately 2194 words per query, whereas Unstructured is returning 560 words per query. There is no size-matched or equal budget comparison provided. The margins are extremely small: under span, DocStruct, its geometry-only variant and PyMuPDF4LLM are essentially tied (p = 0.23). In region, the one with 0.7 appears to be the lead, but there is no explanation of why it is chosen and below 0.6 it is the geometry-only one. The layout-aware baseline is missing, as is the language-model baseline, and there are no ablations of the fusion rule, or the 200-word floor from the chunk-size parameter. The evaluation of the layout recovery which the method relies on is never directly evaluated. If the results of all seven systems in the FinanceBench section is 'exactly zero', it's looks like a harness bug.

The paper presents DOCSTRUCT, a deterministic layout-aware PDF chunking pipeline. Major contribution is the analysis which shows that the chunker rankings depend heavily on the relevance definition. The work is solid and the authors are commendably transparent about negative results. However, the large chunk sizes complicate claims about headline retrieval, necessitating size-matched comparisons. Also, no examined corpus supports the vision component, and the 200/500-word chunk parameters were adjusted using data that the authors now regard as unreliable. There is also a design mismatch in the structural evaluation.

I recommend acceptance with the following changes:
(1) include a size-matched retrieval comparison;
(2) show a corpus where vision is helpful;
(3) eliminate chunk-size parameters on a clean benchmark;
(4) correct Table 1 formatting and standardize naming;
(5) make clear how the FinanceBench limitation affects the OHR-Bench harness.

## Response plan

| # | Point | Status | Where |
|---|---|---|---|
| R1 | Context volume may explain the span/region win | **Done.** Spearman(chunk length, MRR) over eight systems: page −0.95 (p<0.001), span +0.12, region −0.24. Two budget controls (top-k raised; chunks enlarged): region lead survives both, span lead survives top-k but not enlarged chunks. | §5.1, Limitations |
| R2 | Equal-budget comparison: larger chunks or larger k | **Done, both variants.** (a) top-k raised: lead 0.043–0.060 span, 0.047–0.064 region. (b) chunks enlarged: span tie (every p>0.3), region +0.027 to +0.052. Page rule not scored under either control. | §5.1, Limitations |
| R3 | Span top within noise | **Text done.** Span: DocStruct vs PyMuPDF4LLM p=0.33; LumberChunker vs DocStruct p=0.38. Region is the rule the claim rests on. | §5.1 |
| R4 | Why 0.7 region threshold | **Text done.** Fixed before scoring; sweep 0.1–1.0 on the earlier run keeps a DocStruct configuration first among model-free tools at every threshold. Sweep figure removed for space. | §5.1 |
| R5 | FinanceBench all-zero looks like a harness bug | **Corrected now.** The draft overstated it. The measurement is a smoke test on 2 filings, 7 questions and 2 tools. A chunk with region overlap 0.990/1.000 exists, and the retriever ranks it 42nd–299th under three encoders, so the harness does find it, just outside the top five. It is not a full seven-tool result. | §5.4 |
| R6 | How the FinanceBench limit affects OHR-Bench | **Text done.** The same weakness shows up on OHR-Bench as table-sourced questions: 0.19–0.31 span MRR for every tool, with only 35.7% of table gold reachable in raw text. §5.4 now names the two untested remedies (header–value table serialisation behind a default-off flag; vision borderless-table proposals). | §5.4 |
| R7 | No corpus where vision helps | **Resolved by decision.** Vision is off by default; DocStruct in the paper is geometry-only, and DocStruct+vision appears only in §5.4 with its null result. | Fig. 1, §5.4 |
| R8 | Chunk-size floor tuned on withdrawn data; no fusion ablation | **Done for chunk size.** Sweep MIN 100–400 and MAX 800 on OHR-Bench and PMC: 200/500 within noise on retrieval, slightly behind smaller minimums on boundaries. Fusion ablation not run: fusion is inactive in the geometry-only default. | §5.1, Limitations |
| R9 | Missing LumberChunker and a layout-aware baseline | **Done.** Both in Table 1. Docling: 1st under page, behind DocStruct under span/region. LumberChunker (Gemini 3.5 Flash-Lite): 1st under span and region, beats DocStruct significantly under region and page; ~9,500 calls, not repeatable. | §4, Table 1 |
| R10 | Layout recovery never directly evaluated | **Text done.** Limitations says why no detection score is given: the two hand-annotated documents were drafted from the pipeline's own output, and DocLayNet is the vision detector's training data. | Limitations |
| R11 | Multilingual, scanned PDFs, complex layouts | **Out of scope, stated.** No OCR by contract. Word counts and heading-number rules assume space-delimited Latin script. Two-column reading order only. | Limitations |
| R12 | Table 1 formatting and naming | **Done.** Tables and figures use one display name per system (`make_paper_figures.py` re-run); the bold on the ctx column is gone. | Tables 1–2, Figures 2–4 |
| R13 | Design mismatch in structural evaluation | **Text now.** §5.2 states it: 57.4% of gold sections fall below the 200-word floor, so the metric partly measures how the floor merges short sections. | §5.2 |


Note (2026-11-20): the page column in the reviewed draft (DocStruct 0.600, geometry-only 0.470) was stale; it came from benchmark checkpoints written before a page-attribution fix. The revised Table 1 is measured on one install with current code.
