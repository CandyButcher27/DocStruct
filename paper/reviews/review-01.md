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
| R1 | Context volume may explain the span/region win (2,194 vs 560 words) | **Text now, run later.** §5.1 now says the span win cannot be separated from context volume. It reports the measured Spearman ρ between mean chunk width and MRR across the seven tools (span +0.571, region +0.036, page −0.679; n=7, none significant), so region shows no size trend. A size-matched run is still owed (see R2). | §5.1, Limitations |
| R2 | Equal-budget comparison: larger chunks or larger k for small-chunk baselines | **Open, needs a GPU run.** Two variants: (a) every tool's top-k raised until its context words reach about 2,200; (b) LangChain/Unstructured re-run with target chunk sizes matched to ours. Re-score under all three rules. | Limitations names it |
| R3 | Span top three within noise (p = 0.23) | **Text now.** §5.1 already gives p; it now says the top three under span are within 0.007 MRR, and that region is the rule we lead with. | §5.1 |
| R4 | Why the 0.7 region threshold; geometry-only leads below 0.6 | **Text now.** 0.7 was fixed before the sweep and never tuned. It is also where our margin peaks, and the hybrid's first place holds only from 0.6 up. | §5.1 |
| R5 | FinanceBench all-zero looks like a harness bug | **Corrected now.** The draft overstated it. The measurement is a smoke test on 2 filings, 7 questions and 2 tools. A chunk with region overlap 0.990/1.000 exists, and the retriever ranks it 42nd–299th under three encoders, so the harness does find it, just outside the top five. It is not a full seven-tool result. | §5.4 |
| R6 | How the FinanceBench limit affects OHR-Bench | **Text now.** The same weakness shows up on OHR-Bench as table-sourced questions: 0.19–0.31 span MRR for every tool, with only 35.7% of table gold reachable in raw text. | §5.4 |
| R7 | No corpus where vision helps | **Text now, run later.** None of the three corpora show a retrieval gain. On one SEC filing the vision detector finds 122 tables against geometry's 4; that is a detection count, not retrieval. Next corpus to test: borderless-table retrieval. | §5.4, Limitations |
| R8 | Chunk-size floor (200/500) tuned on withdrawn data; no fusion ablation | **Text now, run later.** Limitations already names the inherited floor. Queued: floor sweep on PMC and OHR-Bench; fusion IoU threshold ablation. | Limitations |
| R9 | Missing LumberChunker and a layout-aware baseline | **Text now.** LumberChunker was not run: it needs an LLM call per document. LlamaIndex-semantic is the semantic baseline. Docling (layout-aware) is wired in but never finished a run on our hardware. | §4, Limitations |
| R10 | Layout recovery never directly evaluated | **Text now.** Already in Limitations (parse fidelity); kept. | Limitations |
| R11 | Multilingual, scanned PDFs, complex layouts | **Out of scope, stated.** No OCR by contract. Word counts and heading-number rules assume space-delimited Latin script. Two-column reading order only. | Limitations |
| R12 | Table 1 formatting and naming | **Done.** Tables use one name per system, and the bold on the ctx column is gone (lower is not "best"). Figures still print code names until `make_paper_figures.py` is re-run. | Tables 1–2 |
| R13 | Design mismatch in structural evaluation | **Text now.** §5.2 states it: 57.4% of gold sections fall below the 200-word floor, so the metric partly measures how the floor merges short sections. | §5.2 |
