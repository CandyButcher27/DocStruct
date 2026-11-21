# Claim ledger

Every quantitative claim in `paper/main.tex` resolves to a row here, and every row
names the artifact that produced it. A number with no row ships as `\unverified{}`
or does not ship. Verified against the artifacts on 2026-08-27.

**Regenerate a row, never retype one.** If a rerun changes a value, update the row
and `mimi/memory/results.md` together, then the prose.

## OHR-Bench — the headline

Source: `reports/ohr_results_{span,page,region}.json`, prose tables in
`reports/ohr_report_{span,page,region}.md`.
Protocol: 95 born-digital documents, **3,558** human-authored questions, seven
chunkers, top-5, one fixed embedder. The three runs share identical chunks
(`n_chunks` and `mean_chunk_words` are equal across modes; chunking came from a warm
cache and only scoring changed), so the relevance rule is the only variable.

MRR by mode, ranked:

| rank | span | page | region |
|---|---|---|---|
| 1 | **docstruct 0.7059** | unstructured 0.7950 | **docstruct 0.6657** |
| 2 | docstruct_geo 0.7047 | langchain 0.7562 | docstruct_geo 0.6567 |
| 3 | pymupdf4llm 0.6992 | llamaindex 0.7294 | pymupdf4llm 0.6040 |
| 4 | llamaindex_semantic 0.6540 | pymupdf4llm 0.6684 | langchain 0.6029 |
| 5 | unstructured 0.6539 | llamaindex_semantic 0.6515 | unstructured 0.6006 |
| 6 | llamaindex 0.6483 | **docstruct 0.6004** | llamaindex 0.5887 |
| 7 | langchain 0.6406 | docstruct_geo 0.4703 | llamaindex_semantic 0.5749 |

The inversion claim: docstruct 1st / 6th / 1st, unstructured 5th / 1st / 5th.
That is the mirror pair the abstract refers to. It is exact; do not round it into
"near-inversion" or extend it to any other pair.

Size and cost, identical across modes:

| tool | ctx words | coverage | duplication | n_chunks |
|---|---|---|---|---|
| docstruct | 2194 | 0.9632 | 1.8322 | 9080 |
| docstruct_geo | 2328 | 0.9638 | 1.1320 | 5810 |
| pymupdf4llm | 2424 | 0.9674 | 1.0979 | 3756 |
| llamaindex_semantic | 4698 | 1.0000 | 1.0000 | 3366 |
| unstructured | 560 | 0.9201 | 1.0567 | 18424 |
| llamaindex | 1430 | 1.0000 | 1.0474 | 5794 |
| langchain | 638 | 1.0000 | 1.1005 | 13877 |

Derived, safe to state: context cost $3.9\times$ the leanest baseline
(2194 / 560 = 3.92, leanest is unstructured). Duplication 1.83 is the highest of
the seven. Coverage 0.9632 against LangChain's 1.00.

## OHR-Bench Table 1 on one install (2026-11-20) — supersedes the page column above

Source: `reports/review/table1.json`, `reports/review/summary.md`
(`python scripts/review_summary.py`). Cells: docstruct_geo from
`reports/review/floor_ohr/floor_min200_max500_<mode>.json`; unstructured/langchain/
llamaindex span+region from `reports/budget/ohr_<tool>_k5_<mode>.json`, page from
`reports/review/table1/`; pymupdf4llm and llamaindex_semantic all modes from
`reports/review/table1/`; Docling = 8 merged shards in `reports/review/docling/`;
LumberChunker from `reports/review/lumberchunker/`.

| system | page | span | region | ctx |
|---|---|---|---|---|
| DocStruct (geometry-only) | 0.7235 (6) | 0.7048 (2) | 0.6566 (2) | 2328 |
| PyMuPDF4LLM | 0.6715 (7) | 0.6988 (3) | 0.6036 (4) | 2432 |
| Docling | 0.7969 (1) | 0.6658 (4) | 0.6291 (3) | 631 |
| Unstructured | 0.7957 (2) | 0.6567 (5) | 0.6032 (5) | 556 |
| LangChain | 0.7560 (4) | 0.6407 (8) | 0.6029 (6) | 638 |
| LlamaIndex | 0.7294 (5) | 0.6483 (7) | 0.5887 (7) | 1430 |
| LlamaIndex-sem | 0.6515 (8) | 0.6540 (6) | 0.5749 (8) | 4698 |
| LumberChunker | 0.7852 (3) | 0.7098 (1) | 0.6992 (1) | 1057 |

Safe to state: geo vs pymupdf4llm span +0.0057 p=0.33; geo beats every model-free tool
under region p<=0.001; LumberChunker minus geo: span +0.0050 p=0.38, region +0.0427,
page +0.0628. Spearman(mean chunk words, MRR), n=8: page -0.952 (p<0.001), span +0.119
(p=0.78), region -0.238 (p=0.57). Enlarged-chunk baselines: span diffs -0.0056..-0.0004
(p 0.37..0.97), region +0.0274..+0.0517. Chunk-size sweep and PMC sweep: summary.md.
Coverage 0.9638, duplication 1.1320 (Docling 1.1518 highest). Context ratio 2328/556 =
4.19. Withdrawn: page 0.600 (hybrid) and 0.470 (geo) — stale checkpoints (notes.md Stage 31).

## OHR-Bench equal context budget (review 1, R2)

Source: `reports/budget/ohr_<tool>_k<k>_<mode>.json`, summary and paired bootstrap in
`reports/budget/summary.md` (`python scripts/budget_compare.py`). GitHub Actions run
44727565429, one CPU install for all 14 runs. Hybrid `docstruct` was not rerun; the
comparison is `docstruct_geo` at k=5 (2,328 words/query).

| tool | k | words | span MRR | region MRR | span Recall@k |
|---|---|---|---|---|---|
| docstruct_geo | 5 | 2328 | 0.7048 | 0.6568 | 0.7889 |
| unstructured | 20 | 2193 | 0.6616 | 0.6091 | 0.8002 |
| langchain | 17 | 2137 | 0.6450 | 0.6080 | 0.7648 |
| llamaindex | 8 | 2264 | 0.6525 | 0.5932 | 0.7591 |

Safe to state: lead 0.043-0.060 MRR (span), 0.047-0.064 (region), every paired 95%
interval above zero. k=5 reruns reproduce the headline within 0.003 MRR (largest gap
unstructured span 0.6567 vs 0.6539; this install makes 19,036 chunks vs 18,424).
Loss to state: under span, unstructured k=20 recall 0.800 beats docstruct_geo 0.789.

## OHR-Bench region threshold sweep

Source: `reports/ohr_region_threshold_sweep.json`.

`RELEVANCE_REGION_MIN_OVERLAP` is swept 0.1 to 1.0. The docstruct family leads at
every threshold, so the *family* result is threshold-robust. **But the winner
within the family is not**: `docstruct_geo` ranks 1st at 0.1 through 0.5, and the
headline region table reports `docstruct` 1st at the shipped threshold of 0.7.
Any sentence claiming DocStruct-the-hybrid wins under region relevance is a
claim about 0.7 specifically and must say so. Repo rule 9 already concedes this:
0.7 is where our region margin happens to peak.

## Determinism

Source: `reports/determinism.json`.

95 documents, 2 independent processes, 95 of 95 identical, 0 differing,
**5,810 chunks per run**, 100.0% agreement.

**Mode is `geometry-only`.** The run does not cover the hybrid detector. Prose
saying "DocStruct is deterministic" on this evidence is overreach: the measured
claim is that the geometry-only configuration is byte-identical across processes.
Note also that 5,810 is the `docstruct_geo` chunk count in the OHR table, not the
`docstruct` count of 9,080 — consistent, and the reason the mode caveat matters.

## PMC section boundaries

Source: `reports/section_scores.json`, gold `data/qa/pmc_sections.json`,
scorer `scripts/score_sections.py`. 134 documents, publisher-authored JATS,
no embedder and no relevance rule.

| tool | Pk | WindowDiff | mean chunks | n_docs |
|---|---|---|---|---|
| docstruct_geo | **0.3418** | **0.4226** | 26.8 | 134 |
| docstruct | 0.3531 | 0.4818 | 37.5 | 134 |
| pymupdf4llm | 0.4490 | 0.4800 | 17.7 | 134 |
| lumberchunker | 0.4793 | 0.6486 | 60.6 | 122 |
| llamaindex | 0.5979 | 0.6952 | — | 134 |
| docling | 0.5926 | 0.8706 | 106.5 | 122 |
| unstructured | 0.6025 | 0.8933 | 106.9 | **99** |
| langchain | 0.6200 | 0.8787 | 85.6 | 134 |

Lower is better for both metrics. Docling and lumberchunker rows: Actions run
44773185163 plus retry 44864250733 (`reports/review/pmc_baselines/`), 122 scorable papers;
docstruct_geo on the same 122 is 0.3447 Pk / 0.4240 WindowDiff. Paired bootstrap, Pk diff
(tool minus geo): lumberchunker +0.135 [+0.110, +0.161], docling +0.248 [+0.227, +0.270];
WindowDiff +0.225 and +0.447, all intervals exclude zero.

Two things the prose must get right. **The leader is `docstruct_geo`, not
`docstruct`** — geometry-only beats the hybrid on both metrics, at 2.4x the speed.
"DocStruct leads both" is true of the family and false of the hybrid; write the
variant name. And **unstructured's row is 99 documents, not 134**: it hard-fails
on 35 of the PMC PDFs (26%). Repo rule 9 requires the N in the row.

## Gold reachability — a ceiling, not a score

Source: `reports/gold_reachability.json`, `scripts/gold_reachability.py`.
3,558 items over 95 documents.

Reachable: plain 40.8%, despaced 48.6%, token-fallback 75.4%, **span 80.2%**,
region 80.2%. Region median overlap 0.968.

This is identical for every tool, so it says whether a rule can measure the corpus
at all. It is never a result for DocStruct. The often-quoted "only 1.5% appear
verbatim" figure compared spans to the normalised `gt_text` rather than to
`is_relevant`, and is wrong; the measured number against the real rule is 80.2%.

## FinanceBench — do not quote as a leaderboard

Source: `reports/gold_reachability_financebench.json`, `notes.md` Stage 19.
84 PDFs, 189 rows. The measured zero is a pilot, not a seven-tool run: 2 filings
(`3M_2018_10K`, `3M_2022_10K`, 729 and 1,176 chunks), 7 questions, 2 tools, MRR@5 0.0.
A relevant chunk exists (region overlap 0.990 and 1.000) and is ranked 42nd to 299th
under all-MiniLM-L6-v2, bge-small-en-v1.5 and e5-small-v2, so the harness scores it;
it is outside the top five. Never write "all seven tools" or "every tool". Gold occupies
69% of its page, so span and region reachability are ~100% by construction and
prove nothing — the circularity warning in `scripts/gold_reachability.py` fires
here. Usable for parse fidelity and borderless-table detection only. A published
paper reports MRR 0.700-0.844 on the same files; that contrast is the finding.

## Withdrawn — must never reappear

The internal arXiv corpus (`benchmark_qa_v*`) numbers are withdrawn. The corpus
does not match its own gold: 0 of 65 shared filenames have findable gold, because
a re-fetch reused `doc<N>.pdf` names for different papers, so every ablation
returns MRR 0.0 (`notes.md` Stage 24, repo rule 10). The old coverage 0.817 and
duplication 2.06 came from that withdrawn table; the live numbers are 0.9632 and
1.8322 from OHR-Bench. Any draft, memory file or figure still carrying 0.817 or
2.06 is stale.

The 14-flag gated sweep on the 92-document v6 corpus was environment-blocked and
produced no shippable numbers. Every gated flag stays OFF. Do not cite it.

## Numbers with no artifact yet

Ship these as `\unverified{}` or as stated absences, never as prose values:

- Parse fidelity. No number exists for any tool.
- Detection-layer mAP. Rests on two hand-annotated documents; DocLayNet's
  validation split is the right home and has not been run.
- Token-level IoU and Precision-Omega. The dumped runs record the overlap score
  per retrieved chunk but not the chunk and gold token sets, so this needs a
  re-run, not a re-analysis. Report it missing rather than approximating it.

## Added 2026-11-20 for review 1 (`paper/reviews/review-01.md`)

- Spearman rho(mean chunk width, MRR) across the 7 tools: span +0.571 (p=0.18), region
  +0.036 (p=0.94), page -0.679 (p=0.09). Recomputed 2026-11-20 from
  `reports/ohr_results_{span,region,page}.json`. n=7: quote signs, not as significant.
- Span top three within 0.007 MRR: 0.7059, 0.7047, 0.6992 (`reports/ohr_results_span.json`).
- Region margin over best external tool >= 0.045 from threshold 0.4 up
  (`reports/ohr_region_threshold_sweep.json`).
- 122 tables (hybrid) vs 4 (geometry-only) on `3M_2018_10K`: notes.md Stage 13. A
  detection count, not a retrieval result.
- Table-sourced OHR questions: span MRR 0.19-0.31 for every tool
  (`reports/ohr_slices_span.md`); 35.7% of table gold span-reachable
  (`reports/gold_reachability.json`).
