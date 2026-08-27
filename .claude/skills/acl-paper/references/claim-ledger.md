# Claim ledger

Every quantitative claim in `paper/main.tex` resolves to a row here, and every row
names the artifact that produced it. A number with no row ships as `\unverified{}`
or does not ship. Verified against the artifacts on 2026-08-27.

**Regenerate a row, never retype one.** If a rerun changes a value, update the row
and `memory/results.md` together, then the prose.

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
| llamaindex | 0.5979 | 0.6952 | — | 134 |
| unstructured | 0.6025 | 0.8933 | 106.9 | **99** |
| langchain | 0.6200 | 0.8787 | 85.6 | 134 |

Lower is better for both metrics.

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
84 PDFs, 189 rows. Every tool scores MRR 0.0 at top-5 under three embedders on
identical chunks, which measures the retriever, not the chunker. Gold occupies
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
