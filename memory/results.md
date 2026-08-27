# Results — current numbers and what every config value is worth

**Update this file whenever a benchmark or ablation runs.** It is the reference
any "did this help?" question should be answered from.

> **Gated-feature sweep (2026-07 Fable session): attempted, environment-blocked.**
> *(All MRR values in this block are v6-corpus numbers and are withdrawn — see
> "Internal corpus headline" below. The environment finding stands; the numbers do not.)*
> The 14-flag sweep on the 92-doc v6 corpus is **not runnable in the current
> environment**: one `ablate.py` run is **~69 min** (embedding ~7,100 chunks per run
> dominates; YOLO is cached), so 14 runs ≈ 16 h, and long background jobs here are
> killed roughly hourly while foreground calls cap at 10 min. What was obtained:
> - **Baseline reproduced** on the full 92 docs: MRR **0.8194**, NDCG 0.8313, Recall
>   0.9427, Hit@1 0.7384 — matches the headline docstruct 0.8203, so the harness (and
>   the config-aware-cache fix) are validated.
> - **15-doc arXiv subset** (108 q, ~3.3 min/run, a fast but low-power probe):
>   baseline MRR 0.910 / Hit@1 0.8704; `DEDUPE_CHARS` byte-identical (0.910), as
>   expected — the doubled-glyph bug is doc1-specific, not systemic on clean arXiv.
>
> **Consequence:** every gated flag in `decisions.md` stays **OFF**. None is flipped on
> arXiv-subset signal — the corpus that would actually discriminate them (non-arXiv,
> `benchmark_qa_v7`) needs the broadened-gold run, which also died mid-generation
> (4/23 docs). Run `scripts/_sweep.sh` on a machine that does not kill hour-long jobs,
> against the broadened corpus, to complete this.

## External headline — OHR-Bench, human gold, three relevance modes (2026-08-11)

**This is the number the paper leads with.** 95 born-digital documents (law 60,
manual 15, finance 10, academic 10), **3,558 human-authored questions**, seven
tools. Reports: `reports/ohr_report_{page,span,region}.md`.

The three runs share **identical chunks** (`n_chunks` and `mean_chunk_words` equal
across modes — chunking came from the warm cache, only scoring changed), so the
relevance rule is the only variable.

| tool | `page` | `span` | `region` | ctx words | MRR/1k |
|---|---|---|---|---|---|
| **docstruct** | 0.6004 (6th) | **0.7059 (1st)** | **0.6657 (1st)** | 2194 | 0.322 |
| docstruct_geo | 0.4703 (7th) | 0.7047 (2nd) | 0.6567 (2nd) | 2328 | 0.303 |
| pymupdf4llm | 0.6684 | 0.6992 (3rd) | 0.6040 (3rd) | 2424 | 0.288 |
| unstructured | **0.7950 (1st)** | 0.6539 | 0.6006 | 561 | **1.166** |
| langchain | 0.7562 | 0.6406 | 0.6029 | 638 | 1.005 |
| llamaindex | 0.7294 | 0.6483 | 0.5887 | 1430 | 0.453 |
| llamaindex_semantic | 0.6515 | 0.6540 | 0.5749 | 4698 | 0.139 |

**The ranking inverts with the rule.** Under `region` DocStruct beats all five
external tools significantly; under `span` it beats four of five (`pymupdf4llm`
is inside the noise, p=0.23); under `page` it loses to four of five. Full analysis,
including why each rule is size-biased, in
[`relevance-modes.md`](relevance-modes.md) — read it before quoting any row above.

**Two results that go against us, and belong in the main table:**

- **The vision detector is null on this corpus.** `docstruct` vs `docstruct_geo`
  is +0.0012 under `span` (p=0.80) and +0.0090 under `region` (p=0.12) — neither
  significant. The +0.1305 under `page` is a page-mode artefact (geometry-only
  emits 5,810 chunks against hybrid's 9,080, and page mode rewards chunk count).
  The arXiv +0.0443 below stands for arXiv; it does not generalise here.
- **Context cost.** DocStruct retrieves 2,194 words per query against
  unstructured's 561 — they are 3.6× more efficient on MRR/1k. We win accuracy
  and lose token cost.

**Slices** (`reports/ohr_slices_{page,span,region}.md`, produced by
`scripts/slice_results.py`, joined to gold on `(source_doc, question)`):

- **Academic is our weakest domain, in every mode.** `span`: docstruct 0.4526
  against unstructured 0.5151 and pymupdf4llm 0.5112 — 5th of 7. We win overall by
  winning law (0.8655, 1st) and manual (0.7932). "DocStruct is good for research
  papers" is **not** what this corpus says, and the 10-document academic slice is
  too thin to settle it either way — hence the PMC paper corpus.
- **Tables are ceiling-limited, not simply lost.** Span-mode table MRR is 0.19–0.31
  for *every* tool, and table gold is only 35.7% span-reachable (equation, 8.9%).
  Read the table column against that ceiling, not against the text column.
- **Back matter costs us, but does not explain the gap.** `span` MRR by document
  position: docstruct 0.7534 → 0.6696 across the first-to-last fifth (−0.084),
  unstructured 0.6477 → 0.6451 (flat). Real, and much smaller than any leaderboard gap.

**Reproducibility datapoint:** the 2026-08-11 `page` run reproduces the 2026-08-07
run tool-for-tool on MRR, NDCG, Recall, Hit@1 and chunk counts, on a different
machine and session.

## Section-boundary agreement — PMC papers, publisher JATS gold (2026-08-16)

**The only metric here that does not go through a retriever.** Boundaries against
boundaries, gold written by the publisher for its own purposes. Pk and WindowDiff are
*error* rates — lower is better. Report: `reports/section_scores.md`.

| tool | WindowDiff | Pk | straddle | mean chunks | docs | errors |
|---|---|---|---|---|---|---|
| **docstruct_geo** | **0.4226 (1st)** | **0.3418 (1st)** | 0.5129 | 26.8 | 134 | 0 |
| pymupdf4llm | 0.4800 | 0.4490 | 0.5734 | 17.7 | 134 | 0 |
| **docstruct** | 0.4818 | 0.3531 (2nd) | 0.4385 | 37.5 | 134 | 0 |
| llamaindex_semantic | 0.5337 | 0.5128 | 0.1889 | 29.1 | 134 | 0 |
| llamaindex | 0.6952 | 0.5979 | 0.3660 | 42.7 | 134 | 0 |
| langchain | 0.8787 | 0.6200 | 0.2202 | 85.6 | 134 | 0 |
| unstructured | 0.8933 | 0.6025 | 0.1820 | 106.9 | **99** | **35** |

Ceiling: 138 documents with gold, 3,381 sections, **84.7% body**, of which 134 scored
(4 dropped by the <50%-locatable rule). Ceiling and scores describe the same population.

**Reproduces the 24-document pilot exactly in order, and within 0.02 in value**
(WindowDiff 0.4362 → 0.4226, Pk 0.3525 → 0.3418) on a 5.6× larger corpus. That is the
evidence the metric is not noise.

**How to read it, and the three things that keep it honest:**

- **Chunk count confounds WindowDiff.** langchain (86 chunks) and unstructured (107) are
  scored against a gold averaging ~25 sections, and WindowDiff counts boundaries per
  window, so over-segmentation is punished. Their *straddle* rate is the best in the
  table for exactly the same reason — tiny chunks rarely cross anything. Pk forgives
  over-segmentation and still puts them at 0.60–0.62. Quote both or neither.
- **Straddle rate is not an error.** 57.4% of gold sections are shorter than
  `MIN_CHUNK_TOKENS`; merging them is the design. Not a win to claim.
- **unstructured's row is on 99 documents, not 134** — it hard-failed on 35 (26%). The
  rate matches the pilot (6 of 24), so it is systematic. Its N belongs in the caption,
  and the failure rate itself is a result about unstructured on born-digital PDFs.

**Third corpus where the model detector does not pay for itself.** `docstruct_geo` beats
hybrid `docstruct` on WindowDiff at 2.4× the speed, matching OHR-Bench's +0.0012 span /
+0.0090 region.

### Random-boundary floor (2026-08-28) — the section result survives its control

Pk and WindowDiff both reward matching the gold segment count, and distance from the
~25-section gold predicts the table: **ρ(|chunks − 25|, Pk) = 0.821 (p = 0.023)**,
**ρ for WindowDiff = 0.893 (p = 0.007)**. So the ranking needed a floor. Placing N
boundaries with no knowledge of the document, `scripts/section_random_baseline.py`,
`reports/section_random_baseline.json`, 122 docs, 5 trials:

| n | random Pk | random WD | evenly-spaced Pk | evenly-spaced WD |
|---|---|---|---|---|
| 18 | 0.4672 | 0.5137 | 0.4878 | 0.5228 |
| 27 | 0.4993 | 0.5748 | 0.5501 | 0.6056 |
| 29 | 0.5037 | 0.5858 | 0.5615 | 0.6237 |
| 38 | 0.5295 | 0.6399 | 0.5901 | 0.6868 |
| 43 | 0.5401 | 0.6669 | 0.6050 | 0.7246 |
| 86 | 0.5928 | 0.8293 | 0.6240 | 0.9146 |
| 107 | 0.6051 | 0.8734 | 0.6242 | 0.9526 |

Margin over random at each tool's own chunk count (positive = beats chance):

| tool | chunks | Pk margin | WD margin |
|---|---|---|---|
| docstruct | 37.5 | **+0.177** | **+0.158** |
| docstruct_geo | 26.8 | **+0.157** | **+0.152** |
| pymupdf4llm | 17.7 | +0.018 | +0.034 |
| unstructured | 106.9 | +0.002 | −0.020 |
| llamaindex_sem | 29.1 | −0.009 | +0.052 |
| langchain | 85.6 | −0.027 | −0.050 |
| llamaindex | 42.7 | −0.058 | −0.028 |

**The finding is larger than "we lead the table": five of seven chunkers do not
clearly beat random boundary placement at their own chunk count.** Evenly-spaced
scores *worse* than random everywhere, so the metric is not merely rewarding
regularity.

> **Not yet printable.** The floor is 122 docs; the tool table is 134. The 134-doc
> run's per-document scores are not on this machine (`.cache/section_ckpt/` holds only
> a 6-doc pilot; the real run was the Colab notebook and only aggregates survive in
> `reports/section_scores.json`). Both sides must be computed on one document set
> before this goes in the paper — re-run the PMC scoring keeping per-doc output. The
> 0.15+ margins make the direction safe; the exact numbers are not comparable yet.

### Chunk size vs rank, per relevance rule (2026-08-28)

Spearman ρ between mean chunk width and MRR across the seven tools, from
`reports/ohr_results_{span,region,page}.json`:

| rule | ρ(size, MRR) | p |
|---|---|---|
| span | +0.571 | 0.18 |
| region | **+0.036** | 0.94 |
| page | −0.679 | 0.09 |

With n=7 nothing is significant, so **read the signs, never the magnitudes** — this
cannot carry a headline. Its value: region at ρ≈0 is the first *measurement* that
region relevance is size-neutral, a property the paper previously asserted by design.

### Efficiency, corrected (2026-08-28)

`3.9×` is `2194/560`, the raw **context ratio**, with no rank quality in it. The
efficiency ratio is `1.1664 / 0.3217 = 3.63×` (unstructured's MRR-per-1k-words over
ours). An earlier draft attached 3.9 to an efficiency claim; both numbers now ship
with the right label.

## Region threshold — swept, and the ranking does not move (2026-08-16)

`RELEVANCE_REGION_MIN_OVERLAP = 0.7` was `# unvalidated`; the region headline rested on
it. Swept on **OHR-Bench, not FinanceBench** (whose evidence is unreachable —
`notes.md` Stage 19): 3,558 questions, 7 tools, offline re-scoring of one run's dumped
overlaps, so chunking is identical at every threshold and the rule is the only variable.
Report: `reports/ohr_region_threshold_sweep.json`.

**Precondition met:** the dumping run reproduces the cited 2026-08-11 leaderboard to a
max MRR drift of **0.0002**, with identical chunk counts for all seven tools.

| tool | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| docstruct | 0.9590 | 0.8803 | 0.7888 | **0.6659** | **0.5275** | **0.3890** |
| docstruct_geo | **0.9666** | **0.8926** | **0.7995** | 0.6567 | 0.5076 | 0.3704 |
| pymupdf4llm | 0.9455 | 0.8543 | 0.7468 | 0.6040 | 0.4821 | 0.3263 |
| llamaindex_semantic | 0.9617 | 0.8661 | 0.7400 | 0.5747 | 0.4297 | 0.3098 |
| unstructured | 0.9365 | 0.8339 | 0.7213 | 0.6008 | 0.3911 | 0.2127 |
| llamaindex | 0.9463 | 0.8407 | 0.7263 | 0.5885 | 0.4604 | 0.3401 |
| langchain | 0.9357 | 0.8137 | 0.7111 | 0.6031 | 0.4662 | 0.3370 |

**A DocStruct variant is 1st at all ten thresholds; the two hold both top places at all
ten.** Margin over the best external tool is +0.045 to +0.062 across 0.4–1.0. The region
result does not depend on the constant.

**Two caveats to state before a reviewer does:**

- **0.7 is where our margin peaks** (+0.0619). It predates the sweep by months, but say
  it in the paper rather than wait to be asked. The defence: the margin is +0.045 or
  better everywhere from 0.4 up — a bump on a plateau, not a cliff.
- **The low end is uninformative, not favourable.** At 0.0 every chunk is relevant and
  MRR is 1.0 by definition, so the convergence at 0.1 is that definition asserting
  itself. A metric that climbs as the threshold falls is not evidence for a low one.

**The field below us reorders constantly** — llamaindex_semantic is 2nd at 0.1 and 7th
at 0.7; unstructured 4th at 0.6 and 7th at 1.0. Our *position* is threshold-independent;
a *ranking of the field* is not, and owes the caveat. Same lesson as
[`relevance-modes.md`](relevance-modes.md), one level down, now measured.

**The variants cross over between 0.5 and 0.6**: geometry-only wins the loose half,
hybrid wins the strict half (+0.0092 at 0.7). Still small and still not significant, but
it is the first sign the detector does anything under a strict rule.

## Determinism — measured, not asserted (2026-08-16)

95/95 OHR-Bench documents parse byte-identically across **independent processes**,
5,810 chunks per run, 0 differing. `reports/determinism.json`,
`scripts/verify_determinism.py`.

Unplanned cross-check: 5,810 is exactly the `n_chunks` recorded for `docstruct_geo` in
`reports/ohr_results_span.json` — a run made days earlier on Colab through the
benchmark harness, not through `parse()`. Two code paths, two machines, same count.

Limits that ship with the claim: it holds **within** a version, not across versions;
and the run is **geometry-only**. The hybrid path goes through CUDA, whose kernel
selection is not guaranteed bit-reproducible, and no GPU was available to test it.

Three dense financial filings (120–217 pages) needed a 4-hour cap instead of 30
minutes. That is the performance limitation, measured.

## Internal corpus headline — WITHHELD, being re-measured (2026-08-16)

**Every internal-corpus number is withdrawn.** The 92-document figures, the
five-tool table, the "where the gain came from" deltas (+0.0429 floor, +0.0138
word-gap, -0.0101 XY-cut) and the vision-detector's +0.0443 all measured a corpus
that a re-fetch overwrote: it reused the `docN.pdf` filenames for different papers
(`notes.md` Stage 24/25). Verified again on 2026-08-16 by running the real code
path, not a filename comparison:

```
doc1.pdf   0/8 hits  MRR=0.0     # on disk: "ParVL: ... Multimodal LLMs"
doc10.pdf  0/8 hits  MRR=0.0     # gold asks about air pollution in Macau
doc100.pdf 5/5 hits  MRR=1.0     # one of the 9 newer docs, gold is sound
```

Do not quote any of them, including the 56-document re-measurement — that ran
against the same v6 gold and is being superseded for consistency.

**Repair in progress.** Gold is being regenerated with `gpt-oss:120b` over the 102
PDFs actually held, via `docstruct.cli gen-qa`, which reads each document's full raw
text and never any tool's chunks. Output: `data/qa/benchmark_qa_v9.json`. When it
lands, re-run the five-tool comparison and the 14-flag sweep against it, and restore
this section plus `paper/main.tex` §5.4 from that measurement.

**What is unaffected and still quotable.** OHR-Bench (95 docs, 3,558 human
questions, all three relevance modes), the PMC section metric (134 papers, publisher
JATS), the region-threshold sweep, and the determinism run. Those corpora are fetched
by scripts that key on document identity and their per-document checksums verify
today. Coverage and duplication now come from OHR-Bench — DocStruct 0.9632 / 1.8322
against LangChain's 1.000 / 1.1005 — not from the withdrawn internal table.

### The v5 → v6 reversal (kept for the method, not the number)

`reports/v5_report.md` reported hybrid vs geometry-only at +0.0092 MRR, p = 0.64 —
"the vision model doesn't pay for itself". That was a measurement artefact:
`page.extract_text()` welds the two columns of a paper into one unquotable line, so
gold from two-column pages — exactly where the vision model helps most — was silently
rejected during gold generation. Fixing reference extraction reversed the sign. The
lesson survives the corpus loss and is why a reachability ceiling is now computed
before every leaderboard; the effect sizes do not.

## Chunk-bounds sweep (`reports/ablations/`)

48 docs / 298 questions each, all else fixed, sorted by context cost.

| MIN/MAX | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks | Avg words | Context words | MRR/1k |
|---|---|---|---|---|---|---|---|---|
| baseline (no floor) | 0.6890 | 0.7199 | 0.8490 | 0.5872 | 3905 | 181.3 | — | — |
| 80 / 300 | 0.7022 | 0.7320 | 0.8658 | 0.5973 | 3440 | 218.7 | 1411 | **0.4978** |
| 120 / 400 | 0.7086 | 0.7431 | 0.8859 | 0.5940 | 2945 | 253.6 | 1686 | 0.4203 |
| **200 / 500 (chosen)** | **0.7319** | 0.7560 | 0.8826 | **0.6342** | 2519 | 294.7 | 2050 | 0.3571 |
| 120 / 800 | 0.7203 | 0.7513 | 0.8758 | 0.6141 | 2533 | 291.2 | 2170 | 0.3319 |
| 250 / 800 | 0.7257 | 0.7541 | 0.8792 | 0.6242 | 2174 | 339.6 | 2555 | 0.2841 |
| 400 / 800 | 0.7277 | 0.7612 | 0.8993 | 0.6174 | 1992 | 370.7 | 2873 | 0.2533 |
| 600 / 800 | 0.7584 | 0.7886 | 0.9128 | 0.6477 | 1832 | 403.6 | 3251 | 0.2333 |

Read the trap: **raw MRR rises monotonically with chunk size and MRR/1k falls
monotonically**. 600/800 has the best MRR on the page and is rejected — see
`decisions.md`. 200/500 sits on the Pareto front and strictly dominates the
250/800 it replaced (+0.006 MRR, +0.010 Hit@1, +0.003 recall, **20% less
context**).

**Useful alternate configuration:** if context budget matters more than rank,
80/300 delivers 0.7022 MRR — still above pymupdf4llm — at **43% of the context
cost**. The old flush-at-every-boundary code could not tell that story at all,
because it paid for tiny chunks *and* got the worst MRR.

## Reading order

| | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks |
|---|---|---|---|---|---|
| legacy column split (default) | **0.7457** | **0.7708** | 0.8859 | **0.6409** | 3070 |
| `06_xycut` | 0.7356 | 0.7666 | 0.8859 | 0.6275 | 3132 |
| `07_xycut_rowgap12` (4× row gap) | 0.7356 | 0.7666 | 0.8859 | 0.6275 | 3132 |

Recall identical, rank quality lower. The 4× row-gap run being byte-identical
localises the entire difference to the column cut.

## Report index

| Report | What it is |
|---|---|
| `reports/ohr_report_{page,span,region}.md` | **Current headline.** OHR-Bench, 95 docs, 3,558 human Q, 7 tools, identical chunks across the three modes. |
| `reports/ohr_slices_{page,span,region}.md` | The same runs sliced by evidence source, domain and document position. |
| `reports/gold_reachability.json` | OHR-Bench reachability ceiling per rule (span 80.2%). |
| `reports/gold_reachability_financebench.json` | FinanceBench — circular at 69% gold-to-page; only the 28.0% plain-substring row is informative. |
| `reports/v6_report.md` | Internal-corpus headline. 92 docs, 558 Q, CIs + paired tests + coverage. |
| `reports/v5_report.md` | 55 docs, superseded — scrambled two-column gold, do not cite. |
| `reports/v4_report.md` | 48 docs / 298 Q, pre-significance. Superseded. |
| `reports/rrf40_report.md` | Pre-work baseline: DocStruct 2nd at 0.6773. |
| `reports/ablations/*.json` | Single-variable runs, with `overrides` + full `config` in each file. |
| `reports/dataset_manifest_v2.json` | Provenance of the extended corpus (file, domain, source, sha256). |

## Corpus

`data/raw-pdfs/` is gitignored (rebuild with `scripts/fetch_dataset_v2.py`).
The corpus is **arXiv-heavy born-digital prose**, which is the most important
caveat on every number above — the XY-cut result is direct evidence that corpus
shape decides which algorithm wins. `fetch_dataset_v2.py` targets seven domains
(arxiv, legal, financial, medical, technical, govt, textbook) precisely to break
that homogeneity; the manifest records what has actually landed.
