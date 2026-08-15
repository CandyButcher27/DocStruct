# Metrics — which are standard, which are ours, and what to report

Compiled 2026-08-05. Answers two questions a reviewer will ask:
*"are these standard metrics?"* and *"why these and not the ones everyone else uses?"*
Companions: [`evaluation.md`](evaluation.md) (how they are implemented),
[`related-work.md`](related-work.md) (who reports what).

## Audit of what we currently report

| Metric | Where | Standard? | Verdict |
|---|---|---|---|
| **MRR** | `eval/metrics.py` | Yes — textbook IR; reported by `2604.12047`, every chunking paper | Keep as headline |
| **nDCG@5** | `eval/metrics.py` | Yes — binary-gain nDCG; reported by `2604.12047` | Keep |
| **Recall@5** | `eval/metrics.py` | Yes | Keep. Report Recall@1/3/5 so it lines up with D1's Recall@3 |
| **Hit@1** | `eval/metrics.py` | Yes, but the field calls it **Precision@1** (`2604.12047`) | **Rename to Precision@1** in the paper. Same quantity, standard name |
| **Coverage** (word-instance multiset recall of the doc) | `eval/coverage.py` | **No — ours.** Nothing in the sweep reports it | Keep, but frame it. It is the chunker analogue of parser "vocabulary F1" (READoc) — cite that as the nearest relative |
| **Duplication** (chunk words / doc words) | `eval/coverage.py` | **No — ours** | Keep; it is the honest counterweight to coverage, and it is what makes Precision_Ω interpretable |
| **Context words @ k** | `eval/report.py` | Semi-standard (cost columns are common, the name is not) | Keep as a cost column |
| **MRR / 1k context words** | `eval/report.py` | **No — ours** | **Demote.** Chroma's token-level **IoU** is the published metric that encodes the same idea (rank quality per token spent). Report IoU as primary, keep MRR/1k as a secondary intuition |
| **Pk** (section boundaries) | `scripts/score_sections.py` | Yes — Beeferman et al. 1999, the standard text-segmentation error rate | Keep. Cite the original; note it forgives over-segmentation |
| **WindowDiff** (section boundaries) | `scripts/score_sections.py` | Yes — Pevzner & Hearst 2002, written specifically to fix Pk's blind spots | Keep, and **always report it beside Pk** — it punishes over-segmentation, Pk forgives it, and quoting one alone is quotable in either direction |
| **Straddle rate** | `scripts/score_sections.py` | **No — ours** | Keep as a *descriptive* column, never as an error term. 57.4% of gold sections are under `MIN_CHUNK_TOKENS`, so merging them is intended; it bounds section-label meaningfulness, nothing more |
| **mAP@0.5** (detection) | `eval/runner.py` | Yes — COCO-style; DocLayNet reports mAP@0.5:0.95 | Keep, but move to **mAP@0.5:0.95** to match DocLayNet, and run on its val split, not our 2 annotated docs |
| Bootstrap 95% CI + **paired** bootstrap | `eval/stats.py` | Yes, and **better than most competitors** — D1 reports point estimates only | Keep and lead with it. This is a genuine methodological edge |

**Summary answer to "are we using standard metrics?"** — the retrieval layer is
fully standard (MRR / nDCG@k / Recall@k / P@1) and matches our closest competitor
one-for-one; the section layer (Pk / WindowDiff) is standard too, and is the only
one whose gold no competitor in the table can report against. The *cost* layer is homemade (coverage, duplication, MRR/1k) and
should be re-expressed in the published vocabulary wherever an equivalent exists.

## Metrics to adopt

### 1. Token-level IoU / Precision / Recall / Precision_Ω (Chroma TR, 2024)

The standard metric set for chunking *specifically*, and the one that makes our
"cheap MRR is bought with context" argument citable instead of invented.

- **Recall** — fraction of gold-excerpt tokens present in the retrieved chunks.
- **Precision** — fraction of retrieved tokens that are gold tokens.
- **IoU** — intersection over union of retrieved-token set and gold-token set; penalizes both misses and padding, so a chunker cannot buy score by returning more text.
- **Precision_Ω** — precision under an oracle retriever (all chunks containing gold tokens retrieved). This isolates the **chunker** from the retriever, which is exactly the confound we already control by fixing the embedder. Strongest single metric for our claim.

Reference points from the Chroma report (n=5, text-embedding-3-large):
ClusterSemanticChunker 87.3% recall / 8.0 IoU · RecursiveCharacterTextSplitter
(200) 88.1% / 6.9 · LLMChunker 91.9% / 3.9. Note LLM chunking wins recall and
loses IoU badly — **that is our argument against LLM-in-the-loop, made by
someone else's numbers.**

### 2. Page-level Recall@k (for FinanceBench)

`chunk.page_num == evidence_page_num`. Required to use human gold whose evidence
is a page region rather than a sentence, and it is what `2604.12047` reports, so
it is the one column that can legitimately sit beside their table.

### 3. Determinism / reproducibility

Byte-identical chunk output across N runs, and wall-clock per page on CPU with no
network. Trivial to measure, nobody in the chunking literature reports it, and it
is the operational half of our contract. Report as a table column, not a claim.

### 4. Optional, only if we make table claims

**TEDS** (OmniDocBench / READoc / PubTables-1M). We emit table chunks and never
verify their structure. Either measure it or stop implying table quality.

## Metrics to *not* adopt, with reasons

- **Answer accuracy / Number Match / RAGAs faithfulness** — needs a generator in the loop, reintroduces an LLM and a model-version dependency into the headline. D1 reports it; we deliberately stop at retrieval, and should say so as a scoping decision rather than an omission.
- **Edit distance / CDM** — family A metrics for Markdown converters. DocStruct is not one.
- **BLEU-4** (used by `2410.09871`) — an MT metric applied to extraction; weakly motivated, and superseded by NED in later work.

## The honest-comparison rules for the paper

1. **Never place our MRR beside D1's MRR.** Different corpus, different retrievers, page-level vs span-level gold. Any cross-paper number needs the caveat inline, not in a footnote.
2. **Report the losses in the main table, not the appendix**: coverage 0.817 vs LangChain 1.00; duplication 2.06 (our worst-in-class number); no parse-fidelity number at all; no scanned-document support.
3. **MRR/1k and IoU are tradeoff axes, not rankings** (already stated in `evaluation.md`) — they structurally favour tools that return very little text. Say it in the caption.
4. **Keep the paired bootstrap.** Comparing overlapping marginal CIs is the standard way to wrongly call a real difference insignificant, and our v5→v6 reversal is a concrete case study of a measurement artefact flipping a conclusion — that story is worth a subsection.

## Proposed reporting table for the paper

| Group | Columns |
|---|---|
| Rank quality | MRR, nDCG@5, Recall@{1,3,5}, Precision@1, each with 95% CI + paired p vs DocStruct |
| Token economy | token-level Recall, Precision, **IoU**, Precision_Ω, context words @5 |
| Structure | Pk, WindowDiff vs publisher JATS (+ straddle rate and the reachability ceiling as context columns) |
| Fidelity | coverage, duplication |
| Cost | chunking s/page (CPU, no network), deterministic (Y/N) |
</content>
