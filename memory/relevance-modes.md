# Relevance modes — the measurement that changes the answer

Measured 2026-08-11 on OHR-Bench. Read this before quoting any leaderboard number,
and before choosing a mode for a new corpus. Companion to
[`evaluation.md`](evaluation.md) (how the modes are implemented) and
[`results.md`](results.md) (what the numbers are).

## The finding

The same 95 documents, the same 3,558 questions, the same seven tools, and — this
is the part that makes it a clean experiment — **the same chunks**. `n_chunks` and
`mean_chunk_words` are byte-identical across the three runs, because chunking was
served from the warm cache and only the scoring rule changed. The relevance rule
is the sole variable.

| tool | `page` | `span` | `region` |
|---|---|---|---|
| **docstruct** | 0.6004 (6th) | **0.7059 (1st)** | **0.6657 (1st)** |
| docstruct_geo | 0.4703 (7th) | 0.7047 (2nd) | 0.6567 (2nd) |
| pymupdf4llm | 0.6684 (4th) | 0.6992 (3rd) | 0.6040 (3rd) |
| unstructured | **0.7950 (1st)** | 0.6539 (5th) | 0.6006 (5th) |
| langchain | 0.7562 (2nd) | 0.6406 (7th) | 0.6029 (4th) |
| llamaindex | 0.7294 (3rd) | 0.6483 (6th) | 0.5887 (6th) |
| llamaindex_semantic | 0.6515 (5th) | 0.6540 (4th) | 0.5749 (7th) |

**The ranking does not survive the change of rule. It inverts.** First becomes
fifth and sixth becomes first, on identical chunks.

This is the strongest methodological result the project has, and it generalises
past this corpus: **a leaderboard that reports one relevance mode is reporting the
mode as much as the tool.** Neither closest competitor reports more than one.

## Why each rule is biased, and in which direction

| Mode | Rewards | Mechanism |
|---|---|---|
| `span` | **large** chunks | containment: a bigger chunk contains more spans. Unbounded — "make chunks bigger" buys MRR forever |
| `page` | **small** chunks | more, tighter chunks each carry a clean page label and match a specific question better, so the top-5 lands on the evidence page more often |
| `region` | neither, by construction | Szymkiewicz–Simpson normalises by the smaller side, so containment in *either* direction scores 1.0 |

`page`'s direction was predicted wrong once already, and the wrong prediction is
worth keeping: the expectation was that page mode would favour `pymupdf4llm`,
which emits one chunk per page. It does not. **Unstructured wins page mode with
the smallest chunks in the field** (87 words against DocStruct's 317). Chunk
*count* beats chunk-page alignment.

`region` is the only mode with a defensible claim to size-neutrality, which is why
DocStruct winning it matters more than DocStruct winning `span`. Its threshold
`RELEVANCE_REGION_MIN_OVERLAP = 0.7` is still `# unvalidated` — that caveat rides
along with every region number until it is swept against real chunks.

## Significance, stated precisely

Paired bootstrap, 10,000 resamples, Δ = docstruct − tool on MRR:

| vs | `span` Δ | `span` verdict | `region` Δ | `region` verdict |
|---|---|---|---|---|
| docstruct_geo | +0.0012 | **not significant** (p=0.80) | +0.0090 | **not significant** (p=0.12) |
| pymupdf4llm | +0.0067 | **not significant** (p=0.23) | +0.0616 | significant |
| unstructured | +0.0491 | significant | +0.0626 | significant |
| langchain | +0.0652 | significant | +0.0626 | significant |
| llamaindex | +0.0575 | significant | +0.0768 | significant |
| llamaindex_semantic | +0.0518 | significant | +0.0908 | significant |

So the honest claim is **not** "DocStruct wins":

- Under `region`, DocStruct beats **all five** external tools significantly.
- Under `span`, it beats **four of five** — `pymupdf4llm` is inside the noise.
- Under `page`, it loses to four of five.

## The uncomfortable one: the vision detector is null here

`docstruct` vs `docstruct_geo` is the ablation that asks what YOLO is worth.

| corpus / mode | Δ MRR (hybrid − geometry-only) | verdict |
|---|---|---|
| internal arXiv v6, `span` | +0.0443 | significant, p=0.0026 |
| OHR-Bench, `page` | +0.1305 | significant |
| **OHR-Bench, `span`** | **+0.0012** | **not significant, p=0.80** |
| **OHR-Bench, `region`** | **+0.0090** | **not significant, p=0.12** |

On the two modes DocStruct wins, **the most expensive component in the pipeline —
the model detector, the GPU requirement, the whole hybrid design — buys nothing
measurable.** The +0.13 under `page` is a page-mode artefact: geometry-only emits
5,810 chunks against hybrid's 9,080, and page mode rewards chunk count.

Under `region`, geometry-only is actually *ahead* on table-sourced questions
(0.3868 vs 0.3655), which is the opposite of the story the 122-vs-4 table-detection
count on SEC filings suggested.

This does not make the detector worthless — FinanceBench is the corpus where
borderless financial tables should show its value, and that run has not happened.
But **on OHR-Bench it is a null result and must be reported as one.** Hard rule 1
cuts in both directions.

## Reachability is a ceiling, not a score

`scripts/gold_reachability.py` asks what fraction of a corpus's gold is findable
in raw PDF text *at all*, under each rule, on the gold's own evidence page. It is
identical for every tool, so it says whether a rule can measure the corpus.

OHR-Bench (`reports/gold_reachability.json`), 3,558 items:

| rule | reachable |
|---|---|
| plain substring | 40.8% |
| whitespace-blind | 48.6% |
| token fallback | 75.4% |
| **span (what `is_relevant` applies)** | **80.2%** |
| region | 80.2% |

By `evidence_source`: text **95.5%**, table **35.7%**, equation **8.9%**.

Two things follow. First, `span` is fair on this corpus — the "only 1.5% appear
verbatim" figure that once justified `page` as the only option compared spans to
OHR-Bench's normalised `gt_text`, not to the rule the benchmark applies. Second,
the table and equation slices are ceiling-limited: nobody scores well there
because the gold is largely unreachable in raw text, not because every chunker is
bad at tables.

DocStruct's `span` Recall@5 is **0.7867** against that 80.2% ceiling. Read it as
"close to saturated" rather than as a hard 98%: the ceiling is measured against
*pdfplumber's* page text, and each tool extracts differently, so a tool's own
reachability can differ from the reference.

### The ceiling question can be circular — check before trusting it

On FinanceBench the same script returns `span` 100.0%, `region` 98.4%, median
overlap 1.000. That is not an easy corpus, it is a question that answers itself:
both the token fallback and the overlap coefficient normalise by the gold, and
FinanceBench gold **is** a page region — measured, **69% of its page** by token
count. Asking whether it can be found on the page it was cut from can only return
yes. The script now reports the gold's median share of its page and refuses to let
the span/region rows stand as evidence above 20%. OHR-Bench measures 9%.

One row survives there and is worth having: only **28.0%** of FinanceBench evidence
appears verbatim in pdfplumber's text for its own annotated page — a parse-fidelity
signal about borderless financial tables, measured before any tool touches it.

## How to report this

> On OHR-Bench, DocStruct ranks 1st under span and region relevance and 6th of 7
> under page relevance, with **identical chunks** in all three runs. No relevance
> rule is neutral to chunk size; we report all three and state that the ranking
> does not survive the change.

Leading with `region` is defensible — it is the size-tolerant mode and the one
where the win is significant against every external tool. Leading with `span`
alone is not, and quietly dropping `page` is not.

## Rule

**A single-mode claim is not a claim.** Run `scripts/gold_reachability.py` on any
new corpus before its first leaderboard, check the circularity warning, then report
every mode the corpus can support.

## The region threshold itself, swept (2026-08-16)

`region` is the mode built to be size-tolerant, and its `RELEVANCE_REGION_MIN_OVERLAP =
0.7` was `# unvalidated` — so the obvious attack on the region result was that the
constant chose the winner. It does not.

Swept 0.1–1.0 on OHR-Bench by re-scoring one run's dumped overlaps offline (chunking
identical at every point; the dumping run reproduces the 2026-08-11 leaderboard to a max
MRR drift of 0.0002). **A DocStruct variant is 1st at all ten thresholds, and
{docstruct, docstruct_geo} hold both top places at all ten**, with a +0.045 to +0.062
margin over the best external tool across 0.4–1.0.

Two things to carry into any write-up:

1. **0.7 is where our margin peaks** (+0.0619). Say so first.
2. **The field below us reorders constantly** — llamaindex_semantic 2nd at 0.1 and 7th
   at 0.7, unstructured 4th at 0.6 and 7th at 1.0. So the mode-inversion lesson on this
   page has a smaller sibling *inside* one mode: our position is threshold-independent,
   but a published ranking of the whole field is not.

Numbers: [`results.md`](results.md); raw `reports/ohr_region_threshold_sweep.json`.
