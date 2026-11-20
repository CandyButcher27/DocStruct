# DocStruct — whole-project context

A single-file briefing for an AI assistant or a new collaborator: what this project
is, why it exists, how it is built, how it is evaluated, what it found, and what is
still open. It deliberately avoids a file-by-file code walkthrough — for that, read
the module docstrings under `docstruct/`.

Written 2026-09-15.

---

## 1. What the project is

DocStruct is a **local, deterministic, structure-aware PDF chunking library for
retrieval-augmented generation**. It takes a born-digital PDF and produces retrieval
units ("chunks") that follow the document's own structure — sections, tables, captions
— instead of a fixed token window.

It is **not** a RAG framework. The `indexing/` and `query/` modules exist only to
prove the chunks are retrieval-good; they are not the product. The product is
`docstruct.parse()` and `docstruct run`.

### The contract

> No LLM calls anywhere in the parse path. Same PDF in → same chunks out. Fully
> local, fully auditable.

This is the non-negotiable design constraint, and it is what distinguishes the project
from the commercial alternatives (LlamaParse, Unstructured's hosted API) and from the
"semantic chunking" line of work. Any proposal that puts a model call inside the
pipeline is rejected rather than implemented. Determinism is not aspirational: it was
measured — 95 of 95 documents parse byte-identically across independent processes,
5,810 chunks per run, 0 differing.

### The problem it addresses

Standard RAG chunking splits text on character or token counts. That destroys exactly
the structure that makes a document navigable:

- Tables split mid-row, so a row's label and its value land in different chunks.
- Figure captions detach from what they describe.
- Section headers orphan from the body they introduce.
- On two-column pages, naive text extraction welds the two columns together into
  unreadable interleaved lines before any chunker even sees them.

DocStruct detects layout **first**, orders it into real reading order, and only then
chunks — carrying a `SectionPath` (up to `h1 > h2 > h3`) on every chunk, which no
competing chunker in the benchmark emits. That metadata is what enables filtered
retrieval: `where={"h1": "4. Experiments"}`.

---

## 2. The aim

Two goals, one engineering and one scientific.

**Engineering:** ship a chunker that beats the standard open-source options on
retrieval quality while remaining offline, free and reproducible.

**Scientific:** the project's evaluation work turned into the more interesting
contribution. While building the benchmark we found that **the rule you use to decide
whether a retrieved chunk is "relevant" determines the ranking of the tools** — on
identical chunks, with everything else held constant. That finding became the paper.

---

## 3. How it works — the pipeline

```
PDF ─┬─> geometry detector ─┐
     └─> model detector ────┴─> fusion ─> reading order ─> extraction ─> chunking ─> Chunk[]
```

Detection and fusion run **per page**; reading order is assigned per page then offset
into one document-global sequence; extraction and chunking run once over all blocks.

### 3.1 Two independent detectors

The central design choice. Two layout detectors run **blind to each other** and never
share state:

1. **Geometry detector** — pure rules over pdfplumber primitives. Words group into
   lines by vertical proximity; lines group into blocks; a gap larger than
   `1.6 × line_height` breaks a block. Headers are font size above `1.15 ×` the body
   median, or a short all-bold line. Tables come from pdfplumber's `find_tables()`,
   which is **ruled-line based** — so borderless tables are invisible to it. Figures
   are clusters of graphic primitives covering ≥3% of the page and mostly text-free.
   Captions are matched by a regex on universal prose markers (figure/fig./table/…
   plus a number).

2. **Model detector (optional)** — YOLOv8 trained on DocLayNet, rendered at 150 DPI,
   confidence threshold 0.25. Its 11 classes collapse to DocStruct's 5 (`text`,
   `header`, `table`, `figure`, `caption`). The pixel→point coordinate transform lives
   here, because every downstream stage assumes it has already happened.

Without model weights the pipeline runs geometry-only and degrades gracefully: every
block becomes `unilateral_geometry`, nothing crashes, no network is touched.

**Why blind:** it is what makes the confidence scores mean something — agreement
between two uncorrelated observers, rather than one model's softmax renamed. All
reconciliation is quarantined in `fusion/`.

### 3.2 Fusion

Greedy IoU matching: for each model proposal, take the highest-IoU unused geometry
proposal above 0.35. Then priority NMS over everything (matched > model-only >
geometry-only, confidence breaking ties) at IoU 0.5, so nested duplicates from the two
streams collapse to one region.

| Case | Condition | Final confidence |
|---|---|---|
| Confirmed | both detect, same label, IoU ≥ 0.35 | `0.85 + 0.10·model_conf + 0.05·IoU` |
| Disputed | both detect, different label | `winner_conf × 0.85` |
| Unilateral model | model only | `conf × 0.75`, clamped [0.40, 0.85] |
| Unilateral geometry | geometry only | `conf × 0.60`, clamped [0.25, 0.65] |

For a matched pair, the model's box is taken outright when the model is confident
*and* the boxes agree (`conf ≥ 0.8`, `IoU ≥ 0.6`); otherwise the union.

**Honest caveat:** the unilateral scales and clamps are marked `# unvalidated` — they
were inherited from the v0 prototype and never calibrated against annotated boxes.
Nothing currently consumes them, but anything that starts to (confidence-weighted
ranking) would be building on sand.

### 3.3 Reading order

Split page blocks into at most two columns by the largest gap between block *centres*,
requiring the gap to exceed `0.15 × page_width`; then sort each column top-to-bottom.
Captions attach one-directionally to the nearest figure/table by centre distance, with
a discount when the caption sits below its target (the common layout).

This is **provably wrong** for a full-width element spanning a two-column body — a
title or a wide table has its centre near the page middle, so it joins one column and
reads in the wrong place. The principled fix (recursive XY-cut) is implemented in
`utils/xy_cut.py` and is **turned off**, because it measured worse on this corpus
(MRR 0.7356 vs 0.7457, recall identical). It is the better algorithm on layouts it was
built for and the worse one here; kept in the tree for less uniform corpora.

### 3.4 Extraction

`populate_text()` fills each block's text from pdfplumber within its bbox. The trap
here is word spacing: pdfplumber inserts a space when the inter-character gap exceeds
a flat 3pt default, which is *wider* than a real inter-word gap in small type — so
author lines, footnotes and table cells come out as `IreneAmerini1,ElenaBalashova2`.
Scaling the tolerance with font size fixes it, and it was worth a real MRR gain, not
because it looks nicer but because BM25 cannot match a term glued to its neighbour.

Tables are extracted as grids. Since `extract_tables()` is ruled-line based, a
**partly**-ruled table returns only the ruled fragment — which used to become the
block's entire text, silently dropping every unruled row. Now the rendered grid's word
count is compared against the raw region text, and falls back to raw text below 85%
coverage; the structure is preserved in `table_data` either way.

Several further passes (cross-page header/footer stripping, multi-page table merging,
label-aware containment suppression, dehyphenation, NFKC normalisation) are
implemented, unit-tested, and **gated to config flags that default OFF** — because the
repo rule is that nothing is enabled until it has been measured.

### 3.5 Hierarchy

Header depth comes from two deterministic, document-local signals — no hardcoded
section names, no regex on "Introduction":

1. **Section numbering** wins outright where present: "3 Method" → 1, "3.2 Setup" → 2,
   plus appendix (`A.1`) and Roman (`IV.`) forms, guarded so "A survey of…" does not
   qualify.
2. **Font-size rank** for unnumbered headings: largest distinct size → level 1,
   clamped at 3 levels.

Numbering was added because font size alone collapses the hierarchy entirely on
documents that set every heading at one size. It never moves a chunk boundary, so it
is invisible in the retrieval benchmark — but `section_path` is what filtered
retrieval is built on, and it was wrong on those documents.

### 3.6 Chunk assembly

Walks blocks in reading order maintaining a running `SectionPath`:

| Block | Behaviour |
|---|---|
| header | boundary flush; updates section path; its text opens the body of the chunk it introduces |
| text | accumulates; hard flush at `MAX_CHUNK_TOKENS` (500) with 75 words of overlap |
| table | atomic chunk, never split; does not split surrounding prose |
| caption | `figure_caption` chunk linked to its target; does not split prose |
| figure | skipped — represented through its caption |
| references | dropped entirely |

**The single most important rule in the project:**

> A structural boundary only *ends* the running chunk once it already holds
> `MIN_CHUNK_TOKENS` (200) words. Below the floor, the boundary is crossed and
> accumulation continues.

Without it, a page of prose interleaved with three figures became four unretrievable
stubs — roughly half of all chunks were under 25 words. Tiny chunks hurt ranking
twice: diffuse embeddings, and they crowd the top-5 with near-duplicates of each
other. This was the largest single quality win in the project's history.

A chunk is attributed to the section it *started* in, so crossing a header to reach
the floor never silently relabels the text that came before it.

Chunk sizes are counted in **whitespace words, not tokenizer tokens** — deliberately.
A real tokenizer would tie chunk boundaries to a model version and break "same PDF in
→ same chunks out".

### 3.7 Caching — why the research was affordable

Three content-hash-keyed caches: geometry proposals, model proposals, and fully
populated blocks. The block cache's key covers the PDF, the weights identity and a
**layout-config fingerprint** — but deliberately *excludes* chunking config, because
varying that cheaply is the entire point. A chunking ablation therefore redoes zero
detection work. Full suite 119s → 30s; a benchmark run ~18 min → ~10 min warm.

The corresponding hazard, found and fixed the hard way: any new flag that changes
*block* output must be added to the fingerprint, or an ablation of it silently serves
stale blocks and appears to measure nothing while looking like it worked.

### 3.8 Conventions that prevent whole classes of bug

- **Top-left coordinates everywhere** (`y0` = top, y increases downward), matching
  pdfplumber. Model output is transformed on the way in. Getting this wrong is the
  single most common source of silently-wrong geometry.
- **All thresholds live in `config.py`**, each with a comment naming the measurement
  that chose it. A bare number is a regression waiting to happen.
- **Plain dataclasses, no Pydantic**, so the core package needs only pdfplumber and
  numpy and `dataclasses.asdict` serialises everything.
- **Typed exception hierarchy** (`DocStructError` → `InvalidPDFError`,
  `EncryptedPDFError`, …) so callers never catch pdfminer internals.

---

## 4. Evaluation — the criteria and why each exists

Three layers, answering different questions. They are not interchangeable and should
never be conflated.

### 4.1 Retrieval quality — the headline

**Fair-comparison design:** the embedder and both retrievers are held constant across
every tool; only the chunker varies. So the leaderboard measures chunking and nothing
else.

- Embedder: `all-MiniLM-L6-v2`.
- Retriever A — dense cosine over ChromaDB.
- Retriever B (primary) — hybrid: dense + BM25 fused by Reciprocal Rank Fusion at
  k=60.
- Optional cross-encoder rerank, applied identically to all tools — which is exactly
  why it cannot close a *relative* gap.
- **Per-document indexing**: each document's chunks are indexed and queried in
  isolation, so cross-document confusion never contaminates the chunking signal.

**Metrics:** MRR, NDCG@5, Recall@5, Hit@1 — all standard. Plus two that exist as an
**anti-exploit**: *context words* (text handed to the generator per query, summed over
top-k) and *MRR per 1k context words*. A containment relevance metric always rewards
handing the retriever more text, so "make chunks bigger" is an unbounded way to buy
MRR — including for us. Reporting the price makes that visible in the same table it
improves. MRR/1k is a tradeoff axis, not a ranking: it structurally favours tools that
retrieve very little regardless of rank quality.

**Statistics:** per-question scores are retained; reports carry bootstrap 95% CIs on
each metric plus a **paired** bootstrap test of every tool against DocStruct. Paired,
because all tools answer the same questions — comparing overlapping marginal CIs is
the classic way to call a real difference insignificant.

**Baselines:** langchain (RecursiveCharacterTextSplitter), pymupdf4llm, unstructured,
llamaindex, llamaindex_semantic, plus DocStruct's own single-detector ablations
(`docstruct_geo`, `docstruct_model`). `docling` is implemented but excluded by default
— 10× slower, OOMs on some pages, always last.

### 4.2 Relevance rules — the part that became the paper

A retrieved chunk "counts" if it contains the answer. But *contains* has to be
defined, and the definition is not neutral:

| Mode | Compares | Measured bias |
|---|---|---|
| `span` | chunk text contains the gold sentence | rewards **large** chunks — containment is unbounded in chunk size |
| `region` | Szymkiewicz–Simpson overlap, normalised by the *smaller* side | the only size-tolerant rule (measured ρ(size, MRR) = +0.036) |
| `page` | chunk's pages contain the evidence page | rewards **small** chunks; credits being on the page, not containing the answer |

Span matching escalates through three comparisons: normalised substring →
**whitespace-blind** substring → token overlap at 0.6. Rule 2 exists for a specific
reason: word spacing in a PDF is *inferred*, not stored, and the gold carries whatever
guesses the generator's extractor made. Without it, a chunker that gets spacing *more
right than the gold* scores worse — the benchmark would have graded the
word-spacing extraction fix as a regression. Measured in isolation it changes
DocStruct's score by exactly 0.0000, which is the point: it guards against measuring
tokenizer agreement, not a thumb on the scale.

**Reachability is a ceiling, not a score.** Before a corpus's first leaderboard,
`scripts/gold_reachability.py` reports what fraction of the gold each rule can reach
*at all*, on the gold's own evidence page, identically for every tool. It also warns
when the question is circular — once the gold is a large share of its page
(FinanceBench: 69%), span and region reachability are ~100% by construction and prove
nothing.

### 4.3 Section-boundary agreement — no retriever at all

Pk and WindowDiff (both *error* rates, lower better) against section boundaries the
publishers wrote in JATS XML for PubMed Central papers. No embedder, no relevance
rule, and gold that predates this project and was authored for an unrelated purpose.
This is the cleanest measurement in the project.

It needed a chance floor, because both metrics reward matching the gold segment count
(ρ(|chunks − 25|, Pk) = 0.821). `scripts/section_random_baseline.py` places N
boundaries with no knowledge of the document; evenly-spaced boundaries score *worse*
than random, so the metric is not merely rewarding regularity.

### 4.4 Extraction fidelity — no gold needed

- **coverage** — fraction of the document's word *instances* present in some chunk.
  Silent loss shows up here and nowhere else.
- **duplication** — chunk words over document words; above 1.0 means content is
  emitted more than once, inflating the index and letting two chunks split the
  evidence for one query.

Counted as a **multiset** (a set-based version scores a tool that drops every repeat
as perfect), and the reference is raw pdfplumber defaults, never DocStruct's tuned
settings — scoring against our own extraction would hand us the metric. Neither number
ranks tools alone: a tool reaches coverage 1.0 by emitting the whole document as one
chunk, which is the exact failure the retrieval benchmark exists to catch.

### 4.5 Detection quality

Per-class precision/recall/F1 and mAP@0.5 of fused blocks against hand-corrected
boxes, produced by `docstruct export-annotations` → a browser correction UI. Only two
documents are annotated, which is under-invested and is precisely why the
`# unvalidated` fusion constants stay unvalidated.

### 4.6 Gold generation, and the rule that governs it

> Gold must be tool-agnostic, and preferably not ours.

Our own generator (`eval/qa_generator.py` — the **only** place this project talks to
an LLM, and never on the parse path) produces `(question, verbatim span)` pairs from
each PDF's **full raw pdfplumber text**, never from DocStruct chunks; every span is
validated as an actual substring before it is kept. Spans below 6 words are discarded,
because a two-word span is contained by almost any chunk mentioning the topic and
dissolves the benchmark's ability to discriminate.

Three generator constraints learned the hard way: segment by **characters not words**
(a 3,641-word equation-heavy segment tokenised to 12,228 tokens, so a word budget
silently produced zero questions on every dense document); **cap `max_tokens`**
(providers charge the *reserved* budget against the rate limit); and **pace requests**.

Even so, the headline corpora use **human-authored** gold instead. LLM-generated gold
measures a tool-vs-tool delta acceptably and is weak as an absolute claim.

---

## 5. Corpora

| Corpus | Size | Gold | Role |
|---|---|---|---|
| **OHR-Bench** | 95 born-digital docs (law 60, manual 15, finance 10, academic 10) | 3,558 human-authored questions | **Primary external leaderboard**, reported under all three relevance modes |
| **PubMed Central** | 134 papers | publisher-authored JATS XML section boundaries | Section-boundary agreement; no retriever involved |
| **FinanceBench** | 84 PDFs / 189 rows | human-annotated evidence blocks | **Not a leaderboard.** Kept for parse fidelity and borderless-table detection |
| **Internal arXiv** | 92 docs | LLM-generated | **Withdrawn** — see below. Was used for ablations only |

**Why FinanceBench is not a leaderboard:** measured, not assumed. Its evidence is
unreachable at top-5 under three different embedders on identical chunks, so *every*
tool scores MRR 0.0. That measures the retriever, not the chunker.

**Why the internal corpus is withdrawn:** a re-fetch reused the `docN.pdf` filenames
for *different papers*, so the gold no longer matches the PDFs — 0 of 65 shared
filenames have findable gold, and every ablation against it returns MRR 0.0. Worse,
the manifest was regenerated from the new files, so manifest and disk agree and
neither agrees with the gold; the dataset verifier cannot catch it. Every number
measured on that corpus is withdrawn, including the one significant result the vision
detector ever produced. Recovery is by `arxiv_id` from the manifest.

**The lesson**, recorded so it is not re-learned: *a manifest is evidence about the
files it was written from, and nothing else.*

---

## 6. What the evaluation found

### 6.1 The headline: the relevance rule decides the winner

OHR-Bench, 95 docs, 3,558 human questions, seven chunkers, **identical chunks in all
three columns** — only the relevance rule changes:

| Tool | `page` | `span` | `region` | ctx words |
|---|---|---|---|---|
| **docstruct** | 0.600 (6th) | **0.706 (1st)** | **0.666 (1st)** | 2194 |
| docstruct_geo | 0.470 (7th) | 0.705 (2nd) | 0.657 (2nd) | 2328 |
| pymupdf4llm | 0.668 | 0.699 (3rd) | 0.604 (3rd) | 2424 |
| unstructured | **0.795 (1st)** | 0.654 | 0.601 | **561** |
| langchain | 0.756 | 0.641 | 0.603 | 638 |
| llamaindex | 0.729 | 0.648 | 0.589 | 1430 |
| llamaindex_semantic | 0.652 | 0.654 | 0.575 | 4698 |

**The ranking inverts.** DocStruct is 1st of 7 under two rules and 6th of 7 under the
third; unstructured does the reverse. No relevance rule is size-neutral, and a
leaderboard that reports one rule has reported a ranking *and* a size preference
tangled together. We could find no prior chunking evaluation that varies the rule.

The `region` result does not depend on its threshold: swept 0.1→1.0, a DocStruct
variant is 1st at all ten values, margin +0.045 to +0.062 over the best external tool
across 0.4–1.0. Stated plainly rather than waiting to be asked: 0.7 is where our
margin happens to peak.

### 6.2 Section boundaries

| Tool | WindowDiff | Pk | Chunks | Docs |
|---|---|---|---|---|
| **docstruct_geo** | **0.4226** | **0.3418** | 26.8 | 134 |
| pymupdf4llm | 0.4800 | 0.4490 | 17.7 | 134 |
| **docstruct** | 0.4818 | 0.3531 | 37.5 | 134 |
| llamaindex_semantic | 0.5337 | 0.5128 | 29.1 | 134 |
| llamaindex | 0.6952 | 0.5979 | 42.7 | 134 |
| langchain | 0.8787 | 0.6200 | 85.6 | 134 |
| unstructured | 0.8933 | 0.6025 | 106.9 | **99** |

Against the chance floor, DocStruct beats random boundary placement by +0.16 to +0.18
Pk — and **five of the seven chunkers do not clearly beat random at their own chunk
count**, with one scoring 0.058 Pk *worse* than chance. (Floor and tool table are
currently on different document subsets, 122 vs 134, so the direction is safe and the
exact margin is not yet printable.)

### 6.3 Results that go against us — reported in the main table, not an appendix

- **The vision detector does not pay for itself outside arXiv.** On OHR-Bench it is
  worth +0.0012 span (p=0.80) and +0.0090 region (p=0.12) — neither significant on
  3,558 questions. Its apparent +0.1305 under `page` is an artefact of that rule
  rewarding chunk count. On PMC section boundaries, geometry-only *beats* the hybrid
  at 2.4× the speed. Three corpora, no effect. The one significant arXiv result was
  withdrawn with its corpus.
- **Context cost.** 2,194 words per query against unstructured's 561; they are 3.63×
  more efficient on MRR per 1k words. We win accuracy and lose token cost.
- **Coverage and duplication.** 0.9632 / 1.8322 against LangChain's 1.000 / 1.1005.
  LangChain splits raw text and drops nothing; our duplication comes from inline
  headers and separately emitted table chunks.
- **Academic is our weakest domain** in every mode (span 0.4526, 5th of 7). We win
  overall by winning law (0.8655) and manuals (0.7932). "Good for research papers" is
  not what this corpus says.
- **Tables are ceiling-limited**: span-mode table MRR is 0.19–0.31 for *every* tool,
  and table gold is only 35.7% span-reachable. Read that column against its ceiling.
- **unstructured hard-fails on 26% of PMC PDFs**, so its section row covers 99 of 134
  documents — its N belongs in the caption, and the failure rate is itself a result.
- **Two open extraction defects**: a block spanning both columns can have its text
  extracted across the gutter, interleaving the columns into an unreadable heading;
  and full-width elements above a two-column body sort *after* the columns. Both were
  found by drawing the pipeline's real output, not by the test suite — which was
  green.

---

## 7. What was tried and rejected

Most obvious ideas have already been built and measured. Several lost.

| Idea | Outcome |
|---|---|
| Recursive XY-cut reading order | Built, six tests pass, **measured worse** (0.7356 vs 0.7457 MRR). Off by default, kept for non-arXiv corpora |
| Naive containment suppression | Dropped **28% of content**. Reverted; a label-aware successor is gated off |
| Section-path breadcrumbs in chunk text | Reverted — duplicates metadata into the embedding. Inline header text is the version that survived |
| `MIN_CHUNK_TOKENS = 600` | The **highest MRR measured**, and rejected: the floor nearly equals the ceiling, so structural boundaries stop mattering — it wins by becoming the fixed-window chunking DocStruct exists to replace, at 59% more context |
| Cross-boundary chunk overlap | Measured worse on every metric that moved. Two chunks holding the answer do not outrank one; they split the evidence |
| Cross-encoder reranking as the fix | Lifts every tool identically, so it cannot close a relative gap. The gap was a chunking problem |
| Embedding-similarity semantic chunking | Rejected on principle — needs a model in the parse path |
| Any LLM call in the pipeline | Rejected on principle — breaks the contract |
| Tokenizer-based chunk sizing | Rejected on principle — ties boundaries to a model version, breaks determinism |

Roughly a dozen further features (dehyphenation, NFKC normalisation, multi-page table
merging, page-furniture stripping, k-column reading order, bold-as-depth-signal) are
implemented, unit-tested, and **gated OFF awaiting their ablation**. The default
`parse()` output is byte-identical with all of them present, and the test suite is the
guard.

---

## 8. The paper

`paper/` holds an ACL-template submission: *DocStruct: Layout-Aware PDF Chunking for
Structured Document Retrieval*. Built with `pdflatex → bibtex → pdflatex → pdflatex`;
author-year citations via `acl_natbib`.

The venue limit is **8 pages for the whole PDF, references included** — not 8 body
pages with a free tail. The draft has been through: a port from stock `article` to the
ACL template, a reframing from "the finding" to "the system, and what its evaluation
revealed", a narrative-completeness pass that reached 12 pages, and a compression pass
that cut prose 43% (7,219 → 4,087 words) and reached 9. It is still one page over.

An anti-slop gate (`.claude/skills/acl-paper/scripts/gate.py`) runs on every `.tex`
edit — em-dashes, bolded list lead-ins, weak qualifiers, unresolved `\todo{}` markers.
A claim ledger maps every number in the draft to the report that produced it; it
exists because a real error survived several drafts (an efficiency claim carrying the
raw context ratio 3.9× instead of the efficiency ratio 3.63×) until an outside reader
caught it.

---

## 9. What is open

**Blocking publication:** every number in the paper comes from one retrieval
configuration (`all-MiniLM-L6-v2` + BM25 + RRF, k=5). Two independent reviews arrived
at the same objection: the relevance-rule inversion may be conditional on the encoder.
Re-running the three rules under 2–3 more encoders (bge-base, e5-base) either makes
the central claim much stronger or produces a more interesting finding.

Also queued:

- Re-run PMC section scoring **keeping per-document output**, so the chance floor and
  the tool table sit on one document set. (Lesson: an aggregate-only report cannot be
  re-cut later.)
- `MIN_CHUNK_TOKENS` sweep on PMC — 57.4% of gold sections fall below the floor, and
  the floor was chosen on the withdrawn corpus, making it an *inherited* constant
  tuned to no corpus in the paper.
- Regenerate internal-corpus gold and restore the withdrawn ablations.
- Parse fidelity on OmniDocBench/READoc — a real hole: the system is a structural
  parser with no direct parse measurement.
- DocLayNet val split, to retire the `# unvalidated` fusion constants.
- The two open extraction defects (gutter interleaving, full-width ordering).

---

## 10. The working rules

These govern how the project is changed, and they are the reason the record above is
trustworthy:

1. **Measure before claiming.** Any change to chunking, reading order or extraction is
   worthless until it has run through `scripts/ablate.py` against the current numbers.
   "Should improve retrieval" is not a result.
2. **No LLM in the pipeline.** Confined to gold generation, never on the parse path.
3. **All thresholds in `config.py`**, each carrying the measurement that chose it.
4. **Top-left coordinates everywhere.**
5. **Every corpus needs its relevance rule checked before it is trusted** — not the
   span *length*, the span *reachability*. Both external corpora failed a naive
   assumption here, in opposite directions.
6. **A single-mode leaderboard is not a result.**
7. **Gold must be tool-agnostic, and preferably not ours.**
8. **Report the losses** in the main table, not an appendix.
9. **Every stage ends in a commit, and the engineering log gets the entry.** The log
   is as much the product of this project as the code.

### Where the durable knowledge lives

| File | Holds |
|---|---|
| `README.md` | public claims and the leaderboard |
| `ROADMAP.md` | the work queue |
| `reports/` | every benchmark report, each with its config snapshot |
| `notes.md` | the chronological engineering log, stage by stage |
