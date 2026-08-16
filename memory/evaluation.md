# Evaluation

Two independent layers live in `docstruct/eval/`. They answer different questions
and should not be conflated.

## Layer 1 — Detection quality (`runner.py`, `metrics.py`)

Per-class precision / recall / F1 and confidence-ranked **mAP@0.5** of fused
blocks against hand-corrected ground-truth boxes.

Ground truth format, one JSON per document:

```json
{"boxes": [{"label": "text", "page_num": 0,
            "bbox": [x0, y0, x1, y1, page_width, page_height]}]}
```

Produced by `docstruct export-annotations` (model-assisted pre-fill) →
`tools/annotate.html` (browser correction UI) → `data/annotations/docN.json`.
Only two documents are annotated so far; this layer is under-invested relative to
Layer 2 and is the reason `# unvalidated` constants in `config.py` stay
unvalidated.

## Layer 1b — Extraction fidelity (`coverage.py`) — no gold needed

The detection layer above needs hand-annotated boxes and has two documents.
Coverage needs nothing: the PDF is its own ground truth, so it runs on the whole
corpus and on any corpus.

- **coverage** — fraction of the document's word *instances* present in some
  chunk. Silent loss shows up here and nowhere else. Both extraction bugs this
  project has found (partly-ruled tables dropping unruled rows; headings that
  lived in no chunk) would have moved this number long before a retrieval metric
  noticed.
- **duplication** — chunk words over document words. Above 1.0 means content is
  emitted more than once, inflating the index and letting two chunks split the
  evidence for one query.

Two things that would be easy to get wrong:

- Counted as a **multiset**. A set-based version scores a tool that drops every
  repeat of a term as perfect.
- The reference is raw **pdfplumber defaults**, never DocStruct's tuned extraction
  settings. Scoring against our own extraction hands DocStruct the metric.

Neither number ranks tools on its own — a tool reaches coverage 1.0 by emitting
the whole document as one chunk, which is the exact failure the retrieval
benchmark exists to catch. Read it beside the leaderboard.

Standalone runner (chunking only, no embedder or retriever):
`python scripts/coverage_report.py --limit-docs 20`.

## Layer 2 — Retrieval quality (`benchmark.py`) — the one that matters

This is the headline number and the thing every chunking change is measured on.

### Fair-comparison design

The embedder and both retrievers are **held constant across every tool**; only
the chunker varies. So the leaderboard measures chunking quality and nothing else.

- Embedder: `all-MiniLM-L6-v2` (`config.EMBEDDING_MODEL`), loaded once per run.
- Retriever A — **vector**: dense cosine over ChromaDB.
- Retriever B — **hybrid** (primary): dense + BM25, fused by Reciprocal Rank
  Fusion at `RRF_K = 60`, `BM25_CANDIDATES = 20`. Reported alongside vector so the
  **hybrid lift** column shows what fusion is worth per tool.
- Optional cross-encoder rerank (`--rerank-model`), applied identically to all
  tools. It lifts everyone, so it cannot close a *relative* gap.
- **Per-document indexing**: each document's chunks are indexed and queried in
  isolation, so cross-document confusion never contaminates the chunking signal.

### The gold, and why it is trustworthy

`eval/qa_generator.py` generates `(question, verbatim answer_span)` pairs from the
**full concatenated raw pdfplumber text** of each PDF — never from DocStruct
chunks. This is the non-negotiable rule: generating gold from the output of the
tool being benchmarked inflates that tool's score. Every span is validated as an
actual substring of the raw document text before it is kept, so any tool that
preserves that content can hit it.

Generation uses an OpenAI-compatible endpoint (Ollama cloud `gpt-oss:120b` by
default, GROQ via `--provider groq`) through a stdlib-only client.

Three constraints the generator has to respect, all learned the hard way:

- **Segment by characters, not words** (`QA_MAX_CHARS_PER_REQUEST`). Word count is
  a bad token proxy for scientific PDFs — a 3,641-word segment of an
  equation-heavy paper tokenised to 12,228 tokens (3.4 tokens/word) against ~1.3
  for ordinary prose. A word budget meant every dense document silently produced
  zero questions.
- **Cap `max_tokens`.** Providers charge the *reserved* completion budget against
  the per-minute limit, not what is generated. Unset, the reservation is the
  model's full completion length, which alone can exceed the limit and make a
  request that can never succeed.
- **Pace requests** (`QA_REQUEST_PACING_SECONDS`). One segment is most of a
  free-tier minute's allowance, so consecutive requests collide by construction.
  Waiting up front is cheaper than a rejected round trip plus `Retry-After`.

Answer spans below `QA_MIN_SPAN_WORDS` (6) are discarded. A two-word span like
`"DanceOPD"` is contained by almost any chunk that mentions the topic — it scores
every tool alike and dissolves the benchmark's ability to discriminate. Weaker
generators drift to exactly those spans regardless of the prompt, so the floor is
enforced at validation.

### Relevance (`relevance.py`)

A retrieved chunk counts as relevant if it **contains** the answer span. Three
escalating comparisons:

1. Normalized substring (lowercased, whitespace collapsed, soft hyphens dropped).
2. **Whitespace-blind** substring — both sides with all whitespace removed.
3. Token-overlap fallback at `RELEVANCE_MIN_OVERLAP = 0.6`.

That is `--relevance span`, the default, and it assumes gold marks a *sentence-level*
answer. Public human-annotated corpora mark a **block** instead, and the same rule
applied to block gold silently rigs the comparison — the fraction of FinanceBench
evidence regions a tool's chunks are structurally too small to contain runs from
**3% (pymupdf4llm)** to **74% (unstructured)**, so containment rewards whoever
chunks biggest. `--relevance region` scores by Szymkiewicz–Simpson overlap
coefficient instead, normalising by the smaller side so containment either way
scores 1.0. Use it for any block-level gold; see `benchmark-datasets.md`.

Rule 2 exists for a specific reason. Word spacing in a PDF is *inferred*, not
stored; extractors measure inter-character gaps and disagree, and the gold spans
carry whichever guesses the generator made. Without rule 2 a chunker that gets
spacing **more right than the gold** scores *worse* — the benchmark would have
graded the `TEXT_X_TOLERANCE_RATIO` extraction fix as a regression. Rule 2 was
measured in isolation and changes DocStruct's score by exactly 0.0000, which is
the point: it is a guard against measuring tokenizer agreement, not a thumb on
the scale.

### Relevance modes (`--relevance span|region|page`)

One rule does not fit three corpora, and picking the wrong one produces a
plausible-looking leaderboard rather than an obvious failure.

| Mode | Compares | Use for | Measured bias |
|---|---|---|---|
| `span` (default) | chunk text contains the gold sentence | our generated gold; OHR-Bench (80.2% reachable) | rewards **large** chunks — containment is unbounded in chunk size |
| `region` | Szymkiewicz–Simpson overlap, normalised by the smaller set | FinanceBench; OHR-Bench | the only size-tolerant rule; threshold **swept 2026-08-16** — a DocStruct variant leads at all ten values 0.1–1.0, so the ranking does not depend on it |
| `page` | chunk's pages contain the evidence page | OHR-Bench, as one of three | rewards **small** chunks. Coarse: credits being on the page, not containing the answer, and penalises any tool that drops back matter (DocStruct drops references) |

**`page`'s direction was predicted wrong, and the wrong prediction is instructive.**
The expectation was that page mode favours one-chunk-per-page tools
(`pymupdf4llm`). Measured, it does not: **unstructured wins page mode with the
smallest chunks in the field** (87 words). Chunk *count* beats chunk-page alignment.

**Picking a mode picks a winner.** On OHR-Bench, with identical chunks in all three
runs, DocStruct ranks 1st under `span` and `region` and 6th of 7 under `page`,
while unstructured does the reverse. A single-mode leaderboard reports the mode as
much as the tool. Full result and how to report it: [`relevance-modes.md`](relevance-modes.md).

Before a corpus's first leaderboard, run `scripts/gold_reachability.py`: it reports
what fraction of the gold each rule can reach at all, on the gold's own evidence
page, identically for every tool. It also warns when the question is **circular** —
once the gold is a large share of its page (FinanceBench: 69%), span and region
reachability are ~100% by construction and prove nothing.

`page` needs every adapter to report the pages a chunk drew from. `_pages_of()` in
`benchmark.py` normalises that: Unstructured and Docling count pages from 1, everyone
else from 0, and LangChain/LlamaIndex concatenate before splitting so their adapters
recover page spans by character offset. **The benchmark aborts if an adapter emits no
page metadata** rather than scoring it zero — a silent zero would read as a result.

### Metrics

- **MRR** — reciprocal rank of the first chunk containing the answer.
- **NDCG@5**, **Recall@5**, **Hit@1**.
- **Context words** — words actually handed to the generator per query, summed
  over the retrieved top-k.
- **MRR / 1k context words** — retrieval quality per unit of context spent.

The last two exist as an **anti-exploit**. A containment metric always rewards
handing the retriever more text, so "make chunks bigger" is an unbounded way to
buy MRR — including for us. Reporting the price makes that visible in the same
table it improves. MRR/1k is a **tradeoff axis, not a ranking**: it structurally
favours tools that retrieve very little text regardless of whether they rank well.

### Statistical robustness

Per-question scores are retained (`ToolResult.per_question`), and the report
carries **bootstrap 95% confidence intervals** on each metric plus a **paired
bootstrap test** of every tool against DocStruct. Paired, because all tools answer
the *same* questions — the paired difference has far lower variance than two
independent CIs, and comparing overlapping marginal CIs is the classic way to
call a real difference insignificant.

### Baselines compared

Default set: `langchain` (RecursiveCharacterTextSplitter), `pymupdf4llm`,
`unstructured`. `docling` is implemented but **out of `_ALL`** — 10× slower,
OOM-crashes on some pages, always last; reach it with `--tools docling`. Adapters
live in `eval/adapters/`, all behind one `chunk(pdf_path) -> List[EvalChunk]`
interface; `get_adapters()` silently skips any whose optional dependency is not
installed and the report records which were skipped.

### Single-detector ablation

`docstruct_geo` and `docstruct_model` run the same chunker with one detector
disabled (`run_pipeline(pipeline_mode=...)`). They answer "what is each detector
actually worth?" — the first question a two-detector design invites — and are
**deliberately not in the default tool list**, because that is a different
question from the cross-tool leaderboard.

**The current answer is uncomfortable and is not to be smoothed over.** On the
internal arXiv corpus the model detector is worth +0.0443 MRR (p=0.0026). On
OHR-Bench it is worth +0.0012 under `span` (p=0.80) and +0.0090 under `region`
(p=0.12) — **not significant in either**. Its apparent +0.1305 under `page` is a
page-mode artefact: geometry-only emits fewer chunks and page mode rewards chunk
count. FinanceBench, where borderless financial tables should favour it, has not
been run.

The trap here is the block cache: its key must include the pipeline mode, or a
geometry-only run *with weights present* hashes identically to the hybrid run and
serves its blocks, producing an ablation that measures nothing while appearing to
work.

## Ablation workflow

> **⚠ Ablations cannot run as of 2026-08-16.** The internal arXiv corpus on disk no
> longer matches `benchmark_qa_v6.json`: a re-fetch reused the `doc<N>.pdf` filenames
> for different papers, so every ablation returns `MRR=0.0`. Recover the corpus by
> `arxiv_id` from `dataset_manifest_v2.json` first. `notes.md` Stage 24.


`scripts/ablate.py` runs **one** adapter with `docstruct.config` overrides applied
before chunking, and writes metrics + a per-doc breakdown to
`reports/ablations/<name>.json`.

```bash
python scripts/ablate.py --name min300 --set MIN_CHUNK_TOKENS=300
python scripts/ablate.py --name overlap --set OVERLAP_ON_BOUNDARY=true
```

Two details that matter:

- It passes `cache_dir=None` **to the benchmark** so the checkpoint file cannot
  leak results between variants, while the *adapter* still gets the detector /
  block cache. Confusing these two uses of `cache_dir` silently reuses the previous
  variant's numbers.
- Baselines are unaffected by DocStruct chunking config, so only the DocStruct
  adapter is re-run between stages. The full multi-tool benchmark is re-run once
  at the end of a work pass to confirm.

**The adapter cache is only correct because it fingerprints config.** The block and
geometry-proposal caches key on `layout_config_fingerprint()` (`cache/pdf_cache.py`),
so a config override in an ablation invalidates them. The Fable-review batch exposed
the failure mode: new block-affecting flags were *not* in `_LAYOUT_CONFIG_KEYS`, so
their ablations reused baseline blocks and measured a false null. **Any new flag that
changes block output must be registered there.** The model (YOLO) cache is
deliberately config-independent so inference is reused across ablations. To ablate
config that changes blocks safely, the flag must be in the fingerprint — otherwise
run with `--cache-dir ""` (correct but re-runs YOLO, ~90 s/doc).

**Gated-feature sweep (Fable review):** `scripts/_sweep.sh` runs baseline + 13 gated
flags on the 92-doc/558-q v6 gold against the warm `.bench_cache`. Winners get their
flag flipped to default-on and the numbers recorded in `results.md`.

## Layer 3 — Section-boundary agreement (`scripts/score_sections.py`) — no retriever

The only layer whose gold was not written for this project. `scripts/fetch_pmc.py` pulls
open-access papers **with the publisher's JATS XML**; `scripts/build_jats_gold.py` turns
the XML's `<sec>` structure into `data/qa/pmc_sections.json`. The chunker's boundaries are
then compared to the publisher's, with no embedder and no relevance rule in between — so
none of the size-bias in [`relevance-modes.md`](relevance-modes.md) applies here.

- **Pk** (Beeferman 1999) and **WindowDiff** (Pevzner & Hearst 2002) — standard text
  segmentation *error* rates, lower is better. WindowDiff counts boundaries per window and
  so punishes over-segmentation; Pk only asks whether a window's ends share a segment and
  forgives it. **Report both** — they disagree by design and a single one is quotable in
  either direction.
- **Straddle rate** — ours, and *not* an error term: the fraction of chunks crossing a gold
  boundary. 57.4% of gold sections are below `MIN_CHUNK_TOKENS`, so merging is intended
  behaviour. It bounds how meaningful a per-chunk section *label* can be.
- **Ceiling first.** `scripts/section_reachability.py` locates each gold section in the
  PDF's own text before anything is scored. Back matter is excluded (DocStruct drops
  references by design) and documents under 50% locatable are dropped. Body ceiling is
  84.7% over 138 docs — scores read against that, never against 100%.
- **The ceiling and the score must cover the same documents.** They do as of 2026-08-16
  (138 with gold, 134 scored, 4 dropped by the <50%-locatable rule). They did not on the
  first run — 24 scored against a 126-document ceiling — because `fetch_pmc.py` trusted a
  committed manifest over the disk and `score_sections.py` skips a missing PDF silently.
- **`n_docs` is per tool, not per run.** unstructured hard-fails on ~26% of PMC PDFs, so
  its row covers 99 of 134. Put the N in the caption.

Numbers: `results.md`. Run it with `notebooks/pmc_sections_colab.ipynb` — hybrid
`docstruct` wants a GPU (312 s for 24 papers there; one figure-dense paper measured 475 s
geometry-only on a laptop CPU).

## Report provenance

`eval/report.py::config_snapshot()` dumps every uppercase `config` value into
`meta.config` of the JSON sidecar, and the chunking-relevant subset into a table
in the Markdown report. This exists because `reports/rrf40_results.json` is named
for RRF k=40 while its own prose says k=60, and there was no way to tell which was
true. **A benchmark number without its config is not a result.**

## Running a full benchmark

```bash
# 1. gold (LLM, needs OLLAMA_API_KEY in .env) — warms the block cache too
python -m docstruct.cli gen-qa data/raw-pdfs/*.pdf --out data/qa/benchmark_qa.json \
  --weights weights/yolov8m-doclaynet.pt --per-doc 5 --cache-dir .bench_cache

# 2. leaderboard
python -m docstruct.cli benchmark --pdfs-dir data/raw-pdfs --qa data/qa/benchmark_qa.json \
  --weights weights/yolov8m-doclaynet.pt --cache-dir .bench_cache \
  --report-md reports/vN_report.md --report-json reports/vN_results.json
```

Both commands resume: `gen-qa` skips documents already in the output file,
`benchmark` checkpoints per tool per document into `--cache-dir`. **Delete
`.bench_cache/bench_ckpt_*.json` before re-running a benchmark whose chunking
config changed**, or it will happily resume onto stale numbers.

## Known limits of the benchmark (state these, do not hide them)

- Gold is LLM-generated. In a Stage-0 audit, 7 of 9 spans DocStruct "missed" were
  not present in the raw PDF text at all — paraphrased or hallucinated. This
  penalises every tool identically so the *ranking* holds, but it caps the
  absolute numbers.
- Containment relevance misses genuinely paraphrased answers.
- `Chunk s` is not a fair speed column when `--cache-dir` is set: only the
  DocStruct adapter uses that cache, so it reports cache-hit time against four
  cold tools. Disclaimed in the report rather than quietly left in.
