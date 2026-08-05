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

The trap here is the block cache: its key must include the pipeline mode, or a
geometry-only run *with weights present* hashes identically to the hybrid run and
serves its blocks, producing an ablation that measures nothing while appearing to
work.

## Ablation workflow

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
