# Decisions — what was tried, measured, and rejected

**Read this before proposing anything.** Most of the obvious ideas have already
been built and measured. Several of them lost. Re-proposing one without new
evidence wastes a benchmark run.

---

## Rejected after measurement

### Recursive XY-cut reading order — built, tested, turned off
`utils/xy_cut.py`, `config.XY_CUT = False`.

The legacy column splitter is provably wrong for a full-width title/table across a
two-column body (its centre lands mid-page, so it joins one column). XY-cut fixes
that, has six tests covering exactly those layouts, and they pass.

It still lost: MRR 0.7356 vs 0.7457, Hit@1 0.6275 vs 0.6409, **recall identical**.
Raising `XY_CUT_MIN_ROW_GAP` 4× produced byte-identical output, which localises the
difference entirely to the column cut rather than the band cut.

Kept in the tree, off by default. The corpus is arXiv two-column papers — precisely
where the legacy heuristic's assumptions hold. It is the better algorithm on
layouts it was built for and the worse one here. **Enable it for less uniform
corpora; do not delete it.**

### Containment suppression — enabled, caused 28% content loss, reverted
`fusion/containment.py`. `suppress_contained` and `suppress_table_contained` are
implemented, imported by `pipeline.py`, and never called. Wiring them in dropped
28% of content. Before re-enabling, first explain *which* nested regions are real
duplicates versus real content, because the naive version cannot tell.

### Section-path breadcrumb injection into chunk text — reverted
Prepending `[Section: h1 > h2]` to chunk bodies. Reverted in `c11e091`.
`INLINE_HEADER_TEXT` (the header's own line opening the chunk it introduces) is the
version that survived and it is strictly better: it fixes real content loss rather
than duplicating metadata into the embedding.

### MIN_CHUNK_TOKENS = 600 — the highest MRR on the page, and a trap
The bounds sweep is monotonic: raw MRR rises with chunk size all the way to
MIN=600 (**0.7584**, the best number measured), while MRR-per-1k-context falls
monotonically. Reading only the MRR column picks 600.

At MIN=600 against MAX=800 the floor is nearly the ceiling, so almost every flush
is a token-limit flush and structural boundaries stop mattering. That is
fixed-window chunking with extra steps — it wins by *becoming the thing DocStruct
exists to be an alternative to*, and it costs 59% more retrieved context to buy
+0.027 MRR. **200/500 was chosen instead**, on the Pareto front.

### Cross-encoder reranking as *the* fix for the benchmark gap
It is wired in and available, but it is applied identically to every tool, so it
lifts all of them and cannot close a relative gap. Same objection to tuning RRF
`k`. Both are retrieval-side knobs; the gap was a chunking problem.

---

## Rejected on principle (not measurement)

### Embedding-similarity semantic chunking
Directly contradicts the core contract. It introduces model-version and hardware
float drift (kills determinism), and replaces the black-box heuristic chunking
DocStruct is positioned *against* with exactly that. It is also redundant:
geometry+vision fusion already provides ground-truth structure (headers, tables,
captions), which is a better boundary signal than inferred similarity for any
document that has real structure.

If un-headered long prose ever needs finer splitting than the token cutoff, the
deterministic fallback is **sentence-boundary splitting** (regex / spaCy
boundaries, no model) — not embedding similarity.

### Any LLM call inside the pipeline
Including VLM figure captioning. LLM use is confined to gold generation in
`eval/`. This is the product, not a limitation to route around.

### Tokenizer-based chunk sizing
Word counts are used deliberately. A real tokenizer ties chunk boundaries to a
model version and breaks "same PDF in → same chunks out".

---

## Accepted, with the reasoning that made them non-obvious

### Chunk-size floor on structural boundaries (`MIN_CHUNK_TOKENS`)
The single biggest win, **+0.0429 MRR**. The diagnosis mattered more than the fix:
DocStruct had the **best Recall@5 of any tool** and the **second-worst Hit@1** — it
found the answer more often and then ranked it lower. That is a ranking problem
from over-fragmentation, not an extraction problem. Roughly half of all chunks were
under 25 words, almost none of them token-limit flushes.

### Context-cost metrics in the leaderboard
Added *because* the floor change works partly by making chunks bigger. Left
unmeasured, "make chunks bigger" is an unbounded exploit of our own benchmark.
Reporting context words and MRR/1k prices every future change of that shape,
including ours.

### Whitespace-blind relevance
Added *before* the spacing fix could be credited, and measured in isolation to
prove it contributes exactly 0.0000 on its own. Without that isolation run it would
have been easy to credit a permissive metric for a real improvement.

### Block-level caching
Detection, fusion, reading order and extraction are deterministic in the PDF plus
the *layout* config and dominate wall time (one document took 256 s). Caching at
the block boundary — with chunking keys deliberately excluded from the cache key —
is what makes chunking ablations affordable.

---

## Known-broken / known-missing, deliberately

- **Standalone figures produce no chunk.** A figure with no caption is dropped
  entirely. Skipped because a metadata-only chunk adds index noise for content
  nobody queries, and the benchmark cannot show a gain either way.
- **`# unvalidated` confidence constants.** `UNILATERAL_MODEL_SCALE`,
  `UNILATERAL_GEOMETRY_SCALE` and both `CONFIDENCE_BOUNDS` entries have never been
  calibrated. Nothing that consumes them (e.g. confidence-weighted RRF) should be
  trusted until they are.
- **Header levels from font size alone.** See `pipeline.md` §6. Not measurable on
  the current benchmark, so not scheduled.
- **Doubled interior letters on some PDFs** (`"Trannsfer may hhave many
  mmeanings"` in doc1). Document-specific, not systemic — doc10 extracts cleanly
  through the same code path. Likely faux-bold rendered as duplicate offset glyphs
  that pdfplumber does not dedupe at this tolerance.
- **`data/doclaynet/` is in git history** (84MB, unreferenced). Untracked now;
  removing it from history needs a rewrite that would break existing clones.
