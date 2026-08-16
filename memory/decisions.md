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

### Containment suppression — naive version caused 28% content loss, reverted
`fusion/containment.py`. The naive `suppress_contained` / `suppress_table_contained`
dropped 28% of content and are no longer imported by `pipeline.py` (kept in the
module, unit-tested, unused). The *label-aware* successor `suppress_text_in_tables`
(gated `LABEL_AWARE_CONTAINMENT`, default off) answers the "which nested regions are
real duplicates" question with the one provably-safe case: a text block ≥90% inside a
table whose serialized text already covers its words. Targets our duplication cost
(1.83× on OHR-Bench, the highest of the seven); every other nested case is still left
alone. Awaiting the sweep.

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

### Cross-boundary chunk overlap — measured, lost, stays off
`OVERLAP_ON_BOUNDARY = False`. The plan argued for it on intuition: a new
section's first chunk loses the last sentence of the previous one. Measured
(`reports/ablations/08_overlap_on_boundary.json`, 48 docs / 298 questions) it is
worse on every metric that moves — MRR 0.7432 vs 0.7457, NDCG 0.7658 vs 0.7708,
Recall 0.8826 vs 0.8859, Hit@1 identical — at 86 more chunks and slightly more
retrieved context.

`MIN_CHUNK_TOKENS` is why. With the floor in place most boundaries are *crossed*
rather than cut, so the stranded-opening case largely stopped happening. What the
overlap does now is duplicate text into a chunk that competes with its own source
for the same query; two chunks holding the answer do not outrank one, they split
the evidence.

### Cross-encoder reranking as *the* fix for the benchmark gap
It is wired in and available, but it is applied identically to every tool, so it
lifts all of them and cannot close a relative gap. Same objection to tuning RRF
`k`. Both are retrieval-side knobs; the gap was a chunking problem.

### Open, not rejected: the model detector's value does not replicate (2026-08-11)
Not a decision yet — a measurement that a decision will eventually have to answer,
recorded here so nobody re-derives it or quotes the arXiv number as universal.

`docstruct` vs `docstruct_geo` on OHR-Bench: **+0.0012 MRR under `span` (p=0.80)
and +0.0090 under `region` (p=0.12)** — neither significant, on 3,558 questions.
The internal arXiv corpus said +0.0443 (p=0.0026) — **withdrawn, being re-measured**
(`memory/results.md`) — and OHR-Bench `page` says
+0.1305, but page mode rewards chunk count and geometry-only emits 5,810 chunks
against hybrid's 9,080, so that one is an artefact of the rule. Under `region`,
geometry-only is *ahead* on table-sourced questions (0.3868 vs 0.3655).

So the YOLO layer — the GPU requirement and the largest cost in the pipeline —
currently has **no measured retrieval value outside arXiv**. Before concluding
anything: FinanceBench is the corpus built to show it (borderless financial tables,
122 vs 4 detected tables on `3M_2018_10K`) and has not been run. Do not remove the
detector on this evidence, and do not claim it pays for itself either.

---

## Landed config-gated, default OFF — implemented, awaiting ablation (Fable review)

The Fable review (`fable_suggestions.md`, notes.md Stages 8–9) produced a batch of
deterministic, no-LLM features. Each is **implemented, unit-tested, and gated to a
`config.py` flag that defaults OFF**, so default `parse()` output is byte-identical
and the 184-test suite is the guard. None may be flipped ON until it clears
`scripts/ablate.py` against `results.md` — the measure-before-keeping rule. Flags and
their `[MEASURE]` rationale live in `config.py`:

- **Text quality:** `DEDUPE_CHARS` (faux-bold doubled glyphs, the doc1 bug),
  `DEHYPHENATE` (line-break hyphens), `NORMALIZE_TEXT` (NFKC + soft hyphen).
- **Figures:** `FIGURE_OVERLAP_BY_AREA` (density-independent text-overlap test).
- **Reading order:** `MULTI_COLUMN` (k-column, expected no-op on 2-col arXiv),
  `BAND_SPLIT` (band-then-column — the middle path between the legacy splitter and
  the measured-worse XY-cut; band-cut only at full-width blocks, legacy column split
  within each band, so column detection is unchanged everywhere else).
- **Furniture:** `STRIP_PAGE_FURNITURE` (cross-page repeated header/footer removal).
- **Tables:** `TABLE_TEXT_STRATEGY_FALLBACK` (borderless), `TABLE_SERIALIZATION`
  = keyvalue, `TABLE_SPLIT_ROWS`, `TABLE_SETTINGS`, `MERGE_MULTIPAGE_TABLES`
  (join a table split across a page break; 2-page ceiling).
- **Hierarchy:** `HEADER_RANK_BY_WEIGHT` (bold as a depth signal; `Block.is_bold`).
- **Containment:** `LABEL_AWARE_CONTAINMENT` (see below).
- **References:** `KEEP_REFERENCES` (emit reference chunks, excluded from indexing).

The 14-run sweep measuring all of these on the 92-doc/558-q v6 corpus is what
`results.md` will record; until then treat every flag as unproven.

## Landed ON — deterministic correctness (no measurement needed to keep)

- **Fixed-point graphic clustering.** `_cluster_graphics` did one greedy pass, so
  clusters that grew into overlap after both absorbed primitives never merged —
  figure regions depended on primitive order in the PDF stream. Now iterates to a
  fixed point; order-independent. Unit test bridges two non-touching boxes through a
  third placed last.
- **Confidence-ordered proposal matching.** `_greedy_match` iterated model proposals
  in raw detector order; a low-conf box could claim the geometry box a higher-conf
  box needed. Now sorted by descending confidence (proposal_id tie-break).
- **Appendix / Roman section numbering.** Heading-number regex extended from
  digits-only to `A.`/`A.1`/`B.2.1` and `IV.`/`IX.1`, guarded so `A survey of...` and
  all-caps words are not read as numbered. Changes only section-path metadata.
- **Graphic-primitive cap** (`FIGURE_CLUSTER_MAX_PRIMITIVES`) so a pathological page
  skips the O(n²) merge with a warning instead of hanging.

## The ablation cache was silently config-blind for new flags — fixed

The block cache fingerprinted config, but only over the *original* `_LAYOUT_CONFIG_
KEYS`; the geometry proposal cache was not config-aware at all. So an ablation that
toggled any new flag reused baseline blocks and would have **measured a false null on
every gated feature above**. Fixed before running the sweep: moved the fingerprint
into `cache/pdf_cache.py`, registered every new block-affecting flag, and made the
geometry proposal cache key on it. The model (YOLO) proposal cache is deliberately
kept config-independent (`config_aware = False`) so expensive inference is still
reused across ablations. Chunking-only flags (`TABLE_SPLIT_ROWS`, `KEEP_REFERENCES`)
stay out of the fingerprint by design — the cache exists to vary them cheaply.

**Lesson:** a config-fingerprinted cache is only as correct as its key list; adding a
block-affecting flag without registering it turns every ablation of it into a false
null. Any new flag that changes block output must be added to `_LAYOUT_CONFIG_KEYS`.

## Accepted pragmatic form over the "correct" heavy refactor

### Per-parse config (§1.8) — locked override context, not a threaded config object
`config.override(**values)` (a lock-guarded save/set/restore context manager) plus
`parse(config={...})` / `run_pipeline(config=...)` give non-mutating, thread-safe
per-call overrides with **zero call-site rewrites**. The full Fable proposal — a
frozen `ParseConfig` threaded through every function — was not built: it is a large
mechanical sweep for a benefit (concurrent parses with *different* configs running
fully in parallel) a deterministic research library rarely needs. Accepted ceiling:
overridden parses serialize on the config lock. Upgrade to the threaded object only if
parallel-different-config throughput ever matters.

### SectionPath depth > 3 (§4.3) — deferred, not built
Reworking `SectionPath` to a property-backed `levels` list changes `asdict` output,
breaking the chunk-JSON contract and the golden test, for depth the arXiv corpus never
reaches. Fable agrees ("when the corpus needs it, not before"). New code keeps the
numbering clamp at `HEADER_LEVELS` so the cap does not spread.

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

### OCR off in every baseline adapter — a config choice the paper must state
Twice now a baseline has silently enabled OCR on born-digital PDFs: docling via a
bare `DocumentConverter()`, and pymupdf4llm via its layout path's
`use_ocr=OCRMode.SELECT_KEEP_OLD`, where an ONNX classifier flags pages and RapidOCR
re-derives them. Both were found by reading a run log, not by a test — a green suite
cannot see a tool quietly running a different pipeline.

Two reasons OCR is wrong here, and the second is the one that matters:

1. **Cost.** docling ~50 s/document; pymupdf4llm 270.6 s on `AES_2022_10K` (24/116
   pages OCR'd) against 89.3 s for langchain on the same file.
2. **It measures a pipeline we are not comparing.** Every other tool reads the text
   layer. A tool that OCRs is not a better chunker, it is a different system — and on
   `doc1.pdf` it was strictly worse anyway: 72,425 chars against 75,953 with OCR off,
   on a page that already carried 3,182 characters with 3 replacement characters.

So: **any new adapter must be checked for a default-on OCR path before its numbers
are trusted**, and "OCR disabled in all baselines" is a stated limitation, not a
silent convenience. The classifier's per-page verdicts are preserved rather than
discarded — `scripts/ocr_audit.py` runs it without invoking an OCR engine and writes
`reports/ocr_audit.json`. Its corpus-level `flagged_frac` is third-party evidence for
the "born-digital only" scope claim, which the paper currently asserts unsupported.

---

## Known-broken / known-missing, deliberately

- **Standalone figures produce no chunk.** A figure with no caption is dropped
  entirely. Skipped because a metadata-only chunk adds index noise for content
  nobody queries, and the benchmark cannot show a gain either way.
- **`# unvalidated` confidence constants.** `UNILATERAL_MODEL_SCALE`,
  `UNILATERAL_GEOMETRY_SCALE` and both `CONFIDENCE_BOUNDS` entries have never been
  calibrated. Nothing that consumes them (e.g. confidence-weighted RRF) should be
  trusted until they are.
- **Header levels from font size alone.** See `pipeline.md` §6. A gated fix exists —
  `HEADER_RANK_BY_WEIGHT` ranks by (size, bold) using the new `Block.is_bold` — off
  until the sweep shows it moves anything. Appendix/Roman numbering (landed ON) also
  chips at this: an explicit number states depth where font size only proxies it.
- **Doubled interior letters on some PDFs** (`"Trannsfer may hhave many
  mmeanings"` in doc1). Document-specific, not systemic — doc10 extracts cleanly
  through the same code path. Faux-bold rendered as duplicate offset glyphs that
  pdfplumber does not dedupe at this tolerance. A gated fix now exists —
  `DEDUPE_CHARS` calls `page.dedupe_chars(tolerance=1)` before extraction — off until
  ablated on doc1 + the full corpus (see the gated-features section above).
- **`data/doclaynet/` is in git history** (84MB, unreferenced). Untracked now;
  removing it from history needs a rewrite that would break existing clones.
