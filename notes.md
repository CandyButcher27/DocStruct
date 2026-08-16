# DocStruct — engineering log

> **Read [`ROADMAP.md`](ROADMAP.md) for what to do next. This section is what has been
> done and what still needs you.** Everything below the divider is the chronological
> log, Stages 1–25, kept because it records why each decision was taken.

---

# CURRENT STATUS — 2026-08-16

## The paper is submittable today

`paper/main.tex` → 8 pages, builds clean, 0 undefined references, no `TODO` renders.
Reframed around the finding rather than the system: *The Relevance Rule Decides the
Leaderboard: Seven PDF Chunkers, Identical Chunks, Three Rankings*.

| Result | Number | Corpus |
|---|---|---|
| 1st of 7 under `span` | MRR 0.706 | OHR-Bench, 3,558 human questions |
| 1st of 7 under `region` | MRR 0.666 | same |
| 6th of 7 under `page` | MRR 0.600 | same — **this is the evidence, not an admission** |
| 1st at all 10 region thresholds | +0.045–0.062 margin | same, re-scored offline |
| 1st on WindowDiff and Pk | 0.4226 / 0.3418 | 134 PMC papers, publisher JATS gold |
| Determinism | **95/95 docs, 5,810 chunks, 0 differing** | OHR-Bench, independent processes |

## What was done this session

**Measurement**
- Determinism measured for the first time: 95/95 documents byte-identical across
  independent processes. The chunk total cross-checks exactly against the benchmark's
  own count from a different machine and code path.
- Region threshold swept 0.1–1.0. A DocStruct variant leads at all ten, so the region
  result does not depend on the unvalidated constant. `config.py` now carries the
  measurement instead of a `# unvalidated` marker.
- Section boundaries re-run on the full 134-document PMC corpus (was 24). Order
  identical to the pilot, every value within 0.02.

**Correctness**
- `fetch_pmc.py` trusted a committed manifest over the disk, so a fresh clone
  downloaded nothing and scored 24 of 126 documents while looking finished. Fixed,
  tested, and smoke-verified against the live API.
- `refs.bib` had a **fabricated author attribution** and seven `{Anonymous}`
  placeholders. All 24 arXiv-backed entries verified against the arXiv API.
- PMC gold was redistributing verbatim text from 30 **No-Derivatives** articles on a
  public repo. Now licence-filtered to the 95 papers that permit derivatives.
- **The internal corpus was found to no longer match its own gold** — 0 of 65
  documents. Traced through git: all 31 arXiv ids on disk are August 2026 papers,
  against a gold committed 23 July. **56 of 92 documents recovered** by arXiv id;
  83.8% of their gold spans are reachable.

**Product**
- `pip install docstruct-rag` — packaged, `twine check` clean, verified by a fresh-venv
  install. New API: `parse_bytes`, `parse_many`, `to_langchain`, `to_llamaindex`,
  `to_jsonl`, `stats`. Full reference in [`docs/API.md`](docs/API.md).
- README leads with a demo animation rendered from a real parse, not a mock-up.
- `ROADMAP.md` consolidates `futureplans.md` and `implementation_plan.md` after
  re-checking every item against the code; two claims in the old audit were stale.

**Two defects found by drawing the pipeline's real output**
- Full-width elements sorted *after* the columns. **Fixed in code** by `BAND_SPLIT`;
  needs one ablation to flip the default.
- Full-width blocks extracted across the column gutter. A fix was written, found the
  correct gutter, and measured **14 garbled tokens before and 14 after**. Reverted
  rather than shipped. Recorded as a negative result.

## Running right now

**5-tool benchmark on the recovered 56-document corpus** — started 2026-08-16, several
hours on CPU because the vision detector runs without a GPU.

```bash
tail -f /tmp/bench56.log                     # progress
cat reports/v6_56doc_report.md               # when it lands
```

When it finishes, the paper's internal-corpus table (§5.4, currently the historical
92-document figures with a caveat) should be replaced with the 56-document numbers and
the caveat rewritten to say the subset is recovered and verified.

## What needs you — in priority order

| # | Task | Why it is yours |
|---|---|---|
| 1 | **Send the paper to your professors** | It will not improve by waiting. The open question is framing, and that is what they answer. Tell them Table 3 is historical and being re-measured; every external result is unaffected. |
| 2 | **`twine upload dist/*`** | Needs your PyPI token. Everything else is built and validated. TestPyPI first. |
| 3 | **Affiliation and co-authors** in `paper/main.tex` | The only remaining `\todo`. |
| 4 | **Hand `paper/REFERENCES_TO_VERIFY.md` to another model** | 7 entries have no arXiv id or DOI and could not be machine-checked. |
| 5 | **Check Colab Drive for a corpus backup** | `docstruct_bench/corpora/` may hold the 36 unrecovered documents. If it does, the full 92-document table is restorable. |

Nothing else needs a human. Everything remaining is either running, or is GPU work
listed in `ROADMAP.md` §3.

---

Running log of what was changed, why, what it measured, and whether it was kept.
Newest stage at the bottom. Every stage ends in a commit.

Goal for this pass: **DocStruct should not lose the cross-tool retrieval benchmark
to any provider**, and the package should be installable and usable as
`pip install docstruct-rag` → `import docstruct` (the PyPI name `docstruct` was taken).

---

## Stage 0 — Where we started (diagnosis, no code changes)

### The failure

`reports/rrf40_report.md` (last full run, 48 docs / 298 questions):

| Rank | Tool | MRR (hybrid) | NDCG@5 | Recall@5 | Hit@1 | Avg words/chunk |
|---|---|---|---|---|---|---|
| 1 | pymupdf4llm | 0.6915 | 0.7127 | 0.8289 | 0.6107 | 457.5 |
| 2 | **docstruct** | 0.6773 | 0.7091 | **0.8423** | 0.5738 | **164.8** |
| 3 | langchain | 0.6515 | 0.689 | 0.8221 | 0.5369 | 102.1 |
| 4 | unstructured | 0.6466 | 0.6718 | 0.7819 | 0.5604 | 85.2 |

The shape of the loss matters more than the size of it. DocStruct has the **best
Recall@5 of any tool** (0.8423 vs 0.8289) but the **second-worst Hit@1** (0.5738 vs
0.6107). It finds the answer more often than pymupdf4llm does, and then ranks it
lower. That is a ranking problem caused by chunk granularity, not a
content-extraction problem.

### Root cause: over-fragmentation

Per-doc, DocStruct's MRR tracks its chunk size almost monotonically. Its worst
documents are exactly its most fragmented ones:

| Doc | docstruct avg words | docstruct MRR | pymupdf4llm avg words | pymupdf4llm MRR |
|---|---|---|---|---|
| doc34 | 59.8 | 0.267 | 497.7 | 0.750 |
| doc24 | 62.4 | 0.333 | 834.6 | 0.567 |
| doc32 | 73.5 | 0.306 | 500.4 | 0.454 |
| doc41 | 116.4 | 0.185 | 572.2 | 0.444 |

Direct measurement of the chunk-size distribution on those documents (current HEAD,
hybrid mode) confirms it — the mean is misleading, the **median** is the story:

| Doc | chunks | mean words | median words | `<25` words |
|---|---|---|---|---|
| doc41 | 110 | 123.9 | 40 | 44 (40%) |
| doc34 | 217 | 53.0 | 27 | 104 (48%) |
| doc24 | 113 | 51.3 | 20 | 65 (58%) |
| doc32 | 65 | 77.2 | 23 | 36 (55%) |
| doc33 | 95 | 113.8 | 31 | 44 (46%) |

Roughly **half of all chunks are under 25 words**. `MAX_CHUNK_TOKENS` is 800, so
almost none of these are token-limit flushes — they are *boundary* flushes.
`chunking/assembler.py` calls `flush_text()` unconditionally on every header, every
table and every caption, with no floor on the resulting size. A page of prose
interleaved with three figures becomes four stub chunks instead of one good one.

Tiny chunks hurt ranking two ways: their embeddings are diffuse (few tokens, no
context), and they crowd the top-5 with near-duplicates of each other.

### Secondary finding: content actually lost

Checked every gold answer span that no DocStruct chunk contains, against the
block text and against raw pdfplumber text, to separate "we dropped it" from
"the LLM made it up":

- **7 of 9 misses are not in the raw PDF text at all** — hallucinated/paraphrased
  gold spans. Unfixable here, and they penalise every tool identically, so they do
  not affect the ranking.
- **doc24** — `"Meng-Hao Guo, Cheng-Ze Lu, ... and Shi-Min Hu"` is `in_blocks=True,
  in_chunks=False`. The author line was detected as a `header` block, and
  `assembler.py` **never writes header text into any chunk body** — headers only
  update the running `SectionPath`. Any question whose answer lives in a heading
  or a heading-like line is unanswerable by construction.
- **doc32** — a table row (`"Raw-GRPO 903 82.5 95.9 ..."`) is `in_raw=True,
  in_blocks=False`. Table extraction is losing rows.

Two real bugs, both worth fixing on their own merits.

### What was ruled out

- **Cross-encoder reranking** (plan §7.1) is already wired into
  `eval/benchmark.py` (commit `ac7f7af`) and is applied identically to every tool,
  so it cannot close a *relative* gap — it lifts all four. Still worth having in
  the production retriever, where it is currently missing, but not as the fix for
  this.
- **Tuning RRF `k`** — retrieval-side knob, applied identically to all tools.
  Same objection.

### Plan

1. Floor on boundary flushes + stop tables/captions splitting prose (the main lever).
2. Header text into chunk bodies (fixes the doc24 class of loss, and gives every
   chunk its own section context inline).
3. Table row loss (doc32 class).
4. Config provenance in reports, so any two runs can be diffed (plan §7.9).
5. Production-path parity: reranking and hybrid+`where` in `query/retriever.py`.
6. Packaging for `pip install`.

Measured with `scripts/ablate.py`, which runs one adapter with `docstruct.config`
overrides and writes metrics + per-doc breakdown to `reports/ablations/<name>.json`.
Baselines are unchanged by chunking work, so only the DocStruct adapter is re-run
between stages; the full four-tool benchmark is re-run once at the end to confirm.

---

## Stage 1 — Infrastructure before measurement

Three things had to exist before any change could be honestly evaluated.

**`scripts/ablate.py`** — runs one adapter with `--set KEY=VALUE` config overrides
and no benchmark checkpoint (checkpoints would leak results between variants).

**`docstruct/cache/block_cache.py`** — detection, fusion, reading order and
pdfplumber extraction are deterministic in the PDF plus the layout config, and
dominate wall time (one document took **256 s** to chunk). Caching at the block
boundary means a chunking ablation redoes none of it. The key covers the PDF bytes,
the weights identity and a fingerprint of every config value that can change block
output; chunking keys are deliberately excluded, since varying those cheaply is the
whole point. **Effect: full test suite 119 s → 30 s; benchmark run ~18 min → ~10 min
warm.** Useful, and it is the reason the rest of this pass was affordable.

**Config provenance in reports** (plan §7.9) — report `meta` recorded timestamp,
doc count, question count and LLM model, and nothing about the settings that
produced the numbers. This bit immediately: `reports/rrf40_results.json` is named
for RRF k=40 while its own prose says k=60, and there is no way to tell which is
true. Reports now carry the chunking settings inline and the full config dict in the
JSON sidecar.

**Baseline re-measured at HEAD** (the rrf40 report is old; the code moved under it):

| | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks | Avg words |
|---|---|---|---|---|---|---|
| `00_baseline` | 0.6890 | 0.7199 | 0.8490 | 0.5872 | 3905 | 181.3 |

Still behind pymupdf4llm's 0.6915, and the shape of the loss is unchanged.

---

## Stage 2 — Two real extraction bugs

Found while separating "we dropped it" from "the gold is wrong". Neither is the
main event, both are correctness issues worth fixing on their own.

**Partly-ruled tables dropped their unruled rows.** `extract_tables()` is
ruled-line based; on a table where only part of the grid is ruled it returns that
fragment, and `populate_tables()` rendered the fragment as the block's *entire*
text. Every unruled row vanished. Now the rendered grid is compared against the raw
region text and falls back to raw text below `TABLE_GRID_MIN_COVERAGE` (0.85).
`table_data` keeps the structure either way.

**Headings were in no chunk at all.** `assembler.py` used header text only to update
the running `SectionPath` — it never wrote it into a chunk body. Anything laid out
as a heading (titles, author lines, run-in headers) was unretrievable by
construction. Fixed as part of Stage 3 via `INLINE_HEADER_TEXT`.

Also deleted `table_to_markdown()`, dead since tables switched to plaintext
(`535c595`) — then restored it in Stage 5, which finally gave it a caller.

---

## Stage 3 — The fix: a floor on boundary flushes

`MIN_CHUNK_TOKENS=250`: a structural boundary only ends the running chunk once it
holds that many words. Below the floor the boundary is crossed and accumulation
continues. `BREAK_TEXT_ON_TABLE=False` / `BREAK_TEXT_ON_CAPTION=False`: tables and
captions still emit their own chunk but no longer *also* split the prose around
them. `INLINE_HEADER_TEXT=True`: the header opens the body of the chunk it
introduces. A chunk is attributed to the section it *started* in, so crossing a
header to reach the floor never relabels the text before it.

| | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks | Avg words |
|---|---|---|---|---|---|---|
| `00_baseline` | 0.6890 | 0.7199 | 0.8490 | 0.5872 | 3905 | 181.3 |
| `01_minchunk250` | **0.7257** | **0.7541** | **0.8792** | **0.6242** | 2174 | 339.6 |
| delta | **+0.0367** | +0.0342 | +0.0302 | **+0.0370** | −44% | +87% |

Everything moves the right way, and Hit@1 moves most — which is what the diagnosis
predicted, since the complaint was never that DocStruct couldn't find the answer.
**0.7257 beats pymupdf4llm's 0.6915 by +0.034**, at 339.6 words/chunk against its
457.5. Kept.

Sweeping both bounds to confirm 250/800 was not a lucky pick — see Stage 4. It
wasn't lucky, but it wasn't optimal either.

### The honest objection, and what was done about it

Part of this gain is simply *bigger chunks*, and a containment-based relevance
metric rewards handing the retriever more text per chunk. Left unaddressed, "make
chunks bigger" is an unbounded exploit of our own benchmark, and DocStruct would be
winning by the same move it criticises fixed-window chunkers for.

So the leaderboard now reports **context words** (words actually handed to the
generator per query, summed over the top-k retrieved) and **MRR per 1000 context
words**. A tool that wins MRR by returning 5×500 words is now visibly not the same
as one that matches it with 5×180, and any future "just make chunks bigger" change
shows its price in the same table it improves. This is the metric DocStruct should
want to be measured on anyway: it leads on MRR *while* being cheaper to feed to an
LLM than the tool it beats.

---

## Stage 4 — Sweeping both bounds, and the trap in the results

Grid over `MIN_CHUNK_TOKENS` / `MAX_CHUNK_TOKENS`, all else fixed, 48 docs / 298
questions each. Sorted by retrieved-context cost:

| MIN/MAX | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks | Avg words | Context words | MRR/1k |
|---|---|---|---|---|---|---|---|---|
| baseline (flush everywhere) | 0.6890 | 0.7199 | 0.8490 | 0.5872 | 3905 | 181.3 | — | — |
| 80 / 300 | 0.7022 | 0.7320 | 0.8658 | 0.5973 | 3440 | 218.7 | 1411 | **0.4978** |
| 120 / 400 | 0.7086 | 0.7431 | 0.8859 | 0.5940 | 2945 | 253.6 | 1686 | 0.4203 |
| **200 / 500** | **0.7319** | 0.7560 | 0.8826 | **0.6342** | 2519 | 294.7 | 2050 | 0.3571 |
| 120 / 800 | 0.7203 | 0.7513 | 0.8758 | 0.6141 | 2533 | 291.2 | 2170 | 0.3319 |
| 250 / 800 | 0.7257 | 0.7541 | 0.8792 | 0.6242 | 2174 | 339.6 | 2555 | 0.2841 |
| 400 / 800 | 0.7277 | 0.7612 | 0.8993 | 0.6174 | 1992 | 370.7 | 2873 | 0.2533 |
| 600 / 800 | **0.7584** | 0.7886 | 0.9128 | 0.6477 | 1832 | 403.6 | 3251 | 0.2333 |

**Raw MRR rises monotonically with chunk size, and MRR-per-1k-words falls
monotonically.** Read only the MRR column and the "right" answer is MIN=600 —
0.7584, the best number on the page. That is the trap. At MIN=600 against
MAX=800 the floor is nearly the ceiling, so almost every flush is a token-limit
flush and structural boundaries stop mattering: it is fixed-window chunking with
extra steps, and it wins by becoming the thing DocStruct exists to be an
alternative to. It also costs **59% more retrieved context than 200/500 to buy
+0.027 MRR**.

**Chose 200/500.** It is on the Pareto front and it strictly dominates the 250/800
I had defaulted to in Stage 3 — higher MRR (+0.006), higher Hit@1 (+0.010), higher
recall (+0.003), and **20% less retrieved context**. Against pymupdf4llm's 0.6915 it
is +0.040 MRR, and pymupdf4llm's own chunks average 457.5 words, so DocStruct wins
while handing the generator less text per query.

The two cheap settings (80/300, 120/400) are worth remembering: if context budget
matters more than rank, 80/300 delivers 0.7022 MRR — still above pymupdf4llm — at
**43% of the context cost**. That is a configuration story the old
flush-at-every-boundary code could not tell at all, because it was paying for tiny
chunks *and* getting the worst MRR.

---

## Stage 5 — Packaging, and the bug the packaging found

The library goal (`pip install docstruct` → `import docstruct`) was its own stage,
but it is also what surfaced the largest remaining extraction bug.

### The API

`run_pipeline()` returns the pipeline's internal result — blocks, chunks, fusion
diagnostics. Correct for evaluating the pipeline, wrong for using it. Added
`docstruct.parse()` returning a `Document`:

```python
import docstruct
doc = docstruct.parse("paper.pdf")
doc.text / doc.markdown / doc.pages() / doc.sections() / doc.chunks / doc.to_json()
```

No new pipeline behaviour — a view over what was already there. `markdown` renders
from *blocks*, not chunks, because chunks are sized for retrieval and deliberately
merge across headings, which is the wrong shape for a document meant to be read.
This is what finally gave `table_to_markdown()` a caller, so it came back.

Verified properly rather than assumed: built the wheel and sdist, checked the
wheel contains only `docstruct/` (no data, weights or tests — 58 entries), installed
it into a **fresh venv with only the core dependencies**, and ran `parse()` on a
real PDF. Geometry-only, no model, no network.

### What the smoke test found

The output read `IreneAmerini1,ElenaBalashova2` and `1.Introduction`. pdfplumber
inserts a space when the inter-character gap exceeds a **flat 3pt default**, which
is wider than the real inter-word gap in small type — so author lines, footnotes and
table cells lose their word breaks entirely. `x_tolerance_ratio` scales the
tolerance with font size and fixes it without over-splitting large headings.

This was not only cosmetic. BM25 cannot match a term that has been concatenated to
its neighbour, so it was costing real retrieval hits:

| | MRR | NDCG@5 | Recall@5 | Hit@1 |
|---|---|---|---|---|
| `03_min200_max500` | 0.7319 | 0.7560 | 0.8826 | 0.6342 |
| `04_xtolerance` | **0.7457** | **0.7708** | **0.8859** | **0.6409** |

### A metric that would have punished the fix

Word spacing in a PDF is *inferred*, not stored — extractors measure gaps and
guess, and they disagree. The gold answer spans carry whichever guesses the
generator made at the time. So a chunker that gets spacing *more right than the
gold* scores *worse* on verbatim containment. The benchmark would have graded this
fix as a regression.

Made the containment check whitespace-blind (compare with all whitespace removed,
in addition to the normalized comparison). It is applied identically to every tool,
so it does not favour DocStruct — and measured in isolation it changes DocStruct's
score by **exactly nothing** (0.7319, identical to `03`), which is the point: the
entire +0.0138 is the extraction fix, and the rule is a guard against the metric
measuring tokenizer agreement instead of retrieval quality.

Isolating this mattered. Run `04` mixed both changes; without run `05`
(`TEXT_X_TOLERANCE_RATIO=0`, new relevance rule) the honest attribution was
unavailable, and it would have been easy to credit a permissive metric for a real
improvement.

---

## Stage 6 — Recursive XY-cut reading order: implemented, measured, turned off

Plan §2. The legacy `detect_columns` splits a page into at most two columns by the
largest gap between block **centres**. That is provably wrong for any page whose
layout is not uniformly one- or two-column: a full-width title, abstract, table or
figure sitting across a two-column body has its centre near the page middle, so it
is assigned to whichever column wins and is read in the wrong place.

Implemented recursive XY-cut in `docstruct/utils/xy_cut.py`. Two deliberate
departures from the snippet in `implementation_plan.md`, which does not work as
written:

- Its `find_gaps` accepts a `limit` argument and never uses it, so *any* gap
  triggers a split — including sub-point gaps between consecutive paragraphs. It
  would Y-split down to single blocks on every page and degenerate into a plain
  y-sort, destroying the column handling it exists to provide. Real thresholds are
  required, and the cut must be **vertical-first**: on a two-column region the
  horizontal paragraph gaps are genuine whitespace bands, and cutting on one
  interleaves the two columns.
- It tests membership with `b not in top_half` on `Block`, a non-frozen dataclass
  with generated `__eq__` — that is a field-wise comparison, so two blocks with
  equal fields collide, and it is O(n²) besides. Indices throughout instead.

Six tests cover the cases that motivate it, including the full-width-title case the
legacy splitter gets wrong. They pass.

**And then it lost.**

| | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks |
|---|---|---|---|---|---|
| legacy column split | **0.7457** | **0.7708** | 0.8859 | **0.6409** | 3070 |
| `06_xycut` | 0.7356 | 0.7666 | 0.8859 | 0.6275 | 3132 |
| `07_xycut_rowgap12` | 0.7356 | 0.7666 | 0.8859 | 0.6275 | 3132 |

Recall is identical — the same content is reachable either way — but rank quality
drops. Raising `XY_CUT_MIN_ROW_GAP` 4× produced **byte-identical results**, which
localises the difference entirely to the column cut rather than the band cut: with
vertical cuts tried first, row gaps almost never decide anything.

**Turned off by default; code and tests kept.** It is the better algorithm on the
layouts it was built for and the worse one on this corpus, and a corpus of arXiv
two-column papers is exactly where the legacy heuristic's assumptions hold. Shipping
it on would have been choosing the more elegant implementation over the measurement.
`XY_CUT = True` enables it for anyone whose documents are less uniform.

---

## Final result

Full five-tool run, 48 PDFs / 298 questions, identical embedder and retriever,
only the chunker varying (`reports/v4_report.md`):

| Rank | Tool | MRR | NDCG@5 | Recall@5 | Hit@1 | Avg words/chunk | Context words |
|---|---|---|---|---|---|---|---|
| 1 | **docstruct** | **0.7457** | **0.7708** | **0.8859** | **0.6409** | 355.2 | 2346 |
| 2 | pymupdf4llm | 0.6941 | 0.7160 | 0.8356 | 0.6107 | 455.2 | 2576 |
| 3 | unstructured | 0.6508 | 0.6766 | 0.7886 | 0.5638 | 85.2 | 549 |
| 4 | langchain | 0.6493 | 0.6884 | 0.8221 | 0.5336 | 102.1 | 524 |
| 5 | docling | 0.5652 | 0.5814 | 0.6577 | 0.4966 | 114.2 | 674 |

DocStruct went from **0.6890 → 0.7457** and from second place to first on every
quality metric, while returning **less** retrieved context per query than the tool
it displaced. pymupdf4llm scored 0.6941 here against 0.6915 in the previous run, so
the two runs are comparable and the gap is real movement rather than a changed
measurement.

Where the +0.0567 came from:

| Change | MRR | Δ |
|---|---|---|
| baseline at HEAD | 0.6890 | — |
| chunk-boundary floor + headers in bodies | 0.7319 | **+0.0429** |
| font-scaled word-gap tolerance | 0.7457 | **+0.0138** |
| whitespace-blind relevance | 0.7457 | 0.0000 |
| recursive XY-cut | 0.7356 | −0.0101 (off) |

### Things I am not claiming

- **DocStruct does not win MRR/1k words.** unstructured (1.19) and langchain (1.24)
  lead that column while ranking 3rd and 4th, because they retrieve very little
  text. It is a tradeoff axis, not a ranking, and it is in the report precisely so
  that "make chunks bigger" cannot be used as an unbounded exploit — including by me.
- **`Chunk s` is not a fair speed column** in this run. Only the DocStruct adapter
  uses `--cache-dir`, so its 2.16 s is cache-hit time against four tools measured
  cold. Disclaimed in the report rather than quietly left in the table.
- **The gold is LLM-generated and imperfect.** 7 of 9 spans DocStruct "missed" in
  the Stage 0 audit are not in the raw PDF text at all. This penalises every tool
  identically so the ranking holds, but it caps the absolute numbers.
- **The corpus is arXiv-heavy.** Two-column born-digital papers. The XY-cut result
  is the clearest evidence that corpus shape matters: the principled algorithm lost
  to the heuristic precisely because the heuristic's assumptions hold here.

### Left undone

- §7.4 standalone figures produce no chunk — a figure with no caption is dropped.
  Skipped: a metadata-only chunk adds index noise for content nobody queries, and
  the benchmark cannot show a gain either way.
- §7.5 confidence-formula calibration and §7.11 numbering-based header levels —
  neither feeds retrieval today, so neither is measurable on the current benchmark.
- Regenerating the gold Q&A from correctly-spaced text. The right fix for the
  spacing mismatch; the whitespace-blind relevance rule is the cheap guard.

---

## Stage 7 — Shipping: hygiene, honest statistics, and a corpus that isn't all arXiv

Goal for this pass: make the repository something that can be handed to a
stranger, and make the headline claim survive being questioned.

### 7.1 What was actually redundant

`data/doclaynet/` — 200 page images and an annotations file, 84MB, tracked in git
and referenced by **nothing** in the codebase (verified by grep across every `.py`
and `.md`). Alongside it: `build/`, `*.egg-info/`, `output/`, two stray logs, and a
tracked personal working document. All untracked, and `.gitignore` widened so they
cannot return.

Two things the old `.gitignore` got wrong and had hidden:

- It ignored `.claude` and `CLaude.md` wholesale, which would have silently
  swallowed a tracked `CLAUDE.md`. Narrowed to `.claude/`.
- Its own comment claimed "final GT JSON is tracked" for `data/annotations/`.
  It wasn't — the two hand-corrected detection ground-truth files had never been
  added. They are the only detection gold that exists. Now tracked.

`data/doclaynet/` stays in git *history*; excising it needs a rewrite that would
break every existing clone, which is not a trade worth making for a repository
this size.

### 7.2 A memory/ folder, because the log had outgrown its job

`notes.md` is chronological and `implementation_plan.md` is an audit; neither
answers "what is the current state of X?" without reading all of it. Every session
was re-deriving the same context.

`CLAUDE.md` is now a router — the contract, six hard rules, and a table pointing at
the right file. `memory/` holds the distilled state: architecture, pipeline stages,
evaluation design, current results, conventions, roadmap, and **decisions**.

`decisions.md` is the one that earns its place. XY-cut, containment suppression,
breadcrumb injection and `MIN_CHUNK_TOKENS=600` were each built or seriously
considered, each measured, and each rejected — three of them because they *lost*,
one because it won for the wrong reason. Without that written down, they get
re-proposed, and the second implementation costs as much as the first.

### 7.3 The eval LLM client had never actually worked against GROQ

Extending the corpus needed gold for 47 new documents, and turned up three
separate faults in code that had only ever been exercised against one provider:

- **GROQ was unusable.** Its Cloudflare front end rejects urllib's default
  `Python-urllib/3.x` agent with error 1010 before the request reaches the API.
  The provider was in `_PROVIDERS`, documented in the module docstring, and had
  never served a single request. One header fixed it.
- **Retries could not clear a rate limit.** GROQ reports per-minute token limits
  as HTTP **413**, not 429, so keying off the status code is not enough — the
  `Retry-After` header has to be honoured whenever it appears. Fixed 1.5s steps
  never survive a 60-second window.
- **Long documents lost their second half.** The splitter cut once, into halves,
  against a fixed 60k-word budget. Anything past that was simply not sampled. Now
  split into as many even segments as the budget requires, with the question
  budget spread across them.

Then two failures that were subtler, and mattered more:

**Providers charge the *reserved* completion budget against the per-minute
limit**, not the tokens actually generated. With `max_tokens` unset, that
reservation is the model's full completion length — which alone can exceed the
limit, producing a request that can never succeed however long it waits. That is
what "failed after 6 attempts" meant. Capped at 1500 tokens, plus a 20s pace
before each request so segments stop colliding with each other by construction.

**Weak generators produce spans that quietly break the benchmark.** Switching
generator produced answer spans like `"DanceOPD"` and `"plain velocity MSE loss"`.
A two-word span is contained by almost every chunk that mentions the topic, so it
scores every tool alike — it does not measure retrieval, it dilutes it. Prompting
against it is not enough; the floor is enforced at validation
(`QA_MIN_SPAN_WORDS = 6`). The existing 298-question set already satisfies it
(min 7 words, mean 12.8), so nothing changed retroactively.

### 7.4 Confidence intervals, and the mistake they are there to prevent

The leaderboard was a column of point estimates over 298 questions. It invited
exactly the question it could not answer: is +0.05 MRR real, or resampling noise?
Every claim in the README rested on that.

`eval/stats.py` adds two things. **Percentile bootstrap CIs** per metric, computed
from the per-question scores — no normality assumption, which matters because
per-question reciprocal rank is a spike at 0, a spike at 1, and a handful of
discrete values in between. And a **paired bootstrap** of every tool against
DocStruct.

The pairing is the entire point. All tools answer the *same* questions, so one
resample of question indices is applied to both sides, cancelling the
between-question variance — which is much larger than the between-tool variance.
Two tools can have heavily overlapping marginal CIs and still differ on nearly
every question; "the CIs overlap, so it isn't significant" is the standard way to
get that backwards. The report says so, next to the table.

`benchmark_tool` now retains every metric per question rather than only reciprocal
rank, because a per-question vector cannot be recovered from an average after the
fact. Alignment is keyed on `(doc, question)`, not position: a tool that errors on
a document leaves a hole, and positional pairing would then compare different
questions and report the result with a straight face.

Everything is seeded. A significance number that moves between runs of the same
data is not evidence, and determinism is the contract this project sells.

Verified on a deliberately underpowered 3-document / 16-question subset: DocStruct
+0.0135 MRR over langchain, CI [-0.205, 0.227], p = 0.90, reported as **not
significant**. That is the feature working.

### 7.5 The question the benchmark could not ask

DocStruct's central design claim is that two independent detectors, fused
deterministically, beat either alone. The benchmark had no way to test it:
`run_pipeline` could not disable a detector, so "what is the vision model actually
worth?" — the first thing anyone asks about a two-detector design — was
unanswerable with the code as written.

`pipeline_mode` (`geometry-only` / `model-only`) plus `docstruct_geo` and
`docstruct_model` adapters, kept out of the default tool list because they answer
a different question from the cross-tool leaderboard.

The trap was in the cache. `BlockCache`'s key was (PDF, weights, layout config),
so a geometry-only run *with weights present* hashed identically to the hybrid run
and would have served the other's blocks — an ablation that measured nothing and
looked like it worked. The mode is now part of the key. Tested by pointing
geometry-only at a weights path that cannot load: if the model were reached it
would raise rather than pass.

### 7.6 Cross-boundary overlap: implemented long ago, never measured, and it loses

`OVERLAP_ON_BOUNDARY` had been in `config.py` since the chunk-floor work, off by
default, with no measurement behind that default — the plan (§7.3) argued for it
on intuition: the first chunk of a new section loses the last sentence of the one
before it. Intuition is not a result, so it was run.

| | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks | Context words |
|---|---|---|---|---|---|---|
| off (`04_xtolerance`, current default) | **0.7457** | **0.7708** | **0.8859** | 0.6409 | 3070 | 2346 |
| on (`08_overlap_on_boundary`) | 0.7432 | 0.7658 | 0.8826 | 0.6409 | 3156 | 2354 |

Worse on every metric that moved, Hit@1 identical, and it costs 86 extra chunks
and slightly more retrieved context to get there.

The explanation is the floor. `MIN_CHUNK_TOKENS` already means most structural
boundaries are *crossed* rather than cut, so the case the overlap was designed for
— a section opening stranded without its preceding context — mostly does not
arise any more. What the overlap does instead is duplicate text into a second
chunk that then competes with its own source for the same query. Two chunks
containing the answer do not rank better than one; they split the evidence.

Stays off, now with a number attached to that decision rather than a hunch.

### 7.7 Corpus expansion: the code is ready, the quota was not

The plan was 95 documents. The dataset fetcher had already pulled 47 more PDFs
(doc51-97) across seven domains; only gold was missing. Getting there surfaced,
and fixed, five separate faults in the gold generator — GROQ never actually worked
(Cloudflare rejected urllib's user agent), retries could not clear a rate limit,
long documents lost everything past the halfway point, dense documents lost
everything full stop, and long documents drew all their questions from their own
introduction.

All five are fixed and tested. Then the free tier's **daily** budget ran out:
100,000 tokens per day for `llama-3.3-70b-versatile`, which is roughly seven
papers. The remaining models share buckets or fail differently — `groq/compound`
routes to the same exhausted model, and `gpt-oss-120b` spends its completion
budget on hidden reasoning and returns truncated JSON unless given a much larger
one, which then breaks its 8k/minute limit.

So the corpus grew from 48 documents / 298 questions to **55 / 322**, not 95, and
the rest is a quota problem rather than an engineering one. Recorded plainly
rather than quietly reported as if 95 had been the target all along.

What this does mean: the gold is now generated by two models — the original 298
questions by `gpt-oss:120b`, the 24 new ones by `llama-3.3-70b-versatile`. Both
are validated verbatim against raw document text, both are tool-agnostic, and
every tool answers every question, so the tool-vs-tool comparison is unaffected.
Span-length distributions match closely (mean 12.8 vs 12.7 words), which is the
property that would actually distort scoring if it drifted.

### 7.8 The finale: 92 docs, 558 questions, and a null result that wasn't

The corpus finally reached 92 documents / 558 questions once the column-aware
reference extraction landed and gold generation moved wholesale to `gpt-oss:120b`
on Ollama (the better generator, and the one that wrote the original 298).

Full run, five tools (`reports/v6_report.md`), hybrid retriever, top-5:

| Tool | MRR | 95% CI | Hit@1 | Coverage |
|---|---|---|---|---|
| **docstruct** | **0.8203** | [0.794, 0.846] | **0.7401** | 0.817 |
| docstruct_geo | 0.7760 | [0.747, 0.804] | 0.6756 | 0.822 |
| pymupdf4llm | 0.7646 | [0.736, 0.793] | 0.6577 | 0.768 |
| langchain | 0.7009 | [0.669, 0.734] | 0.5986 | 1.000 |
| unstructured | 0.6948 | [0.662, 0.727] | 0.5920 | 0.833 |

DocStruct's lead over every external tool is significant on MRR/NDCG/Hit@1
(p 0.0008 → 0.0001). That part is the expected result at more than double the
question count of v4.

**The part that matters is a correction of my own earlier claim.** In Stage 7.5 I
added the single-detector ablation and, on the v5 gold, it said the vision model
was worth +0.0092 MRR at p = 0.64 — not significant. I wrote that up as a genuine
finding: "the expensive half of this architecture doesn't pay for itself." It was
wrong, and it was wrong for a reason worth recording.

The v5 gold was built on `page.extract_text()`, which sorts words by (top, x) and
so welds the left and right column of every two-column line into one string. Gold
drawn from those lines is unquotable, so it was silently rejected — and the
rejected questions were disproportionately the ones about dense two-column body
text, which is exactly where a vision model beats a geometry heuristic. The broken
measurement was suppressing the very signal the ablation was meant to detect.

With column-aware reference text (§7.7's fix, extended here to a proper gutter
detector) the same ablation on v6 gives **+0.0443 MRR, 95% CI [0.016, 0.073],
p = 0.0026, and Hit@1 +0.0645, p = 0.003.** The vision model pays for itself, and
significantly. Five times the effect the broken run showed.

Two lessons, both already baked into the tooling rather than just noted:

1. The ablation path and the paired test did their job twice over — first by
   letting the question be asked at all, then by making the reversal legible
   instead of a silent number change. A leaderboard of point estimates would have
   shown +0.009 then +0.044 and given no reason to trust either.
2. A retrieval benchmark is only as good as its gold, and gold generated from
   scrambled text fails *toward* a specific conclusion, not at random. The
   coverage metric (§ "extraction fidelity") and the column-aware reference both
   exist now so this class of error is visible rather than load-bearing.

Also dropped docling from the default tool set. It runs 10× slower than anything
else, OOM-crashes on some pages, and is invariably last; keeping it in the default
loop cost most of every run's wall time to reconfirm a fixed result. Still there
under `--tools docling`.

### What I am not claiming, v6 edition

- **DocStruct does not win extraction coverage.** langchain keeps 100% of the
  document's words by splitting raw text and dropping nothing; DocStruct keeps
  81.7% and duplicates the most (2.06×, inline headers plus separate table
  chunks). It wins retrieval, which is a different and — for RAG — more useful
  thing, but the coverage table says so plainly.
- **The corpus is still arXiv-heavy.** 92 born-digital two-column papers. The
  seven-domain fetcher exists; those documents are not yet scored.
- **The gold generator changed between v4 and v6** (gpt-oss:120b throughout now,
  vs a v5 interim that mixed two models). v4/v5 numbers are not comparable to v6
  and are marked superseded in `memory/results.md`.

## Stage 8 — PyPI-release hardening (Fable review, batch 1)

Fable reviewed the whole pipeline (`fable_suggestions.md`). Its list is ~40 items
across nine sections; most of the structural ones are marked **[MEASURE]** and are
worthless until they clear `scripts/ablate.py` on a corpus that is not all arXiv —
so they are staged, not landed here. This stage is the subset that changes *no chunk
output at all*: packaging and the library-facing API. None of it can move a
benchmark number, so none of it needed one.

- **§1.1 version single-source.** `__init__.py` hard-coded `0.3.0` while
  `pyproject.toml` said `0.4.0`. Now read from installed metadata
  (`importlib.metadata.version`) with a dev fallback; `test_errors.py` asserts the
  runtime version equals the `pyproject` version so they can never drift again.
  (Editable installs must be reinstalled on a version bump for the metadata to
  refresh — did that.)
- **§1.2 `model` extra missing pymupdf.** `ModelDetector` imports `fitz` to
  rasterize pages, but the extra only pulled `ultralytics`. Added `pymupdf>=1.24`.
- **§1.3 typed exceptions.** New `docstruct/errors.py`:
  `DocStructError` → `InvalidPDFError` / `EncryptedPDFError` / `EmptyDocumentError`,
  plus one `open_pdf()` context manager that every `pdfplumber.open` site
  (geometry detector, both extractors) now routes through. It translates the ragged
  pdfminer surface — including the wrapped `PdfminerException` whose real cause hides
  in `__context__` with an empty message — into these three. Callers no longer need
  to import pdfminer internals to catch a bad file. Tested against zero-byte,
  corrupt, and AES-256-encrypted fixtures.
- **§1.4 scanned-PDF diagnostic.** A wordless image-only PDF used to return an empty
  `Document` with no explanation. `run_pipeline` now sets
  `diagnostics["likely_scanned"]` when a majority of pages yield no extractable text
  and logs a warning pointing at `ocrmypdf`. No OCR in the pipeline — that stays a
  documented pre-processing step (contract).
- **§1.5 logging hygiene.** `NullHandler` on the `docstruct` logger so importing the
  library never spams a host app that has not configured logging.
- **§1.6 `py.typed`.** Added the marker + `package-data` so the annotations in
  `schema.py` are actually visible to downstream type checkers.
- **§1.7 input flexibility (partial).** `parse()` now takes `str | Path` and a
  `password=` that threads through `run_pipeline` → detector/extractors →
  `pdfplumber.open`. Left `pages=` for later: it needs page-index remapping through
  the detectors, extractors and cache keys, which is real work for a speculative
  feature.
- **§8 re-exports.** `run_pipeline` and `PipelineResult` are the documented
  diagnostics surface; they are now in `__all__`, alongside the error classes.

Deferred deliberately: §1.8 `ParseConfig` (its own mechanical branch), §1.9 CI, and
every §2–§5 correctness/structural item — those are behind config flags and must be
ablated before being kept on, which is blocked on the same corpus grind as Stage 7.

Full suite: 149 passed (143 + 6 new in `test_errors.py`), no regressions.

## Stage 9 — Fable review, batches 2–4 (bugs, gated features, hardening)

Continued from Stage 8. Everything here either changes no default output (config-
gated, default off) or is a deterministic correctness fix covered by unit tests, so
none of it needed a benchmark to land — and the ones that *would* move a benchmark
number are gated off precisely because they haven't cleared `scripts/ablate.py` yet.
The contract stays intact: default `docstruct.parse` output is byte-identical except
for the two deterministic bug fixes below, and the golden test pins that.

**Deterministic bug fixes (default on):**
- `_cluster_graphics` merges to a fixed point — figure regions were order-dependent
  and non-transitive (two clusters that only grew into overlap never merged).
- Proposal matching is confidence-ordered — a low-conf model box could steal the
  geometry box a higher-conf box needed.

**Config-gated, default off — each carries its `[MEASURE]` justification in
`config.py` and must clear `scripts/ablate.py` against `memory/results.md` before its
flag is flipped on:**
- Text quality: `DEDUPE_CHARS`, `DEHYPHENATE`, `NORMALIZE_TEXT` (NFKC + soft hyphen).
- Figures: `FIGURE_OVERLAP_BY_AREA` (density-independent text-overlap semantics).
- Reading order: `MULTI_COLUMN` (k-column), `BAND_SPLIT` (band-then-column — the
  middle path between the legacy splitter and the measured-worse XY-cut).
- Furniture: `STRIP_PAGE_FURNITURE` (cross-page repeated header/footer removal).
- Tables: `TABLE_TEXT_STRATEGY_FALLBACK` (borderless), `TABLE_SERIALIZATION`
  (keyvalue), `TABLE_SPLIT_ROWS`, `TABLE_SETTINGS`.
- Hierarchy: `HEADER_RANK_BY_WEIGHT` (bold as a depth signal, `is_bold` on Block).
- Containment: `LABEL_AWARE_CONTAINMENT` (the safe text-in-table subset only).
- `KEEP_REFERENCES` (emit reference chunks, excluded from indexing by default).

**Landed on by default (no scored-content change):** appendix/Roman section
numbering; the `FIGURE_CLUSTER_MAX_PRIMITIVES` cap; open-the-PDF-once perf.

**Hardening:** golden determinism tripwire (chunk hash of doc11), malformed-PDF
fuzz corpus (zero-byte / corrupt / truncated → typed error or empty Document, never
a crash or hang), GitHub Actions CI, seeded CHANGELOG.

**Deferred deliberately, with reasons:**
- §1.8 `ParseConfig` — a frozen per-parse config threaded through every module. The
  review itself calls it "its own branch, purely mechanical"; it is a sweep across
  every `config.*` read and does not belong bundled with feature work. Not started.
- §3.4 multi-page table merge, §4.3 `SectionPath` depth > 3 — build when the corpus
  needs them; designed-around, not built.
- §5.3 confidence calibration, §6.1 corpus broadening, §6.2 structure-targeted gold
  — data tasks blocked on annotation tooling and LLM quota (same blocker as Stage 7),
  not code.
- §8 PyPI README / mkdocs site, `on_page` progress callback — doc/UX work, not
  behaviour.

Full suite: 184 passed.

## Stage 10 — Measurement session: cache bug, ablation sweep, corpus broadening

The Fable batches (Stages 8–9) left ~14 features implemented but unmeasured. This
stage is about turning them into measurements, and it opened with a bug that would
have poisoned every one of those measurements.

### The ablation cache was config-blind for the new flags

`BlockCache` fingerprints config so a layout change invalidates the entry — but the
fingerprint (`_LAYOUT_CONFIG_KEYS`) predated the new flags, and the geometry proposal
cache was never config-aware at all. So an ablation toggling `DEDUPE_CHARS`,
`MULTI_COLUMN`, `STRIP_PAGE_FURNITURE`, etc. would have reused **baseline** blocks and
reported a false null on every gated feature. Caught before running the sweep.

Fix (`a1aa345`): moved `layout_config_fingerprint` + `_LAYOUT_CONFIG_KEYS` into
`cache/pdf_cache.py`, registered every new block-affecting flag, and made the geometry
proposal cache key on the fingerprint (`config_aware = True`). The model (YOLO) cache
is deliberately kept config-independent (`config_aware = False`) so the expensive
inference is reused across ablations — the whole reason the sweep is affordable.
Verified: flipping a flag changes the block + geo keys, model key stable. Chunking-only
flags (`TABLE_SPLIT_ROWS`, `KEEP_REFERENCES`) stay out of the fingerprint by design.

Lesson, now in `decisions.md`/`evaluation.md`/`architecture.md`: a config-fingerprinted
cache is only as correct as its key list; any new block-affecting flag MUST be
registered or its ablation is a false null.

### The sweep

`scripts/_sweep.sh`: baseline + 13 gated flags, each `scripts/ablate.py` on the 92-doc
/558-q v6 gold against the warm `.bench_cache` (YOLO reused, geometry+populate+chunk
recomputed per fingerprint). Smoke-tested first — 3 docs with cache 26 s, without cache
271 s (YOLO on CPU is ~90 s/doc), which is exactly why the config-aware-cache fix
matters. Running in background; winners get flipped to default-on with numbers in
`results.md`.

### Corpus broadening (roadmap #1) — unblocked

Ollama cloud (`gpt-oss:120b`, `OLLAMA_API_KEY`) verified live — quota is no longer the
blocker it was in Stage 7. `scripts/fetch_dataset_v2.py` across the six non-arXiv
domains added ~20 docs (95 → 115); several sources rate-limited or 404'd, so it
under-delivered the ~150 planned. `gen-qa` (Ollama) smoke-tested on doc100 — a Berkshire
financial report — produced clean tool-agnostic gold from raw text (real spans, real
financial-domain questions). Now generating gold for all 23 new docs into
`data/qa/benchmark_qa_v7_extra.json`; a full multi-tool re-baseline on the broadened
corpus follows once it lands.

### Still queued (gated on the running jobs)

Sweep → flip winners. gen-qa → merge gold + re-baseline. Then the remaining code work:
§1.8 `ParseConfig`, §3.4 multi-page table merge, §4.3 SectionPath depth > 3 — none can
touch core modules while the sweep's per-run subprocesses are re-importing from disk.

### Sweep outcome — environment-blocked

The sweep and the broadened-gold run were both **killed mid-flight** by the
environment (long unattended jobs do not survive here — the baseline ran ~69 min,
then the kill landed during run 2; gen-qa reached 4/23 docs). Measured facts:
one `ablate.py` run is ~69 min on 92 docs (embedding ~7,100 chunks dominates, YOLO
cached), so the 14-run sweep is ~16 h — not viable with hourly kills and a 10-min
foreground cap. A 15-doc arXiv subset runs in ~3.3 min and reproduces a high baseline
(MRR 0.910), but arXiv is exactly the corpus where most gated flags are no-ops
(`DEDUPE_CHARS` measured byte-identical). The full baseline reproduced the headline
(MRR 0.8194 ≈ 0.8203), which validates the harness and the config-aware-cache fix.

**Decision:** no flag is flipped on this evidence. All gated features stay OFF (the
honest default), `scripts/_sweep.sh` is committed for a capable machine, and effort
moves to the measurement-independent code items (§1.8, §3.4, §4.3). See `results.md`.

## Stage 11 — Measurement-independent code items (§3.4, §1.8, §4.3)

With the sweep environment-blocked, the remaining Fable items that need no benchmark
were built. No background jobs running, so core code was safe to touch.

### §3.4 Multi-page table merge (gated, off)

`merge_multipage_tables()` joins the last table on page N with the first on page N+1
when their column counts match and their x-extents align within
`MULTIPAGE_TABLE_X_TOLERANCE`, dropping a repeated header row. Runs after table
population, before furniture/containment/chunking. Registered in the cache
fingerprint (it changes block output). `MERGE_MULTIPAGE_TABLES = False`.
**Ceiling (ponytail comment in code):** 2-page merges only — a 3+-page table joins
its first two pages and leaves the rest, upgrade to a running merge if such tables
show up in the broadened corpus.

### §1.8 ParseConfig — pragmatic, not the full threaded refactor

Fable's §1.8 asks for a frozen config object threaded through every function for
thread-safety and non-mutating overrides. The full version is a sweep across every
`config.*` read — high risk for a benefit (concurrent parses with *different* configs)
that a deterministic research library rarely needs. Built the pragmatic version that
delivers the two real asks with **zero call-site rewrites**:

`config.override(**values)` — a context manager that sets module globals under a
`threading.Lock` and restores them in a `finally`, validating unknown keys.
`parse(config={...})` / `run_pipeline(config=...)` apply it for the call only, then
re-enter with `config=None` so the existing body reads the overridden globals (and the
cache fingerprint, which reads the same globals, keys correctly). No permanent global
mutation; concurrent overridden parses serialize on the lock instead of racing.
**Ceiling:** overridden parses serialize rather than running fully parallel — the
accepted trade for not rewriting the pipeline. The full frozen-dataclass version
remains a future branch if true parallel-with-different-configs is ever needed.

### §4.3 SectionPath depth > 3 — deferred (YAGNI)

Reworking `SectionPath` from `h1/h2/h3` fields to a property-backed `levels` list
changes `dataclasses.asdict` output — it would break the chunk JSON format and the
golden test — for a capability the arXiv corpus never exercises (headings rarely go
past 3 levels). Fable itself says "do this when the corpus needs it, not before."
Deferred; new code already keeps the numbering-depth clamp at `HEADER_LEVELS` so the
cap does not spread.

### §8 on_page progress callback

`parse(on_page=fn)` / `run_pipeline(on_page=fn)` calls `fn(page_index, total)` per
page during the fusion loop, for progress reporting on long documents. Threads
through the config-override re-entry too.

## Stage 12 — Literature sweep, metric audit, paper draft (2026-08-05)

No pipeline code changed. This stage is about where the project stands against
published work, and what has to be true before any of it can be submitted.

**Literature sweep.** Seeded from `pdf_parsing_papers.html`, extended by search.
Four adjacent literatures, only one of which is our competition: parser fidelity
(OmniDocBench, READoc, Docling/MinerU/Marker), layout datasets (DocLayNet,
PubLayNet, DocBank, PubTables-1M), chunking evaluation on plain text (Chroma TR,
LumberChunker, the 2025–26 systematic studies), and end-to-end parse×chunk×RAG —
where DocStruct actually lives. Nearest published neighbours: El Bachyr et al.
(ICSE-SEIP 2026, `arXiv:2604.12047`) and OHR-Bench (ICCV 2025). Written up in
`memory/related-work.md`, including a table of the axes where competitors beat us.

**The gap we can own.** Nobody evaluates a *deterministic layout-aware* chunker
against generic splitters with the embedder and retriever held fixed on public
PDFs. El Bachyr et al. vary six parsers × six chunkers × four retrievers on
FinanceBench and have **no structure-aware chunker in the grid**; the chunking
literature runs on plain text with no layout at all. Also unreported anywhere:
coverage/duplication (what the chunker silently drops) and determinism.

**Metric audit** (`memory/metrics-justification.md`). The retrieval layer is fully
standard and matches the closest competitor one-for-one (MRR, nDCG@k, Recall@k,
Hit@1 — which should be *renamed* Precision@1, the field's name for it). The cost
layer is homemade: `MRR/1k context words` is our independent reinvention of
Chroma's token-level **IoU**, and should be demoted in favour of the citable
version plus **Precision_Ω** (precision under an oracle retriever — it isolates the
chunker from the retriever, exactly our confound). Coverage/duplication stay, framed
as the chunker analogue of READoc's vocabulary F1. Our paired bootstrap is a genuine
methodological edge: `arXiv:2604.12047` reports point estimates only.

**Dataset migration** (`memory/benchmark-datasets.md`). Our gold is LLM-authored,
not public, and arXiv-heavy — three separate attack surfaces. FinanceBench is the
fix: 150 human-annotated questions over 84 born-digital SEC filings, CC-BY-NC-4.0,
verbatim evidence text + page number, table-heavy, non-arXiv, and it is the corpus
our closest competitor used. `scripts/fetch_financebench.py` (stdlib only) fetches
the PDFs and rewrites the gold into our QA schema; **smoke-tested on 2 documents:
7 gold rows, span chars 173–2874, mean 1186**.

That mean is the catch and the one real design consequence: FinanceBench evidence
is a *page region*, not a sentence span, so `relevance.py`'s containment rule
cannot score it — a 500-word chunk will never contain a 6k-char region. The
benchmark needs a `--relevance page|span|token` switch before the full run: page
for comparability with `arXiv:2604.12047`, token for comparability with Chroma.

**Paper draft.** `paper/main.tex` + `paper/refs.bib`, plain `article` class (venue
undecided). Carries the v6 numbers, the ablation table, the negative results
(XY-cut lost; the v5→v6 gold artefact that reversed the vision-model conclusion),
and an explicit limitations section listing every axis where we lose. `\todo{}`
marks the open work; `refs.bib` has a header naming the entries whose author lists
are still unverified — do not submit on those.

**Environment note:** the project `.venv` is absent from the working directory;
only the system Python is on PATH. Recreate before any test or benchmark run.

## Stage 13 — FinanceBench on CPU: what the smoke runs found (2026-08-06)

Environment rebuilt first: the project `.venv`, `weights/yolov8m-doclaynet.pt`
(52 MB, `hantian/yolo-doclaynet`) and the whole of `data/raw-pdfs/` had not
survived the directory move. Venv and weights are back; **the internal corpus is
still missing** and needs `scripts/fetch_dataset_v2.py` before v6 can be reproduced.

**`--relevance span|region` (`relevance.py`, `benchmark.py`, `cli.py`).** Public
human-annotated gold marks a *block*, not a sentence. The assumption going in was
that containment would score ~0 on FinanceBench for everyone; that was wrong, and
the truth is worse. Measured over all 189 evidence regions (median 167 words), the
share each tool is structurally too small to contain: pymupdf4llm 3%, docstruct
11%, langchain 68%, unstructured 74%. Containment fails *in proportion to how small
a tool chunks*, so the default rule would have handed DocStruct a large unearned
win on a leaderboard that looked entirely plausible. `region` mode scores by
Szymkiewicz–Simpson overlap coefficient (normalise by the smaller set), so
containment either way scores 1.0.

The first implementation of that mode was a **no-op** — it reused span mode's own
token-overlap ratio with a stricter threshold. That ratio is capped by the very
size mismatch it is meant to tolerate, so no threshold could fix it; the unit test
caught it. `RELEVANCE_REGION_MIN_OVERLAP = 0.7` remains a guess, marked unvalidated.

**All five adapters verified on a 15-page arXiv PDF.** `unstructured` needed
`pip install unstructured-inference` — `unstructured[pdf]` does not pull it in this
resolution and `partition_pdf` imports it at module load even under
`strategy="fast"`. **Not yet added to `pyproject.toml`'s `benchmark-heavy` extra**;
the Colab run will hit the same failure until it is. Measured chunk sizes
(unstructured 101 w, langchain 99 w, pymupdf4llm 404 w) track the corpus means used
in the fairness calculation above, which is a useful independent check on it.

**The headline finding.** On `3M_2018_10K` (160 pages), hybrid vs geometry-only:

| | chunks | mean | table chunks | time |
|---|---|---|---|---|
| docstruct | 384 | 397 w | **122** | 367.7 s (2.30 s/page) |
| docstruct_geo | 169 | 529 w | **4** | 168.5 s (1.05 s/page) |

**30× more tables.** SEC tables are borderless and `find_tables()` is ruled-line
based. arXiv papers rule their tables, so an arXiv-only evaluation has been
systematically under-selling the vision detector: on arXiv it is worth +0.044 MRR,
here it is the difference between representing the document and not. This is the
strongest single argument for the two-detector design the project has produced, and
it arrived from changing the corpus, not the code — which is the same lesson the
XY-cut result taught.

Also corrects the stale `~90 s/doc` YOLO figure in `measurement-environment.md`:
measured 2.30 s/page (~35 s on a 15-page paper) with ultralytics 8.4.115.

## Stage 14 — OHR-Bench investigated; it is the corpus (2026-08-06)

The internal scraper under-delivered twice (40, then 68 of a planned ~150) with 82
dead sources, and it carries a real bug: it dedupes against the committed manifest
rather than the disk, so a wiped `data/raw-pdfs/` can never rebuild. Rather than
keep repairing it, went back to the dataset survey and actually downloaded the
candidate instead of reading its abstract.

**OHR-Bench is much more than the survey credited.** Three artefacts: 1,261 PDFs,
a v1 parquet carrying 5,039 QA, a v2 parquet carrying human-verified `gt_text` for
all 8,561 pages, and a retrieval bundle that includes MinerU and Qwen2.5-VL parses
of the same pages. QA lives **only** in v1; v2 has no `qas` column.

Filtered to what we can actually use — has QA, multi-page domain, born-digital —
it is **95 documents, 3,787 pages, 3,558 QA across law (60), manual (15), finance
(10), academic (10)**, and **all 95 are born-digital**. Law and manual are exactly
the domains the scraper has never managed to deliver.

The property that decides it: **evidence is span-level, median 25 words**, only 1.3%
larger than LangChain's mean chunk. So `--relevance span` is fair here, with none of
the size bias that made `region` mode mandatory on FinanceBench — and there are 24×
more questions. `evidence_source` is typed (text 2,666 / table 847 / equation 45),
so the 24% table slice measures table handling directly for the first time.

Corpus set is now OHR-Bench (primary), FinanceBench (head-to-head with
`arXiv:2604.12047`), internal arXiv (ablations). Three provenances, which is more
than either closest competitor reports.

Also this session: `openai` provider added for gold generation, defaulting to
gpt-4.1 rather than gpt-5 because that family accepts only temperature 1 and gold is
generated at 0 for reproducibility. On a like-for-like single document gpt-4.1 kept
2 of 6 pairs against gpt-oss:120b's 3 of 6 — **no quality argument for switching
generators**, only a speed one.

## Stage 15 — page relevance, seven baselines, and a finding about relevance itself (2026-08-06)

`--relevance page` shipped (see Stage 14 for why OHR-Bench needs it). Adapters now
emit the pages each chunk drew from; `_pages_of()` normalises the fact that
Unstructured and Docling count pages from 1 while everything else counts from 0, and
LangChain/LlamaIndex recover page spans by character offset because they concatenate
before splitting. The benchmark **aborts** if an adapter reports no pages rather than
scoring it zero — a silent zero reads as a result.

**The finding, from the 3-doc page-mode smoke (not a result, but a real signal):**

| tool | MRR | chunks | mean |
|---|---|---|---|
| unstructured | **0.817** | 397 | 84 w |
| docstruct | 0.598 | 154 | 290 w |
| pymupdf4llm | 0.595 | 66 | 484 w |
| llamaindex | 0.572 | 140 | 99 w |
| langchain | 0.544 | 251 | 57 w |
| docstruct_geo | 0.343 | 96 | 333 w |

The prediction going in was that page mode would favour `pymupdf4llm`, which emits
one chunk per page. Wrong: **it favours small chunks.** Unstructured wins by 0.22 MRR
with the smallest chunks in the field — more, tighter chunks each carry a clean page
label and match a specific question better, so the top-5 lands on the evidence page
more often.

So the generalisation is bigger than "one relevance rule per corpus":

> **No relevance rule is neutral with respect to chunk size.** `span` rewards large
> chunks, `page` rewards small ones, `region` was built to be size-tolerant and is
> the only one with a defensible claim to neutrality — and its threshold is still
> unvalidated.

That is a methodological contribution in its own right, and it means no single mode
can carry a claim. Report all three and show whether the ranking survives; if
DocStruct wins only under `span`, that is uncomfortable and has to be said.

Also visible: docstruct 0.598 vs docstruct_geo 0.343, i.e. the vision detector worth
**+0.255 MRR** here against +0.044 on arXiv — consistent with the 122-vs-4 table
result from Stage 13.

**Baselines now seven** (docstruct, docstruct_geo, langchain, pymupdf4llm,
unstructured, llamaindex, llamaindex_semantic) plus docling. `llamaindex_semantic`
is the semantic-chunking baseline the comparison was missing, and it is handed
`config.EMBEDDING_MODEL` so the contest stays about where boundaries land rather than
who embeds better. **docling has still never run**: locally it dies with
`InvalidCxxCompiler` (no MSVC) and `std::bad_alloc`, both Windows/CPU problems, so
Colab is its first real test.

Two process failures worth recording. A LangChain adapter edit **silently failed to
apply** and was committed with a message claiming it worked; the page-mode guard
caught it at runtime, which is exactly why the guard exists. And the golden
determinism test was pinned to a *filename* — the corpus rebuild reassigns
`docN.pdf` sequentially, so it had silently repointed at a different paper and its
failure read like a chunking regression. It now pins its input by sha256.

---

## Stage 16 — pymupdf4llm was also OCRing born-digital pages (2026-08-07)

The same defect as docling in Stage 15's fix, in a different tool, found by reading
the OHR-Bench Colab log rather than by any test. `pymupdf4llm.to_markdown` routes
through its layout path, which defaults to `use_ocr=OCRMode.SELECT_KEEP_OLD`: an
ONNX classifier (`ocr_decision_model.onnx`, probability threshold 0.93) scores every
page, and flagged pages are re-derived by RapidOCR. Nothing in our adapter asked for
this.

Measured on `doc1.pdf` locally:

| | pages | chars | wall |
|---|---|---|---|
| `use_ocr=False` | 14 | 75,953 | 9.7 s |
| default | 14 | 72,425 | 13.1 s |

OCR was **35% slower and lost 3,528 characters**. It did not fill a gap; it replaced
a good text layer with a worse transcription. On Colab with RapidOCR the cost is far
worse — `AES_2022_10K` spent 270.6 s with 24 of 116 pages OCR'd, against 89.3 s for
langchain on the same document.

The audit script (below) shows why the classifier fires: the page it flagged in
doc1 already carried 3,182 characters with 3 replacement characters. It is tuned to
be eager, which is right for a general-purpose tool and wrong for this comparison.

**Only pymupdf4llm's numbers are affected.** The tools queued behind it in that run
share no code path — `unstructured` partitions with `strategy="fast"` (pdfminer text
layer), and both llamaindex variants read through pdfplumber. Adapters are
independent and the benchmark caches per tool, so pymupdf4llm re-runs alone.

### The verdicts are worth keeping, so they are now recorded

Disabling OCR throws away a signal that is useful precisely because it is not ours:
a tool-agnostic, per-page estimate of how extractable a page's text is.
`scripts/ocr_audit.py` runs the classifier alone — no OCR engine is invoked — and
writes `reports/ocr_audit.json` with per-page `needs_ocr`, `ocr_spans`,
`chars_total`, `chars_bad`, `img_area`, `txt_area`, plus a corpus-level
`flagged_frac`.

That last number is the point. The paper asserts "born-digital only" as a scope
limit; `flagged_frac` is the first thing in this repo that can put evidence behind
it, from a third-party model with no stake in the result. On the first three docs of
`data/raw-pdfs` it is 4/47 pages (0.085).

---

## Stage 17 — the relevance rule decides the winner (2026-08-11/12)

The OHR-Bench Colab session finished all three relevance modes. The three runs
share **identical chunks** — `n_chunks` and `mean_chunk_words` are equal across
modes, because chunking came from the warm cache and only scoring changed. So the
relevance rule is the sole variable, and the result is the cleanest experiment this
project has run.

| tool | `page` | `span` | `region` |
|---|---|---|---|
| docstruct | 0.6004 (6th) | **0.7059 (1st)** | **0.6657 (1st)** |
| docstruct_geo | 0.4703 (7th) | 0.7047 | 0.6567 |
| pymupdf4llm | 0.6684 | 0.6992 | 0.6040 |
| unstructured | **0.7950 (1st)** | 0.6539 | 0.6006 |
| langchain | 0.7562 | 0.6406 | 0.6029 |
| llamaindex | 0.7294 | 0.6483 | 0.5887 |
| llamaindex_semantic | 0.6515 | 0.6540 | 0.5749 |

**First becomes fifth and sixth becomes first, on identical chunks.** Stated
precisely: under `region` DocStruct beats all five external tools significantly;
under `span` it beats four of five (`pymupdf4llm` +0.0067, p=0.23, inside the
noise); under `page` it loses to four of five. Full write-up, including why each
rule is size-biased and how to report it, is now `memory/relevance-modes.md`.

The premise that sent us to `page` in the first place was wrong. "Only 1.5% of
OHR-Bench spans appear verbatim in raw PDF text" compared spans to the corpus's
normalised `gt_text`, not to what `is_relevant` actually applies.
`scripts/gold_reachability.py` measures the real rule: **80.2% span-reachable**
(text 95.5%, table 35.7%, equation 8.9%). `span` was always fair here.

**The uncomfortable finding.** `docstruct` vs `docstruct_geo` is +0.0012 under
`span` (p=0.80) and +0.0090 under `region` (p=0.12) — the model detector, the GPU
requirement, the largest cost in the pipeline, buys **nothing measurable on the two
modes we win**. Its +0.1305 under `page` is an artefact: geometry-only emits 5,810
chunks against hybrid's 9,080 and page mode rewards chunk count. Under `region`
geometry-only is actually ahead on table questions (0.3868 vs 0.3655). Recorded in
`decisions.md` as open, not as a verdict — FinanceBench is the corpus built to test
it and has not run.

**Slices** (`scripts/slice_results.py`, joining `per_question` to gold on
`(source_doc, question)`, no re-run needed): academic is our **weakest** domain in
every mode (`span` 0.4526 against unstructured's 0.5151, 5th of 7) — we win overall
by winning law and manual. Table MRR is 0.19–0.31 for every tool, which reads as a
ceiling effect given 35.7% table reachability rather than as everyone being bad at
tables. Back matter costs us 0.084 MRR from first fifth to last, against
unstructured's 0.003 — real, and far smaller than any leaderboard gap.

The 2026-08-11 `page` run reproduces the 2026-08-07 one tool-for-tool across MRR,
NDCG, Recall, Hit@1 and chunk counts, on a different machine. The older root-level
`ohr_report.md` / `ohr_results.json` were strictly redundant and are deleted.

### Also this session

`gold_reachability.py` learned to detect its own circularity. Run on FinanceBench
it returns span 100.0% / region 98.4% / median overlap 1.000 — not an easy corpus
but a question that answers itself, since both rules normalise by the gold and
FinanceBench gold **is** a page region (measured: 69% of its page). It now reports
the gold's median share of its page and refuses to let those rows stand as evidence
above 20%; OHR-Bench measures 9%. One row survives: only **28.0%** of FinanceBench
evidence appears verbatim in pdfplumber's text for its own annotated page.

FinanceBench is now fetched in full — 84 PDFs, 189 evidence rows — after adding
backoff for GitHub raw tearing the socket down eight files into the run.

Corpus breadth beyond arXiv moved to Europe PMC (`scripts/fetch_pmc.py`) after
OpenAlex's anonymous pool throttled to a standstill (25 min wall, 0.2 s CPU, all of
it sleeping in a 429 backoff). PMC serves a rendered PDF *and* `fullTextXML` — the
publisher's JATS, carrying the real section hierarchy, verified on the smoke at
19–24 nested titled sections plus `table-wrap` and `fig` elements per article. That
is gold which predates the benchmark, and the only route to scoring section paths
as a metric rather than a claim. IEEE Access is not reachable this way, so IEEEtran
two-column remains a gap.

---

## Stage 18 — an 18-page paper takes 8 minutes (2026-08-12)

Found by accident: two section-scorer runs made zero progress because each spent
its whole life inside one document. `natcomm__PMC12855844.pdf`, **18 pages**, takes
**475 s** geometry-only. The next document takes 14.3 s.

The document carries **422,340 vector primitives** against a typical 5,763, with
**134,164 on one page** — a Nature Communications figure page. Stage breakdown:

| stage | time |
|---|---|
| `detect` | 362.5 s |
| `fuse` + reading order | 0.0 s |
| `populate_text` | 271.4 s |
| `populate_tables` | 1.7 s |

**Two separate costs, and one of them is pure waste.** `detect_page` collects
`images + curves + rects + lines`, finds the count over `FIGURE_CLUSTER_MAX_PRIMITIVES`
(5,000), and discards the lot. The cap does its job — it exists to stop the O(n²)
fixed-point clustering hanging — but it fires *after* pdfplumber has materialised
the objects, measured at **53.1 s for one page**. We pay full price to learn we
should have skipped.

The second is `populate_text`: 216 blocks × `page.crop()`, and pdfplumber's crop
filters every object on the page, so each block pays against 134k objects.

**A measurement caveat worth keeping.** The first timing pass said pdfplumber was
fast (`extract_words` 1.0 s, `find_tables` 1.9 s) and pointed the finger at our
code. Wrong: those numbers were taken after a `len(page.curves)` line had already
warmed pdfplumber's lazy object cache. Cold, the same `detect_page` is 57.7 s of
which 53.1 s is the parse. Any per-stage timing on pdfplumber has to run on a
freshly opened page or it attributes the parse to whoever touched it first.

Candidate fixes, neither applied — both are parse-path changes and want their own
measurement plus an ablation before they land:

1. Decide to skip figure clustering from a cheap proxy (content-stream size) *before*
   touching `page.curves`, instead of after.
2. `populate_text` could crop from a page filtered to chars once per page, so 216
   crops do not each filter 134k vector objects.

Neither is on the retrieval path's critical list, but both are on `parse()`, so a
user with a figure-heavy paper pays this today. The arXiv corpus never contained a
document like this — the same blind spot corpus broadening exists to close, showing
up as a performance bug rather than a quality one.

---

## Stage 19 — FinanceBench is not a retrieval benchmark for us (2026-08-13)

The Colab smoke returned `MRR=0.0` for both tools on two documents, 7 questions.
Not a bug, and not the chunkers.

**The scoring rule is fine.** A relevant chunk exists: best region overlap against
langchain's chunks is 0.990 and 1.000, well over the 0.7 threshold.

**The retrieval never reaches it.** Rank of the first relevant chunk, hybrid RRF
over dense + BM25, five questions across `3M_2018_10K` (729 chunks) and
`3M_2022_10K` (1,176 chunks):

| embedder | ranks | recall@5 | recall@50 | recall@100 | MRR@5 |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 81, 83, 163, 239, 299, 299, 299 | 0/5 | 0/5 | 2/5 | 0.0000 |
| BAAI/bge-small-en-v1.5 | 63, 146, 173, 173, 173, 181, 222 | 0/5 | 0/5 | 1/5 | 0.0000 |
| intfloat/e5-small-v2 | 42, 62, 128, 128, 128, 175, 252 | 0/5 | 1/5 | 2/5 | 0.0000 |

Chunking was done once and every embedder scored on the identical chunk set, so
the embedder is the only variable. A stronger one moves the best rank from 81 to
42 and changes nothing: **top-5 is unreachable under all three.**

Why this corpus behaves unlike OHR-Bench, whose questions quote distinctive
phrases: FinanceBench questions are analyst prompts ("Assume that you are a public
equities analyst...") with almost no lexical overlap with the evidence; the
evidence is a numeric financial table, which small dense encoders embed poorly;
and a 10-K yields 729-1,176 chunks, so the target is one in a thousand.

**Decision: FinanceBench is not run as a retrieval leaderboard.** Every tool would
score ~0, which discriminates nothing -- and the whole point of holding the
embedder fixed is to isolate chunking. Reporting a table of zeros would be worse
than reporting nothing, because it reads as a chunking result.

It keeps three uses, all of which it has already delivered or can:

1. **Parse fidelity** — only 28.0% of its evidence appears verbatim in pdfplumber's
   text for its own annotated page (Stage 17), measured before any tool runs.
2. **Borderless-table detection** — 122 detected tables against pdfplumber's 4 on
   `3M_2018_10K`. This is the corpus where the model detector should pay for itself,
   and that can be measured without a retrieval leaderboard.
3. **A methodological point worth stating**: `arXiv:2604.12047` reports MRR
   0.700-0.844 on this corpus. Our protocol reaches 0.0 on the same PDFs, which
   makes the "not comparable -- different retrievers, page-level gold" caveat in
   `related-work.md` concrete rather than hedged.

The notebook gates the run behind `RUN_FINANCEBENCH = False` rather than deleting
it, so the measurement can be repeated if the retrieval side ever changes.

---

## Stage 20 -- section-boundary agreement: the first metric we win outright (2026-08-13)

The Colab section run came back (`docstruct_results (1).zip`, now unpacked into
`reports/`). Seven tools, PMC papers, scored against the publisher's own JATS section
boundaries. Pk and WindowDiff are *error* rates -- lower is better.

| Tool | WindowDiff | Pk | Straddle | Mean chunks | Docs | Errors | Seconds |
|---|---|---|---|---|---|---|---|
| docstruct_geo **(ours)** | **0.4362** | **0.3525** | 0.527 | 25.5 | 24 | 0 | 132.5 |
| pymupdf4llm | 0.4928 | 0.4661 | 0.6005 | 15.9 | 24 | 0 | 253.0 |
| docstruct **(ours)** | 0.4934 | 0.3641 | 0.4484 | 35.4 | 24 | 0 | 312.0 |
| llamaindex_semantic | 0.5334 | 0.5134 | 0.2496 | 24.7 | 24 | 0 | 273.3 |
| llamaindex | 0.6959 | 0.5938 | 0.3828 | 37.9 | 24 | 0 | 115.0 |
| langchain | 0.8821 | 0.6183 | 0.2227 | 73.9 | 24 | 0 | 117.7 |
| unstructured | 0.8839 | 0.5974 | 0.1924 | 92.4 | 18 | 6 | 91.3 |

**What is new here.** Every external number so far has been retrieval, which means it
is entangled with an embedder and a relevance rule -- and Stage 17 showed the ranking
inverts when that rule changes. This metric touches neither. It compares boundaries to
boundaries against gold a publisher wrote, for a purpose that has nothing to do with
this benchmark. It is the most tool-agnostic gold in the project.

**We are 1st on both metrics, and the two of ours disagree in a readable way.**
`docstruct_geo` (geometry only, no model detector) wins WindowDiff; hybrid `docstruct`
wins nothing but is close on Pk and is 2.4x slower (312 s vs 132 s). That is the third
corpus in a row where the model detector does not pay for itself -- consistent with
OHR-Bench's +0.0012 span / +0.0090 region. Three corpora agreeing is no longer a
null result that can be waved off as corpus-specific.

**Chunk count is the confound, and it cuts against the naive reading.** langchain and
unstructured emit 74 and 92 chunks against a gold that averages ~21 sections, so they
are penalised hard by WindowDiff (which counts boundaries per window) while scoring a
*low* straddle rate -- of course, since tiny chunks rarely span a boundary. Pk forgives
over-segmentation and their Pk is still 0.59-0.62. Both metrics have to be read
together or the table lies in one direction or the other; this is the same size-neutrality
problem as the relevance modes, in a new coat.

**Straddle rate is not an error term.** 57.4% of gold sections are shorter than
`MIN_CHUNK_TOKENS`, so merging them is the design working. It bounds how meaningful a
per-chunk section *label* can be, nothing more. Our 0.527 against pymupdf4llm's 0.60 is
not a win to claim.

**The caveat that keeps this off the headline table today: 24 documents, not 126.**
The Colab session had only 24 PMC PDFs on disk when `score_sections.py` ran, so the
scores cover a subset. `reports/section_reachability.json` (126 docs, 3,144 sections,
84.5% body ceiling) is the full-corpus ceiling; the 24-doc ceiling is 86.9% and now
sits in `reports/section_reachability_colab24.json`. The two files are not the same
population. `notebooks/pmc_sections_colab.ipynb` re-runs the table on the whole corpus
and asserts the count before spending the GPU time.

**FinanceBench still has not run as a retrieval leaderboard**, and Stage 19 is why. The
zip's name is misleading -- it contains no FinanceBench output at all.

**Why it was 24 documents, found 2026-08-15.** Not NCBI throttling. `fetch_pmc.py` built
its "already have" set from `reports/pmc_manifest.json`, which is committed -- so a fresh
Colab clone claimed to own 133 papers it did not have a byte of, took the `fname in have`
branch on every search hit, counted each one toward the per-journal quota *without
downloading*, and exited reporting success having fetched nothing. The only PDFs present
were the handful cached on Drive from earlier sessions.

The script's own docstring already promised the right behaviour -- "re-running skips what
is already **on disk**" -- so this was a bug against its documented contract, not a design
choice. `prune_to_disk()` now drops manifest entries whose PDF *or* XML is missing before
the set is built; disk is the only authority. Five tests, and a fresh-clone smoke against
the live API: 133-entry manifest, empty `data/pmc/`, one paper requested, one PDF + one XML
downloaded, manifest rewritten to 1 entry.

**This is the second time this exact class has bitten** -- `fetch_dataset_v2.py` dedupes
the internal arXiv corpus against its committed manifest the same way, which is why 27 of
that gold's documents are missing on disk and will not re-download (to-do item 7). A
committed manifest is a record of what was once fetched, never evidence of what is present.
Both failures were silent and both produced a smaller corpus that looked like a complete
run.

---

## Stage 21 -- both GPU jobs landed, and neither result moved (2026-08-16)

One Colab session, ~4 h, `notebooks/pmc_sections_colab.ipynb`. Two open measurement
questions, both now closed. The headline is that **nothing changed**, which is the
outcome that was worth paying for.

### 1. Section boundaries: 24 documents became 134, and the table held

The `fetch_pmc.py` manifest fix (Stage 20 addendum) did what it was supposed to --
the corpus went from 29 paired documents to the full set.

| Tool | WindowDiff | Pk | Straddle | Mean chunks | Docs | Errors |
|---|---|---|---|---|---|---|
| docstruct_geo **(ours)** | **0.4226** | **0.3418** | 0.5129 | 26.8 | 134 | 0 |
| pymupdf4llm | 0.4800 | 0.4490 | 0.5734 | 17.7 | 134 | 0 |
| docstruct **(ours)** | 0.4818 | 0.3531 | 0.4385 | 37.5 | 134 | 0 |
| llamaindex_semantic | 0.5337 | 0.5128 | 0.1889 | 29.1 | 134 | 0 |
| llamaindex | 0.6952 | 0.5979 | 0.3660 | 42.7 | 134 | 0 |
| langchain | 0.8787 | 0.6200 | 0.2202 | 85.6 | 134 | 0 |
| unstructured | 0.8933 | 0.6025 | 0.1820 | 106.9 | **99** | **35** |

Against the 24-document run: order identical, every value within 0.02. WindowDiff
0.4362 -> 0.4226, Pk 0.3525 -> 0.3418. A 5.6x larger corpus that reproduces the small
one is the cheapest kind of evidence that the metric is not noise, and it is the reason
this table can now go in the paper as a headline rather than an appendix curiosity.

Reachability: 138 documents with gold, 3,381 sections, 84.7% body ceiling -- and 134 of
those 138 scored, the four dropped by the <50%-locatable rule. **The ceiling and the
score finally describe the same population**, which was the specific defect of the last
run (24 scored against a 126-document ceiling file). `section_reachability_colab24.json`
is deleted; it existed only to keep those two populations apart.

**One number in that table is not on the same population as the rest.** unstructured
errored on 35 of 134 documents (26%) and its row covers 99. The rate is unchanged from
the small run (6 of 24, 25%), so it is systematic rather than a session accident. Its
row needs that N in the caption, and the failure itself is worth a sentence -- a 26%
hard-failure rate on born-digital PMC PDFs is a result about unstructured, not a
footnote about our harness.

### 2. The region threshold: swept, and the ranking does not move

`RELEVANCE_REGION_MIN_OVERLAP = 0.7` was `# unvalidated` and one of the paper's two
headline wins rested on it. Swept on OHR-Bench (not FinanceBench -- Stage 19), 3,558
questions, offline re-scoring of one run's dumped overlaps so chunking is identical at
every threshold.

**The dumping run reproduces the cited 2026-08-11 leaderboard: max MRR drift 0.0002,
identical chunk counts for all seven tools.** That check was the precondition -- a dump
that disagreed with the published numbers would have been describing a different run,
and the sweep over it would have looked exactly as convincing while meaning nothing.

MRR by threshold:

| tool | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| docstruct | 0.9590 | 0.8803 | 0.7888 | **0.6659** | **0.5275** | **0.3890** |
| docstruct_geo | **0.9666** | **0.8926** | **0.7995** | 0.6567 | 0.5076 | 0.3704 |
| pymupdf4llm | 0.9455 | 0.8543 | 0.7468 | 0.6040 | 0.4821 | 0.3263 |
| llamaindex_semantic | 0.9617 | 0.8661 | 0.7400 | 0.5747 | 0.4297 | 0.3098 |
| unstructured | 0.9365 | 0.8339 | 0.7213 | 0.6008 | 0.3911 | 0.2127 |
| llamaindex | 0.9463 | 0.8407 | 0.7263 | 0.5885 | 0.4604 | 0.3401 |
| langchain | 0.9357 | 0.8137 | 0.7111 | 0.6031 | 0.4662 | 0.3370 |

**A DocStruct variant is 1st at all ten thresholds, and the two of them hold both top
places at all ten.** Margin over the best external tool: +0.045 to +0.062 across
0.4-1.0. The region result does not depend on the constant, which is the claim the
sweep existed to test, and `config.py` now carries the measurement instead of the
`# unvalidated` marker (hard rule 5).

**Read two things honestly before quoting this.**

First, **0.7 is where our margin peaks** (+0.0619). We did not choose it that way -- it
predates the sweep by months -- but a reviewer will check, so the paper says it rather
than waiting to be asked. The defence is that the margin is +0.045 or better everywhere
from 0.4 up, so the peak is a bump on a plateau, not a cliff we are standing on.

Second, **the low end is uninformative, not favourable**. At 0.0 every chunk counts as
relevant and MRR is 1.0 by definition, so the convergence at 0.1 (all seven tools inside
0.03) is that definition asserting itself. A metric climbing as the threshold falls is
not evidence for a low threshold.

**The field below us reorders constantly** -- llamaindex_semantic runs 2nd at 0.1 and
7th at 0.7; unstructured 4th at 0.6 and 7th at 1.0. So the *leaderboard* is
threshold-sensitive even though *our position* is not. Anyone reporting a region
ranking of the field owes that caveat; this is the relevance-mode lesson
(`relevance-modes.md`) one level down, and it now has its own measurement.

**Small rehabilitation of the model detector.** The variants cross over between 0.5 and
0.6: geometry-only wins the loose half, hybrid wins the strict half (+0.0092 at 0.7).
Still not significant, still tiny, and still against three corpora that found nothing --
but it is the first sign the detector does something at all under a strict rule, and it
is the direction a table-heavy corpus would predict.

### Where that leaves the paper

Compute is done. Every headline the draft needs is now measured, and the remaining work
is that `paper/main.tex` is dated 2026-08-05, mentions OHR-Bench once, and still lists
FinanceBench as the planned external corpus. Sections 4 and 5 are rewrites, not edits.

---

## Stage 22 -- building the README visual found two real defects (2026-08-16)

`scripts/make_readme_gif.py` walks one page through `run_pipeline()` and draws what
comes back. Drawing the actual output, rather than a diagram of the intended output,
surfaced two bugs that a green test suite and every leaderboard had missed.

### 1. Full-width blocks are extracted across the column gutter

On `data/raw-pdfs/doc1.pdf` page 0, a block at `y0=332, x0=64, width=494` -- the full
page width, over a two-column body -- carries the text:

```
bacyroresussminugltitphlee vexisiisotinnagnVdilTanagnudagLeL...
```

That is "abstract"/"scaling..." and its right-column neighbour interleaved character by
character. A block whose bbox spans both columns has its text populated without column
awareness, so extraction walks across the gutter and welds the two columns together.

This is **the exact failure the paper's introduction attributes to naive parsers**,
occurring inside DocStruct. It reaches the output: the affected block is classified
`header`, so the garbage becomes a `SectionPath` level, and 2 of 76 chunks on that
document carry a section path that fails a vowel-ratio sanity check.

Impact is narrow but the class is not: it fires wherever fusion emits a full-width
block over a multi-column region, which is exactly where a unilateral model detection
is most likely. It also partly explains why section *labels* are less trustworthy than
section *boundaries* -- and boundaries are what the PMC metric scores, which is why
that result is unaffected.

### 2. Full-width elements above columns are ordered after them

Same page, reading order: `1 Abstract, 2 body, 3 sidebar, 4 Introduction, 5, 6, then
7 = the paper title, 8 = the author block`. The title is at `y0=97`, above everything
else on the page, and comes out seventh.

The column-aware sweep assigns full-width elements to a column band and reads bands in
order, so anything spanning the page -- title, author block, a full-width figure --
sorts after the column it was assigned to instead of before both. Body pages are
unaffected (verified on page 1: left column top-to-bottom, then right, correctly), so
this is a title-page and section-break defect rather than a general one.

### Neither is fixed here

Repo rule 1: a change to reading order or extraction is worthless until it has been run
through `scripts/ablate.py` against the numbers in `memory/results.md`. Both are logged
in `to-do.md` with the reproduction. Recording them is also the honest option --
the README now names both defects rather than shipping a demo that hides them.

The demo uses page 1 (a body page) because that is the representative case, and says so.

---

## Stage 23 — determinism, finally measured (2026-08-16)

The contract's first line — *same PDF in, same chunks out* — had one unit test behind
it: `test_parse_is_deterministic`, one PDF, parsed twice, **in one process**. That
cannot observe anything that varies across a process boundary, which is where
non-determinism actually comes from: hash seeds, iteration order over addresses,
thread scheduling, a model's kernel selection.

`scripts/verify_determinism.py` parses every document in a **fresh subprocess** and
hashes the whole chunk structure — id, type, page, reading order, section path, and a
digest of the content. Two runs agree only if every chunk matches on all of them.

**Result, OHR-Bench, geometry-only, 2 runs:**

| | |
|---|---|
| documents scored | **95 of 95** |
| byte-identical across independent processes | **95 (100.0%)** |
| differing | **0** |
| chunks per run | 5,810 |
| wall time | 6,810 s for 92 docs + a 4h-cap re-run for the last 3 |

**This is now a table row no LLM- or embedding-based chunker can fill**, and it is
cheap to re-verify. It also closes the largest hole in the paper: a contract asserted
in the abstract with no experiment behind it.

**Three documents timed out at the 30-minute-per-parse cap**, all dense financial
filings: `JPMORGAN_2022Q2_10Q` (197 pages), `JPMORGAN_2023Q2_10Q` (217), and
`VERIZON_2021_10K` (120), against a corpus median of 18 pages. Page count is not the
whole story — the corpus's largest document, at 382 pages, parsed fine. It is table
density, which matches Stage 18: the cost is in materialising vector primitives, and
financial filings are wall-to-wall ruled tables.

Re-run with a 4-hour cap: **3/3 identical, 1,338 chunks**, so the corpus total is
**95/95 and 5,810 chunks**. And that total is an unplanned cross-check --- 5,810 is
exactly the `n_chunks` recorded for `docstruct_geo` in `reports/ohr_results_span.json`,
a run made days earlier on Colab through the benchmark harness rather than through
`parse()`. Two independent paths agreeing to the chunk is better evidence than either
alone. **The timeout is still reportable**: these are the same documents FinanceBench is made
of, so "DocStruct is slow on long financial filings" is now measured on two independent
corpora rather than asserted from one anecdote.

**What this does not cover, and the paper says so:** determinism holds *within* a
version, not across versions — a release that changes chunking changes boundaries, so
a persisted index needs a pinned version. And this run is geometry-only. The hybrid
path goes through CUDA, whose kernel selection is not guaranteed bit-reproducible, and
this machine has no GPU (`torch.cuda.is_available()` is False), so the hybrid claim
stays unverified rather than assumed. Geometry-only is pure Python and NumPy and has
no such caveat.

---

## Stage 24 — the internal corpus no longer matches its own gold (2026-08-16)

Trying to ablate the two extraction defects surfaced something worse than either.

`scripts/ablate.py` returned `MRR=0.0` on document after document. Not a bad
invocation: **the arXiv corpus on disk is a different set of documents from the one
`benchmark_qa_v6.json` was generated against.** `doc1.pdf`'s gold asks about air
pollution monitoring in Macau; `doc1.pdf` on disk is *ParVL: Parallel Scaling and
Expandable Compute Allocation for Multimodal LLMs*.

Measured across every document the gold and the disk share a filename with:

| | |
|---|---|
| documents whose gold is findable in the PDF | **0** |
| documents whose gold matches nothing | **65** |

Not partial drift. The filenames were reused for entirely different papers, almost
certainly by a re-fetch that renumbered `doc<N>.pdf` positionally instead of keying on
identity. `datasets/verify.py` passes 68/68 because the manifest was regenerated from
the *new* files, so manifest and disk agree with each other and neither agrees with the
gold.

**Consequences, stated plainly.**

1. **The internal-corpus table in the paper is not reproducible today.** MRR 0.8203 and
   the rest were valid when measured — they are consistent with each other and with the
   two independent reproductions recorded in `results.md` — but no one can re-run them
   against the corpus now on disk.
2. **Three ablation deltas the paper quotes are in the same position**: the chunk-floor
   +0.0429, XY-cut −0.0101, and the vision detector's +0.0443 (p=0.0026), which is the
   *only* corpus where the detector shows an effect.
3. **No ablation can run until this is repaired.** That blocks measuring `BAND_SPLIT`,
   which is the fix for the reading-order defect.

**What is unaffected:** every external number. OHR-Bench, the PMC section metric, the
threshold sweep and the determinism run all use corpora fetched by scripts that key on
document identity (`ohrbench_manifest.json` and `pmc_manifest.json` carry per-document
checksums that verify today). The paper's headline results do not depend on the broken
corpus.

**Repair.** `fetch_dataset_v2.py` must key on `arxiv_id`, not on positional filename,
and the gold must be regenerated or the original PDFs recovered by id from
`dataset_manifest_v2.json` — which still records `arxiv_id` per entry, so the original
document set is recoverable even though the files are not.

Until then the honest move is to say so in the paper rather than present internal
numbers as reproducible, and this is a second instance of the failure class from
Stage 20: a manifest treated as evidence about files it does not actually describe.

---

## Stage 25 — where the corpus went, traced through git (2026-08-16)

**The 65 documents were never committed.** `data/raw-pdfs/` has been in `.gitignore`
since `03a6bc9`, and the only PDFs ever tracked are four arXiv samples and three
fixtures from the initial commits. So the originals cannot be restored from git
directly. What *is* tracked is the manifest that describes them, and that is enough to
date the break and to recover most of the corpus.

**Timeline, from the manifests' own history:**

| date | commit | state |
|---|---|---|
| 2026-06-26 | `d82c25d` | v1 manifest, 24 entries, `doc25 = 2606.27377v1` |
| 2026-07-23 | `a0151b9` | **v6 gold committed** — 558 questions, 92 documents |
| 2026-07-23 | `00e9325` | v6 benchmark run, MRR 0.8203. Corpus and gold agreed here |
| 2026-07-23 | `03a6bc9` | manifest_v2, 47 entries |
| 2026-07-24 | `4cc88ab` | manifest_v2, 67 entries, `doc51 = 2606.27377v1` |
| **2026-08-06** | `dbbc979` | manifest_v2, 68 entries, **`doc1 = 2608.04010v1` (ParVL)** |

**The proof is in the arXiv identifiers.** All 31 ids in the corpus now on disk are
`2608.*` — posted in **August 2026**. The gold was committed on **23 July**. A July
corpus cannot contain August papers, so the swap is dated to between 2026-07-24 and
2026-08-06, and the current manifest is byte-identical to `dbbc979`'s for all 68
entries. The v6 benchmark ran on 23 July, before the swap, which is why its numbers
were valid and reproduced twice at the time.

**The same paper appears under two filenames across manifests** — `2606.27377v1` is
`doc25` in v1 and `doc51` in the July v2 — which is the renumbering bug visible in the
record rather than inferred: a re-fetch that had lost its dedupe key assigned new
positional names to papers already in the corpus.

**56 of 92 documents are recoverable, and verified.** `reports/corpus_recovery_map.json`
maps filename to arXiv id, built only from pre-August manifests and accepted only where
the manifest's title shares content words with that document's own gold questions --
so each entry is checked against the gold rather than assumed from filename order. The
remaining 36 appear in no committed manifest and are lost unless they exist in a
backup outside this repository.

**Consequence.** A partial re-fetch restores 56 documents with correct identity, which
is enough to re-run ablations on a 56-document subset and enough to state the internal
table's provenance honestly. It is not enough to reproduce the 92-document numbers
exactly, and the paper should not claim otherwise.
