# DocStruct work notes

Running log of what was changed, why, what it measured, and whether it was kept.
Newest stage at the bottom. Every stage ends in a commit.

Goal for this pass: **DocStruct should not lose the cross-tool retrieval benchmark
to any provider**, and the package should be installable and usable as
`pip install docstruct` → `import docstruct`.

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
