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
