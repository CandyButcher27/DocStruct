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
