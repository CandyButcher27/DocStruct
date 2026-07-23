# Results — current numbers and what every config value is worth

**Update this file whenever a benchmark or ablation runs.** It is the reference
any "did this help?" question should be answered from.

## Current headline (`reports/v4_report.md`)

48 born-digital PDFs, 298 LLM-generated Q&A, identical embedder and retriever for
every tool, only the chunker varying. Hybrid retriever, top-5.

| Rank | Tool | MRR | NDCG@5 | Recall@5 | Hit@1 | Avg words/chunk | Context words |
|---|---|---|---|---|---|---|---|
| 1 | **docstruct** | **0.7457** | **0.7708** | **0.8859** | **0.6409** | 355.2 | 2346 |
| 2 | pymupdf4llm | 0.6941 | 0.7160 | 0.8356 | 0.6107 | 455.2 | 2576 |
| 3 | unstructured | 0.6508 | 0.6766 | 0.7886 | 0.5638 | 85.2 | 549 |
| 4 | langchain | 0.6493 | 0.6884 | 0.8221 | 0.5336 | 102.1 | 524 |
| 5 | docling | 0.5652 | 0.5814 | 0.6577 | 0.4966 | 114.2 | 674 |

DocStruct leads on every quality metric **while returning less retrieved context
per query than the tool it displaced**. pymupdf4llm scored 0.6941 here against
0.6915 in the previous run, so the two runs are comparable and the movement is
real rather than a changed measurement.

## Where the gain came from

| Change | MRR | Δ |
|---|---|---|
| baseline at HEAD (flush on every boundary) | 0.6890 | — |
| chunk-boundary floor + headers in chunk bodies | 0.7319 | **+0.0429** |
| font-scaled word-gap tolerance | 0.7457 | **+0.0138** |
| whitespace-blind relevance | 0.7457 | 0.0000 (guard, not a gain) |
| recursive XY-cut | 0.7356 | −0.0101 → **off** |

Starting point before this work: 0.6773, second place behind pymupdf4llm.

## Chunk-bounds sweep (`reports/ablations/`)

48 docs / 298 questions each, all else fixed, sorted by context cost.

| MIN/MAX | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks | Avg words | Context words | MRR/1k |
|---|---|---|---|---|---|---|---|---|
| baseline (no floor) | 0.6890 | 0.7199 | 0.8490 | 0.5872 | 3905 | 181.3 | — | — |
| 80 / 300 | 0.7022 | 0.7320 | 0.8658 | 0.5973 | 3440 | 218.7 | 1411 | **0.4978** |
| 120 / 400 | 0.7086 | 0.7431 | 0.8859 | 0.5940 | 2945 | 253.6 | 1686 | 0.4203 |
| **200 / 500 (chosen)** | **0.7319** | 0.7560 | 0.8826 | **0.6342** | 2519 | 294.7 | 2050 | 0.3571 |
| 120 / 800 | 0.7203 | 0.7513 | 0.8758 | 0.6141 | 2533 | 291.2 | 2170 | 0.3319 |
| 250 / 800 | 0.7257 | 0.7541 | 0.8792 | 0.6242 | 2174 | 339.6 | 2555 | 0.2841 |
| 400 / 800 | 0.7277 | 0.7612 | 0.8993 | 0.6174 | 1992 | 370.7 | 2873 | 0.2533 |
| 600 / 800 | 0.7584 | 0.7886 | 0.9128 | 0.6477 | 1832 | 403.6 | 3251 | 0.2333 |

Read the trap: **raw MRR rises monotonically with chunk size and MRR/1k falls
monotonically**. 600/800 has the best MRR on the page and is rejected — see
`decisions.md`. 200/500 sits on the Pareto front and strictly dominates the
250/800 it replaced (+0.006 MRR, +0.010 Hit@1, +0.003 recall, **20% less
context**).

**Useful alternate configuration:** if context budget matters more than rank,
80/300 delivers 0.7022 MRR — still above pymupdf4llm — at **43% of the context
cost**. The old flush-at-every-boundary code could not tell that story at all,
because it paid for tiny chunks *and* got the worst MRR.

## Reading order

| | MRR | NDCG@5 | Recall@5 | Hit@1 | Chunks |
|---|---|---|---|---|---|
| legacy column split (default) | **0.7457** | **0.7708** | 0.8859 | **0.6409** | 3070 |
| `06_xycut` | 0.7356 | 0.7666 | 0.8859 | 0.6275 | 3132 |
| `07_xycut_rowgap12` (4× row gap) | 0.7356 | 0.7666 | 0.8859 | 0.6275 | 3132 |

Recall identical, rank quality lower. The 4× row-gap run being byte-identical
localises the entire difference to the column cut.

## Report index

| Report | What it is |
|---|---|
| `reports/v4_report.md` | **Current headline.** Five tools, 48 docs, 298 Q. |
| `reports/rrf40_report.md` | Pre-work baseline: DocStruct 2nd at 0.6773. |
| `reports/docstruct_v3_report.md` | Table-plaintext-fix era. |
| `reports/baseline_report.md` | Original four-tool run. |
| `reports/ablations/*.json` | Single-variable runs, with `overrides` + full `config` in each file. |
| `reports/dataset_manifest_v2.json` | Provenance of the extended corpus (file, domain, source, sha256). |

## Corpus

`data/raw-pdfs/` is gitignored (rebuild with `scripts/fetch_dataset_v2.py`).
The corpus is **arXiv-heavy born-digital prose**, which is the most important
caveat on every number above — the XY-cut result is direct evidence that corpus
shape decides which algorithm wins. `fetch_dataset_v2.py` targets seven domains
(arxiv, legal, financial, medical, technical, govt, textbook) precisely to break
that homogeneity; the manifest records what has actually landed.
