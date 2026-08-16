# Future plans — what the next session does

> **SUPERSEDED 2026-08-16 by [`ROADMAP.md`](ROADMAP.md).** Kept for history. Its live
> items were folded into that file after being re-checked against the code; several
> were already done. Its §4 MinerU suggestion and §5 open decisions survived the check.
> Its warning about `fetch_dataset_v2.py` deduping against the manifest rather than the
> disk turned out to be the exact mechanism that destroyed the internal corpus
> (`notes.md` Stage 24/25) — the bug was documented here and its consequence was not
> foreseen.



Written 2026-08-06 at the end of the infrastructure session. Read this first, then
`memory/benchmark-datasets.md` and `memory/evaluation.md`. `to-do.md` is the older
scratchpad; where they disagree, this file is newer.

**One-line state (as written 2026-08-06):** the harness is ready and nothing has been
measured. Every number produced on 2026-08-06 came from a 3-document smoke test and
**none of it is citable**. Tomorrow's job is to produce the first real leaderboards
on a GPU.

> **Update 2026-08-12 — §1 is done, §2 is not.** OHR-Bench ran on 2026-08-11 for all
> **three** relevance modes, seven tools (docling excluded by choice), 95 docs, 3,558
> questions: `reports/ohr_report_{page,span,region}.md`. The `--relevance page`
> command below is superseded — the premise that page was the only viable mode was
> wrong, and the ranking inverts between modes. Read `memory/relevance-modes.md`
> before using anything in §1. §2 (FinanceBench) is still accurate and still the next
> GPU job; its corpus is now fetched in full (84 PDFs / 189 rows).
>
> `to-do.md` was rewritten on 2026-08-12 and is newer than this file again.

---

## 0. Before anything: 15-minute local sanity pass

Do not skip this even though it "worked yesterday". Five failures in the last
session were invisible to a green test suite and only surfaced by running the real
CLI. The suite passing is not evidence the run will start.

```bash
.venv/Scripts/python.exe -m pytest -q            # expect 201 passed, ~13 skipped
python scripts/fetch_ohrbench.py --limit 3
python -m docstruct.cli benchmark --pdfs-dir data/ohrbench --qa data/qa/ohrbench.json \
  --weights weights/yolov8m-doclaynet.pt --tools docstruct,langchain --relevance page \
  --report-md /tmp/x.md --report-json /tmp/x.json
```

If both run, the environment is good. If `.venv` or `weights/` or `data/` are
missing again (they are gitignored and did not survive the last directory move):

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[all,benchmark-heavy]" pyarrow
curl -L -o weights/yolov8m-doclaynet.pt \
  https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8m-doclaynet.pt
```

---

## 1. The main event — OHR-Bench leaderboard on Colab

**Why this one first:** it is the only corpus with enough questions (3,558) and
domains (4) to support a headline claim, and it is public human gold, which is the
whole reason for the migration. FinanceBench is smaller and slower (15,000 pages
against 3,787).

```python
!git clone -b feat/paper-draft https://github.com/CandyButcher27/DocStruct
%cd DocStruct
!pip install -e ".[all,benchmark-heavy]" unstructured-inference pyarrow \
   llama-index-core llama-index-embeddings-huggingface
!mkdir -p weights && wget -P weights \
   https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8m-doclaynet.pt
!python scripts/fetch_ohrbench.py                      # ~1.6 GB, self-fetching
!python -m docstruct.cli benchmark \
   --pdfs-dir data/ohrbench --qa data/qa/ohrbench.json \
   --weights weights/yolov8m-doclaynet.pt \
   --tools docstruct,docstruct_geo,langchain,pymupdf4llm,unstructured,llamaindex,llamaindex_semantic,docling \
   --relevance page --cache-dir .bench_cache \
   --report-md reports/ohr_report.md --report-json reports/ohr_results.json
```

**Traps, in the order they will bite:**

1. **Mount Drive and put `.bench_cache` on it.** Free Colab reclaims sessions without
   warning. The benchmark checkpoints per tool per document and resumes, but only if
   the cache survives. A 90-minute job that dies at minute 85 with the cache on local
   disk loses everything.
2. **`unstructured-inference` must be installed explicitly.** `unstructured[pdf]`
   does not pull it, and `partition_pdf` imports it at module load even under
   `strategy="fast"`. It is now in `pyproject.toml`'s `benchmark-heavy`, but verify.
3. **`ultralytics` ships a top-level `tests` package** that shadows ours. `tests/__init__.py`
   fixes it in-repo; if pytest collection explodes on Colab, that is why.
4. **docling has never run successfully.** It failed locally with `InvalidCxxCompiler`
   (no MSVC) and `std::bad_alloc` — both Windows/CPU problems that Linux+GPU should
   not have. **Run it first on 2–3 documents before including it in the full run**, or
   it will consume the session. If it fails on Colab too, that is a reportable result
   about docling; drop it with `--tools` and say so.
5. **`llamaindex_semantic` embeds during chunking**, so it is much slower than the
   other splitters. Time it on 2 documents before trusting the full-run estimate.

**Expected cost:** 3,787 pages at ~2.30 s/page CPU for the DocStruct adapter alone;
on a T4 the whole 8-tool run should be well under 2 hours.

**Done looks like:** `reports/ohr_results.json` with 8 tools × 3,558 questions, CIs
and paired bootstrap populated, and `meta.config` filled in.

---

## 2. Then FinanceBench — different corpus, different relevance rule

```bash
!python scripts/fetch_financebench.py
!python -m docstruct.cli benchmark --pdfs-dir data/financebench \
   --qa data/qa/financebench.json --weights weights/yolov8m-doclaynet.pt \
   --relevance region --cache-dir .bench_cache_fb \
   --report-md reports/fb_report.md --report-json reports/fb_results.json
```

**`--relevance region` is mandatory here.** Under the default `span` rule, 74% of
FinanceBench evidence regions are structurally too large for Unstructured's chunks to
contain and only 11% for ours — the leaderboard would hand DocStruct a large unearned
win and look entirely plausible. See `memory/benchmark-datasets.md`.

**Sweep `RELEVANCE_REGION_MIN_OVERLAP` before quoting any FinanceBench number.** It is
0.7, chosen by guess, marked `# unvalidated`. Re-scoring uses cached retrievals, so
the sweep is cheap once the run exists.

**Cost warning:** 84 documents but ~15,000 pages (10-Ks run 160–250 pages each).
This is ~4× the OHR-Bench run. Budget a separate session.

---

## 3. The analysis that decides the next month of work

Do these on the JSON, no GPU needed.

**a. Slice OHR-Bench by `evidence_source`.** The gold carries `text` (2,666),
`table` (847), `equation` (45) per question. If DocStruct loses specifically on table
or equation questions, that is the measurement justifying an expanded label set — see
§5. If it does not, do not expand.

**b. Report all three relevance modes and check the ranking survives.** This is the
methodological finding from the smoke, and it is a paper subsection:

> No relevance rule is neutral with respect to chunk size. `span` rewards large
> chunks (containment is easier), `page` rewards small ones (more chunks, more page
> coverage — Unstructured beat everything at 0.817 MRR with 84-word chunks), and
> `region` was designed to be size-tolerant and is the only one with a defensible
> claim to neutrality.

If DocStruct wins only under `span`, that is uncomfortable and must be reported.

**c. Check the back-matter penalty.** DocStruct drops references by design. Under
`page` relevance, any question whose evidence page is in the back matter is
structurally unreachable for us and reachable for everyone else. On the smoke,
DocStruct's chunks stopped at page 11 of a 15-page paper while every other tool
reached 14. **Quantify what fraction of OHR-Bench questions this costs us** before
concluding anything from the page-mode numbers.

---

## 4. Still-missing baselines

| Baseline | State | Note |
|---|---|---|
| docling | never run | environment-blocked locally; first real test is Colab |
| **MinerU** | not started | OHR-Bench's `retrieval.zip` already contains MinerU's parse of the exact pages — read from disk, no inference, no GPU. Label the row "MinerU parse + page chunking", not "MinerU" |
| TableQuest | unverified | paper calls it public; no repo found. Verify before promising it |
| ClusterSemanticChunker | superseded | `llamaindex_semantic` covers the semantic-chunking argument and shares our embedder |

---

## 5. Open decisions (do not decide these without the numbers)

- **Expand the 5-label set?** The model already detects 11 DocLayNet classes; we
  collapse 5 into `text`. Only `formula` and `footnote` have a plausible chunking
  consequence — page furniture is already handled deterministically by
  `STRIP_PAGE_FURNITURE`, and title/section-header both feed the hierarchy builder.
  Decide from §3a. Note `Chunk.chunk_type` is public API and in the JSON format.
- **The gated-flag sweep** (14 flags, `scripts/_sweep.sh`) is still outstanding from
  two sessions ago. All 14 features are shipped and unit-tested but default-OFF
  because nothing measured them. Several plausibly matter more on 160-page filings
  than on arXiv papers.
- **Full v8 gold regeneration with OpenAI?** Provider is wired (`--provider openai`,
  defaults to gpt-4.1 because gpt-5 rejects temperature 0 and gold must be
  reproducible). On a like-for-like document gpt-4.1 kept 2 of 6 pairs against
  gpt-oss:120b's 3 of 6 — **no quality argument for switching**, only speed. Do it
  only as a deliberate one-shot, since it invalidates comparability with every v6
  number in the paper draft.

---

## 6. Paper work, once numbers exist

`paper/main.tex` carries `\todo{}` markers. The edit list lives in
`memory/paper-structure-survey.md` §5 — add explicit RQs, split limitations into
internal/external/construct validity, turn the corpus prose into a table, add an
artifact statement, rename Hit@1 → Precision@1.

**Verify `paper/refs.bib` before submission.** Several author lists are placeholders;
the file header names them.

---

## What not to do

- Do not add another dataset. Three corpora, six domains and ~4,270 questions already
  exceed both closest competitors. The gap is measurements, not data.
- Do not repair `scripts/fetch_dataset_v2.py`. It has under-delivered twice and 82 of
  its sources are dead. Treat the internal corpus as ablation-only at whatever size it
  is. (Known bug if revived: it dedupes against the committed manifest rather than the
  disk, so a wiped `data/raw-pdfs/` can never rebuild.)
- Do not quote any number from 2026-08-06. All of it is 3-document smoke output.
