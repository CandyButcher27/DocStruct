# DocStruct — consolidated roadmap

Written 2026-08-16. **This file supersedes `futureplans.md` and
`implementation_plan.md` §1–§7 as the work queue.** Both are kept for their history;
where they disagree with this file, this file is newer. `to-do.md` remains the short
"where we are" scratchpad, `memory/` remains the durable reference, and
`paper/IMPROVING-THIS-PAPER.md` remains the paper-specific plan.

Every item below was checked against the code on 2026-08-16, not carried forward on
faith. Several items in the older files were already done and are marked as such.

---

## 0. Done — do not redo

| Item | Evidence |
|---|---|
| OHR-Bench, three relevance modes, 7 tools | `reports/ohr_report_{page,span,region}.md` |
| Region threshold swept 0.1–1.0 | `reports/ohr_region_threshold_sweep.json` |
| Section boundaries vs publisher JATS, 134 papers | `reports/section_scores.md` |
| Determinism, 95/95 documents, 5,810 chunks | `reports/determinism.json` |
| Paper reframed around the relevance-rule finding, cut to 8pp | `paper/main.tex` |
| Bibliography verified against the arXiv API | `paper/REFERENCES_TO_VERIFY.md` |
| `pip install docstruct-rag` packaging + API | `docs/API.md`, `dist/` |
| Corpus break traced and 56 documents recovered | `notes.md` Stage 24/25 |

**Stale claims corrected while consolidating.** `implementation_plan.md` §10 says
`pipeline_mode` "never landed" and bootstrap CIs are "entirely unbuilt". Both are
wrong as of today: `pipeline_mode` has 15 references in `docstruct/`, and
`eval/stats.py` provides `bootstrap_ci` and `paired_bootstrap`, which the paper
already reports. Two other claims in that audit are still accurate and appear below.

---

## 1. Blocking / in flight

### 1.1 Internal corpus — recovered to 56 of 92
`scripts/recover_v6_corpus.py` re-fetches by arXiv id into `data/arxiv-v6/`, verified
against the gold. The 36 documents that appear in no committed manifest are lost
unless a backup exists outside the repo — **check any Colab Drive
`docstruct_bench/corpora/` folder before writing them off.**

Consequence for the paper: internal numbers are re-measured on the 56-document subset
and reported as such. The 92-document figures are not reproducible and are not
presented as though they were.

### 1.2 Re-run internal benchmark and ablations on the 56
The reason the corpus mattered. Unblocks `BAND_SPLIT`, which is the fix for the
reading-order defect.

---

## 2. CPU-only, no new data — do these next

### 2.1 `BAND_SPLIT` measurement, then flip the default
Already implemented and verified to fix the reading-order defect (a full-width title
sorts 1st instead of 7th). Default-OFF only because nothing measured it. This is now
a measurement task.

### 2.2 MinerU baseline — free, and the field expects it
OHR-Bench's `retrieval.zip` **already contains MinerU's parse of the exact pages**, so
this is a read from disk: no inference, no GPU, no new dependency. Label the row
"MinerU parse + page chunking" rather than "MinerU", because that is what it is.
Carried over from `futureplans.md` §4 and still accurate.

### 2.3 Remove or wire `suppress_contained`
`pipeline.py` imports `suppress_contained` and `suppress_table_contained` from
`fusion/containment.py` and **never calls either**. Still true today (1 reference: the
import). Either wire it behind a measured flag or delete the import — a dead import
that looks load-bearing is worse than neither.

### 2.4 Diagnose the worst documents rather than tuning blind
`implementation_plan.md` §7.2: the outlier documents sit far below the corpus range.
Dump their block/chunk output and find the real failure mode before touching general
heuristics. Re-identify the outliers on the recovered corpus first — the old document
numbers refer to papers that are no longer on disk.

### 2.5 Paper edit list (`memory/paper-structure-survey.md` §5)
Explicit research questions; limitations split into internal / external / construct
validity; corpus prose turned into a table; an artifact statement. All writing.

---

## 3. Needs a GPU

### 3.1 The 14-flag gated sweep
Every gated feature in `config.py` is shipped, unit-tested and default-OFF because
nothing measured it. `scripts/_sweep.sh`. Outstanding for three sessions.

### 3.2 Hybrid-path determinism
The determinism result covers geometry-only. The hybrid path runs a model through
CUDA, whose kernel selection is not guaranteed bit-reproducible. `verify_determinism.py
--weights ...` on a GPU box answers it. **Be prepared for a negative answer** — it
would qualify the claim to `docstruct_geo`, which the paper already says.

### 3.3 FinanceBench for borderless tables only
Not a retrieval leaderboard (`notes.md` Stage 19). It is still the one corpus where
the model detector could justify itself: 122 detected tables against pdfplumber's 4 on
`3M_2018_10K`. Needs a detection metric, not a retrieval one.

### 3.4 DocLayNet val split
The detection-layer mAP rests on two hand-annotated documents. Also the only cheap
route to calibrating the `# unvalidated` fusion constants.

---

## 4. Open decisions — do not settle these without numbers

- **Demote or defend the vision detector.** Null on three corpora, and *loses* to
  geometry-only on section boundaries at 2.4× the cost. Either give it §3.3 or cut it
  from the contributions and present DocStruct as a geometric chunker with an optional
  model path. The second is the stronger paper.
- **Expand the 5-label set?** The model detects 11 DocLayNet classes; we collapse 5
  into `text`. Only `formula` and `footnote` have a plausible chunking consequence.
  `Chunk.chunk_type` is public API, so this is a breaking change.
- **v8 gold regeneration?** Measured: gpt-4.1 kept 2 of 6 pairs against gpt-oss:120b's
  3 of 6. No quality argument, only speed — and it would invalidate comparability with
  every v6 number.

---

## 5. Known defects

| Defect | State |
|---|---|
| Full-width blocks extracted across the column gutter | **Open.** A gutter-splitting fix was written, found the correct gutter, and measured 14 garbled tokens before and 14 after. Reverted. Next question: why does the right-hand crop stay interleaved when the split point is correct? |
| Full-width elements sort after the columns | **Fixed in code** by `BAND_SPLIT`; needs §2.1 |
| `fetch_dataset_v2.py` dedupes against the manifest, not disk | **Known since `futureplans.md`**, and it is what destroyed the corpus. Do not revive that script without fixing it to key on `arxiv_id`. |
| 8 min to parse a figure-dense paper; 4h for a 200-page filing | Open. Sharpest practical limitation; two candidate fixes in `notes.md` Stage 18. |

---

## 6. Needs you, not me

1. **Publish to PyPI** — `twine upload dist/*` with your token. Everything else is done and validated.
2. **Affiliation and co-authors** in `paper/main.tex`.
3. **Send the paper to your professors** — it is submittable now.
4. **Hand `paper/REFERENCES_TO_VERIFY.md` to another model** — 7 entries have no arXiv id or DOI and were not machine-checkable.
5. **Check for a corpus backup** on Colab Drive for the 36 unrecovered documents.
