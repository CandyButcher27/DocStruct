# CLAUDE.md — DocStruct

Guidance for Claude Code sessions working in this repository.

## What this is

DocStruct is a **local, deterministic, structure-aware PDF chunking library for RAG**.
It is *not* a RAG framework. The core contract is:

> No LLM calls in the pipeline. Same PDF in → same chunks out. Fully local, auditable.

`indexing/`, `query/` and `eval/` exist to *prove* the chunks are retrieval-good,
not as product surfaces. `docstruct run` (bare chunking) and `docstruct.parse()`
are the primary entry points.

Any proposal that puts a model call inside the pipeline violates the contract and
should be rejected, not implemented. See `memory/decisions.md`.

## Read this first

The `memory/` folder is the durable context for this project. Read the file that
matches the task before writing code:

| File | Read it when |
|---|---|
| [`memory/architecture.md`](memory/architecture.md) | You need the module map, data model, or where a responsibility lives |
| [`memory/pipeline.md`](memory/pipeline.md) | You are changing detection, fusion, reading order, extraction or chunking |
| [`memory/evaluation.md`](memory/evaluation.md) | You are touching `eval/`, running benchmarks, or adding a metric |
| [`memory/relevance-modes.md`](memory/relevance-modes.md) | **Before quoting any leaderboard number** — the OHR-Bench ranking inverts between relevance modes on identical chunks |
| [`memory/related-work.md`](memory/related-work.md) | You are writing the paper, positioning against competitors, or asked "has this been done?" |
| [`memory/paper-structure-survey.md`](memory/paper-structure-survey.md) | You are editing `paper/` — how comparable papers are *organised*, venue conventions, and the edit list for the draft |
| [`memory/benchmark-datasets.md`](memory/benchmark-datasets.md) | You need a public corpus, or are touching gold generation / the FinanceBench migration |
| [`memory/metrics-justification.md`](memory/metrics-justification.md) | You are adding, renaming or defending a metric — says which are standard and which are ours |
| [`memory/results.md`](memory/results.md) | You need to know what a config value is worth, or what the current numbers are |
| [`memory/decisions.md`](memory/decisions.md) | Before proposing anything — it lists what was already tried and rejected, with measurements |
| [`memory/roadmap.md`](memory/roadmap.md) | You are picking the next piece of work |
| [`memory/measurement-environment.md`](memory/measurement-environment.md) | You are running the sweep/benchmark or gen-qa — GPU needs, why jobs got killed, how to resume |
| [`memory/conventions.md`](memory/conventions.md) | Always — commit style, test policy, how to run things |

`notes.md` is the chronological engineering log (what changed, what it measured,
whether it was kept). `to-do.md` (repo root) is the short "where we are / what's next"
scratchpad. `implementation_plan.md` is the standing plan-vs-code audit. `memory/` is
the distilled, current-state version; when they disagree, `memory/` is the one that
was updated last.

## Hard rules for this repo

1. **Measure before claiming.** Any change to chunking, reading order or
   extraction is worthless until it has been run through `scripts/ablate.py` and
   compared against the current numbers in `memory/results.md`. "Should improve
   retrieval" is not a result.
2. **No LLM in the pipeline.** LLM use is confined to `eval/qa_generator.py`
   (gold generation) and is never on the parse path.
3. **All thresholds live in `config.py`.** No magic numbers in detector, fusion
   or chunking code. Values inherited from the v0 prototype are marked
   `# unvalidated` — do not silently trust them.
4. **Top-left coordinates everywhere** (`y0` = top, y increases downward),
   matching pdfplumber. Model output is transformed on the way in.
5. **Config changes carry their justification.** Every tuned constant in
   `config.py` has a comment naming the measurement that chose it. Keep it that
   way; a bare number is a regression waiting to happen.
6. **Every stage ends in a commit, and `notes.md` gets the entry.** The log is the
   product of this project as much as the code is.
7. **Every corpus needs its relevance rule checked before it is trusted.** Not the
   span *length* — the span *reachability*. Ask: is this gold findable in raw PDF
   text at all, and is it findable equally for a tool that chunks small? Both
   external corpora failed a naive assumption here, in opposite directions.
   **A single-mode leaderboard is not a result** — measured on OHR-Bench, the
   ranking inverts between modes on identical chunks (`memory/relevance-modes.md`).
8. **Gold must be tool-agnostic, and preferably not ours.** Never generate Q&A from
   the output of a tool being benchmarked. Prefer a public human-annotated corpus
   (FinanceBench first — see `memory/benchmark-datasets.md`) over LLM-generated gold
   for any headline number in the paper.
9. **Report the losses.** Coverage 0.9632 vs LangChain's 1.00 and duplication 1.83
   (both OHR-Bench; the old 0.817/2.06 came from the withdrawn internal table), no
   parse-fidelity number, born-digital only, **6th of 7 under page relevance**, and
   **the model detector is not significant on any corpus but arXiv** — +0.0012 span /
   +0.0090 region on OHR-Bench, and on PMC section boundaries geometry-only *beats*
   the hybrid at 2.4× the speed. Three corpora, no effect. Also: unstructured
   hard-fails on 26% of PMC PDFs so its section row covers 99 of 134 (say the N), and
   `RELEVANCE_REGION_MIN_OVERLAP = 0.7` is where our region margin happens to peak.
   These belong in the main table, not an appendix.
   `memory/related-work.md` keeps the list of who beats us where.

10. **The internal arXiv corpus does not match its own gold** (measured 2026-08-16,
    `notes.md` Stage 24): 0 of 65 shared filenames have findable gold, because a
    re-fetch reused `doc<N>.pdf` names for different papers. **Every ablation returns
    MRR=0.0 until this is repaired**, and `datasets/verify.py` will not catch it — the
    manifest was regenerated from the new files, so manifest and disk agree and neither
    agrees with the gold. Recover by `arxiv_id` from `dataset_manifest_v2.json`.
    A manifest is evidence about the files it was written from, and nothing else.

11. **Two known extraction defects are open and must not be forgotten**
    (`to-do.md` 6 and 7, `notes.md` Stage 22): a block spanning both columns has its
    text extracted across the gutter, interleaving the columns into an unreadable
    section heading; and full-width elements above a two-column body sort *after* the
    columns. Both were found by drawing the pipeline's real output, not by the test
    suite. Neither is fixed, because rule 1 applies to them like anything else.

## The paper

The draft lives in `paper/` and is **on the ACL 2023 template** as of 2026-08-28:
`main.tex` + `refs.bib` + `ACL2023.sty` + `acl_natbib.bst`, built with
`pdflatex → bibtex → pdflatex → pdflatex`. Title: *Relevance Rules Confound PDF
Chunker Evaluation*. Citations are author-year via `acl_natbib`, not numeric.

**The venue limit is 8 pages for the whole PDF, references included** — not 8 body
pages with the tail free. It lands at exactly 8, so every addition must be paid for
by a cut.

`.claude/skills/acl-paper/` is the contract for editing it: the ACL layout diff, a
claim ledger mapping every number in the draft to the report that produced it, and
`scripts/gate.py`, a runnable anti-slop gate. **Run the gate on every `.tex` edit and
report its counts.** `\todo{}` marks open work; `refs.bib` has a header listing the
entries whose author lists are still unverified. The three research memory files
above remain the source of truth for content — update them, then the draft.

## Running things

```bash
.venv/Scripts/python.exe -m pytest -q          # 220 tests (215+5 skipped), ~4 min
python -m docstruct.cli run data/raw-pdfs/doc1.pdf
python scripts/ablate.py --name try --set MIN_CHUNK_TOKENS=300

# corpora (all self-fetching, so they work on Colab too)
python scripts/fetch_ohrbench.py --limit 3     # primary external corpus
python scripts/fetch_financebench.py           # 84 PDFs / 189 rows, fetched
python scripts/fetch_pmc.py --per-journal 2    # papers + publisher JATS XML

# reading a run without re-running it
python scripts/gold_reachability.py --gold data/qa/ohrbench.json --pdfs-dir data/ohrbench
python scripts/slice_results.py --results reports/ohr_results_span.json \
    --out reports/ohr_slices_span.md          # by evidence source / domain / position
```

**Relevance mode is not optional — pick it per corpus, or the leaderboard lies.**

| Corpus | Mode | Because |
|---|---|---|
| internal (`benchmark_qa_v*`) | `span` | gold marks a verbatim sentence |
| FinanceBench | **n/a — do not run as a leaderboard** | measured 2026-08-13: evidence unreachable at top-5 under three embedders on identical chunks, so every tool scores MRR 0.0. That measures the retriever, not the chunker (`notes.md` Stage 19). Keep it for parse fidelity and borderless-table detection only |
| OHR-Bench | report **all three** — done 2026-08-11 | the "only 1.5% appear verbatim" figure compared spans to the normalised `gt_text`, not to `is_relevant`. Measured against the real rule: **80.2% span-reachable**. And the three modes disagree about who wins |
| PMC papers | **no relevance rule at all** | section-boundary agreement (Pk / WindowDiff) against publisher-authored JATS. No embedder, no rule, gold that predates the benchmark. 134 docs, we lead both metrics. `scripts/score_sections.py` |

Reachability is a *ceiling*, not a score: it is the same number for every tool, so it
says whether a rule can measure the corpus at all. Run `scripts/gold_reachability.py`
on any new corpus **before** its first leaderboard, and heed its circularity warning —
once the gold is a large share of its page (FinanceBench: 69%), span and region
reachability are ~100% by construction and prove nothing.

No rule is size-neutral: `span` rewards large chunks, `page` rewards small ones
(measured — unstructured wins `page` with the smallest chunks in the field),
`region` is the only one built to be size-tolerant and its threshold is still
`# unvalidated`. **Measured consequence: on OHR-Bench the ranking inverts between
modes on identical chunks** — DocStruct 1st under span and region, 6th of 7 under
page. A single-mode claim is not a claim. See `memory/relevance-modes.md`.

**Before any GPU session, run the 3-document smoke.** Five failures this session were
invisible to a green test suite and only surfaced by running the real CLI: a missing
`unstructured-inference`, `QAItem(**d)` rejecting external gold, `--relevance page`
absent from argparse choices, a silently-unapplied adapter patch, and `ultralytics`
shipping a top-level `tests` package that shadows ours.

Always use the project `.venv`. The `docstruct` console-script shim can be stale
after the project directory moves — `python -m docstruct.cli` always works.
Full command reference: `memory/conventions.md`.
