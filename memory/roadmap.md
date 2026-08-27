# Roadmap

Ranked by expected value. Everything here has been audited against the actual code
(`implementation_plan.md` §10 holds the full plan-vs-code table).

## Next up

> **Three GPU/Colab jobs, queued 2026-08-28 from two independent paper reviews.**
> Both reviewers made the first one their blocking item, arriving at it separately.
>
> 1. **Embedder × retriever sweep.** Every number in the paper comes from one
>    configuration (`all-MiniLM-L6-v2` + BM25 + RRF, k=5). The relevance-rule
>    inversion may be conditional on it, and §5.5 is our own evidence that swapping
>    small encoders can move results from a ranking to all zeros. Run the three rules
>    under 2–3 more encoders (bge-base, e5-base). If the inversion survives, the
>    paper's central claim is much stronger; if it does not, that is the more
>    interesting finding. **This is the gate on submitting anywhere.**
> 2. **Re-run PMC section scoring keeping per-document output.** Needed to print the
>    random-boundary floor beside the tool table — the two are currently on different
>    document sets (122 vs 134) and the per-doc scores for the 134 run were lost with
>    the Colab session. Cheapest of the three; same job that produced the table.
> 3. **`MIN_CHUNK_TOKENS` floor sweep on PMC.** 57.4% of gold sections fall below the
>    floor, so the section metric partly measures how our floor negotiates with
>    publisher structure. Worse: the floor was chosen on the withdrawn internal
>    corpus, so it is an *inherited* constant, not one tuned to any corpus in the
>    paper. Report Pk/WindowDiff across 3–4 floors.
>
> Also open from the reviews, no run required: parse fidelity on OmniDocBench/READoc
> (a real hole — the system is a structural parser with no direct parse measurement),
> and the DocLayNet val split to retire the `# unvalidated` fusion constants.

> **Blocking, added 2026-08-16: the internal arXiv corpus does not match its own gold.**
> 0 of 65 shared filenames have findable gold; a re-fetch reused `doc<N>.pdf` names for
> different papers. This blocks *every* ablation, including the `BAND_SPLIT` measurement
> that would land the reading-order fix. `dataset_manifest_v2.json` still records
> `arxiv_id` per entry, so the original set is recoverable — `fetch_dataset_v2.py` must
> key on that, not on positional filename. `notes.md` Stage 24.
>
> **Done since the last revision:** determinism measured (95/95, `results.md`); the
> region threshold swept; section boundaries scored on 134 PMC papers; the paper
> reframed around the relevance-rule finding and cut to 8 pages; the library made
> pip-installable as `docstruct-rag`.


> **Item 0 is done (2026-08-11).** OHR-Bench, not FinanceBench, became the primary
> external corpus: 95 docs, 3,558 human questions, seven tools, **all three
> relevance modes**. DocStruct is 1st under `span` and `region`, 6th of 7 under
> `page`, on identical chunks — the ranking inverts with the rule, which is a
> result in its own right (`relevance-modes.md`). The `--relevance` switch that
> was the blocker below shipped and now has three modes, not two.
>
> **Updated 2026-08-16. (a) and (b) below are resolved; the ordering changed.**
>
> **(a) FinanceBench is not a retrieval leaderboard** — measured, not deferred. Its
> evidence is unreachable at top-5 under three embedders on identical chunks, so every
> tool scores MRR 0.0 (`notes.md` Stage 19). It keeps two uses that need no leaderboard:
> parse fidelity, and borderless-table detection as the one place the model detector
> could still pay for itself.
>
> **(b) The region threshold is swept** (2026-08-16, on OHR-Bench). A DocStruct variant
> is 1st at all ten thresholds 0.1–1.0; the constant does not choose the winner.
>
> **(c) The model detector still does not replicate outside arXiv** (+0.0012 span,
> +0.0090 region), and a third corpus now agrees: on PMC section boundaries,
> geometry-only *beats* the hybrid at 2.4× the speed. Three corpora, no effect. The
> open decision is no longer "measure it again" but "demote it or give it
> FinanceBench's borderless tables".
>
> **New since:** section-boundary agreement against publisher JATS gold on 134 PMC
> papers — we lead both Pk and WindowDiff, with no retriever involved
> (`results.md`). And two extraction defects found while building the README figure
> (`to-do.md` 6 and 7).

0. ~~**Adopt FinanceBench as the external corpus**~~ (added 2026-08-05, paper track).
   Superseded by the OHR-Bench run above, and partly done: 150 human-annotated
   questions / 84 born-digital SEC filings, public, table-heavy, and the corpus our
   closest competitor (`arXiv:2604.12047`) used. `scripts/fetch_financebench.py`
   has now fetched all 84 documents and 189 evidence rows; what remains is the GPU
   run under `--relevance region`. See `benchmark-datasets.md` and `related-work.md`.
   Item 1 below stays valuable (558 paired questions is statistical power
   FinanceBench's 150 cannot match) but is no longer the *first* move.

1. **Broaden the corpus beyond arXiv.** Every number in `results.md` comes from
   born-digital two-column papers, and the XY-cut result proves corpus shape
   decides which algorithm wins. This is the highest-value open item because it is
   the one that could *invalidate* current conclusions rather than incrementally
   improve them.

   **Status: IN PROGRESS (2026-07 Fable session).** Quota is no longer the blocker —
   Ollama cloud (`gpt-oss:120b`, `OLLAMA_API_KEY`) is verified live and has room. A
   background fetch of the six non-arXiv domains (legal/financial/medical/technical/
   govt/textbook, ~150 docs via `scripts/fetch_dataset_v2.py`) is running. Next:
   `gen-qa` on the new docs via Ollama, then re-baseline `results.md`. Until that
   lands, every number stays arXiv-only.

2. **Regenerate gold from correctly-spaced text.** The whitespace-blind relevance
   rule is a cheap guard around a mismatch whose real fix is regenerating the
   spans now that `TEXT_X_TOLERANCE_RATIO` is in place. Same quota constraint as
   item 1 — and note that `_extract_full_text` deliberately uses **pdfplumber
   defaults**, not DocStruct's tuned tolerance: making the gold match DocStruct's
   own extraction would bias the benchmark toward DocStruct.

3. **Calibrate the `# unvalidated` confidence constants** against the annotated
   set (`UNILATERAL_*_SCALE`, both `CONFIDENCE_BOUNDS`). This unblocks
   confidence-weighted retrieval ranking, which is otherwise built on untuned
   numbers. Needs more than two annotated documents first.

## In flight — Fable review (2026-07)

Two batches landed on branch `feat/pypi-hardening` (merged to main): PyPI-release
hardening (typed errors, Path/password, scanned diagnostic, version single-source,
py.typed, CI, CHANGELOG) and ~14 config-gated deterministic features + 2 bug fixes.
See `notes.md` Stages 8–9 and `decisions.md`. Remaining from that review:

- **Run the gated-feature sweep** — attempted, **environment-blocked** (~69 min/run,
  ~16 h total; long jobs killed hourly). `scripts/_sweep.sh` is ready for a capable
  machine + the broadened corpus. No flag flipped on the low-power arXiv subset. See
  `results.md`. This is the one open action that converts the batch to "worth keeping".
- **§1.8 `ParseConfig`** — DONE (pragmatic): `config.override()` + `parse(config=...)`
  give non-mutating, thread-safe per-parse overrides with zero call-site rewrites.
  The full frozen-dataclass-threaded version is a future branch, only if
  parallel-with-different-configs is ever needed. See `decisions.md`.
- **§3.4 multi-page table merge** — DONE (gated `MERGE_MULTIPAGE_TABLES`, 2-page).
- **§4.3 SectionPath depth > 3** — deferred (YAGNI): breaks chunk-JSON serialization
  for a capability arXiv never exercises; build when a deep-hierarchy corpus lands.

## Lower value, well understood

6. **Table size cap.** DONE (gated) — `TABLE_SPLIT_ROWS` splits an oversized table
   into header-repeating row segments. Off pending the sweep.

8. **Embedding cache across benchmark reruns.** Would speed up retrieval-only
   changes; does nothing for chunking changes, which legitimately re-chunk. The
   block cache already captured most of the available win.

9. **Standalone figure chunks.** See `decisions.md` — deliberately skipped, listed
   here only so it is not rediscovered as an oversight.

## Done since this list was written

- Single-detector ablation (`pipeline_mode`, `docstruct_geo` / `docstruct_model`).
- Bootstrap CIs and a paired significance test on every benchmark report.
- Numbering-pattern header levels (`HEADER_NUMBERING_LEVELS`); appendix/Roman too.
- Cross-boundary overlap: measured, lost, stays off — see `decisions.md`.
- PyPI-release hardening batch (typed errors, Path/password, py.typed, CI, ...).

## Explicitly not planned

Embedding-similarity semantic chunking, VLM captioning, any LLM call on the parse
path, tokenizer-based chunk sizing. Reasons in `decisions.md`.

## Standing rule

Nothing on this list ships without a measurement against the current numbers in
`results.md`. Items 5-8 are cheap; item 1 is the one that changes what we know.
