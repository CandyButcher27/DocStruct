# Roadmap

Ranked by expected value. Everything here has been audited against the actual code
(`implementation_plan.md` §10 holds the full plan-vs-code table).

## Next up

0. **Adopt FinanceBench as the external corpus** (added 2026-08-05, paper track).
   Supersedes item 1 as the cheapest way to break arXiv homogeneity *and* removes
   the "LLM-generated gold" objection at the same time: 150 human-annotated
   questions / 84 born-digital SEC filings, public, table-heavy, and the corpus our
   closest competitor (`arXiv:2604.12047`) used. `scripts/fetch_financebench.py` is
   written and smoke-tested; the only code work is a `--relevance page|span|token`
   switch in the benchmark, because FinanceBench evidence is a page region rather
   than a sentence span. See `benchmark-datasets.md` and `related-work.md`.
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
