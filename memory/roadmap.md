# Roadmap

Ranked by expected value. Everything here has been audited against the actual code
(`implementation_plan.md` §10 holds the full plan-vs-code table).

## Next up

1. **Broaden the corpus beyond arXiv.** Every number in `results.md` comes from
   born-digital two-column papers, and the XY-cut result proves corpus shape
   decides which algorithm wins. This is the highest-value open item because it is
   the one that could *invalidate* current conclusions rather than incrementally
   improve them.

   **Status: unblocked in code, blocked on quota.** 47 further PDFs are fetched
   and the generator faults that made bulk gold impossible are all fixed (see
   `notes.md` §7.3, §7.7). What stopped it is a free tier's 100,000 tokens **per
   day** — roughly seven papers. Resuming needs either a paid tier or several days
   of running `gen-qa`, which resumes per document and can be left to grind.
   Nothing in the codebase needs to change first.

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

## Lower value, well understood

6. **Table size cap.** Tables are atomic with no size guard, so a table exceeding
   `MAX_CHUNK_TOKENS` produces an oversized outlier chunk. No observed failure yet.

8. **Embedding cache across benchmark reruns.** Would speed up retrieval-only
   changes; does nothing for chunking changes, which legitimately re-chunk. The
   block cache already captured most of the available win.

9. **Standalone figure chunks.** See `decisions.md` — deliberately skipped, listed
   here only so it is not rediscovered as an oversight.

## Done since this list was written

- Single-detector ablation (`pipeline_mode`, `docstruct_geo` / `docstruct_model`).
- Bootstrap CIs and a paired significance test on every benchmark report.
- Numbering-pattern header levels (`HEADER_NUMBERING_LEVELS`).
- Cross-boundary overlap: measured, lost, stays off — see `decisions.md`.

## Explicitly not planned

Embedding-similarity semantic chunking, VLM captioning, any LLM call on the parse
path, tokenizer-based chunk sizing. Reasons in `decisions.md`.

## Standing rule

Nothing on this list ships without a measurement against the current numbers in
`results.md`. Items 5-8 are cheap; item 1 is the one that changes what we know.
