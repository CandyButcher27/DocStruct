# Roadmap

Ranked by expected value. Everything here has been audited against the actual code
(`implementation_plan.md` §10 holds the full plan-vs-code table).

## Next up

1. **Broaden the corpus beyond arXiv.** Every number in `results.md` comes from
   born-digital two-column papers, and the XY-cut result proves corpus shape
   decides which algorithm wins. `scripts/fetch_dataset_v2.py` already targets
   seven domains; the blocker is gold generation and wall time, not code. This is
   the highest-value open item because it is the one that could *invalidate*
   current conclusions rather than incrementally improve them.

2. **Regenerate gold from correctly-spaced text.** The whitespace-blind relevance
   rule is a cheap guard around a mismatch whose real fix is regenerating the
   spans now that `TEXT_X_TOLERANCE_RATIO` is in place.

3. **Calibrate the `# unvalidated` confidence constants** against the annotated
   set (`UNILATERAL_*_SCALE`, both `CONFIDENCE_BOUNDS`). This unblocks
   confidence-weighted retrieval ranking, which is otherwise built on untuned
   numbers. Needs more than two annotated documents first.

4. **`pipeline_mode` ablation path** (`geometry-only` / `model-only`) with
   `docstruct_geo` / `docstruct_model` adapters. Would let the benchmark answer
   "what is the vision model actually worth?" — currently unanswerable, and it is
   the obvious first question anyone asks about a hybrid design.

## Lower value, well understood

5. **Overlap across structural boundaries.** `OVERLAP_ON_BOUNDARY` is implemented;
   whether it helps is an ablation nobody has run yet. Cheap to answer.

6. **Table size cap.** Tables are atomic with no size guard, so a table exceeding
   `MAX_CHUNK_TOKENS` produces an oversized outlier chunk. No observed failure yet.

7. **Numbering-pattern header levels.** Regex `^\d+(\.\d+)*\s` as a deterministic
   tiebreaker when font size does not separate levels. Not measurable on the
   current benchmark (header level feeds only `section_path` metadata), so it can
   only be justified on correctness grounds.

8. **Embedding cache across benchmark reruns.** Would speed up retrieval-only
   changes; does nothing for chunking changes, which legitimately re-chunk. The
   block cache already captured most of the available win.

9. **Standalone figure chunks.** See `decisions.md` — deliberately skipped, listed
   here only so it is not rediscovered as an oversight.

## Explicitly not planned

Embedding-similarity semantic chunking, VLM captioning, any LLM call on the parse
path, tokenizer-based chunk sizing. Reasons in `decisions.md`.

## Standing rule

Nothing on this list ships without a measurement against the current numbers in
`results.md`. Items 5-8 are cheap; item 1 is the one that changes what we know.
