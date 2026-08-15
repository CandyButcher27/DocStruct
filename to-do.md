# TO-DO

Short "where we are / what's next". Durable detail lives in `notes.md` (Stages
8–21) and `memory/`. `memory/` wins when they disagree.

## Where we are (2026-08-16)

**OHR-Bench is done.** All three relevance modes, seven tools, 95 docs, 3,558
human questions, identical chunks across modes. DocStruct is **1st under `span`
and `region`, 6th of 7 under `page`** — the ranking inverts with the rule. Reports
in `reports/ohr_report_{page,span,region}.md`, slices alongside, full read in
`memory/relevance-modes.md`. This is the paper's headline.

Two results that go against us and are now recorded, not buried: **the model
detector is not significant outside arXiv** (+0.0012 span, +0.0090 region), and we
retrieve 2,194 context words per query against unstructured's 561.

**Compute is done. The blocker is now writing, not measurement.** The 2026-08-16 GPU
session closed both remaining gaps and neither result moved, which is what was worth
paying for. `notes.md` Stage 21.

**Section-boundary agreement is in, and we win it.** Seven tools, **134 documents**, against the
PMC papers' publisher-authored JATS boundaries: `docstruct_geo` 1st on both WindowDiff
(0.4226) and Pk (0.3418). No retriever, no relevance rule, gold nobody wrote for us --
the most tool-agnostic result in the project. Reproduces the 24-document pilot in order
and within 0.02 in value. Ceiling and scores now cover the same population (138 with
gold, 134 scored, 84.7% body ceiling). `reports/section_scores.md`, `memory/results.md`.

**The region threshold is validated.** `RELEVANCE_REGION_MIN_OVERLAP = 0.7` was
`# unvalidated` and one headline rested on it. Swept 0.1-1.0 on OHR-Bench: **a DocStruct
variant is 1st at all ten thresholds**, margin +0.045 to +0.062 over the best external
tool across 0.4-1.0. `config.py` now carries the measurement. Two things the paper must
say anyway: 0.7 is where our margin peaks (+0.0619), and the field below us reorders
constantly even though our position does not.

**Corpora on disk:** OHR-Bench 95 docs (+ 12-doc subset), FinanceBench 84 docs /
189 evidence rows, PMC papers 133 (7 journals, PDF + JATS XML each), internal
arXiv 68 PDFs.

## Next

1. **Write the paper.** This is the only thing on the critical path.
   `paper/main.tex` is dated 2026-08-05: it mentions OHR-Bench once, still lists
   FinanceBench as the planned external corpus, and carries `\todo{run it. This section
   is the difference between an internal report and a paper.}` over a Results section
   that is internal-arXiv-only. §4 Setup and §5 Results are rewrites around the three
   external results (OHR-Bench three modes, section boundaries, the threshold sweep),
   not edits. Everything they need is measured and in `memory/`.

2. **FinanceBench — NOT a retrieval leaderboard. Downgraded, see `notes.md` Stage 19.**
   Measured: its evidence is unreachable at top-5 under three embedders on identical
   chunks, so every tool scores MRR 0.0. That measures the retriever, not the chunkers.
   Gated behind `RUN_FINANCEBENCH = False` in `notebooks/financebench_sections_colab.ipynb`;
   ~8 GPU-hours to reproduce a table of zeros. It keeps two uses that need no leaderboard:
   **parse fidelity** (28.0% of evidence verbatim in pdfplumber text) and
   **borderless-table detection** (122 detected vs pdfplumber's 4 on `3M_2018_10K`) --
   the latter is still the cleanest remaining test of the model detector, now that three
   corpora have called it null.
3. **Report-only metric work — no GPU, no new runs.** Rename Hit@1 -> Precision@1
   (matching `arXiv:2604.12047`); adopt token-level IoU / Precision_Omega to replace the
   homemade MRR-per-1k-context-words; add Beeferman 1999 and Pevzner & Hearst 2002 to
   `refs.bib` and verify the placeholder author lists flagged in its header. All
   computable from dumps already on disk.

4. **Section *hierarchy* (paths), not just boundaries.** Boundary agreement is done
   (item 1); scoring the nesting — does chunk N sit under the right `SectionPath`? — is
   still open, and is still the metric no competitor in the table can report. The JATS
   gold already carries the tree, so this is CPU-only scoring work.
5. **Academic is our weakest domain in every mode** (`span` 0.4526 vs
   unstructured's 0.5151, 5th of 7), on a 10-document slice too thin to settle it.
   PMC answers the *structure* half — we win section boundaries there. The *retrieval*
   half on academic PDFs is still unanswered, because PMC has no Q&A gold.
6. **Full-width blocks are extracted across the column gutter.** A block spanning
   both columns has its text populated without column awareness, interleaving the two
   columns into unreadable text. Reproduce:
   `run_pipeline('data/raw-pdfs/doc1.pdf', weights='weights/yolov8m-doclaynet.pt')`,
   page 0, block at `y0=332 x0=64 w=494`. It is labelled `header`, so the garbage
   becomes a `SectionPath` level (2 of 76 chunks on that doc carry a bad path).
   This is the failure the paper's intro attributes to naive parsers, inside our own
   pipeline. Fix in `populate_text`: detect column bands within the block bbox and
   extract per band. Needs `scripts/ablate.py` before landing (rule 1). `notes.md`
   Stage 22.

7. **Full-width elements above a two-column body sort after the columns.** On
   `doc1.pdf` page 0 the paper title (`y0=97`) comes out 7th in reading order, after
   the whole left column. Body pages are correct, so this is a title-page/section-break
   defect. Same file, same measurement requirement. `notes.md` Stage 22.

8. **Gated-feature ablation sweep — GPU.** 14 flags, all still default OFF.
   `scripts/_sweep.sh`, steps in `memory/measurement-environment.md`.
9. **Internal arXiv corpus is broken on disk**: gold covers 92 docs, 68 PDFs
   present, **27 of the gold's documents missing**. `fetch_dataset_v2.py` dedupes
   against the committed manifest rather than the disk, so it will not re-download
   them. Prune the manifest to files that exist, or fix it to check `os.path.exists`.
   Blocks nothing headline — this corpus is for ablations now.
10. **An 18-page paper takes 8 minutes to parse** (`notes.md` Stage 18). 422k vector
   primitives; `detect` 362 s and `populate_text` 271 s, and the figure-clustering
   cap discards the objects *after* pdfplumber spent 53 s/page materialising them.
   This is on `parse()`, so users pay it, not just the benchmark. Two candidate
   fixes recorded; both need their own measurement before landing.

11. **IEEE Access / IEEEtran two-column is unrepresented.** Not in PMC, and its OA
   PDFs 403 outside a browser. The biggest typographic contrast with arXiv is still
   a corpus gap.

## Paper track

Draft: `paper/main.tex` + `paper/refs.bib`. Source of truth is `memory/` —
`relevance-modes.md`, `results.md`, `related-work.md`, `benchmark-datasets.md`,
`metrics-justification.md`, `paper-structure-survey.md`.

- **Verify `refs.bib`** — several author lists are placeholders (see file header).
  Two new citations are now required: **Beeferman et al. 1999** (Pk) and
  **Pevzner & Hearst 2002** (WindowDiff).
- **New losses to report in the main table, not the appendix**: unstructured hard-fails
  on 26% of PMC PDFs so its section row is on 99 of 134 documents (say the N); 0.7 is
  where our region margin peaks; and the region *field* ranking is threshold-sensitive
  even though our position is not.
- **Rename Hit@1 → Precision@1**, matching `arXiv:2604.12047`.
- **Token-level IoU / Precision_Ω** (Chroma TR metric set) to replace the homemade
  MRR-per-1k-context-words.
- **DocLayNet val split** for the detection layer, replacing the 2 hand-annotated
  docs; also the only cheap path to calibrating the `# unvalidated` fusion constants.

## Deferred (with reasons — see decisions.md/roadmap.md)

- §4.3 deeper SectionPath — breaks chunk-JSON for depth arXiv never reaches (YAGNI).
- §5.3 confidence calibration — needs ~20 hand-annotated docs (user chose to skip).
- §7.2/§7.3 perf, §8 mkdocs site — low value / P3.
- Full §1.8 threaded ParseConfig — pragmatic version shipped; full refactor only if
  parallel-different-config throughput is ever needed.
