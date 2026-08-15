# TO-DO

Short "where we are / what's next". Durable detail lives in `notes.md` (Stages
8–17) and `memory/`. `memory/` wins when they disagree.

## Where we are (2026-08-13)

**OHR-Bench is done.** All three relevance modes, seven tools, 95 docs, 3,558
human questions, identical chunks across modes. DocStruct is **1st under `span`
and `region`, 6th of 7 under `page`** — the ranking inverts with the rule. Reports
in `reports/ohr_report_{page,span,region}.md`, slices alongside, full read in
`memory/relevance-modes.md`. This is the paper's headline.

Two results that go against us and are now recorded, not buried: **the model
detector is not significant outside arXiv** (+0.0012 span, +0.0090 region), and we
retrieve 2,194 context words per query against unstructured's 561.

**Section-boundary agreement is in, and we win it.** Seven tools against the PMC
papers' publisher-authored JATS boundaries: `docstruct_geo` 1st on both WindowDiff
(0.4362) and Pk (0.3525). No retriever, no relevance rule, gold nobody wrote for us --
the most tool-agnostic result in the project. `reports/section_scores.md`,
`memory/results.md`, `notes.md` Stage 20. **Caveat: 24 documents, not 126** -- the Colab
session had only 24 PMC PDFs fetched. Re-run before it goes in the paper.

**Corpora on disk:** OHR-Bench 95 docs (+ 12-doc subset), FinanceBench 84 docs /
189 evidence rows, PMC papers fetching (7 journals, PDF + JATS XML each), internal
arXiv 68 PDFs.

## Next

1. **Re-run the section table on the full 126-doc PMC corpus — GPU.**
   `notebooks/pmc_sections_colab.ipynb`, one-click. The current table is 24 docs and its
   ceiling file covers a different population (`reports/section_reachability.json` = 126
   docs, `reports/section_reachability_colab24.json` = the scored 24). The notebook
   asserts the fetched count before spending GPU time. This is the cheapest remaining
   headline in the project -- everything else needs a corpus or a metric we do not have.

2. **FinanceBench — NOT a retrieval leaderboard. Downgraded, see `notes.md` Stage 19.**
   Measured: its evidence is unreachable at top-5 under three embedders on identical
   chunks, so every tool scores MRR 0.0. That measures the retriever, not the chunkers.
   Gated behind `RUN_FINANCEBENCH = False` in `notebooks/financebench_sections_colab.ipynb`;
   ~8 GPU-hours to reproduce a table of zeros. It keeps two uses that need no leaderboard:
   **parse fidelity** (28.0% of evidence verbatim in pdfplumber text) and
   **borderless-table detection** (122 detected vs pdfplumber's 4 on `3M_2018_10K`) --
   the latter is still the cleanest remaining test of the model detector, now that three
   corpora have called it null.
3. **Sweep `RELEVANCE_REGION_MIN_OVERLAP`** against real chunks. Every region
   number, including our best result, currently rides on an unvalidated 0.7. The
   reachability script cannot settle it — that question is circular on region gold.
4. **Section *hierarchy* (paths), not just boundaries.** Boundary agreement is done
   (item 1); scoring the nesting — does chunk N sit under the right `SectionPath`? — is
   still open, and is still the metric no competitor in the table can report. The JATS
   gold already carries the tree, so this is CPU-only scoring work.
5. **Academic is our weakest domain in every mode** (`span` 0.4526 vs
   unstructured's 0.5151, 5th of 7), on a 10-document slice too thin to settle it.
   PMC answers the *structure* half — we win section boundaries there. The *retrieval*
   half on academic PDFs is still unanswered, because PMC has no Q&A gold.
6. **Gated-feature ablation sweep — GPU.** 14 flags, all still default OFF.
   `scripts/_sweep.sh`, steps in `memory/measurement-environment.md`.
7. **Internal arXiv corpus is broken on disk**: gold covers 92 docs, 68 PDFs
   present, **27 of the gold's documents missing**. `fetch_dataset_v2.py` dedupes
   against the committed manifest rather than the disk, so it will not re-download
   them. Prune the manifest to files that exist, or fix it to check `os.path.exists`.
   Blocks nothing headline — this corpus is for ablations now.
8. **An 18-page paper takes 8 minutes to parse** (`notes.md` Stage 18). 422k vector
   primitives; `detect` 362 s and `populate_text` 271 s, and the figure-clustering
   cap discards the objects *after* pdfplumber spent 53 s/page materialising them.
   This is on `parse()`, so users pay it, not just the benchmark. Two candidate
   fixes recorded; both need their own measurement before landing.

9. **IEEE Access / IEEEtran two-column is unrepresented.** Not in PMC, and its OA
   PDFs 403 outside a browser. The biggest typographic contrast with arXiv is still
   a corpus gap.

## Paper track

Draft: `paper/main.tex` + `paper/refs.bib`. Source of truth is `memory/` —
`relevance-modes.md`, `results.md`, `related-work.md`, `benchmark-datasets.md`,
`metrics-justification.md`, `paper-structure-survey.md`.

- **Verify `refs.bib`** — several author lists are placeholders (see file header).
  Two new citations are now required: **Beeferman et al. 1999** (Pk) and
  **Pevzner & Hearst 2002** (WindowDiff).
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
