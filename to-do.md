# TO-DO

Short "where we are / what's next". Durable detail lives in `notes.md` (Stages
8–17) and `memory/`. `memory/` wins when they disagree.

## Where we are (2026-08-12)

**OHR-Bench is done.** All three relevance modes, seven tools, 95 docs, 3,558
human questions, identical chunks across modes. DocStruct is **1st under `span`
and `region`, 6th of 7 under `page`** — the ranking inverts with the rule. Reports
in `reports/ohr_report_{page,span,region}.md`, slices alongside, full read in
`memory/relevance-modes.md`. This is the paper's headline.

Two results that go against us and are now recorded, not buried: **the model
detector is not significant outside arXiv** (+0.0012 span, +0.0090 region), and we
retrieve 2,194 context words per query against unstructured's 561.

**Corpora on disk:** OHR-Bench 95 docs (+ 12-doc subset), FinanceBench 84 docs /
189 evidence rows, PMC papers fetching (7 journals, PDF + JATS XML each), internal
arXiv 68 PDFs.

## Next

1. **FinanceBench run — GPU.** Corpus and gold are fetched; nothing blocks it but
   hardware. ~15,000 pages, `--relevance region` mandatory. It is also the corpus
   built to test the model detector (122 vs 4 tables detected on `3M_2018_10K`),
   so it decides whether the null result above is corpus-specific.
2. **Sweep `RELEVANCE_REGION_MIN_OVERLAP`** against real chunks. Every region
   number, including our best result, currently rides on an unvalidated 0.7. The
   reachability script cannot settle it — that question is circular on region gold.
3. **JATS → section-hierarchy gold** from the PMC XML. CPU-only. The only route to
   scoring section paths as a metric rather than the qualitative claim it is now,
   and no competitor in the table can report it.
4. **Academic is our weakest domain in every mode** (`span` 0.4526 vs
   unstructured's 0.5151, 5th of 7), on a 10-document slice too thin to settle it.
   The PMC corpus is the follow-up.
5. **Gated-feature ablation sweep — GPU.** 14 flags, all still default OFF.
   `scripts/_sweep.sh`, steps in `memory/measurement-environment.md`.
6. **Internal arXiv corpus is broken on disk**: gold covers 92 docs, 68 PDFs
   present, **27 of the gold's documents missing**. `fetch_dataset_v2.py` dedupes
   against the committed manifest rather than the disk, so it will not re-download
   them. Prune the manifest to files that exist, or fix it to check `os.path.exists`.
   Blocks nothing headline — this corpus is for ablations now.
7. **IEEE Access / IEEEtran two-column is unrepresented.** Not in PMC, and its OA
   PDFs 403 outside a browser. The biggest typographic contrast with arXiv is still
   a corpus gap.

## Paper track

Draft: `paper/main.tex` + `paper/refs.bib`. Source of truth is `memory/` —
`relevance-modes.md`, `results.md`, `related-work.md`, `benchmark-datasets.md`,
`metrics-justification.md`, `paper-structure-survey.md`.

- **Verify `refs.bib`** — several author lists are placeholders (see file header).
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
