# How the competing papers are written — structure, rhetoric, conventions

Compiled 2026-08-06. Companion to [`related-work.md`](related-work.md) (what they
claim) and [`metrics-justification.md`](metrics-justification.md) (what they
measure). **This file is about form, not content**: section skeletons, how a
contribution is framed, where the losses go, what an artifact statement looks like.

Scope note: skeletons below are read off the papers' own section lists and their
venue templates. Where a detail was not directly verified it is marked *(assumed
from venue norm)* — check before imitating.

---

## 1. There are three genres, and they are written differently

| Genre | Example | What the paper *is* | Section shape |
|---|---|---|---|
| **Benchmark / resource** | OmniDocBench, READoc, DocLayNet, FinanceBench, OHR-Bench | "here is a dataset + a protocol; here is how everything scores on it" | Intro → Related datasets → **Dataset construction** (the longest section) → Evaluation protocol/metrics → Results across many systems → Analysis → Limitations → Ethics/licence |
| **Empirical study** | El Bachyr et al. (ICSE-SEIP 2026), the 2025–26 chunking studies | "here is a grid of existing components; here is what wins and why" | Intro → Background → **Research questions (RQ1..RQn)** → Study design → Results *per RQ* → Discussion → **Threats to validity** → Conclusion |
| **System / method** | Docling, LumberChunker, TopoChunker, MultiDocFusion | "here is a thing we built; here is the ablation proving each part earns its place" | Intro → Related work → **Method** (architecture + design decisions) → Experimental setup → Results + **ablations** → Limitations → Conclusion |

**DocStruct is a system paper with an unusually strong empirical-study core.**
That hybrid is a real choice, not a hedge: the pipeline is the contribution, but
the protocol (fix embedder + retriever, vary only the chunker, paired bootstrap)
is what makes the numbers believable. The draft in `paper/` follows the system
skeleton and imports two habits from the empirical genre: explicit RQs and a
threats-to-validity discipline.

**Recommendation:** add explicit RQs to `paper/main.tex`. Software-engineering and
empirical venues expect them, they make the results section self-organizing, and
they cost a paragraph. Proposed:

- **RQ1** Does structure-aware chunking improve retrieval over generic splitters when the retriever is held fixed?
- **RQ2** What is each detector worth? (geometry-only vs hybrid)
- **RQ3** What does the rank-quality gain cost in retrieved context, coverage and duplication?
- **RQ4** Does the result hold on a public, human-annotated, non-arXiv corpus?

RQ4 is currently unanswerable — that is exactly the FinanceBench work item, and
naming it as an RQ makes the gap visible to us instead of to a reviewer.

## 2. The moves these papers make, that our draft should copy

**A one-sentence gap statement in the first or second paragraph.** Every one of
these papers names the hole before naming the fix. OmniDocBench: existing
benchmarks are single-task and narrow-domain. READoc: extraction is evaluated on
fragments, not realistic whole documents. El Bachyr: no comprehensive study of how
RAG components interact on PDFs. Ours, already drafted: parsing and chunking are
measured separately and never as a composition.

**Contributions as an explicit numbered list at the end of the intro.** Universal
in this literature. Ours has five; three or four reads stronger — merge the weaker
two.

**Research questions or claims that the results section then answers in order.**
The empirical papers make this structural; system papers usually do it implicitly.
Doing it explicitly is free rigour.

**A dataset/protocol table early.** Reviewers look for the corpus size, the domain
mix and the licence before reading anything else. Our draft buries corpus
description in prose — it should be a table: docs, questions, domains, gold
provenance, licence, public y/n.

**Ablations that remove one thing at a time, with the removed thing named in the
row label.** Our Table 2 already does this and is one of the draft's strengths.

**Threats to validity / limitations as a named section, not a paragraph.** In SE
venues (ICSE-SEIP) this is mandatory and is conventionally split into internal,
external and construct validity. Our `Limitations` section has the content; adding
that three-way structure costs nothing and signals genre fluency:
- *internal* — cache correctness, config fingerprinting, the v5→v6 gold artefact;
- *external* — arXiv-heavy corpus, born-digital only;
- *construct* — containment relevance misses paraphrase, coverage is not a ranking.

**An artifact / reproducibility statement.** El Bachyr et al. publish code,
prompts, datasets and configs, and say so in the abstract. OmniDocBench, READoc,
DocLayNet, OHR-Bench and FinanceBench all ship data. **This is now table stakes,
and it is the one axis where we are already ahead of most system papers**: the
pipeline is deterministic, offline, config-fingerprinted, and every reported number
carries its full config snapshot in the JSON sidecar. Say that in the abstract.

**Licence and ethics stated for every dataset used.** FinanceBench is CC-BY-NC-4.0;
that belongs in the paper, not just in `benchmark-datasets.md`.

## 3. Conventions worth matching precisely

| Convention | What they do | What we should do |
|---|---|---|
| Metric naming | `Precision@1`, `Recall@k`, `MRR`, `nDCG@k`, `TEDS`, `NED` | Rename our `Hit@1` → `Precision@1`. Never invent a name where one exists |
| Bold in tables | Best per column bolded; arrows (↑/↓) on the header when direction is not obvious | Our draft bolds; add ↑/↓ |
| Significance | Rarely reported in this literature — El Bachyr gives point estimates only | **Keep our CIs and paired bootstrap and say so explicitly.** It is a differentiator, so it should be a sentence in the intro, not a line in the setup |
| Negative results | Usually buried or omitted | Ours are a named section. Keep it — it is credibility, and the XY-cut result is genuinely informative about corpus dependence |
| Cross-paper numbers | Compared only within a shared benchmark | Never put our MRR beside El Bachyr's. If we want that comparison, we must run *their* corpus — which is precisely the FinanceBench item |
| Length | CVPR/ICCV 8pp + refs; ACL 8pp; ICSE-SEIP ~10pp; arXiv-only studies 10–20pp | Draft the arXiv version long, cut to venue |
| Related work | Grouped by *what is measured*, not chronologically | Our draft already does this |

## 4. Venue options, and what each demands

| Venue | Fit | Demands |
|---|---|---|
| **ICSE-SEIP / FSE-Industry** | Strong — El Bachyr et al. landed there with a near-identical framing | RQs, threats-to-validity, practitioner guidance, artifact |
| **ACL / EMNLP Findings, or the industry track** | Strong for the retrieval framing | Tight 8pp, strong baselines (a semantic chunker is non-negotiable), significance testing (we have it) |
| **ECIR / SIGIR short** | Good for the fixed-retriever protocol | IR reviewers will demand standard metric names and multiple corpora |
| **DocEng / ICDAR** | Natural home for the layout half | Will expect parse-fidelity numbers (READoc/OmniDocBench) we do not have |
| **arXiv preprint first** | Recommended | Costs nothing, dates the contribution, and the 2026 chunking wave (`2603.18409`, `2605.00318`, `2604.12352`) is moving fast enough that priority matters |

## 5. Concrete edits this survey implies for `paper/main.tex`

1. Add an RQ paragraph at the end of §1; restructure §5 (Results) to answer RQ1–RQ4 in order.
2. Merge contributions 4 and 5 → four contributions.
3. Turn the corpus prose in §4.2 into a table (docs / questions / domains / gold provenance / licence / public).
4. Split Limitations into internal / external / construct validity.
5. Add an artifact-and-reproducibility statement; put determinism in the abstract.
6. Add ↑/↓ to table headers; rename Hit@1 → Precision@1 throughout.
7. Move the "we report paired bootstrap tests, the closest comparable study does not" claim from §4.3 into §1 as a contribution.
</content>
