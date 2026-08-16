# What this paper needs, and in what order

Written 2026-08-16, against `paper/main.tex` at commit `ffa0fea`.

This is a work plan, not a critique. The question behind it — *is this a good paper or
a list of failures?* — is answered first, because the answer changes what the work is
for.


---

## STATUS — 2026-08-16 (read this first; the sections below predate it)

**The paper is submittable to a workshop today.** 8 pages, builds clean, 0 undefined
references, one `\todo` (affiliation). Reframed around the relevance-rule finding.

### Done since this document was written

| | |
|---|---|
| Determinism | **95/95** documents byte-identical across independent processes, 5,810 chunks per run. Cross-checks exactly against the benchmark's own chunk count from a different machine and code path. |
| Region threshold | Swept 0.1–1.0; a DocStruct variant leads at all ten. The constant does not choose the winner. |
| Section boundaries | Full corpus: 134 PMC papers, we lead Pk and WindowDiff. |
| Reframe | Title, abstract, contributions, §6 retitle, §6+§7 merged. |
| Length | 9 → 8 pages. |
| Bibliography | All 24 arXiv entries verified against the API; one fabricated attribution corrected, seven placeholders filled. |
| Library | `pip install docstruct-rag` — built, `twine check` clean, verified in a fresh venv. **Not yet published.** |

### Blocked, and it is the top priority

**The internal arXiv corpus does not match its own gold.** 0 of 65 shared filenames
have findable gold; a re-fetch reused `doc<N>.pdf` names for different papers
(`notes.md` Stage 24). Consequences:

- Table 3 and three ablation deltas — including the vision detector's +0.0443, the
  only corpus where it shows an effect — are **not reproducible today**. The paper now
  says so where the table appears.
- **No ablation can run**, which blocks the reading-order fix below.

Repair: recover by `arxiv_id` from `dataset_manifest_v2.json`, which still records it.
Estimated half a day, mostly download time.

### The two extraction defects, after attempting both

**Reading order (full-width elements sort after columns): solved in code, unmeasured.**
`BAND_SPLIT=True` fixes it — verified, the title moves from 7th to 1st. It is
default-OFF pending an ablation that cannot run until the corpus is repaired. This is
now a measurement task, not a coding one.

**Column-gutter extraction: attempted, measured, reverted.** A gutter-splitting
extractor was written and did find the correct gutter, and the left column came out
clean — but document-wide the garbled-token count was **14 before and 14 after**. It
moved text without fixing the defect, so it was reverted rather than shipped behind a
flag. Recorded as a negative result: *a naive gutter split does not solve this.*
Whoever takes it next should start by asking why the right-hand crop remains
interleaved even when the split point is correct.

### Recommended order from here

1. **Send the paper for review now.** It will not get better by waiting for items 2–4,
   and the feedback most likely to change the work is framing feedback, which is
   available today.
2. **Repair the internal corpus** (half a day). Unblocks everything else.
3. **Measure `BAND_SPLIT`** and, if it holds, flip the default. Lands a real fix.
4. **Publish to PyPI** — one command, needs your token.
5. Then Tier 2 below, if aiming beyond a workshop.

---

## 1. Is this a failure paper? No. But it reads like one, and that is a bug.

Count what the measurements actually say.

**Results in our favour (all on external, human or publisher-authored gold):**

| Result | Number | Corpus |
|---|---|---|
| 1st of 7 under `span` relevance | MRR 0.706 | OHR-Bench, 3,558 human questions |
| 1st of 7 under `region` relevance | MRR 0.666 | same |
| Beats 4 of 5 external tools significantly | paired bootstrap | same |
| 1st on WindowDiff | 0.423 | 134 PMC papers, publisher JATS gold |
| 1st on Pk | 0.342 | same |
| 1st at **all ten** relevance thresholds | sweep 0.1–1.0 | same corpus, re-scored |
| Deterministic, local, no LLM on the parse path | — | by construction |

**Results against us:**

| Result | Number |
|---|---|
| 6th of 7 under `page` relevance | 0.600 |
| Vision detector null outside arXiv | +0.0012 span, +0.0090 region |
| Context cost | 3.9× unstructured |
| Word coverage | 0.817 vs LangChain 1.00 |
| Chunk duplication | 2.06×, worst in field |

That is **seven wins against five caveats**, on gold nobody involved in this project
wrote. A paper with that ledger is a *normal, decent* paper. Most published chunking
work reports fewer wins on weaker gold and mentions none of the caveats.

**So why does the draft feel like a confession?** Because of framing choices I made,
not because of the data. The abstract literally says *"Our central finding is
methodological and negative"* — leading with the word *negative* about a paper that
wins two of three relevance rules and both segmentation metrics. §6 is titled
"Negative and Cautionary Results" and runs a full page before Limitations adds another.
The honest ledger got written up in a self-flagellating register.

Fix that first, in §5 below. It costs nothing and it is the single highest-leverage
change available.

---

## 2. Is there a breakthrough? No. Is there a real contribution? Yes — one, clearly.

Be precise, because the two words are not the same and inflating one is how papers get
rejected.

**Not a breakthrough.** Nothing here changes what is possible. DocStruct is a
well-engineered rules-based chunker; the field already knew structure-aware chunking
beats fixed-size splitting (`jimeno2024financialchunking` said so in 2024). Being 1st
on a leaderboard is an incremental engineering result, and the vision detector — the
component that would have been novel — is null on three of four corpora.

**But there is one genuine, citable contribution**, and it is not the leaderboard:

> On identical chunks, the ranking of seven chunkers **inverts** depending on which
> relevance rule scores them. DocStruct is 1st of 7 under two rules and 6th of 7 under
> the third. No prior chunking evaluation we can find reports more than one rule.

This is a finding *about the field's method*, not about our tool. Its implication is
that a large body of published chunking comparisons — including the ones we cite — are
underdetermined by their own evaluation design. That is the kind of result that gets
cited by everyone who builds a benchmark afterwards, and it is true whether or not
DocStruct is any good.

**A second, smaller contribution**, currently undersold at §5.3: scoring RAG chunkers
against **publisher-authored JATS section boundaries** with Pk and WindowDiff. Pk and
WindowDiff are decades old and JATS is not a secret, but putting them together to get
retriever-free, tool-agnostic chunking gold appears to be new. It neatly sidesteps the
"who wrote your gold" objection that dogs every other number in the field.

**Honest verdict:** one solid methodological contribution, one neat evaluation idea, and
a competent system that wins where it should. That is a good workshop paper today, and a
credible full paper after §4 below. It is not a breakthrough, and claiming one would be
the fastest way to get desk-rejected.

---

## 3. Tier 1 — cheap, high payoff. Do these first.

### 3.1 ~~Prove determinism~~ — **DONE 2026-08-16**

**95/95** documents byte-identical across independent processes, **5,810 chunks per
run**, 0 differing. Now §5.5 of the paper and `reports/determinism.json`. The three
dense financial filings that blew a 30-minute cap finished under a 4-hour one and also
matched; that cost is reported in Limitations.

Unplanned cross-check: 5,810 is exactly the `n_chunks` recorded for `docstruct_geo` in
the OHR-Bench retrieval run days earlier, on different hardware through a different
code path.

Still open on this item: the **hybrid path is unverified** (no GPU on the dev machine;
CUDA kernel selection is not guaranteed bit-reproducible). The paper claims determinism
for `docstruct_geo` and says the hybrid case is untested rather than assuming it.

<details><summary>original entry</summary>

**The single largest hole.** Determinism is the first line of the contract, in the
abstract, in the title's spirit, and in the README — and there is *no experiment*.
What exists is `tests/test_golden_and_fuzz.py::test_parse_is_deterministic`: one PDF,
parsed twice, in one process. That is a unit test, not evidence for a paper claim.

Do this:

```bash
# 95 OHR-Bench docs, parsed twice in SEPARATE processes, chunk-boundary hashes compared
python scripts/verify_determinism.py --pdfs-dir data/ohrbench --runs 2
```

Report: *N* documents, *M* chunks, byte-identical chunk boundaries across independent
processes. Then a table row no competitor can fill, because LLM- and embedding-based
chunkers cannot.

**Test the GPU path too, and be ready for a bad answer.** YOLO inference on CUDA is not
guaranteed bit-reproducible across runs; cuDNN algorithm selection can vary. If the
hybrid path is *not* deterministic, that is a genuine finding and it must go in the
paper — it would mean the determinism guarantee holds for `docstruct_geo` and is
qualified for `docstruct`. Better to find that yourself than have a reviewer find it.

### 3.2 ~~Reframe~~ — **DONE 2026-08-16**

Abstract rewritten to lead with the finding; §6 retitled "Where the approach does
not pay"; Limitations merged into it so the paper stops apologising twice;
contributions reordered so the method finding is #1 and DocStruct is #3, "the
instrument". Original plan below.

<details><summary>original entry</summary>


- Rewrite the abstract to lead with what was found, not with the word *negative*.
  Current: *"Our central finding is methodological and negative."*
  Better: *"We find that the ranking of seven chunkers inverts depending on the
  relevance rule used to score them — on identical chunks."* Same fact, no cringe.
- Retitle §6 from "Negative and Cautionary Results" to something that states content,
  e.g. "Where the approach does not pay, and why". Keep every number.
- Merge §6 and §7 (Limitations). Right now the paper apologises twice.
- Promote §5.3 (section boundaries). It is the most defensible result in the paper and
  it is buried third.
- Cut the total self-criticism budget by roughly a third. Every caveat currently in
  there is *true*; the problem is the ratio, not the content.

</details>

### 3.3 ~~Sharpen the title and the claim~~ — **DONE 2026-08-16**

Title is now *The Relevance Rule Decides the Leaderboard: Seven PDF Chunkers,
Identical Chunks, Three Rankings*. Original reasoning below.

<details><summary>original entry</summary>


The current title sells the system. The contribution is the method finding. Consider:

> *Relevance Rules Decide Chunking Leaderboards: Evidence from Seven Chunkers on
> Identical Chunks* — with DocStruct as the vehicle rather than the headline.

This is the strategic call in the whole document. Framing the paper around the method
finding makes a modest system result into a supporting exhibit, instead of making the
method finding into a caveat on a modest system result. **Recommended.**

</details>

---

## 4. Tier 2 — real work, real payoff. Pick two.

### 4.1 A parse-fidelity number. (~1 day, CPU)

We claim structural superiority and never measure structure extraction directly.
READoc-arXiv (`li2025readoc`) is the cheapest credible option: gold derives from LaTeX
source, no human annotation needed, and our corpus is already arXiv-heavy. Report text
edit distance and vocabulary F1.

Expect this to be *uncomfortable*: coverage is 0.817 and duplication 2.06×, so a
fidelity metric may not flatter us. Run it anyway. A reviewer who works on parsing will
ask, and "we didn't measure it" is a worse answer than a mediocre number with an
explanation.

### 4.2 Fix the two known extraction defects, then re-measure. (~1 day + a sweep)

`to-do.md` items 6 and 7, found while building the README figure:

- full-width blocks extracted across the column gutter, interleaving columns
- full-width elements above a two-column body ordered after the columns

Both are real, both are reproducible, and the first one is *embarrassing in context*
— it is the exact failure the introduction attributes to naive parsers, occurring
inside our own pipeline. Fixing them plausibly improves the academic-domain slice,
which is currently our weakest (`span` 0.453, 5th of 7).

Repo rule 1 applies: measure with `scripts/ablate.py` before claiming anything.

### 4.3 Token-level IoU and Precision_Ω. (~1 GPU session)

The published vocabulary (`smith2024chunking`) for the cost/quality trade we currently
express with a homemade MRR-per-1k-words. Needs a re-run that dumps chunk and gold
token sets, not just overlap scores — the current dumps cannot support it.

Do this **only if** you also do 4.1; on its own it is a metric-cosmetics change.

### 4.4 A second retrieval corpus. (~1 GPU session + gold work)

The weakest structural part of the paper is that one human-gold retrieval corpus
carries every retrieval claim. FinanceBench is measured-unusable and that is defensible,
but "we tried two and one failed" is a harder sell than "we ran two successfully".

Cheapest real option: **MMLongBench-Doc** (`ma2024mmlongbenchdoc`), already in the
bibliography. Check gold reachability *first* (`scripts/gold_reachability.py`) — that
check has now saved this project twice, and a corpus that fails it costs a GPU session
to discover late.

---

## 5. Tier 3 — expensive. Only for a top-tier full-paper push.

- **DocLayNet val split for the detection layer.** The current mAP rests on two
  hand-annotated documents, which is indefensible if anyone asks. Also the only cheap
  route to calibrating the `# unvalidated` fusion constants. Large download, GPU eval.
- **Decide the vision detector's fate.** It is null on three of four corpora and
  *loses* to geometry-only on section boundaries at 2.4× the cost. Two honest options:
  (a) measure it on FinanceBench's borderless tables — the one place it should pay,
  122 detected tables against pdfplumber's 4 — and report a real answer; or (b) cut it
  from the paper's contributions and present DocStruct as a geometric chunker with an
  optional model path. **Option (b) is the stronger paper** and nobody wants to hear it.
- **Scanned-document support.** Out of scope; state as scope, not limitation.

---

## 6. Venue strategy

| Target | Needs | Realistic? |
|---|---|---|
| Workshop (DocEng, SDU, RAG workshops) | Tier 1 only | **Yes, ~1 week** |
| Short paper (ACL/EMNLP Findings) | Tier 1 + §3.3 reframe | Yes, ~2 weeks |
| Full paper (ECIR / CIKM / DocEng full) | Tier 1 + two of Tier 2 | ~4–6 weeks |
| Top-tier (SIGIR / ACL main) | All of the above + Tier 3 + a second corpus | Not with the current scope |

**Recommendation: reframe around the mode-inversion finding (§3.3), do Tier 1, add 4.1
and 4.2, and target a full paper at DocEng or ECIR.** That is a paper whose central
claim is true, novel, well-measured, and useful to other people — which is the actual
bar, and one this work clears.

---

## 7. What not to do

- **Do not claim a breakthrough.** The data does not support it and reviewers punish it
  harder than modesty.
- **Do not delete the negative results to look stronger.** They are the paper's
  character and its most unusual quality. Reframe the *register*, keep every number.
  In particular keep the FinanceBench section: reporting that a published paper gets
  MRR 0.700–0.844 where our protocol gets 0.0 on the same PDFs is a genuine service to
  the field.
- **Do not add corpora without running reachability first.** That check has caught two
  corpus-level errors already.
- **Do not let the vision detector stay in the contributions list unmeasured.** Either
  give it a corpus where it can prove itself, or demote it. Leaving it ambiguous invites
  the reviewer question you least want.
