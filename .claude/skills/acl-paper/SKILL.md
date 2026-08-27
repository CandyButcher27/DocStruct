---
name: acl-paper
description: Write, port or revise the DocStruct paper in paper/ for ACL submission. Layers the ACL 2023 layout contract, the DocStruct claim ledger, and this repo's reporting rules on top of the generic paper-writing skills. Use whenever editing paper/main.tex, paper/refs.bib, an abstract, a section draft, a rebuttal, or when quoting any benchmark number in prose.
---

# acl-paper — DocStruct's paper contract

This skill owns only what a generic paper skill cannot know: the venue's layout,
this project's evidence, and the reporting rules the repo committed to. Prose
craft and the writing pipeline belong to the parent skills. Do not restate them.

## Load order (non-negotiable)

1. `~/.claude/skills/paper-writing-skill/author_profile/gate_mechanical.md`
   — the greppable style audit. Run it on every `.tex` edit.
2. `~/.claude/skills/claude-latex-paper-skill/references/style.md`
   — de-AI prose rules and the rewrite safety zones.
3. Then the overrides in this file, which win where they conflict.
4. `references/acl2023-layout.md` when touching the preamble or section order.
5. `references/claim-ledger.md` before writing any number into prose.

**Report grep counts, not "audited".** A mental pass is not a run.

## Overrides on the parent gates

The parents are tuned for systems and ML papers. Three of their rules are wrong
for a measurement paper and are amended here.

**M11 passive voice: BANNED -> AUDIT.** Measurement prose has a legitimate
agentless passive when the actor is the protocol and naming it adds nothing:
"5,810 chunks were byte-identical across two processes". Keep passive only when
the agent is the experiment itself. Every other hit is still fixed.

**M2 antithesis: BANNED, with one whitelist.** The ban holds everywhere except
the paper's central claim, where the negation carries the finding rather than
decorating it: the ranking inverts, the chunks do not. The title, one abstract
sentence and one topic sentence per relevant subsection may use it. Everywhere
else, apply the parent's keep-vs-cut test and assert the positive. Note that
`measured, not asserted` is the parent's own worked example of a rhetorical
negation to cut, and it currently appears in our contributions list.

**style.md "prose over lists": AMENDED.** The contributions `enumerate` stays;
its items are genuinely enumerable. Strip the `\textbf{}` lead-in from each
item. A bolded phrase opening every bullet is the tell, not the list itself.

## Reporting rules this repo committed to

These come from `CLAUDE.md` rules 7 and 9 and are not negotiable in prose.

**No leaderboard number without its relevance mode.** Any MRR, NDCG, Recall or
Hit@1 from OHR-Bench is meaningless unless the sentence, the table column or the
caption names `span`, `page` or `region`. A number carried across modes is a
wrong number. See `memory/relevance-modes.md`.

**Losses go in the main text.** Coverage 0.9632 against LangChain's 1.00, the
highest duplication of the seven, 6th of 7 under page relevance, the vision
detector null on every corpus but arXiv, no parse-fidelity number, born-digital
only. These belong in the body and in Limitations, never buried in an appendix.
`memory/related-work.md` keeps the list of who beats us where.

**Name the N when it moved.** `unstructured` hard-fails on 26% of PMC PDFs, so
its section row covers 99 of 134 documents. Any table row with a different N
says so in the row, not in a footnote.

**Say which detector produced a number.** The hybrid (`docstruct`) and the
geometry-only variant (`docstruct_geo`) are different systems with different
results, and on two of our own metrics the geometry-only variant wins. Prose
that says "DocStruct" where the evidence file says `docstruct_geo` is wrong.

## Before reporting an edit done

- Mechanical gate run, counts reported.
- Every new number traced to `references/claim-ledger.md`, or shipped as
  `\unverified{}`. Never polish an unsupported claim into a supported-sounding one.
- Every OHR number carries its mode.
- `\todo{}` and `\unverified{}` counts reported, not silently left.
- `pdflatex` + `bibtex` clean, or the errors quoted.
