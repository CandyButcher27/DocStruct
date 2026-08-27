# ACL 2023 layout contract

Template lives in `acl2023.zip` at the repo root: `acl2023.sty`, `acl_natbib.bst`,
`anthology.bib` (43 MB), `custom.bib`, and `acl2023.tex` as the worked example.
Unzip to a scratch dir; do not commit `anthology.bib` into the repo.

`paper/main.tex` is currently stock `article` with hand-rolled two-column geometry.
Porting is not a one-line class swap. The list below is the whole diff.

## Preamble

Delete and replace:

| Current line | Do |
|---|---|
| `\documentclass[10pt,twocolumn,letterpaper]{article}` | `\documentclass[11pt]{article}` |
| — | add `\usepackage[review]{ACL2023}` (switch `review` to `final` for camera-ready) |
| `\usepackage[margin=0.75in]{geometry}` | **delete.** The sty sets the geometry. Loading geometry fights it |
| `twocolumn` class option | **delete.** The sty makes the page two-column |
| `\usepackage[numbers,sort&compress]{natbib}` | **delete.** `ACL2023.sty` line 202 does `\RequirePackage{natbib}` itself and line 206 redefines `\cite` as `\citep`. Loading it again with `numbers` is an option clash |
| `\usepackage[hidelinks]{hyperref}` | **delete.** The sty loads hyperref by default with `breaklinks` |
| `\bibliographystyle{plainnat}` | `\bibliographystyle{acl_natbib}` |
| `\bibliography{refs}` | `\bibliography{anthology,refs}` once anthology entries are used |
| `\date{}` | **delete.** ACL's `\maketitle` has no date |

Add, in this order, before anything else:

```latex
\pdfoutput=1                      % MUST be in the first 5 lines, arXiv reads it
\documentclass[11pt]{article}
\usepackage[review]{ACL2023}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
```

Keep unchanged: `booktabs`, `graphicx` + `\graphicspath`, `amsmath`, `amssymb`,
`xcolor`, `multirow`, `url`, `caption`, and all five custom macros
(`\todo`, `\unverified`, `\ds`, `\dsg`, `\best`).

## Citations

Numeric style is gone. `acl_natbib` is author-year:

- `\citep{lewis2020rag}` renders `(Lewis et al., 2020)` — parenthetical.
- `\citet{lewis2020rag}` renders `Lewis et al. (2020)` — as a sentence subject.
- Bare `\cite` is aliased to `\citep`, so existing `\cite` calls are safe, but
  prose written for numeric citations reads wrong once the numbers become names.
  **Sweep every citation site after porting**: a sentence like "prior work [12]
  reports" becomes "prior work (Author et al., 2024) reports" and usually wants
  rewriting to `\citet`.

For any paper published at an *ACL venue (ACL, EMNLP, NAACL, EACL, COLING, TACL,
Findings), take the entry from `anthology.bib` rather than writing it by hand.
That is the canonical, author-list-verified entry, and it clears most of
`paper/REFERENCES_TO_VERIFY.md`. Non-ACL venues stay in `refs.bib`.

## Section order

ACL fixes the tail of the paper. Current order is wrong in three places.

Required order after the last numbered section:

```latex
\section{Conclusion}          % last numbered section
\section*{Limitations}        % MANDATORY, unnumbered
\section*{Ethics Statement}   % unnumbered
\section*{Acknowledgements}   % unnumbered, omit in review version
\bibliography{anthology,refs}
\appendix
\section{...}                 % appendix sections, numbered A, B, ...
```

**Limitations is mandatory. A submission without it is desk-rejected without
review** (`acl2023.tex` line 241). It must sit after the conclusion and before
the references, and it does not count toward the page limit.

Consequences for our draft:

- `\section{Where the approach does not pay}` currently sits numbered in the body
  as §6. Its five subsections are the Limitations section. Move it, unnumber it,
  drop the `\subsection` levels to `\paragraph` run-ins. Repo rule 9 still holds:
  the headline losses (coverage 0.9632, duplication, 6th of 7 under page
  relevance, the null vision detector) stay quoted in the Results body too. The
  Limitations section discusses them; it is not where the reader first meets them.
- `\section{Reproducibility}` is not an ACL section. Fold the corpus-fetch list
  and the determinism command into the appendix, and keep one sentence with the
  repository URL in the body.
- No Ethics Statement exists. Write one: public corpora only, no human subjects,
  no personal data, the only model use is gold generation in `eval/`, and the
  compute cost of the GPU runs.

## Page budget

ACL long paper: **8 pages of body**. Limitations, Ethics, Acknowledgements,
References and Appendix are all free.

Current PDF is 8 pages in `article` at 10pt with 0.75in margins, and the tail
comment admits the bibliography was squeezed with `\small` and
`\setlength{\bibsep}{2pt}` to land there. Under ACL both of those hacks are
unnecessary, because references cost nothing. **Delete the squeeze on port** —
a shrunk bibliography in an ACL submission reads as someone who missed the rule.

The port moves roughly 1.5 pages out of the body (negative results, reproducibility)
and 11pt with ACL's measure adds some back. Expect to land near the limit and need
one compression pass. Use `paper-writing-skill/author_profile/compression_patterns.md`
for it, not ad-hoc cutting.

## Verify the port

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
grep -c 'Overfull' main.log        # column overflows, fix each
grep -n 'Citation.*undefined' main.log
```

Body page count must be checked against where `\section*{Limitations}` starts,
not against the total page count of the PDF.
