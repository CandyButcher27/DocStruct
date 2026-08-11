"""How much of a corpus's gold is scoreable by each relevance rule?

Relevance mode is chosen per corpus, and choosing it wrong makes the leaderboard
lie (see CLAUDE.md rule 7). The choice rests on one number -- what fraction of the
gold spans can be found in the raw PDF text at all -- and that number has to be
measured against the rule the benchmark actually applies, not a simpler one.

This measures each rule independently on the gold's own evidence page, so it is a
ceiling: no chunker can score an item this finds unreachable, and the ceiling is
identical for every tool.

    python scripts/gold_reachability.py --gold data/qa/ohrbench.json --pdfs-dir data/ohrbench

`span` decomposes because the sub-rules disagree by an order of magnitude on
reflowed gold: plain substring is the pessimistic reading, whitespace-blind is
what `is_relevant` really does.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct import config  # noqa: E402
from docstruct.eval.relevance import (  # noqa: E402
    _despaced,
    _overlap_coefficient,
    _token_overlap,
    normalize_text,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="data/qa/ohrbench.json")
    p.add_argument("--pdfs-dir", default="data/ohrbench")
    p.add_argument("--out", default="reports/gold_reachability.json")
    args = p.parse_args()

    import pdfplumber

    gold = json.load(open(args.gold, encoding="utf-8"))
    by_doc = collections.defaultdict(list)
    for g in gold:
        by_doc[g["source_doc"]].append(g)

    counts = collections.Counter()
    overlaps = []
    ratios = []
    by_source = collections.defaultdict(collections.Counter)
    n = missing_pdf = bad_page = 0

    for d, (doc, items) in enumerate(sorted(by_doc.items()), 1):
        path = os.path.join(args.pdfs_dir, doc)
        if not os.path.exists(path):
            missing_pdf += len(items)
            continue
        # only the evidence pages: FinanceBench filings run to 250 pages against a
        # handful of gold rows, so extracting every page is ~50x the work needed
        want = {int(g["page_num"]) for g in items}
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            pages = {i: (pdf.pages[i].extract_text() or "") for i in want if 0 <= i < n_pages}
        for g in items:
            pn = int(g["page_num"])
            if pn not in pages:
                bad_page += 1
                continue
            text, span = pages[pn], g["answer_span"]
            n += 1
            plain = normalize_text(span) in normalize_text(text)
            desp = _despaced(span) in _despaced(text)
            tok = _token_overlap(span, text) >= config.RELEVANCE_MIN_OVERLAP
            region = _overlap_coefficient(span, text) >= config.RELEVANCE_REGION_MIN_OVERLAP
            counts["plain"] += plain
            counts["despaced"] += desp
            counts["token_fallback"] += tok
            counts["span"] += plain or desp or tok
            counts["region"] += region
            overlaps.append(_overlap_coefficient(span, text))
            page_tokens = len(set(normalize_text(text).split()))
            ratios.append(len(set(normalize_text(span).split())) / page_tokens if page_tokens else 0.0)
            src = g.get("evidence_source", "?")
            by_source[src]["n"] += 1
            by_source[src]["span"] += plain or desp or tok
        if d % 20 == 0:
            print(f"  {d}/{len(by_doc)} docs")

    if not n:
        print("no gold items scored -- are the PDFs fetched?")
        return 1

    pct = {k: round(100 * v / n, 1) for k, v in counts.items()}
    print(f"\n{n} gold items ({len(by_doc)} docs)"
          + (f"; {missing_pdf} skipped (pdf absent), {bad_page} bad page ids" if missing_pdf or bad_page else ""))
    print(f"  plain substring     : {pct['plain']:5.1f}%   <- the pessimistic reading")
    print(f"  whitespace-blind    : {pct['despaced']:5.1f}%")
    print(f"  token fallback      : {pct['token_fallback']:5.1f}%   (thr={config.RELEVANCE_MIN_OVERLAP})")
    print(f"  span (is_relevant)  : {pct['span']:5.1f}%   <- what the benchmark applies")
    print(f"  region              : {pct['region']:5.1f}%   (thr={config.RELEVANCE_REGION_MIN_OVERLAP}), "
          f"median overlap {statistics.median(overlaps):.3f}")
    print("\n  by evidence_source:")
    for src, c in sorted(by_source.items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {src:12} n={c['n']:5}  span-reachable {100*c['span']/c['n']:5.1f}%")

    # The reference here is the whole evidence page, so these rules are only a real
    # test when the gold is much smaller than the page. Once the gold *is* a page
    # region, both the token fallback and the overlap coefficient normalise by the
    # gold, every item scores ~1.0 by construction, and a 100% reading means the
    # question was circular -- not that the corpus is easy.
    gold_frac = statistics.median(ratios)
    print(f"\n  gold is {gold_frac:.0%} of its page (median)")
    if gold_frac > 0.2:
        print("  ** span/region reachability above is CIRCULAR at this gold size and is not\n"
              "     evidence that a chunker can score this corpus. Only the plain-substring\n"
              "     row is informative here; validate the region threshold against real\n"
              "     chunks instead.")

    out = {
        "gold": args.gold,
        "n_items": n,
        "n_docs": len(by_doc),
        "skipped_missing_pdf": missing_pdf,
        "skipped_bad_page": bad_page,
        "reachable_pct": pct,
        "region_median_overlap": round(statistics.median(overlaps), 4),
        "gold_frac_of_page_median": round(gold_frac, 4),
        "circular": gold_frac > 0.2,
        "by_evidence_source": {
            s: {"n": c["n"], "span_pct": round(100 * c["span"] / c["n"], 1)}
            for s, c in by_source.items()
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
