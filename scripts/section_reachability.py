"""Can the JATS section boundaries be found in the PDF's own text at all?

Rule 7: a corpus gets its relevance rule checked before it gets a leaderboard.
The section-boundary metrics have the same exposure. Gold comes from the
publisher's XML and the chunks come from the PDF, and the two texts are never
byte-identical -- hyphenation, ligatures, table serialisation and figure captions
all differ. If a gold boundary cannot be located on the PDF's own token stream,
no chunker can be scored against it, and a Pk that quietly skipped a third of the
boundaries would look like a result.

This measures the ceiling: what fraction of gold section starts are findable in
raw pdfplumber text. It is identical for every tool, so it says whether the
corpus can be measured at all -- not how well anyone does.

    python scripts/section_reachability.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import collections
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.eval.segmentation import (  # noqa: E402
    _PROBE,
    cached_spine,
    locate_boundaries,
    spine_of,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/qa/pmc_sections.json")
    ap.add_argument("--pdfs-dir", default="data/pmc")
    ap.add_argument("--out", default="reports/section_reachability.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)

    docs = sorted(gold)
    if args.limit:
        docs = docs[: args.limit]

    per_doc = []
    cats: collections.Counter = collections.Counter()
    total = found = 0
    missing_pdf = 0

    for i, doc in enumerate(docs, 1):
        path = os.path.join(args.pdfs_dir, doc)
        if not os.path.exists(path):
            missing_pdf += 1
            continue
        spine = cached_spine(path)
        if len(spine) < 50:
            per_doc.append({"doc": doc, "n_sections": len(gold[doc]), "located": 0,
                            "pct": 0.0, "spine_chars": len(spine), "note": "no text layer"})
            total += len(gold[doc])
            continue
        secs = gold[doc]
        offsets = locate_boundaries(spine, [s["text"] for s in secs])
        n = len(offsets)
        ok = sum(1 for o in offsets if o is not None)
        total += n
        found += ok

        # A miss is not one thing, and averaging the three together understates the
        # usable ceiling. A section shorter than the probe cannot be located by
        # construction -- that is an artefact of the method, not a property of the
        # corpus -- and back matter is excluded from the metric anyway, since
        # DocStruct drops references by design.
        for s, o in zip(secs, offsets):
            short = len(spine_of(s["text"])) < _PROBE
            cat = ("back_matter" if s["in_back_matter"]
                   else "too_short" if short else "body")
            cats[f"{cat}_{'found' if o is not None else 'miss'}"] += 1

        per_doc.append({"doc": doc, "n_sections": n, "located": ok,
                        "pct": round(100 * ok / n, 1) if n else 0.0,
                        "spine_chars": len(spine)})
        if i % 25 == 0:
            print(f"  {i}/{len(docs)} docs", flush=True)

    if not total:
        print("nothing scored -- are the PDFs fetched?")
        return 1

    pcts = [d["pct"] for d in per_doc if d["n_sections"]]
    body = cats["body_found"] + cats["body_miss"]
    print(f"\n{len(per_doc)} documents, {total} gold section boundaries")
    print(f"  locatable in raw pdfplumber text: {found} ({100 * found / total:.1f}%)")
    print(f"  per-document: median {statistics.median(pcts):.1f}%, "
          f"worst {min(pcts):.1f}%, best {max(pcts):.1f}%")
    print("\n  by kind of section:")
    for kind in ("body", "too_short", "back_matter"):
        f_, m_ = cats[f"{kind}_found"], cats[f"{kind}_miss"]
        if f_ + m_:
            print(f"    {kind:12} {f_:5}/{f_ + m_:<5} {100 * f_ / (f_ + m_):5.1f}%")
    if body:
        print(f"\n  ** the usable ceiling is body prose: {cats['body_found']}/{body} "
              f"= {100 * cats['body_found'] / body:.1f}%")
        print(f"     Sections shorter than the {_PROBE}-character probe cannot be located by "
              f"construction;\n     back matter is out of scope because DocStruct drops "
              f"references by design.")
    if missing_pdf:
        print(f"  {missing_pdf} documents skipped (PDF absent)")

    worst = sorted((d for d in per_doc if d["n_sections"]), key=lambda d: d["pct"])[:5]
    print("\n  worst documents:")
    for d in worst:
        note = f"  [{d['note']}]" if d.get("note") else ""
        print(f"    {d['doc']:34} {d['located']:3}/{d['n_sections']:<3} {d['pct']:5.1f}%{note}")

    out = {"gold": args.gold, "n_docs": len(per_doc), "n_sections": total,
           "located": found, "located_pct": round(100 * found / total, 1),
           "by_kind": dict(cats),
           "body_ceiling_pct": round(100 * cats["body_found"] / body, 1) if body else None,
           "median_doc_pct": round(statistics.median(pcts), 1), "per_doc": per_doc}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
