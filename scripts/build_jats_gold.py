"""Turn publishers' JATS XML into section-hierarchy gold.

Every chunk DocStruct emits carries a `SectionPath` (h1/h2/h3), and no competitor
in the leaderboard exposes anything comparable. That has only ever been claimed,
never scored, because scoring it needs to know which section a piece of text
*really* belongs to -- and deriving that from our own parse would be marking our
own exam.

Europe PMC serves the publisher's JATS next to the PDF. Its `<sec>` nesting and
`<title>` elements are the document's real hierarchy, authored by the publisher
before this benchmark existed. That is the strongest form of the tool-agnostic
rule: the gold does not merely come from someone else, it predates the question.

    python scripts/build_jats_gold.py                       # all of data/pmc
    python scripts/build_jats_gold.py --limit 3 --verbose

Writes data/qa/pmc_sections.json: one record per section, with the h1/h2/h3 path
it sits at, its own text (excluding nested subsections), and flags for the parts
DocStruct deliberately treats differently.

**Depth is truncated to three levels on purpose.** `schema.SectionPath` holds
h1/h2/h3 and nothing deeper, so gold at depth 4+ could never be matched and would
score as a loss that no configuration could fix. Sections deeper than three are
recorded at depth 3 with `truncated: true`, which keeps them scoreable and keeps
the count of them visible -- that count is the evidence for or against ever
raising the limit (see decisions.md, SectionPath depth > 3).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

try:
    # These files are third-party documents fetched over the network, and JATS
    # carries a DTD reference, so entity-expansion attacks are in scope even though
    # the source is reputable. defusedxml is already in the environment; the
    # fallback keeps the script runnable where it is not, since stdlib ElementTree
    # refuses undefined external entities anyway and only the billion-laughs class
    # is left open.
    from defusedxml.ElementTree import parse as _parse
except ImportError:  # pragma: no cover
    _parse = ET.parse

XML_DIR = os.path.join("data", "pmc")
OUT = os.path.join("data", "qa", "pmc_sections.json")

# JATS wrappers whose text is not body prose. Tables and figures are pulled out
# rather than deleted: they are content DocStruct emits as its own chunk types, so
# a metric that silently dropped them would flatter every tool that drops them too.
_NON_PROSE = {"table-wrap", "fig", "disp-formula", "supplementary-material"}


def _text_of(el: ET.Element, skip_nested_sec: bool = True) -> str:
    """Visible text of an element, minus nested sections and non-prose blocks."""
    parts = []
    for child in el:
        tag = child.tag.split("}")[-1]
        if tag == "title" or (skip_nested_sec and tag == "sec") or tag in _NON_PROSE:
            continue
        parts.append("".join(child.itertext()))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _title_of(sec: ET.Element) -> str:
    t = sec.find("title")
    return re.sub(r"\s+", " ", "".join(t.itertext())).strip() if t is not None else ""


def _walk(sec: ET.Element, path: list[str], out: list[dict], counts: collections.Counter,
          in_back: bool) -> None:
    title = _title_of(sec)
    full_path = path + [title]
    counts["depth_%d" % min(len(full_path), 9)] += 1
    truncated = len(full_path) > 3
    if truncated:
        counts["deeper_than_3"] += 1

    text = _text_of(sec)
    if text:
        # h1/h2/h3 only -- see the module docstring. A depth-4 section keeps its own
        # title at h3 rather than its grandparent's: the leaf is what a chunk of its
        # text would most plausibly be labelled with.
        levels = full_path if not truncated else [full_path[0], full_path[1], full_path[-1]]
        out.append({
            "h1": levels[0] or None,
            "h2": levels[1] if len(levels) > 1 else None,
            "h3": levels[2] if len(levels) > 2 else None,
            "text": text,
            "n_words": len(text.split()),
            "depth": len(full_path),
            "truncated": truncated,
            # DocStruct drops references and labels abstracts as their own chunk
            # type, so a scorer that ignores this would charge us for a design choice
            # rather than an error.
            "in_back_matter": in_back,
        })

    for child in sec.findall("sec"):
        _walk(child, full_path, out, counts, in_back)


def _body_words(xml_path: str) -> int:
    try:
        body = _parse(xml_path).getroot().find(".//body")
    except ET.ParseError:
        return -1
    return 0 if body is None else len(re.sub(r"\s+", " ", "".join(body.itertext())).split())


def sections_of(xml_path: str) -> tuple[list[dict], collections.Counter]:
    root = _parse(xml_path).getroot()
    out: list[dict] = []
    counts: collections.Counter = collections.Counter()

    abstract = root.find(".//front//abstract")
    if abstract is not None:
        text = _text_of(abstract, skip_nested_sec=False)
        if text:
            out.append({"h1": "Abstract", "h2": None, "h3": None, "text": text,
                        "n_words": len(text.split()), "depth": 1,
                        "truncated": False, "in_back_matter": False})

    for parent, in_back in ((root.find(".//body"), False), (root.find(".//back"), True)):
        if parent is None:
            continue
        for sec in parent.findall("sec"):
            _walk(sec, [], out, counts, in_back)

    return out, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", default=XML_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no XML in {args.xml_dir} -- run scripts/fetch_pmc.py first")
        return 1

    gold: dict[str, list[dict]] = {}
    totals: collections.Counter = collections.Counter()
    empty: list[tuple[str, int]] = []

    for p in paths:
        doc = os.path.basename(p)[:-4] + ".pdf"
        try:
            secs, counts = sections_of(p)
        except ET.ParseError as e:
            print(f"  {doc}: unparseable XML ({e})")
            totals["unparseable"] += 1
            continue
        if not secs:
            # Distinguish "the parser found nothing" from "there is nothing": these
            # are stub deposits (corrections, abstract-only records) whose <body>
            # holds a handful of <p> and no <sec> at all. Printing the word count
            # makes that visible instead of reading as 7 silent failures.
            empty.append((doc, _body_words(p)))
            continue
        gold[doc] = secs
        totals.update(counts)
        totals["sections"] += len(secs)
        totals["words"] += sum(s["n_words"] for s in secs)
        if args.verbose:
            print(f"\n{doc}: {len(secs)} sections")
            for s in secs[:10]:
                # print the whole path, not the leaf: indentation alone reads as if
                # every subsection belonged to the heading printed above it
                path = " > ".join(x for x in (s["h1"], s["h2"], s["h3"]) if x)
                print(f"    {path}  ({s['n_words']}w)")

    n_docs = len(gold)
    print(f"\n{n_docs} documents, {totals['sections']} sections, "
          f"{totals['words']:,} words of gold")
    if n_docs:
        print(f"  median sections/doc: "
              f"{sorted(len(v) for v in gold.values())[n_docs // 2]}")
    print("  depth histogram: " + ", ".join(
        f"{k.split('_')[1]}:{v}" for k, v in sorted(totals.items()) if k.startswith("depth_")))
    deep = totals["deeper_than_3"]
    print(f"  deeper than h3: {deep}"
          + (f"  ({100 * deep / totals['sections']:.1f}% -- recorded at depth 3, "
             f"flagged `truncated`)" if totals["sections"] else ""))
    back = sum(1 for v in gold.values() for s in v if s["in_back_matter"])
    print(f"  back matter: {back} sections (DocStruct drops references by design)")
    if empty:
        print(f"  {len(empty)} documents have no <sec> structure at all "
              f"(stub deposits -- body word counts "
              f"{sorted(w for _, w in empty)}); excluded from the gold")
    if totals["unparseable"]:
        print(f"  {totals['unparseable']} unparseable")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
