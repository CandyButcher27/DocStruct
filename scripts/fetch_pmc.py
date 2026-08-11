"""Fetch open-access papers from Europe PMC -- PDF *and* the publisher's JATS XML.

Two problems, one source.

Breadth: the internal corpus is arXiv-only, so "works on research papers" currently
means "works on one LaTeX template family". Publisher matters more than field --
PLOS, Frontiers, PeerJ, BMC and the Nature family each run a different typesetting
engine, so column geometry, header weights, table rules and back matter differ in
exactly the places the detector and reading-order layers key off.

Gold: every article here also serves `fullTextXML`, the publisher's own JATS. It
carries the real section hierarchy, titles and table markup, authored by the
publisher and never touched by a chunker. That satisfies the tool-agnostic rule at
its strongest -- the gold is not merely not-ours, it predates the benchmark -- and
it is the only route to scoring section paths as a metric rather than a claim.

Europe PMC needs no key and does not throttle the way OpenAlex's anonymous pool does.

    python scripts/fetch_pmc.py --per-journal 2      # smoke
    python scripts/fetch_pmc.py --per-journal 20     # ~140 papers

Writes data/pmc/<slug>__<PMCID>.pdf, the matching .xml, and reports/pmc_manifest.json.
Re-running skips what is already on disk, so it resumes after a killed job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

PDF_DIR = os.path.join("data", "pmc")
MANIFEST = os.path.join("reports", "pmc_manifest.json")
SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
PDF = "https://europepmc.org/articles/{pmcid}?pdf=render"

# journal (as Europe PMC indexes it) -> filename slug. Picked for typesetting
# diversity, not subject.
JOURNALS = {
    "PLOS ONE": "plos",
    "Scientific Reports": "scirep",
    "PeerJ": "peerj",
    "Frontiers in Psychology": "frontiers",
    "eLife": "elife",
    "BMC Bioinformatics": "bmc",
    "Nature Communications": "natcomm",
}


def get(url: str, tries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "docstruct-bench/0.1"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 2**attempt
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                wait = max(int(e.headers.get("Retry-After") or 0), 10 * (attempt + 1))
            print(f"    {type(e).__name__}: {e} -- retry in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def search(journal: str, n: int, from_date: str, to_date: str) -> list[dict]:
    query = (
        f'(JOURNAL:"{journal}") AND (OPEN_ACCESS:Y) AND (HAS_FT:Y) '
        f"AND (FIRST_PDATE:[{from_date} TO {to_date}])"
    )
    q = urllib.parse.urlencode(
        {"query": query, "format": "json", "pageSize": min(n, 100), "resultType": "core"}
    )
    return json.loads(get(f"{SEARCH}?{q}"))["resultList"]["result"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-journal", type=int, default=20)
    ap.add_argument("--from-date", default="2023-01-01")
    ap.add_argument("--to-date", default="2026-01-01")
    ap.add_argument("--journals", default="", help="comma-separated subset of the journal names")
    args = ap.parse_args()

    wanted = JOURNALS
    if args.journals:
        names = [j.strip() for j in args.journals.split(",")]
        wanted = {k: v for k, v in JOURNALS.items() if k in names}

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    manifest = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else []
    have = {m["file"] for m in manifest}

    for journal, slug in wanted.items():
        print(f"{journal} ...", flush=True)
        # one journal's failure is a skip, not the end of the run
        try:
            # ask for extra: some hits have no rendered PDF despite HAS_FT
            hits = search(journal, args.per_journal * 2, args.from_date, args.to_date)
        except Exception as e:  # noqa: BLE001
            print(f"  search failed ({type(e).__name__}), journal skipped")
            continue

        kept = sum(1 for m in manifest if m["journal"] == journal)
        for r in hits:
            if kept >= args.per_journal:
                break
            pmcid = r.get("pmcid")
            if not pmcid:
                continue
            fname = f"{slug}__{pmcid}.pdf"
            if fname in have:
                kept += 1
                continue
            pdf_path = os.path.join(PDF_DIR, fname)
            xml_path = pdf_path[:-4] + ".xml"
            try:
                if not os.path.exists(pdf_path):
                    data = get(PDF.format(pmcid=pmcid), tries=2)
                    if not data.startswith(b"%PDF"):
                        print(f"  {pmcid}: no rendered PDF, skipped")
                        continue
                    with open(pdf_path, "wb") as f:
                        f.write(data)
                if not os.path.exists(xml_path):
                    # the PDF alone is just another corpus; the XML is the reason
                    # this source was chosen, so an article without it is not kept
                    with open(xml_path, "wb") as f:
                        f.write(get(XML.format(pmcid=pmcid), tries=2))
            except Exception as e:  # noqa: BLE001
                print(f"  {pmcid}: {type(e).__name__}, skipped")
                for p in (pdf_path, xml_path):
                    if os.path.exists(p) and not os.path.getsize(p):
                        os.remove(p)
                continue

            manifest.append(
                {
                    "file": fname,
                    "xml": os.path.basename(xml_path),
                    "journal": journal,
                    "pmcid": pmcid,
                    "doi": r.get("doi"),
                    "title": (r.get("title") or "")[:200],
                    "pub_date": r.get("firstPublicationDate"),
                    "license": r.get("license"),
                    "source": PDF.format(pmcid=pmcid),
                    "sha256": hashlib.sha256(open(pdf_path, "rb").read()).hexdigest(),
                }
            )
            have.add(fname)
            kept += 1
            print(f"  {fname}  [{r.get('license')}]  xml {os.path.getsize(xml_path)//1024} KB")
            time.sleep(0.5)

        json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    by: dict[str, int] = {}
    for m in manifest:
        by[m["journal"]] = by.get(m["journal"], 0) + 1
    print(f"\n{len(manifest)} papers -> {MANIFEST}")
    for j, c in sorted(by.items()):
        print(f"  {j:28} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
