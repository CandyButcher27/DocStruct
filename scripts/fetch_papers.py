"""Fetch open-access research papers from non-arXiv publishers.

The internal corpus is arXiv-heavy, and `memory/results.md` records that the XY-cut
result flipped on corpus shape alone -- so "works on research papers" currently means
"works on one LaTeX template family". Publisher matters more than field here: MDPI,
IEEE, PLOS and Frontiers each run a different typesetting engine, so their column
geometry, header weights, table rules and back matter differ in exactly the ways the
detector and reading-order layers key off.

Sources are resolved through OpenAlex (free, no key, `mailto` for the polite pool)
rather than per-publisher scrapers: one API covers every venue, and the venue id is
looked up by name at runtime so nothing here rots when an id changes.

    python scripts/fetch_papers.py --per-venue 2          # smoke
    python scripts/fetch_papers.py --per-venue 15         # ~90 papers

Writes data/papers/<venue>__<openalex-id>.pdf and reports/papers_manifest.json.
Re-running skips anything already on disk, so it resumes after a killed job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

PDF_DIR = os.path.join("data", "papers")
MANIFEST = os.path.join("reports", "papers_manifest.json")
API = "https://api.openalex.org"
# OpenAlex's polite pool keys off a real, reachable address; example.org gets the
# same throttling as an anonymous caller. Set OPENALEX_MAILTO (or --mailto) to your
# own -- deliberately not hardcoded, so nobody's address ships in the repo.
MAILTO = os.environ.get("OPENALEX_MAILTO", "")

# venue -> short slug used in the filename. Chosen for typesetting diversity, not
# subject: each of these renders through a different template family.
VENUES = {
    "PLOS ONE": "plos",
    "Scientific Reports": "scirep",
    "IEEE Access": "ieee",           # IEEEtran two-column -- the biggest contrast with arXiv
    "PeerJ": "peerj",
    "Frontiers in Psychology": "frontiers",
    "eLife": "elife",
}
# MDPI (Sensors) was in this list and is not: every PDF fetch returns 403, it blocks
# non-browser agents. Not worth spoofing a User-Agent to get around a publisher's
# explicit refusal -- PeerJ fills the same "third template family" slot and serves.


def get(url: str, tries: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"docstruct-bench/0.1 (mailto:{MAILTO})"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            # OpenAlex 429s for minutes, not seconds; a 1-2-4-8 ladder exhausts itself
            # inside its cooldown and the run dies on a throttle it could have waited out.
            wait = 2**attempt
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                wait = max(int(e.headers.get("Retry-After") or 0), 15 * (attempt + 1))
            print(f"    {type(e).__name__}: {e} -- retry in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def resolve_source(name: str) -> str | None:
    time.sleep(1)  # OpenAlex 429s on back-to-back calls even in the polite pool
    q = urllib.parse.urlencode({"search": name, "per-page": 1, "mailto": MAILTO})
    hits = json.loads(get(f"{API}/sources?{q}"))["results"]
    return hits[0]["id"].rsplit("/", 1)[-1] if hits else None


def works(source_id: str, n: int, from_date: str) -> list[dict]:
    time.sleep(1)
    q = urllib.parse.urlencode(
        {
            "filter": ",".join(
                [
                    f"primary_location.source.id:{source_id}",
                    "open_access.is_oa:true",
                    "type:article",
                    f"from_publication_date:{from_date}",
                    "has_fulltext:true",
                ]
            ),
            # deterministic ordering, so a re-run with a bigger --per-venue is a superset
            "sort": "publication_date:desc",
            "per-page": n,
            "mailto": MAILTO,
        }
    )
    return json.loads(get(f"{API}/works?{q}"))["results"]


def main() -> int:
    global MAILTO
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-venue", type=int, default=15)
    ap.add_argument("--from-date", default="2023-01-01")
    ap.add_argument("--venues", default="", help="comma-separated subset of the venue names")
    ap.add_argument("--mailto", default=MAILTO, help="your address, for OpenAlex's polite pool")
    args = ap.parse_args()

    MAILTO = args.mailto
    if not MAILTO:
        print("no --mailto/OPENALEX_MAILTO: using the anonymous pool, expect 429s\n")

    wanted = VENUES
    if args.venues:
        names = [v.strip() for v in args.venues.split(",")]
        wanted = {k: v for k, v in VENUES.items() if k in names}

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    manifest = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else []
    have = {m["file"] for m in manifest}

    for name, slug in wanted.items():
        print(f"{name} ...", flush=True)
        # one venue's throttle used to kill the whole run and discard every venue
        # queued behind it. Each venue is independent; failure here is a skip.
        try:
            sid = resolve_source(name)
            # ask for a big multiple: most OA records point at a landing page, not a
            # PDF, and Nature-family records almost always do (measured 3/3)
            candidates = works(sid, min(args.per_venue * 8, 200), args.from_date) if sid else []
        except Exception as e:  # noqa: BLE001
            print(f"  OpenAlex query failed ({type(e).__name__}), venue skipped")
            continue
        if not sid:
            print("  no OpenAlex source, skipped")
            continue
        for w in candidates:
            got = sum(1 for m in manifest if m["venue"] == name)
            if got >= args.per_venue:
                break
            loc = w.get("best_oa_location") or {}
            url = loc.get("pdf_url")
            if not url:
                continue
            wid = w["id"].rsplit("/", 1)[-1]
            fname = f"{slug}__{wid}.pdf"
            dest = os.path.join(PDF_DIR, fname)
            if fname in have:
                continue
            if not os.path.exists(dest):
                try:
                    data = get(url, tries=2)
                except Exception as e:  # noqa: BLE001
                    print(f"  {wid}: {type(e).__name__}")
                    continue
                if not data.startswith(b"%PDF"):
                    print(f"  {wid}: not a PDF (landing page), skipped")
                    continue
                with open(dest, "wb") as f:
                    f.write(data)
                time.sleep(1)
            manifest.append(
                {
                    "file": fname,
                    "venue": name,
                    "openalex_id": wid,
                    "doi": w.get("doi"),
                    "title": re.sub(r"\s+", " ", w.get("title") or "")[:200],
                    "publication_date": w.get("publication_date"),
                    "license": loc.get("license"),
                    "source": url,
                    "sha256": hashlib.sha256(open(dest, "rb").read()).hexdigest(),
                }
            )
            have.add(fname)
            print(f"  {fname}  [{loc.get('license')}]")

        json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    by_venue: dict[str, int] = {}
    for m in manifest:
        by_venue[m["venue"]] = by_venue.get(m["venue"], 0) + 1
    print(f"\n{len(manifest)} papers -> {MANIFEST}")
    for v, c in sorted(by_venue.items()):
        print(f"  {v:28} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
