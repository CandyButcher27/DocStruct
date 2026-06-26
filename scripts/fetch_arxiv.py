"""Fetch born-digital arXiv PDFs to grow the benchmark dataset (reproducible).

Downloads recent papers across several categories into data/raw-pdfs/ (gitignored)
continuing the docN.pdf numbering, and records provenance to
reports/dataset_manifest.json (tracked) so the set can be reproduced.

Usage: python scripts/fetch_arxiv.py
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

RAW_DIR = os.path.join("data", "raw-pdfs")
MANIFEST = os.path.join("reports", "dataset_manifest.json")
ATOM = "{http://www.w3.org/2005/Atom}"

# category -> how many papers
PLAN = {
    "cs.CL": 6,
    "cs.LG": 5,
    "stat.ML": 4,
    "q-bio.NC": 4,
    "econ.EM": 4,
    "math.OC": 3,
}


def _next_index() -> int:
    nums = [
        int(m.group(1))
        for f in os.listdir(RAW_DIR)
        if (m := re.match(r"doc(\d+)\.pdf$", f))
    ]
    return max(nums, default=0) + 1


def _query(category: str, n: int):
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=cat:{category}&start=0&max_results={n}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = entry.find(f"{ATOM}id").text.strip().split("/abs/")[-1]
        title = " ".join(entry.find(f"{ATOM}title").text.split())
        pdf = None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf = link.get("href")
        out.append({"arxiv_id": arxiv_id, "title": title, "pdf_url": pdf or f"http://arxiv.org/pdf/{arxiv_id}"})
    return out


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    idx = _next_index()
    manifest = []
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST, encoding="utf-8"))

    for category, n in PLAN.items():
        print(f"querying {category} ({n}) ...", flush=True)
        try:
            entries = _query(category, n)
        except Exception as err:  # noqa: BLE001
            print(f"  query failed: {err}")
            continue
        for e in entries:
            name = f"doc{idx}.pdf"
            dest = os.path.join(RAW_DIR, name)
            try:
                req = urllib.request.Request(e["pdf_url"], headers={"User-Agent": "docstruct-bench/0.1"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if not data.startswith(b"%PDF"):
                    print(f"  skip {e['arxiv_id']} (not a PDF)")
                    continue
                with open(dest, "wb") as fh:
                    fh.write(data)
                manifest.append({"file": name, "category": category, **e})
                print(f"  {name} <- {e['arxiv_id']} ({len(data)//1024} KB)")
                idx += 1
            except Exception as err:  # noqa: BLE001
                print(f"  download failed {e['arxiv_id']}: {err}")
            time.sleep(3)  # be polite to arXiv

    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nmanifest: {len(manifest)} entries -> {MANIFEST}")


if __name__ == "__main__":
    main()
