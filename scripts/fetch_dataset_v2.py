"""Fetch ~200 born-digital PDFs across 7 domains for the v2 benchmark dataset.

Domains and sources:
  arxiv      -- academic papers (arXiv API, 10 diverse categories)
  legal      -- US court opinions (CourtListener REST API, free)
  financial  -- curated direct PDF URLs (investor-relations annual reports)
  medical    -- PubMed Central open-access articles (PMC OA FTP / API)
  technical  -- NIST docs, RFCs, hardware datasheets, open project manuals
  govt       -- Federal Register, WHO, NIST policy PDFs
  textbook   -- OpenStax CC-licensed textbooks

Usage:
  python scripts/fetch_dataset_v2.py                     # all domains
  python scripts/fetch_dataset_v2.py --domain arxiv      # one domain
  python scripts/fetch_dataset_v2.py --dry-run           # plan only
  python scripts/fetch_dataset_v2.py --smoke             # 1 doc per domain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

MANIFEST_V2 = os.path.join("reports", "dataset_manifest_v2.json")
ATOM = "{http://www.w3.org/2005/Atom}"

DOMAIN_TARGETS = {
    "arxiv":     50,
    "legal":     35,
    "financial": 35,
    "medical":   30,
    "technical": 25,
    "govt":      15,
    "textbook":  10,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _next_index(raw_dir: str) -> int:
    if not os.path.isdir(raw_dir):
        return 1
    nums = [int(m.group(1)) for f in os.listdir(raw_dir) if (m := re.match(r"doc(\d+)\.pdf$", f))]
    return max(nums, default=0) + 1


def _load_manifest() -> list:
    if os.path.exists(MANIFEST_V2):
        with open(MANIFEST_V2, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def _save_manifest(manifest: list) -> None:
    os.makedirs(os.path.dirname(MANIFEST_V2), exist_ok=True)
    with open(MANIFEST_V2, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def _download(url: str, dest: str, delay: float = 2.0) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "docstruct-bench/0.2 (research)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            print(f"    not a PDF ({len(data)} bytes) — skip")
            return None
        with open(dest, "wb") as fh:
            fh.write(data)
        time.sleep(delay)
        return data
    except Exception as err:
        print(f"    download failed: {err}")
        return None


# ── arxiv ─────────────────────────────────────────────────────────────────────

ARXIV_PLAN = {
    "cs.CL": 8, "cs.LG": 7, "cs.CV": 6, "stat.ML": 5,
    "q-bio.NC": 5, "econ.EM": 5, "physics.optics": 4,
    "math.OC": 4, "cs.RO": 3, "astro-ph.CO": 3,
}


def _arxiv_entries(category: str, n: int) -> list:
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
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        out.append({"arxiv_id": arxiv_id, "title": title, "pdf_url": pdf_url})
    return out


def fetch_arxiv(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[arxiv] target={n}")
    done = 0
    seen = {e.get("arxiv_id") for e in manifest}
    for category, count in ARXIV_PLAN.items():
        if done >= n:
            break
        print(f"  querying {category} ({count})...")
        try:
            entries = _arxiv_entries(category, count)
        except Exception as err:
            print(f"  query failed: {err}")
            continue
        for e in entries:
            if done >= n:
                break
            if e["arxiv_id"] in seen:
                print(f"  skip {e['arxiv_id']} (dup)")
                continue
            name = f"doc{idx}.pdf"
            print(f"  {name} <- {e['arxiv_id']} [{category}] {e['title'][:50]}")
            if not dry_run:
                data = _download(e["pdf_url"], os.path.join(raw_dir, name), delay=3)
                if data is None:
                    continue
                manifest.append({"file": name, "domain": "arxiv", "source": "arxiv.org",
                                  "arxiv_id": e["arxiv_id"], "title": e["title"],
                                  "category": category, "sha256": _sha256(data)})
                _save_manifest(manifest)
            seen.add(e["arxiv_id"])
            idx += 1
            done += 1
    return idx


# ── legal ─────────────────────────────────────────────────────────────────────
# CourtListener REST API v4 — precedential US federal opinions with PDF

CL_SEARCH = (
    "https://www.courtlistener.com/api/rest/v4/search/"
    "?type=o&order_by=score+desc&precedential_status=Published"
    "&filed_after=2020-01-01&filed_before=2024-01-01"
)


def fetch_legal(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[legal] target={n}")
    done = 0
    cursor = None
    while done < n:
        url = CL_SEARCH + ("&cursor=" + urllib.request.quote(cursor) if cursor else "")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docstruct-bench/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read())
        except Exception as err:
            print(f"  API failed: {err}")
            break

        results = payload.get("results", [])
        if not results:
            break

        for op in results:
            if done >= n:
                break
            # v4 returns cluster with opinions list; grab first opinion's PDF
            opinions = op.get("opinions", [])
            pdf_url = None
            for o in opinions:
                if o.get("download_url", "").endswith(".pdf"):
                    pdf_url = o["download_url"]
                    break
            if not pdf_url:
                continue
            case_name = op.get("case_name", "unknown")[:60]
            court = op.get("court_id", "")
            name = f"doc{idx}.pdf"
            print(f"  {name} <- {case_name} [{court}]")
            if not dry_run:
                data = _download(pdf_url, os.path.join(raw_dir, name), delay=1.5)
                if data is None:
                    continue
                manifest.append({"file": name, "domain": "legal", "source": "courtlistener.com",
                                  "case_name": case_name, "court": court, "sha256": _sha256(data)})
                _save_manifest(manifest)
            idx += 1
            done += 1

        cursor = payload.get("next")
        if not cursor:
            break
        time.sleep(1)
    return idx


# ── financial ─────────────────────────────────────────────────────────────────
# Direct PDF annual reports from investor relations pages — reliable born-digital
# EDGAR 10-Ks are HTML-primary; these are the actual PDF annual reports

FINANCIAL_PDFS = [
    # Technology
    ("https://s2.q4cdn.com/470004039/files/doc_earnings/2023/ar/2023-annual-report.pdf", "Apple 2023 Annual Report", "technology"),
    ("https://s2.q4cdn.com/240708110/files/doc_financials/2023/ar/msft-20230630.pdf", "Microsoft FY2023 Annual Report", "technology"),
    ("https://abc.xyz/assets/annual-reports/alphabet-annual-report-2023.pdf", "Alphabet 2023 Annual Report", "technology"),
    ("https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/amzn-20231231.pdf", "Amazon 2023 Annual Report", "technology"),
    ("https://investor.fb.com/annual-report/2022/doc/meta-2022-annual-report.pdf", "Meta 2022 Annual Report", "technology"),
    # Pharmaceutical
    ("https://www.pfizer.com/sites/default/files/investors/financial_reports/annual_reports/2023/index.html", "Pfizer 2023 AR", "pharma"),  # HTML, skip
    ("https://s21.q4cdn.com/834003165/files/doc_financials/2023/ar/jnj-20231231.pdf", "J&J 2023 Annual Report", "pharma"),
    ("https://www.abbvie.com/content/dam/abbvie-dotcom/uploads/PDFs/investors/2023-annual-report.pdf", "AbbVie 2023 Annual Report", "pharma"),
    # Energy
    ("https://corporate.exxonmobil.com/-/media/Global/Files/investor-relations/annual-report/ExxonMobil-2023-Annual-Report.pdf", "ExxonMobil 2023 Annual Report", "energy"),
    ("https://chevroncorp.gcs-web.com/static-files/a2d0cc3c-5b93-4b1e-b7e4-6d3be5b4b745", "Chevron 2023 Annual Report", "energy"),
    # Financial services
    ("https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2023.pdf", "JPMorgan 2023 Annual Report", "finance"),
    ("https://www.berkshirehathaway.com/2022ar/2022ar.pdf", "Berkshire 2022 Annual Report", "finance"),
    ("https://s23.q4cdn.com/407969754/files/doc_financials/2023/ar/gs-2023-annual-report.pdf", "Goldman Sachs 2023 Annual Report", "finance"),
    # Healthcare
    ("https://s21.q4cdn.com/834003165/files/doc_financials/2023/ar/unh-20231231.pdf", "UnitedHealth 2023 Annual Report", "healthcare"),
    # Retail/Consumer
    ("https://s2.q4cdn.com/056532643/files/doc_financials/2024/ar/wmt-20240131.pdf", "Walmart FY2024 Annual Report", "retail"),
    ("https://s2.q4cdn.com/710621382/files/doc_financials/2023/ar/tsla-20231231.pdf", "Tesla 2023 Annual Report", "auto"),
    # Semiconductor
    ("https://s21.q4cdn.com/463672886/files/doc_financials/2023/ar/nvda-20240128.pdf", "NVIDIA FY2024 Annual Report", "semiconductor"),
    ("https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2023-intel-annual-report.pdf", "Intel 2023 Annual Report", "semiconductor"),
    # Manufacturing
    ("https://s1.q4cdn.com/714491698/files/doc_financials/2023/ar/cat-2023-annual-report.pdf", "Caterpillar 2023 Annual Report", "manufacturing"),
    ("https://s22.q4cdn.com/910638702/files/doc_financials/2023/ar/ge-2023-annual-report.pdf", "GE 2023 Annual Report", "manufacturing"),
]

FINANCIAL_PDFS_CLEAN = [(u, t, s) for u, t, s in FINANCIAL_PDFS if not u.endswith(".html")]


def fetch_financial(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[financial] target={n}")
    done = 0
    for url, title, sector in FINANCIAL_PDFS_CLEAN[:n]:
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        print(f"  {name} <- {title} [{sector}]")
        if not dry_run:
            data = _download(url, os.path.join(raw_dir, name), delay=2)
            if data is None:
                print(f"    NOTE: if URL is dead, manually download from investor-relations page")
                continue
            manifest.append({"file": name, "domain": "financial", "source": url,
                              "title": title, "sector": sector, "sha256": _sha256(data)})
            _save_manifest(manifest)
        idx += 1
        done += 1
    if done < n:
        print(f"  [financial] only {done}/{n} downloaded — supplement manually from:")
        print(f"  https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K")
    return idx


# ── medical ───────────────────────────────────────────────────────────────────
# PubMed Central Open Access — free, varied medical/bio content

PMC_SEARCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    "?db=pmc&retmode=json&retmax={n}&term={query}+AND+open+access[filter]"
)
PMC_FETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pmc&rettype=pdf&retmode=file&id={pmc_id}"
)

MEDICAL_QUERIES = [
    "clinical trial randomized controlled",
    "systematic review meta-analysis",
    "cardiovascular disease treatment outcomes",
    "cancer immunotherapy response",
    "machine learning medical imaging diagnosis",
    "COVID-19 long-term outcomes",
]


def fetch_medical(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[medical] target={n}")
    done = 0
    seen = {e.get("pmc_id") for e in manifest if e.get("domain") == "medical"}
    per_query = max(1, (n // len(MEDICAL_QUERIES)) + 2)

    for query in MEDICAL_QUERIES:
        if done >= n:
            break
        search_url = PMC_SEARCH.format(n=per_query, query=urllib.request.quote(query))
        try:
            req = urllib.request.Request(search_url, headers={"User-Agent": "docstruct-bench/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as err:
            print(f"  PMC search failed: {err}")
            continue

        ids = data.get("esearchresult", {}).get("idlist", [])
        for pmc_id in ids:
            if done >= n:
                break
            if pmc_id in seen:
                continue
            pdf_url = PMC_FETCH.format(pmc_id=pmc_id)
            name = f"doc{idx}.pdf"
            print(f"  {name} <- PMC{pmc_id} [{query[:35]}]")
            if not dry_run:
                dl = _download(pdf_url, os.path.join(raw_dir, name), delay=1)
                if dl is None:
                    continue
                manifest.append({"file": name, "domain": "medical", "source": "pubmedcentral",
                                  "pmc_id": pmc_id, "query": query, "sha256": _sha256(dl)})
                _save_manifest(manifest)
            seen.add(pmc_id)
            idx += 1
            done += 1
        time.sleep(0.5)
    return idx


# ── technical ─────────────────────────────────────────────────────────────────

TECHNICAL_PDFS = [
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc9110.txt.pdf", "RFC 9110 HTTP Semantics"),
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc8446.txt.pdf", "RFC 8446 TLS 1.3"),
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc7540.txt.pdf", "RFC 7540 HTTP/2"),
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc9000.txt.pdf", "RFC 9000 QUIC"),
    ("https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.04162018.pdf", "NIST CSF 1.1"),
    ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf", "NIST SP 800-53r5"),
    ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf", "NIST SP 800-63b Auth"),
    ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf", "NIST Zero Trust"),
    ("https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf", "RP2040 Datasheet"),
    ("https://datasheets.raspberrypi.com/raspberry-pi-5/raspberry-pi-5-product-brief.pdf", "RPi5 Product Brief"),
    ("https://www.ti.com/lit/ds/symlink/lm317.pdf", "TI LM317 Datasheet"),
    ("https://www.ti.com/lit/ds/symlink/ina219.pdf", "TI INA219 Datasheet"),
    ("https://www.st.com/resource/en/datasheet/stm32f103c8.pdf", "STM32F103 Datasheet"),
    ("https://gcc.gnu.org/onlinedocs/gcc.pdf", "GCC Manual"),
    ("https://git-scm.com/book/en/v2/book.pdf", "Pro Git Book"),
    ("https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf", "PostgreSQL 16 Manual"),
    ("https://downloads.apache.org/httpd/docs/httpd-docs-2.4.57.en.pdf", "Apache HTTPD Manual"),
    ("https://www.sqlite.org/docsrc/doc/trunk/art/sqlite370_banner.gif", "SQLite Docs"),  # image, skip
    ("https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Assigned_Numbers/out/en/Assigned_Numbers.pdf", "Bluetooth Assigned Numbers"),
    ("https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.pdf", "MQTT v5.0 Spec"),
    ("https://spec.graphql.org/October2021/GraphQL.pdf", "GraphQL Spec Oct 2021"),  # may 404
    ("https://www.w3.org/TR/2023/REC-css-color-4-20230718/", "CSS Color 4"),  # HTML
    ("https://www.ietf.org/archive/id/draft-ietf-quic-http-34.pdf", "HTTP/3 Draft"),
    ("https://openocd.org/doc/pdf/openocd.pdf", "OpenOCD Manual"),
    ("https://www.freedesktop.org/software/systemd/man/systemd.pdf", "systemd Manual"),  # may 404
]

TECHNICAL_PDFS_CLEAN = [(u, t) for u, t in TECHNICAL_PDFS if not u.endswith((".gif", "/"))]


def fetch_technical(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[technical] target={n}")
    done = 0
    for url, title in TECHNICAL_PDFS_CLEAN[:n + 5]:  # extra buffer for 404s
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        print(f"  {name} <- {title}")
        if not dry_run:
            data = _download(url, os.path.join(raw_dir, name), delay=2)
            if data is None:
                continue
            manifest.append({"file": name, "domain": "technical", "source": url,
                              "title": title, "sha256": _sha256(data)})
            _save_manifest(manifest)
        idx += 1
        done += 1
    return idx


# ── government ────────────────────────────────────────────────────────────────

GOVT_PDFS = [
    ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "NIST AI RMF 1.0"),
    ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf", "NIST AI Adversarial ML"),
    ("https://www.gao.gov/assets/gao-23-105980.pdf", "GAO AI Accountability Report"),
    ("https://www.gao.gov/assets/gao-21-519sp.pdf", "GAO Technology Assessment"),
    ("https://www.gao.gov/assets/gao-24-106538.pdf", "GAO Generative AI Report"),
    ("https://apps.who.int/iris/bitstream/handle/10665/341091/9789240021327-eng.pdf", "WHO AI Ethics Health"),
    ("https://unesdoc.unesco.org/ark:/48223/pf0000381137.locale=en", "UNESCO AI Recommendation"),  # may 404
    ("https://www.federalregister.gov/documents/full_text/pdf/2023-01055.pdf", "Fed Register 2023-01055"),
    ("https://www.federalregister.gov/documents/full_text/pdf/2022-27382.pdf", "Fed Register 2022-27382"),
    ("https://www.cisa.gov/sites/default/files/2023-11/AI_ROADMAP_508C.pdf", "CISA AI Roadmap"),
    ("https://www.ftc.gov/system/files/ftc_gov/pdf/P201200_report_-_ai_and_algos_in_the_economy.pdf", "FTC AI Algorithms Report"),
    ("https://crsreports.congress.gov/product/pdf/R/R46795", "CRS AI Regulation"),
    ("https://crsreports.congress.gov/product/pdf/IF/IF11937", "CRS Machine Learning"),
    ("https://www.dhs.gov/sites/default/files/2023-09/23_0913_ia_ai-roadmap-final-508.pdf", "DHS AI Roadmap"),
    ("https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf", "WH AI Bill of Rights"),
]


def fetch_govt(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[govt] target={n}")
    done = 0
    for url, title in GOVT_PDFS[:n + 3]:
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        print(f"  {name} <- {title}")
        if not dry_run:
            data = _download(url, os.path.join(raw_dir, name), delay=2)
            if data is None:
                continue
            manifest.append({"file": name, "domain": "govt", "source": url,
                              "title": title, "sha256": _sha256(data)})
            _save_manifest(manifest)
        idx += 1
        done += 1
    return idx


# ── textbook ──────────────────────────────────────────────────────────────────

OPENSTAX_PDFS = [
    ("https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume1-WEB.pdf", "OpenStax University Physics Vol 1"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Calculus-WEB.pdf", "OpenStax Calculus"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Microbiology-WEB.pdf", "OpenStax Microbiology"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/IntroductiontoSociology3e-WEB.pdf", "OpenStax Sociology"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-WEB.pdf", "OpenStax Organic Chemistry"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/IntroductiontoStatistics-WEB.pdf", "OpenStax Statistics"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/AmericanGovernment3e-WEB.pdf", "OpenStax American Government"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/BusinessEthics-WEB.pdf", "OpenStax Business Ethics"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Macroeconomics3e-WEB.pdf", "OpenStax Macroeconomics"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Microeconomics3e-WEB.pdf", "OpenStax Microeconomics"),
]


def fetch_textbook(n: int, idx: int, raw_dir: str, manifest: list, dry_run: bool) -> int:
    print(f"\n[textbook] target={n}")
    done = 0
    for url, title in OPENSTAX_PDFS[:n]:
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        print(f"  {name} <- {title}")
        if not dry_run:
            data = _download(url, os.path.join(raw_dir, name), delay=3)
            if data is None:
                continue
            manifest.append({"file": name, "domain": "textbook", "source": url,
                              "title": title, "sha256": _sha256(data)})
            _save_manifest(manifest)
        idx += 1
        done += 1
    return idx


# ── main ──────────────────────────────────────────────────────────────────────

FETCHERS = {
    "arxiv":     fetch_arxiv,
    "legal":     fetch_legal,
    "financial": fetch_financial,
    "medical":   fetch_medical,
    "technical": fetch_technical,
    "govt":      fetch_govt,
    "textbook":  fetch_textbook,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch diverse PDF dataset for DocStruct v2 benchmark")
    parser.add_argument("--domain", default=None, choices=list(FETCHERS), help="single domain (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="print plan, no downloads")
    parser.add_argument("--smoke", action="store_true", help="1 doc per domain (connectivity test)")
    parser.add_argument("--out-dir", default=os.path.join("data", "raw-pdfs"))
    args = parser.parse_args()

    raw_dir = args.out_dir
    os.makedirs(raw_dir, exist_ok=True)

    manifest = _load_manifest()
    idx = _next_index(raw_dir)
    domains = [args.domain] if args.domain else list(DOMAIN_TARGETS)

    print(f"Starting index : doc{idx}.pdf")
    print(f"Domains        : {domains}")
    print(f"Dry run        : {args.dry_run}")
    print(f"Smoke test     : {args.smoke}")
    print(f"Out dir        : {raw_dir}")

    for domain in domains:
        target = 1 if args.smoke else DOMAIN_TARGETS[domain]
        idx = FETCHERS[domain](target, idx, raw_dir, manifest, dry_run=args.dry_run)

    if not args.dry_run and not args.smoke:
        counts = {}
        for e in manifest:
            counts[e.get("domain", "?")] = counts.get(e.get("domain", "?"), 0) + 1
        print(f"\nmanifest: {len(manifest)} total -> {MANIFEST_V2}")
        for d, c in sorted(counts.items()):
            print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
