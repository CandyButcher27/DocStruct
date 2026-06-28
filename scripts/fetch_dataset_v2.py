"""Fetch ~200 born-digital PDFs across 7 domains for the v2 benchmark dataset.

Domains and sources:
  arxiv      -- academic papers (arXiv API, continuation of existing doc*.pdf set)
  legal      -- US court opinions (CourtListener REST API, free)
  financial  -- SEC EDGAR 10-K annual filings (EDGAR full-text search, free)
  medical    -- PubMed Central open-access articles (PMC OA API, free)
  technical  -- Apache/Linux Foundation project docs, open product manuals (direct URL list)
  govt       -- Federal Register notices + EU policy PDFs (free)
  textbook   -- OpenStax textbook chapters (CC-licensed, free)

Usage:
  python scripts/fetch_dataset_v2.py                  # fetch everything
  python scripts/fetch_dataset_v2.py --domain legal   # one domain only
  python scripts/fetch_dataset_v2.py --dry-run        # print plan, no downloads
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

RAW_DIR = os.path.join("data", "raw-pdfs")
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

# ── helpers ──────────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _next_index() -> int:
    if not os.path.isdir(RAW_DIR):
        return 1
    nums = [int(m.group(1)) for f in os.listdir(RAW_DIR) if (m := re.match(r"doc(\d+)\.pdf$", f))]
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
            print(f"    skip {url} (not a PDF)")
            return None
        with open(dest, "wb") as fh:
            fh.write(data)
        time.sleep(delay)
        return data
    except Exception as err:
        print(f"    download failed: {err}")
        return None


# ── arxiv ─────────────────────────────────────────────────────────────────────

# Diverse arXiv categories: CS, stats, physics, bio, economics — ensures
# varied vocabulary and PDF complexity (double-column, math-heavy, bio figures)
ARXIV_PLAN = {
    "cs.CL": 8,
    "cs.LG": 7,
    "cs.CV": 6,
    "stat.ML": 5,
    "q-bio.NC": 5,
    "econ.EM": 5,
    "physics.optics": 4,
    "math.OC": 4,
    "cs.RO": 3,
    "astro-ph.CO": 3,
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
        out.append({"arxiv_id": arxiv_id, "title": title, "pdf_url": pdf_url, "category": category})
    return out


def fetch_arxiv(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[arxiv] target={n}")
    done = 0
    seen_ids = {e.get("arxiv_id") for e in manifest}
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
            if e["arxiv_id"] in seen_ids:
                print(f"  skip {e['arxiv_id']} (duplicate)")
                continue
            name = f"doc{idx}.pdf"
            dest = os.path.join(RAW_DIR, name)
            print(f"  {name} <- {e['arxiv_id']} [{category}]")
            if not dry_run:
                data = _download(e["pdf_url"], dest, delay=3)
                if data is None:
                    continue
                manifest.append({"file": name, "domain": "arxiv", "source": "arxiv.org",
                                  "arxiv_id": e["arxiv_id"], "title": e["title"],
                                  "category": category, "sha256": _sha256(data)})
                _save_manifest(manifest)
            seen_ids.add(e["arxiv_id"])
            idx += 1
            done += 1
    return idx


# ── legal ─────────────────────────────────────────────────────────────────────
# CourtListener free API — US federal court opinions (PDF)
# Mix of circuit courts for varied doc length and structural complexity

COURTLISTENER_CLUSTERS = [
    # (cluster_id, description)  — sample of landmark/interesting opinions
    ("4214664", "9th Cir 2022"),  ("4123456", "2nd Cir 2022"),
    ("4300000", "DC Cir 2023"),   ("4400000", "1st Cir 2023"),
    ("4500000", "3rd Cir 2023"),  ("4600000", "4th Cir 2023"),
    ("4700000", "5th Cir 2023"),  ("4800000", "6th Cir 2022"),
    ("4900000", "7th Cir 2022"),  ("5000000", "8th Cir 2022"),
    ("5100000", "10th Cir 2022"), ("5200000", "11th Cir 2023"),
]
# Use search API instead of hardcoded IDs for robustness
COURTLISTENER_SEARCH = "https://www.courtlistener.com/api/rest/v3/search/?type=o&order_by=score+desc&stat_Precedential=on&filed_after=2020-01-01&filed_before=2024-01-01&format=json"


def fetch_legal(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[legal] target={n}")
    done = 0
    page = 1
    while done < n:
        url = f"{COURTLISTENER_SEARCH}&page={page}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docstruct-bench/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as err:
            print(f"  search failed: {err}")
            break

        results = data.get("results", [])
        if not results:
            break

        for opinion in results:
            if done >= n:
                break
            pdf_url = opinion.get("download_url") or ""
            if not pdf_url.endswith(".pdf"):
                continue
            case_name = opinion.get("caseName", "unknown")
            court = opinion.get("court", "")
            name = f"doc{idx}.pdf"
            dest = os.path.join(RAW_DIR, name)
            print(f"  {name} <- {case_name[:60]} [{court}]")
            if not dry_run:
                dl_data = _download(pdf_url, dest, delay=1.5)
                if dl_data is None:
                    continue
                manifest.append({"file": name, "domain": "legal", "source": "courtlistener.com",
                                  "case_name": case_name, "court": court, "pdf_url": pdf_url,
                                  "sha256": _sha256(dl_data)})
                _save_manifest(manifest)
            idx += 1
            done += 1
        page += 1
        time.sleep(1)
    return idx


# ── financial ─────────────────────────────────────────────────────────────────
# SEC EDGAR full-text search for 10-K filings — mix of industries
# Heavy tables, standardized section structure (Item 1-9), borderless tables

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom&startdt=2022-01-01&enddt=2024-01-01&forms=10-K"
EDGAR_BASE = "https://www.sec.gov"

# Diverse sectors: tech, pharma, energy, retail, finance, manufacturing
EDGAR_QUERIES = [
    ("technology", "10-K"),
    ("pharmaceutical", "10-K"),
    ("energy", "10-K"),
    ("retail", "10-K"),
    ("financial services", "10-K"),
    ("manufacturing", "10-K"),
    ("healthcare", "10-K"),
]


def fetch_financial(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[financial] target={n}")
    done = 0
    per_sector = max(1, n // len(EDGAR_QUERIES))

    for sector, form in EDGAR_QUERIES:
        if done >= n:
            break
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{urllib.request.quote(sector)}%22"
            f"&forms={form}&dateRange=custom&startdt=2022-01-01&enddt=2024-06-01"
        )
        try:
            req = urllib.request.Request(search_url, headers={"User-Agent": "docstruct-bench/0.2 research@example.com"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as err:
            print(f"  EDGAR search failed [{sector}]: {err}")
            continue

        hits = data.get("hits", {}).get("hits", [])
        count = 0
        for hit in hits:
            if done >= n or count >= per_sector:
                break
            src = hit.get("_source", {})
            file_date = src.get("file_date", "")
            entity = src.get("entity_name", "unknown")
            # Get the actual 10-K document URL
            accession = src.get("accession_no", "").replace("-", "")
            cik = str(src.get("entity_id", "")).zfill(10)
            if not accession or not cik:
                continue
            # Primary document is typically the 10-K htm; we need the PDF version
            # Try filing index to find PDF
            index_url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=1&output=atom"
            name = f"doc{idx}.pdf"
            dest = os.path.join(RAW_DIR, name)
            print(f"  {name} <- {entity[:50]} [{sector}] {file_date}")
            # Direct PDF link via EDGAR viewer
            pdf_url = f"{EDGAR_BASE}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession}-index.htm"
            # Try the annual report PDF if available — fall back to htm→pdf conversion note
            # For now record and skip if no direct PDF
            if not dry_run:
                # EDGAR doesn't always have PDFs; try direct PDF in filing
                filing_url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=1"
                print(f"    (EDGAR PDF extraction — may need manual download for {entity})")
                # Skip: EDGAR 10-Ks are primarily HTML; leave for manual or use alternative source
                continue
            idx += 1
            done += 1
            count += 1

    # Better financial source: use direct PDF annual reports from investor relations pages
    # These are more reliably PDF and structurally rich
    DIRECT_FINANCIAL_PDFS = [
        ("https://www.apple.com/newsroom/pdfs/FY2023_Annual_Report.pdf", "Apple 10-K 2023", "technology"),
        ("https://ir.aboutamazon.com/annual-reports-proxies-and-shareholder-letters/annual-reports/2022/default.aspx", "Amazon AR 2022", "technology"),
        # Add more as needed — these are stable public PDFs
    ]

    print(f"  [financial] EDGAR HTML-only issue — recommend manual curation of 10-K PDFs")
    print(f"  Best source: download directly from SEC EDGAR 'View Filing' PDF links")
    print(f"  Or use: https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&dateb=&owner=include&count=40&search_text=")
    return idx


# ── medical ───────────────────────────────────────────────────────────────────
# PubMed Central Open Access — free, varied medical/bio content
# Stress-tests: tables with lab values, figures, clinical protocols

PMC_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax={n}&term={query}+AND+open+access[filter]"
PMC_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&rettype=pdf&id={pmc_id}"

MEDICAL_QUERIES = [
    "clinical trial randomized controlled",
    "systematic review meta-analysis",
    "cardiovascular disease treatment",
    "cancer immunotherapy outcomes",
    "machine learning medical imaging",
    "COVID-19 long-term outcomes",
]


def fetch_medical(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[medical] target={n}")
    done = 0
    per_query = max(1, n // len(MEDICAL_QUERIES))

    for query in MEDICAL_QUERIES:
        if done >= n:
            break
        search_url = PMC_SEARCH.format(n=per_query + 2, query=urllib.request.quote(query))
        try:
            req = urllib.request.Request(search_url, headers={"User-Agent": "docstruct-bench/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as err:
            print(f"  PMC search failed [{query[:30]}]: {err}")
            continue

        ids = data.get("esearchresult", {}).get("idlist", [])
        for pmc_id in ids:
            if done >= n:
                break
            pdf_url = PMC_FETCH.format(pmc_id=pmc_id)
            name = f"doc{idx}.pdf"
            dest = os.path.join(RAW_DIR, name)
            print(f"  {name} <- PMC{pmc_id} [{query[:30]}...]")
            if not dry_run:
                dl_data = _download(pdf_url, dest, delay=1)
                if dl_data is None:
                    continue
                manifest.append({"file": name, "domain": "medical", "source": "pubmedcentral",
                                  "pmc_id": pmc_id, "query": query, "sha256": _sha256(dl_data)})
                _save_manifest(manifest)
            idx += 1
            done += 1
        time.sleep(0.5)
    return idx


# ── technical ─────────────────────────────────────────────────────────────────
# Open-source project documentation PDFs — numbered steps, lists, cross-refs
# Linux kernel docs, Apache manuals, RFCs, hardware datasheets

TECHNICAL_PDFS = [
    # Linux kernel and POSIX docs
    ("https://www.kernel.org/doc/html/latest/_downloads/8ef9b58e2b2e6aed3c32a1cb2c1c8ef6/kernel-doc-guide.pdf", "Linux Kernel Doc Guide"),
    # Apache project manuals
    ("https://downloads.apache.org/httpd/docs/httpd-docs-2.4.57.en.pdf", "Apache HTTPD Manual"),
    # PostgreSQL documentation
    ("https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf", "PostgreSQL 16 Manual"),
    # Python documentation
    ("https://docs.python.org/3/archives/python-3.11.0-docs-pdf-a4.zip", "Python 3.11 Docs"),  # zip, skip
    # RFC documents (IETF) — plain text structure, numbered sections
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc9110.txt.pdf", "RFC 9110 HTTP Semantics"),
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc8446.txt.pdf", "RFC 8446 TLS 1.3"),
    ("https://www.rfc-editor.org/rfc/pdfrfc/rfc7540.txt.pdf", "RFC 7540 HTTP/2"),
    # Docker documentation
    ("https://docs.docker.com/get-started/overview/", "Docker Overview"),  # HTML, skip
    # Open hardware datasheets (Texas Instruments, STM)
    ("https://www.ti.com/lit/ds/symlink/lm317.pdf", "TI LM317 Datasheet"),
    ("https://www.ti.com/lit/ds/symlink/ina219.pdf", "TI INA219 Datasheet"),
    ("https://www.st.com/resource/en/datasheet/stm32f103c8.pdf", "STM32F103 Datasheet"),
    # USB specification (structured, table-heavy)
    ("https://www.usb.org/sites/default/files/usb_20_20230901.zip", "USB 2.0 Spec"),  # zip, skip
    # Wi-Fi Alliance documents
    ("https://www.wi-fi.org/download.php?file=/sites/default/files/private/Wi-Fi_Easy_Connect_Specification_v3.0.pdf", "WiFi Easy Connect Spec"),
    # NIST cybersecurity framework
    ("https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.04162018.pdf", "NIST CSF 1.1"),
    ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf", "NIST SP 800-53"),
    # IEEE standards samples (free)
    ("https://standards.ieee.org/wp-content/uploads/import/documents/tutorials/eee_services.pdf", "IEEE Services Tutorial"),
    # OpenAPI specification
    ("https://spec.openapis.org/oas/v3.1.0", "OpenAPI 3.1"),  # HTML, skip
    # Kubernetes docs
    ("https://kubernetes.io/docs/concepts/", "K8s Concepts"),  # HTML, skip
    # Real technical PDFs: Raspberry Pi hardware specs
    ("https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf", "RP2040 Datasheet"),
    ("https://datasheets.raspberrypi.com/raspberry-pi-5/raspberry-pi-5-product-brief.pdf", "RPi 5 Product Brief"),
    # Arduino
    ("https://content.arduino.cc/assets/UNO_V3_With_328P_schematic.pdf", "Arduino UNO Schematic"),
    # GCC manual
    ("https://gcc.gnu.org/onlinedocs/gcc.pdf", "GCC Manual"),
    # Git documentation
    ("https://git-scm.com/book/en/v2/book.pdf", "Pro Git Book"),
    # Ansible docs
    ("https://docs.ansible.com/ansible/2.9/pdf/ansible_2.9_user_guide.pdf", "Ansible User Guide"),
    # Tensorflow guide
    ("https://www.tensorflow.org/guide", "TF Guide"),  # HTML
]

TECHNICAL_PDFS_CLEAN = [(url, title) for url, title in TECHNICAL_PDFS if not url.endswith(".zip") and not url.endswith("/")]


def fetch_technical(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[technical] target={n}")
    done = 0
    for url, title in TECHNICAL_PDFS_CLEAN[:n]:
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        dest = os.path.join(RAW_DIR, name)
        print(f"  {name} <- {title}")
        if not dry_run:
            dl_data = _download(url, dest, delay=2)
            if dl_data is None:
                continue
            manifest.append({"file": name, "domain": "technical", "source": url,
                              "title": title, "sha256": _sha256(dl_data)})
            _save_manifest(manifest)
        idx += 1
        done += 1
    return idx


# ── government ────────────────────────────────────────────────────────────────
# Federal Register + EU policy — single-column dense prose, formal hierarchy
# Stress-tests long-section detection and reference section skipping

GOVT_PDFS = [
    # Federal Register rules (dense, long)
    ("https://www.federalregister.gov/documents/full_text/pdf/2023-00001.pdf", "Fed Register 2023-00001"),
    ("https://www.federalregister.gov/documents/full_text/pdf/2023-10000.pdf", "Fed Register 2023-10000"),
    ("https://www.federalregister.gov/documents/full_text/pdf/2022-27382.pdf", "Fed Register 2022-27382"),
    ("https://www.federalregister.gov/documents/full_text/pdf/2023-04800.pdf", "Fed Register AI Policy"),
    # FTC reports
    ("https://www.ftc.gov/system/files/documents/reports/federal-trade-commission-report-2022/ftc-2022-annual-report.pdf", "FTC Annual Report 2022"),
    # SEC reports
    ("https://www.sec.gov/files/sec-2022-annual-report.pdf", "SEC Annual Report 2022"),
    # NIST guidelines (already some in technical — include policy-facing ones here)
    ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "NIST AI RMF"),
    # EU AI Act (structured legislative document)
    ("https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52021PC0206", "EU AI Act Proposal"),
    # White House AI EO
    ("https://www.whitehouse.gov/wp-content/uploads/2023/10/Biden-Harris-Administration-Fact-Sheet-on-AI-EO.pdf", "WH AI EO Fact Sheet"),
    # GAO technology reports
    ("https://www.gao.gov/assets/gao-23-105980.pdf", "GAO AI Accountability"),
    ("https://www.gao.gov/assets/gao-21-519sp.pdf", "GAO Technology Assessment"),
    # Congressional research service
    ("https://crsreports.congress.gov/product/pdf/R/R46795", "CRS AI Regulation"),
    ("https://crsreports.congress.gov/product/pdf/IF/IF11937", "CRS Machine Learning"),
    # WHO health policy
    ("https://apps.who.int/iris/bitstream/handle/10665/341091/9789240021327-eng.pdf", "WHO AI Ethics Health"),
    # UNESCO AI recommendation
    ("https://unesdoc.unesco.org/ark:/48223/pf0000381137/PDF/381137eng.pdf.multi", "UNESCO AI Recommendation"),
]


def fetch_govt(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[govt] target={n}")
    done = 0
    for url, title in GOVT_PDFS[:n]:
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        dest = os.path.join(RAW_DIR, name)
        print(f"  {name} <- {title}")
        if not dry_run:
            dl_data = _download(url, dest, delay=2)
            if dl_data is None:
                continue
            manifest.append({"file": name, "domain": "govt", "source": url,
                              "title": title, "sha256": _sha256(dl_data)})
            _save_manifest(manifest)
        idx += 1
        done += 1
    return idx


# ── textbook ──────────────────────────────────────────────────────────────────
# OpenStax CC-licensed textbooks — chapter/section/subsection hierarchy,
# exercises, mixed content; stress-tests deep header hierarchy (h1/h2/h3)

OPENSTAX_PDFS = [
    ("https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume1-WEB.pdf", "OpenStax University Physics Vol 1"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Calculus-WEB.pdf", "OpenStax Calculus"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Microbiology-WEB.pdf", "OpenStax Microbiology"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/IntroductiontoSociology3e-WEB.pdf", "OpenStax Intro Sociology"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/OrganicChemistry-WEB.pdf", "OpenStax Organic Chemistry"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/IntroductiontoStatistics-WEB.pdf", "OpenStax Intro Statistics"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/AmericanGovernment3e-WEB.pdf", "OpenStax American Government"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/BusinessEthics-WEB.pdf", "OpenStax Business Ethics"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Macroeconomics3e-WEB.pdf", "OpenStax Macroeconomics"),
    ("https://assets.openstax.org/oscms-prodcms/media/documents/Microeconomics3e-WEB.pdf", "OpenStax Microeconomics"),
]


def fetch_textbook(n: int, idx: int, manifest: list, dry_run: bool) -> int:
    print(f"\n[textbook] target={n}")
    done = 0
    for url, title in OPENSTAX_PDFS[:n]:
        if done >= n:
            break
        name = f"doc{idx}.pdf"
        dest = os.path.join(RAW_DIR, name)
        print(f"  {name} <- {title}")
        if not dry_run:
            dl_data = _download(url, dest, delay=3)
            if dl_data is None:
                continue
            manifest.append({"file": name, "domain": "textbook", "source": url,
                              "title": title, "sha256": _sha256(dl_data)})
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
    parser.add_argument("--domain", default=None, help="single domain to fetch (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="print plan without downloading")
    parser.add_argument("--out-dir", default=RAW_DIR)
    args = parser.parse_args()

    global RAW_DIR
    RAW_DIR = args.out_dir
    os.makedirs(RAW_DIR, exist_ok=True)

    manifest = _load_manifest()
    idx = _next_index()
    domains = [args.domain] if args.domain else list(DOMAIN_TARGETS)

    print(f"Starting index: doc{idx}.pdf")
    print(f"Domains: {domains}")
    print(f"Dry run: {args.dry_run}")

    for domain in domains:
        if domain not in FETCHERS:
            print(f"Unknown domain: {domain}")
            continue
        target = DOMAIN_TARGETS[domain]
        idx = FETCHERS[domain](target, idx, manifest, args.dry_run)

    if not args.dry_run:
        print(f"\nmanifest: {len(manifest)} entries -> {MANIFEST_V2}")
        domain_counts = {}
        for e in manifest:
            domain_counts[e.get("domain", "unknown")] = domain_counts.get(e.get("domain", "unknown"), 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
