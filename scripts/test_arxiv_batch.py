import os
import subprocess
import requests
from pathlib import Path
import hashlib
import json

# Configuration
ARXIV_IDS = [
    "1706.03762",  # Attention Is All You Need
    "2302.04062",  # Synthetic Data Generation Survey
    "2408.01747",  # Classical ML survey
    "2407.12220",  # Questionable practices in ML
]

BASE_DIR = Path("arxiv_test")
PDF_DIR = BASE_DIR / "pdfs"
OUT_DIR = BASE_DIR / "outputs"

DOCSTRUCT_MAIN = "main.py"  # adjust if needed

# Setup
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Download PDFs
def download_pdf(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = PDF_DIR / f"{arxiv_id}.pdf"

    if output_path.exists():
        print(f"[SKIP] {arxiv_id} already downloaded")
        return output_path

    print(f"[DOWNLOAD] {arxiv_id}")
    r = requests.get(url)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(r.content)

    return output_path

# Run DocStruct
def run_docstruct(pdf_path):
    output_path = OUT_DIR / (pdf_path.stem + ".json")

    cmd = [
        "python",
        DOCSTRUCT_MAIN,
        str(pdf_path),
        str(output_path),
        "--detector",
        "stub"
    ]

    subprocess.run(cmd, check=True)
    return output_path


# Determinism Check
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def check_determinism(pdf_path):
    print(f"[DETERMINISM] {pdf_path.name}")

    out1 = run_docstruct(pdf_path)
    hash1 = file_hash(out1)

    out2 = run_docstruct(pdf_path)
    hash2 = file_hash(out2)

    if hash1 == hash2:
        print("  OK: deterministic")
    else:
        print("  FAIL: non-deterministic output")


# Main
def main():
    for arxiv_id in ARXIV_IDS:
        pdf_path = download_pdf(arxiv_id)
        run_docstruct(pdf_path)
        check_determinism(pdf_path)

if __name__ == "__main__":
    main()