#!/usr/bin/env bash
set -u
PDFS=$(cat reports/_newdocs.txt)
.venv/Scripts/python.exe -m docstruct.cli gen-qa $PDFS \
  --out data/qa/benchmark_qa_v7_extra.json \
  --provider ollama --per-doc 5 --cache-dir .bench_cache
echo "=== GENQA DONE ==="
