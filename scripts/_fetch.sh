#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
for d in legal financial medical technical govt textbook; do
  echo "=== FETCH $d ==="
  $PY scripts/fetch_dataset_v2.py --domain "$d" 2>&1 | tail -4
done
echo "=== FETCH DONE ==="
ls data/raw-pdfs/*.pdf | wc -l
