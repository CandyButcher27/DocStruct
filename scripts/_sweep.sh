#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
QA=data/qa/benchmark_qa_v6.json
runs=(
  "baseline:"
  "dedupe:DEDUPE_CHARS=True"
  "dehyphen:DEHYPHENATE=True"
  "normalize:NORMALIZE_TEXT=True"
  "fig_area:FIGURE_OVERLAP_BY_AREA=True"
  "multicol:MULTI_COLUMN=True"
  "bandsplit:BAND_SPLIT=True"
  "furniture:STRIP_PAGE_FURNITURE=True"
  "tbl_borderless:TABLE_TEXT_STRATEGY_FALLBACK=True"
  "tbl_keyvalue:TABLE_SERIALIZATION=keyvalue"
  "tbl_split:TABLE_SPLIT_ROWS=True"
  "hdr_bold:HEADER_RANK_BY_WEIGHT=True"
  "keep_refs:KEEP_REFERENCES=True"
  "label_contain:LABEL_AWARE_CONTAINMENT=True"
)
for r in "${runs[@]}"; do
  name="${r%%:*}"; ov="${r#*:}"
  args=(--name "ab_$name" --qa "$QA" --cache-dir .bench_cache)
  [ -n "$ov" ] && args+=(--set "$ov")
  echo "=== RUN $name  ($ov) ==="
  $PY scripts/ablate.py "${args[@]}" 2>&1 | tail -3
done
echo "=== SWEEP DONE ==="
