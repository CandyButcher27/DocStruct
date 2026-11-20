# Review 1 follow-up runs

GitHub Actions run 44739832582 (OHR-Bench, 95 docs, 3,558 questions, one CPU install) and a local PMC run. Diffs are paired bootstrap over questions (2000 resamples, seed 0), 95% interval. Built by `scripts/review_summary.py`.

## Chunk-size floor on OHR-Bench (docstruct_geo)

Diff = row MRR minus the shipped 200/500 setting.

| MIN/MAX | chunks | words/query (span) | span MRR | region MRR | page MRR | span diff | region diff |
|---|---|---|---|---|---|---|---|
| 100/500 | 6959 | 2107 | 0.7011 | 0.6584 | 0.7329 | -0.0036 [-0.0094, +0.0024] | +0.0020 [-0.0050, +0.0090] |
| 150/500 | 6246 | 2215 | 0.7097 | 0.6617 | 0.7337 | +0.0048 [-0.0006, +0.0105] | +0.0052 [-0.0007, +0.0110] |
| 200/500 | 5810 | 2328 | 0.7048 | 0.6566 | 0.7235 |  |  |
| 300/500 | 5256 | 2501 | 0.7019 | 0.6492 | 0.7102 | -0.0029 [-0.0095, +0.0031] | -0.0074 [-0.0149, -0.0003] |
| 400/500 | 4898 | 2692 | 0.7030 | 0.6450 | 0.6998 | -0.0017 [-0.0088, +0.0059] | -0.0115 [-0.0201, -0.0028] |
| 200/800 | 5489 | 2749 | 0.7083 | 0.6564 | 0.7115 | +0.0037 [-0.0021, +0.0096] | +0.0001 [-0.0082, +0.0078] |

## Chunk-size floor on PMC section boundaries (docstruct_geo)

Pk and WindowDiff are error rates (lower is better).

| MIN/MAX | Pk | WindowDiff | straddle | chunks/doc | docs |
|---|---|---|---|---|---|
| 100/500 | 0.3291 | 0.4238 | 0.4805 | 31.9 | 122 |
| 150/500 | 0.3341 | 0.4200 | 0.4982 | 28.9 | 122 |
| 200/500 | 0.3447 | 0.4240 | 0.5082 | 27.0 | 122 |
| 300/500 | 0.3655 | 0.4316 | 0.5147 | 24.4 | 122 |
| 400/500 | 0.3760 | 0.4341 | 0.5130 | 23.0 | 122 |
| 200/800 | 0.3356 | 0.4129 | 0.5424 | 25.3 | 122 |

## Docling (8 shards merged) against docstruct_geo 200/500

| mode | Docling MRR | chunks | words/query | errors | docstruct_geo MRR | geo minus Docling |
|---|---|---|---|---|---|---|
| span | 0.6658 | 20863 | 631 | 0 | 0.7048 | +0.0388 [+0.0264, +0.0515] |
| region | 0.6291 | 20863 | 631 | 0 | 0.6566 | +0.0272 [+0.0133, +0.0418] |
| page | 0.7969 | 20863 | 631 | 0 | 0.7235 | -0.0749 [-0.0884, -0.0613] |

## Baselines resized to docstruct_geo's context budget

| tool | setting | words/query | span MRR | region MRR | geo minus tool (span) | geo minus tool (region) |
|---|---|---|---|---|---|---|
| unstructured | max_characters=4400, combine_text_under_n_chars=3200 | 2592 | 0.7050 | 0.6281 | -0.0004 [-0.0119, +0.0102] | +0.0274 [+0.0138, +0.0415] |
| langchain | chunk_size=8500 | 4488 | 0.7104 | 0.6083 | -0.0056 [-0.0169, +0.0062] | +0.0484 [+0.0344, +0.0634] |
| llamaindex | chunk_size=2400 | 6031 | 0.7068 | 0.6045 | -0.0026 [-0.0146, +0.0094] | +0.0517 [+0.0362, +0.0666] |
