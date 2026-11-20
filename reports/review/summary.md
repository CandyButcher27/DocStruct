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

## Table 1 on one install (current code)

| system | span | region | page | ctx | geo minus (span / region / page) |
|---|---|---|---|---|---|
| docstruct_geo | 0.7048 (2) | 0.6566 (2) | 0.7235 (6) | 2328 |  |
| pymupdf4llm | 0.6988 (3) | 0.6036 (4) | 0.6715 (7) | 2432 | +0.0057 p=0.327, +0.0529 p=0.0, +0.0511 p=0.0 |
| unstructured | 0.6567 (5) | 0.6032 (5) | 0.7957 (2) | 556 | +0.0478 p=0.0, +0.0523 p=0.0, -0.0740 p=0.0 |
| langchain | 0.6407 (8) | 0.6029 (6) | 0.7560 (4) | 638 | +0.0641 p=0.0, +0.0537 p=0.0, -0.0337 p=0.0 |
| llamaindex | 0.6483 (7) | 0.5887 (7) | 0.7294 (5) | 1430 | +0.0567 p=0.0, +0.0680 p=0.0, -0.0064 p=0.34 |
| llamaindex_semantic | 0.6540 (6) | 0.5749 (8) | 0.6515 (8) | 4698 | +0.0505 p=0.0, +0.0816 p=0.0, +0.0722 p=0.0 |
| docling | 0.6658 (4) | 0.6291 (3) | 0.7969 (1) | 631 | +0.0388 p=0.0, +0.0272 p=0.0, -0.0749 p=0.0 |
| lumberchunker | 0.7098 (1) | 0.6992 (1) | 0.7852 (3) | 1057 | -0.0050 p=0.379, -0.0427 p=0.0, -0.0628 p=0.0 |

Spearman(mean chunk words, MRR): span rho=0.119 p=0.779 (n=8), region rho=-0.238 p=0.57 (n=8), page rho=-0.952 p=0.0 (n=8)

## Span MRR by domain and evidence type

| system | academic | finance | law | manual | text | table | coverage | duplication |
|---|---|---|---|---|---|---|---|---|
| docstruct_geo | 0.4574 | 0.5017 | 0.8656 | 0.7966 | 0.8448 | 0.2857 | 0.9638 | 1.1320 |
| pymupdf4llm | 0.5128 | 0.4662 | 0.8569 | 0.7715 | 0.8678 | 0.1882 | 0.9679 | 1.1003 |
| unstructured | 0.5151 | 0.4474 | 0.7935 | 0.7101 | 0.7995 | 0.2245 | 0.9224 | 1.0548 |
| langchain | 0.1934 | 0.4938 | 0.8312 | 0.7915 | 0.7778 | 0.2404 | 1.0000 | 1.1005 |
| llamaindex | 0.2016 | 0.5199 | 0.8372 | 0.7820 | 0.7771 | 0.2746 | 1.0000 | 1.0474 |
| llamaindex_semantic | 0.2857 | 0.5352 | 0.8412 | 0.7077 | 0.7748 | 0.3074 | 1.0000 | 1.0000 |
| docling | 0.5137 | 0.4385 | 0.8159 | 0.7190 | 0.8254 | 0.1830 | 0.9336 | 1.1518 |
| lumberchunker | 0.5349 | 0.5157 | 0.8298 | 0.8154 | 0.8431 | 0.3106 | 0.9706 | 1.0801 |
