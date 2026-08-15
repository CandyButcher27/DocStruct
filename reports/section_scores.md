# Section-boundary agreement (PMC corpus)

134 documents with publisher JATS gold. Pk and WindowDiff are **error** rates -- lower is better, 0.0 is perfect agreement with the publisher's own section boundaries.

| Tool | WindowDiff | Pk | Straddle rate | Mean chunks | Docs | Errors |
|---|---|---|---|---|---|---|
| docstruct_geo **(ours)** | 0.4226 | 0.3418 | 0.5129 | 26.8 | 134 | 0 |
| pymupdf4llm | 0.48 | 0.449 | 0.5734 | 17.7 | 134 | 0 |
| docstruct **(ours)** | 0.4818 | 0.3531 | 0.4385 | 37.5 | 134 | 0 |
| llamaindex_semantic | 0.5337 | 0.5128 | 0.1889 | 29.1 | 134 | 0 |
| llamaindex | 0.6952 | 0.5979 | 0.366 | 42.7 | 134 | 0 |
| langchain | 0.8787 | 0.62 | 0.2202 | 85.6 | 134 | 0 |
| unstructured | 0.8933 | 0.6025 | 0.182 | 106.9 | 99 | 35 |

- **WindowDiff** compares the *number* of boundaries in each window, so it penalises a tool that puts three splits where the document has one. **Pk** only asks whether the window's ends fall in the same segment, so it forgives over-segmentation; read them together.
- **Straddle rate** is the fraction of chunks crossing a gold boundary. It is not an error by itself -- 57.4% of gold sections are shorter than `MIN_CHUNK_TOKENS`, so merging them is the design working as intended -- but it bounds how meaningful a per-chunk section *label* can be.
- Back matter is excluded (DocStruct drops references by design), as are documents with under 50% of their gold locatable in the PDF's own text; see `reports/section_reachability.json`.
