# Section-boundary agreement (PMC corpus)

24 documents with publisher JATS gold. Pk and WindowDiff are **error** rates -- lower is better, 0.0 is perfect agreement with the publisher's own section boundaries.

| Tool | WindowDiff | Pk | Straddle rate | Mean chunks | Docs | Errors |
|---|---|---|---|---|---|---|
| docstruct_geo **(ours)** | 0.4362 | 0.3525 | 0.527 | 25.5 | 24 | 0 |
| pymupdf4llm | 0.4928 | 0.4661 | 0.6005 | 15.9 | 24 | 0 |
| docstruct **(ours)** | 0.4934 | 0.3641 | 0.4484 | 35.4 | 24 | 0 |
| llamaindex_semantic | 0.5334 | 0.5134 | 0.2496 | 24.7 | 24 | 0 |
| llamaindex | 0.6959 | 0.5938 | 0.3828 | 37.9 | 24 | 0 |
| langchain | 0.8821 | 0.6183 | 0.2227 | 73.9 | 24 | 0 |
| unstructured | 0.8839 | 0.5974 | 0.1924 | 92.4 | 18 | 6 |

- **WindowDiff** compares the *number* of boundaries in each window, so it penalises a tool that puts three splits where the document has one. **Pk** only asks whether the window's ends fall in the same segment, so it forgives over-segmentation; read them together.
- **Straddle rate** is the fraction of chunks crossing a gold boundary. It is not an error by itself -- 57.4% of gold sections are shorter than `MIN_CHUNK_TOKENS`, so merging them is the design working as intended -- but it bounds how meaningful a per-chunk section *label* can be.
- Back matter is excluded (DocStruct drops references by design), as are documents with under 50% of their gold locatable in the PDF's own text; see `reports/section_reachability_colab24.json` (the ceiling for **these 24 docs**:
  84.1% of sections located, 86.9% body ceiling). `reports/section_reachability.json` is
  the ceiling over the **full 126-doc** PMC corpus (80.7% / 84.5% body) — the scores above
  cover a 24-doc subset of it, because the Colab session that produced them had only 24
  PMC PDFs fetched. Re-run on the full corpus before this table goes in the paper.
