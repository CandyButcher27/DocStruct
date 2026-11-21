# Section-boundary agreement (PMC corpus)

6 documents with publisher JATS gold. Pk and WindowDiff are **error** rates -- lower is better, 0.0 is perfect agreement with the publisher's own section boundaries.

| Tool | WindowDiff | Pk | Straddle rate | Mean chunks | Docs | Errors |
|---|---|---|---|---|---|---|
| lumberchunker | 0.6994 | 0.5083 | 0.3024 | 63.8 | 6 | 0 |

- **WindowDiff** compares the *number* of boundaries in each window, so it penalises a tool that puts three splits where the document has one. **Pk** only asks whether the window's ends fall in the same segment, so it forgives over-segmentation; read them together.
- **Straddle rate** is the fraction of chunks crossing a gold boundary. It is not an error by itself -- 57.4% of gold sections are shorter than `MIN_CHUNK_TOKENS`, so merging them is the design working as intended -- but it bounds how meaningful a per-chunk section *label* can be.
- Back matter is excluded (DocStruct drops references by design), as are documents with under 50% of their gold locatable in the PDF's own text; see `reports/section_reachability.json`.
