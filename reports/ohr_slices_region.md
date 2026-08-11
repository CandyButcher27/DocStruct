# Slices of `reports/ohr_results_region.json`

relevance mode: **region**  |  3558 questions, 95 docs

## MRR by evidence source

| tool                | equation (n=45) | table (n=847) | text (n=2666) | unjoined |
|---------------------|-----------------|---------------|---------------|----------|
| docstruct           | 0.1111          | 0.3655        | 0.7704        | 0        |
| docstruct_geo       | 0.1333          | 0.3868        | 0.7512        | 0        |
| pymupdf4llm         | 0.0933          | 0.1224        | 0.7656        | 0        |
| langchain           | 0.0333          | 0.3258        | 0.7006        | 0        |
| unstructured        | 0.1364          | 0.2542        | 0.7227        | 0        |
| llamaindex          | 0.0333          | 0.3495        | 0.674         | 0        |
| llamaindex_semantic | 0.0111          | 0.3611        | 0.6523        | 0        |

## MRR by domain

| tool                | academic (n=631) | finance (n=736) | law (n=1515) | manual (n=676) | unjoined |
|---------------------|------------------|-----------------|--------------|----------------|----------|
| docstruct           | 0.4157           | 0.5484          | 0.7922       | 0.743          | 0        |
| docstruct_geo       | 0.4459           | 0.5292          | 0.7634       | 0.753          | 0        |
| pymupdf4llm         | 0.4326           | 0.3624          | 0.7458       | 0.7094         | 0        |
| langchain           | 0.1034           | 0.5434          | 0.7762       | 0.7456         | 0        |
| unstructured        | 0.4893           | 0.452           | 0.6961       | 0.6651         | 0        |
| llamaindex          | 0.1175           | 0.5561          | 0.7498       | 0.7028         | 0        |
| llamaindex_semantic | 0.1795           | 0.5092          | 0.7444       | 0.6354         | 0        |

## MRR by relative position in the document

Gold in the last fifth is the back-matter check: DocStruct drops references by design, so a falling right-hand column for us and a flat one for everyone else is that design decision showing up as a measured cost.

| tool                | 0-20%  | 20-40% | 40-60% | 60-80% | 80-100% |
|---------------------|--------|--------|--------|--------|---------|
| docstruct           | 0.6825 | 0.6961 | 0.6612 | 0.6463 | 0.6355  |
| docstruct_geo       | 0.692  | 0.676  | 0.649  | 0.6161 | 0.6498  |
| pymupdf4llm         | 0.6461 | 0.6181 | 0.5731 | 0.5771 | 0.6088  |
| langchain           | 0.5762 | 0.6241 | 0.623  | 0.6004 | 0.5849  |
| unstructured        | 0.5868 | 0.6331 | 0.6009 | 0.5884 | 0.5885  |
| llamaindex          | 0.5723 | 0.5992 | 0.6048 | 0.5813 | 0.5832  |
| llamaindex_semantic | 0.6124 | 0.5551 | 0.5883 | 0.579  | 0.5314  |
