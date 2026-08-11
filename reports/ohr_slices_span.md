# Slices of `reports/ohr_results_span.json`

relevance mode: **span**  |  3558 questions, 95 docs

## MRR by evidence source

| tool                | equation (n=45) | table (n=847) | text (n=2666) | unjoined |
|---------------------|-----------------|---------------|---------------|----------|
| docstruct           | 0.3333          | 0.2883        | 0.8448        | 0        |
| docstruct_geo       | 0.3019          | 0.2851        | 0.8448        | 0        |
| pymupdf4llm         | 0.2756          | 0.1876        | 0.8689        | 0        |
| llamaindex_semantic | 0.0222          | 0.3074        | 0.7748        | 0        |
| unstructured        | 0.3409          | 0.223         | 0.8012        | 0        |
| llamaindex          | 0.0556          | 0.2746        | 0.7771        | 0        |
| langchain           | 0.0556          | 0.2404        | 0.7776        | 0        |

## MRR by domain

| tool                | academic (n=631) | finance (n=736) | law (n=1515) | manual (n=676) | unjoined |
|---------------------|------------------|-----------------|--------------|----------------|----------|
| docstruct           | 0.4526           | 0.5141          | 0.8655       | 0.7932         | 0        |
| docstruct_geo       | 0.4574           | 0.501           | 0.8656       | 0.7966         | 0        |
| pymupdf4llm         | 0.5112           | 0.4675          | 0.8573       | 0.7726         | 0        |
| llamaindex_semantic | 0.2857           | 0.5352          | 0.8412       | 0.7077         | 0        |
| unstructured        | 0.5151           | 0.4474          | 0.7935       | 0.7056         | 0        |
| llamaindex          | 0.2016           | 0.5199          | 0.8372       | 0.782          | 0        |
| langchain           | 0.1934           | 0.4929          | 0.8312       | 0.7915         | 0        |

## MRR by relative position in the document

Gold in the last fifth is the back-matter check: DocStruct drops references by design, so a falling right-hand column for us and a flat one for everyone else is that design decision showing up as a measured cost.

| tool                | 0-20%  | 20-40% | 40-60% | 60-80% | 80-100% |
|---------------------|--------|--------|--------|--------|---------|
| docstruct           | 0.7534 | 0.7407 | 0.6868 | 0.6723 | 0.6696  |
| docstruct_geo       | 0.7676 | 0.7276 | 0.6877 | 0.6674 | 0.6674  |
| pymupdf4llm         | 0.7558 | 0.7095 | 0.6811 | 0.6633 | 0.6856  |
| llamaindex_semantic | 0.7163 | 0.6434 | 0.6607 | 0.618  | 0.6288  |
| unstructured        | 0.6477 | 0.6781 | 0.6504 | 0.6446 | 0.6451  |
| llamaindex          | 0.6604 | 0.6701 | 0.647  | 0.6225 | 0.6398  |
| langchain           | 0.6338 | 0.6698 | 0.6441 | 0.615  | 0.6387  |
