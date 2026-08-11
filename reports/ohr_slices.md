# Slices of `ohr_results.json`

relevance mode: **page**  |  3558 questions, 95 docs

## MRR by evidence source

| tool                | equation (n=45) | table (n=847) | text (n=2666) | unjoined |
|---------------------|-----------------|---------------|---------------|----------|
| unstructured        | 0.8239          | 0.6504        | 0.8421        | 0        |
| langchain           | 0.6767          | 0.6165        | 0.8019        | 0        |
| llamaindex          | 0.6637          | 0.6313        | 0.7617        | 0        |
| llamaindex_semantic | 0.4378          | 0.5834        | 0.6767        | 0        |
| docstruct           | 0.7074          | 0.5349        | 0.6194        | 0        |
| docstruct_geo       | 0.3619          | 0.4082        | 0.4919        | 0        |

## MRR by domain

| tool                | academic (n=631) | finance (n=736) | law (n=1515) | manual (n=676) | unjoined |
|---------------------|------------------|-----------------|--------------|----------------|----------|
| unstructured        | 0.7733           | 0.6865          | 0.8398       | 0.8425         | 0        |
| langchain           | 0.5142           | 0.7092          | 0.8512       | 0.8201         | 0        |
| llamaindex          | 0.5355           | 0.6741          | 0.8152       | 0.7786         | 0        |
| llamaindex_semantic | 0.5596           | 0.5364          | 0.737        | 0.6711         | 0        |
| docstruct           | 0.5181           | 0.5501          | 0.6679       | 0.5809         | 0        |
| docstruct_geo       | 0.3804           | 0.3891          | 0.5291       | 0.5109         | 0        |

## MRR by relative position in the document

Gold in the last fifth is the back-matter check: DocStruct drops references by design, so a falling right-hand column for us and a flat one for everyone else is that design decision showing up as a measured cost.

| tool                | 0-20%  | 20-40% | 40-60% | 60-80% | 80-100% |
|---------------------|--------|--------|--------|--------|---------|
| unstructured        | 0.7683 | 0.8568 | 0.7906 | 0.7894 | 0.7568  |
| langchain           | 0.7245 | 0.7923 | 0.7722 | 0.7541 | 0.7294  |
| llamaindex          | 0.6976 | 0.7673 | 0.7503 | 0.7444 | 0.6723  |
| llamaindex_semantic | 0.696  | 0.6364 | 0.6415 | 0.6721 | 0.6032  |
| docstruct           | 0.6164 | 0.6678 | 0.5762 | 0.5994 | 0.5254  |
| docstruct_geo       | 0.5248 | 0.523  | 0.4287 | 0.46   | 0.4021  |
