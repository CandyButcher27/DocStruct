# Claude Prompt — Rewrite the DocStruct Research Paper

## Role

You are helping rewrite an existing research paper about **DocStruct**, a local, deterministic, structure-aware PDF processing and chunking system.

The attached/current paper is the source of truth for existing claims, experiments, citations, and numbers. Do not invent experiments, results, datasets, implementation details, or claims that are not supported by the paper or the project context provided below.

The goal is **not** to lightly edit the existing paper. The goal is to rebuild its narrative so that the paper explains the **entire DocStruct pipeline as one coherent system**, while preserving the same formal section structure required for the submission.

The current paper is overly centered on chunking evaluation and relevance-rule inversion. That finding is important, but it should no longer dominate the identity of the paper. The main story should be the complete pipeline: why PDF structure matters, how DocStruct recovers and preserves that structure, how the resulting representation becomes retrieval-oriented chunks, and what the evaluation tells us about the approach.

---

# 1. Core narrative to follow

The paper should read as one continuous story:

> PDFs are not naturally flat text. They contain spatial and structural information such as columns, headings, tables, figures, captions, and section relationships. Conventional PDF-to-text pipelines can lose this information before retrieval-oriented chunking even begins. DocStruct was built to preserve this information through an end-to-end deterministic pipeline. It obtains layout evidence from PDF geometry and, optionally, visual detection; fuses those observations into structured blocks; reconstructs reading order; extracts content; identifies section hierarchy; and finally assembles retrieval-oriented chunks. We then evaluate whether this structure-aware representation is useful, whether it recovers document structure, whether it is deterministic, and where it fails. During this evaluation, we also discover that retrieval leaderboard results depend substantially on the relevance rule used to define a correct retrieved chunk.

The reader should finish the paper understanding **the system first and the evaluation findings second**.

Do not frame DocStruct merely as an "instrument" used to investigate chunker evaluation.

---

# 2. Central problem statement

The problem must be stated in simple language very early.

The paper should clearly communicate:

> The problem is not simply that PDFs need to be split into chunks. The deeper problem is that useful document structure can be lost when a PDF is converted into a flat text stream, and this lost structure affects reading order, content extraction, section boundaries, and ultimately the units presented to a retrieval system.

Avoid beginning with the leaderboard/evaluation problem.

The reader should first understand **why structure-aware PDF processing is needed**.

---

# 3. What DocStruct actually is

Describe DocStruct as:

> A local, deterministic, structure-aware PDF processing pipeline for producing retrieval-oriented chunks.

Important properties:

- No language model is used on the parse path.
- The same PDF should produce the same chunks within a fixed version/configuration.
- Processing is local and auditable.
- Geometry and visual layout detection are separate sources of evidence.
- Vision is optional.
- The final product of the parse pipeline is a sequence of structured chunks, not merely extracted text.
- Indexing, retrieval, and LLM-based gold-question generation are evaluation/downstream components, not part of the core parsing path.

Do not call DocStruct a general RAG framework.

---

# 4. Exact pipeline that the paper should explain

Use this conceptual pipeline throughout the paper:

```text
                         PDF
                          │
              ┌───────────┴───────────┐
              │                       │
        PDF Geometry             Page Image
              │                       │
     Geometry Detector         Vision Detector
              │                       │
              └───────────┬───────────┘
                          │
                        Fusion
                          │
                 Structured Blocks
                          │
                   Reading Order
                          │
                 Content Extraction
                          │
                Section Hierarchy
                          │
                   Chunk Assembly
                          │
                       Chunks
                          │
                 Retrieval / RAG
```

The exact figure can be redesigned later, but the paper's prose should follow this order.

The five high-level pipeline stages used in the current paper can still be retained, but the explanation should make the intermediate objects and purpose of each stage much clearer.

---

# 5. Important technical facts to preserve

## Detection

Two independent detectors operate on the same page:

### Geometry detector
Uses PDF primitives through pdfplumber, including:
- character/word positions
- font names and sizes
- ruling lines
- whitespace and geometric relationships

It proposes structural regions such as:
- text
- header
- table
- figure
- caption

### Vision detector
Uses YOLOv8 trained on DocLayNet and operates from page appearance/raster information.

It is optional.

The important architectural idea is that the detectors are **mutually blind**: they read different evidence from the same page. Their agreement is therefore useful as an independent structural signal.

Do not claim that vision consistently improves results. The experiments do not support that.

---

# 6. Fusion

Explain that detection produces proposals, not yet the final document representation.

Fusion:
- matches geometry and model proposals using IoU
- resolves same-label and conflicting-label matches
- preserves unilateral detections
- produces structured Blocks
- attaches confidence/source information

The paper should explain the intuition rather than dumping formulas.

You may mention the IoU matching threshold and confidence formulas when technically useful, but do not make the section read like a list of constants.

Important limitation:
- unilateral confidence scaling/clamping is inherited from a prototype and remains explicitly unvalidated.

Also mention that an aggressive containment suppression approach was tried and caused substantial content loss, so it was reverted/gated. This is a useful example of an engineering decision, but do not overemphasize the exact 28% figure unless it is needed.

---

# 7. Reading order

Explain why reading order is necessary.

A PDF can contain:

```text
Column 1      Column 2

A             D
B             E
C             F
```

A naive text extractor can produce something like:

```text
A D B E C F
```

or otherwise interleave the columns.

DocStruct reconstructs page-level reading order using a column-aware heuristic:
- infer column bands from block positions
- read each band top-to-bottom
- move to the next band

The current default supports at most two columns under the main heuristic.

Recursive XY-cut was implemented and evaluated but was not selected as the default because it reduced retrieval performance on the tested arXiv-heavy corpus.

Known limitation:
- full-width blocks spanning multiple columns can still be assigned imperfectly.

Do not hide this limitation.

---

# 8. Content extraction

Make the distinction explicit:

> Detection determines what a region is and where it is. Extraction determines what content is actually inside that region.

Text is extracted from the detected block geometry using pdfplumber.

Important improvement:
- word-spacing tolerance is scaled using font size rather than relying on one fixed tolerance.
- This corrected spacing problems on small/display text and produced a measured retrieval improvement.

Tables:
- structured grids are extracted where possible
- tables can be rendered as structured representations
- a raw-text coverage guard prevents partially ruled tables from silently losing unruled rows
- table chunks preserve useful row/header information

Do not overstate table extraction quality.

Also explain that human-facing Markdown rendering and retrieval-oriented table chunk text can serve different purposes.

Running headers/footers and some multi-page table handling exist as gated functionality.

---

# 9. Section hierarchy

DocStruct builds a SectionPath of up to three heading levels.

Heading hierarchy is inferred primarily from:
1. section numbering patterns
2. font-size ranking as fallback
3. optional boldness-related logic where relevant

Example:

```text
2. Methodology       → h1
2.1 Dataset          → h2
2.1.1 Preprocessing  → h3
```

The SectionPath travels with the resulting chunk as metadata.

Do not inject artificial breadcrumb strings into the chunk text. The heading hierarchy is represented structurally as metadata.

---

# 10. Chunk assembly

Explain chunking as the **final consequence of the recovered structure**, not as the whole system.

The assembler walks the ordered blocks while maintaining the current SectionPath.

Default values:
- minimum chunk size: 200 words
- maximum chunk size: 500 words
- overlap: 75 words for hard size boundaries

Important design insight:
- enforcing every structural boundary directly creates tiny fragments
- the minimum chunk floor allows a chunk to cross a structural boundary when necessary to preserve enough context
- this improved retrieval substantially in the development experiments

The paper can mention the +0.0429 MRR improvement associated with the minimum floor, but explain what the result means before presenting the number.

A larger minimum of 600 produced the highest raw MRR in one experiment but was rejected because it moved the system toward fixed-size chunking and substantially increased retrieved context.

Tables are treated as atomic structured objects.

Captions are handled separately where possible.

References are not included as normal retrieval chunks.

A chunk that starts in one section and crosses into another remains attributed to the section in which it started.

---

# 11. Caching

The paper does not need a giant caching section, but caching is important enough to explain in Method or Engineering Details.

There are three conceptual cache levels:

```text
Proposal Cache
    ↓
Model Proposal Cache
    ↓
Block Cache
    ↓
Chunking
```

The caches are content/configuration keyed.

Most importantly:
- raw geometry proposals can be reused when downstream settings change
- model proposals do not depend on layout configuration
- fully populated blocks are cached before chunking
- chunk-only settings are excluded from the BlockCache fingerprint so chunking experiments do not rerun expensive detection/extraction stages

This is useful because it explains how the extensive ablation work was made practical.

---

# 12. Evaluation story

The evaluation should answer questions in this order:

### Question 1 — Does the resulting chunk representation work for retrieval?

Use OHR-Bench:
- 95 documents
- 3,558 human-authored questions
- multiple domains
- comparison against seven systems

### Question 2 — Does the pipeline actually recover document structure?

Use PMC:
- 134 papers
- publisher-authored JATS section structure
- Pk and WindowDiff
- no retriever and no relevance rule

Be careful:
This evaluates structural agreement, not whether section boundaries are necessarily optimal for RAG.

### Question 3 — Is the deterministic claim actually true?

Use the independent-process experiment:
- all 95 OHR-Bench documents
- two separate processes
- byte-identical geometry-only output

Important caveat:
The full hybrid GPU path was not verified for bit-level determinism because no GPU was available.

### Question 4 — What does not work?

Discuss:
- vision detector gives no significant retrieval benefit in tested corpora
- geometry-only actually beats hybrid on the PMC structural metric and is faster
- context cost is high
- word coverage is not perfect
- academic/finance performance is weaker
- FinanceBench was unusable as a retrieval leaderboard under the tested protocol
- parsing speed is a practical limitation on dense financial documents

The paper should be comfortable reporting these negative findings.

---

# 13. Relevance-rule finding

This is important, but it should be presented as a **finding that emerged from evaluating the system**, not as the paper's opening premise.

The key observation:

The same chunks were evaluated under:
- span relevance
- region relevance
- page relevance

The ranking changes.

DocStruct:
- 1st under span
- 1st under region
- 6th under page

Unstructured shows the approximate mirror pattern.

The underlying issue is that relevance definitions have implicit preferences for chunk size:
- span tends to favor larger chunks
- page tends to favor smaller chunks
- region is closer to size-neutral in the observed data

Therefore, a leaderboard using only one relevance rule can entangle system quality with chunk-size preference.

This is an important contribution/finding.

However, do not let it swallow the whole paper.

---

# 14. Contribution section

Rewrite the contributions so they describe both the system and the findings.

Suggested structure:

1. **End-to-end structure-aware PDF pipeline.**
   DocStruct recovers layout evidence, fuses structural observations, reconstructs reading order, extracts content, builds section hierarchy, and assembles retrieval-oriented chunks in one deterministic pipeline.

2. **Structure-preserving chunk representation.**
   The resulting chunks retain section paths, block/source information, page information, and structural metadata instead of being produced from a flat text stream alone.

3. **Empirical evaluation across retrieval and structural views.**
   The system is evaluated against existing chunkers on human-authored retrieval questions and independently against publisher-authored document structure.

4. **Evidence about evaluation methodology.**
   The retrieval ranking changes when the relevance rule changes, showing that chunking evaluations can mix chunk quality with an implicit preference for chunk size.

5. **Measured limitations and negative results.**
   The work explicitly reports the limited benefit of the vision detector, context cost, speed limitations, and benchmark limitations.

Do not make the contribution section a statistical scoreboard.

---

# 15. Abstract requirements

The abstract must be **one paragraph only**.

Its logical order should be:

1. Problem in simple language.
2. Why existing PDF processing can fail.
3. What DocStruct is.
4. What the pipeline does.
5. What makes it distinct: deterministic/local/no LM on parse path.
6. What was evaluated.
7. Most important results.
8. Main contribution/implication.

Do not begin with leaderboard inversion.

Do not make the abstract a list of metrics.

Do not claim the vision detector improves the system generally.

Do not say DocStruct is "state of the art" or "best" unless directly supported.

---

# 16. Writing style requirements

The current paper feels too compressed and formulaic. Rewrite it to sound like researchers explaining a system they actually built and evaluated.

Use these rules:

- Prefer plain, direct English.
- Explain a technical decision before giving its numerical result.
- Connect paragraphs logically.
- Use transitions that reflect causality: "This creates...", "To address this...", "Once this...", "We then...", "This allows...", etc.
- Avoid excessive rhetorical one-liners.
- Avoid repeatedly starting paragraphs with "We show", "We demonstrate", "Notably", "Furthermore", "Importantly".
- Avoid semicolon-heavy sentences.
- Avoid stacking multiple claims into one sentence.
- Avoid excessive parenthetical qualifications.
- Do not make every paragraph follow "claim → statistic → caveat".
- Use numbers selectively.
- Keep necessary technical details.
- Do not make the prose artificially sophisticated.
- Do not use vague phrases such as "leveraging advanced techniques", "holistic framework", "robust solution", "novel paradigm", "seamlessly integrates", etc.
- Do not write like marketing material.
- Do not oversell.
- Do not hide negative results.
- Do not use generic AI-style filler.
- Do not repeat the same idea in multiple sections.
- Keep terminology consistent: use **DocStruct**, **geometry detector**, **vision detector**, **fusion**, **Block**, **SectionPath**, and **Chunk** consistently.
- Fix all grammar, spelling, punctuation, and typographical errors.

Most importantly, the writing should feel like a **single explanation**, not independently generated sections.

---

# 17. Paragraph-by-paragraph plan

The following is the intended content flow. Where enough information is available below, draft the paragraph. Where the source does not provide enough evidence, leave an explicit instruction for Claude rather than inventing content.

## Abstract

### Paragraph 1 — WRITE
Write one paragraph covering:
- PDFs contain spatial/structural information that is often lost when converted into flat text.
- This can affect reading order, extraction, and chunk construction for RAG.
- Introduce DocStruct as a local deterministic structure-aware pipeline.
- Briefly describe geometry + optional vision → fusion → reading order → extraction → hierarchy → chunking.
- State no language model is used on the parse path.
- Mention evaluation on 95 OHR-Bench documents / 3,558 human questions and 134 PMC papers with publisher-authored JATS structure.
- Give the main retrieval result in a compact way without turning the abstract into a leaderboard table.
- Mention the relevance-rule inversion finding.
- End with the actual contribution/implication.

Do not write more than one paragraph.

---

# 18. Introduction

## Paragraph 1 — WRITE
Explain the practical problem:

PDFs are designed as visual documents rather than as linear text streams. A human reader can use columns, whitespace, headings, tables, figures, and spatial relationships to understand how information is organized. When a PDF is converted directly into plain text, much of that organization can disappear before a retrieval system ever sees the document.

## Paragraph 2 — WRITE
Explain the consequence:

If reading order is wrong, text from different columns can be interleaved. If table structure is lost, rows and headers can become disconnected. If headings disappear, the relationship between a passage and its surrounding section is lost. Chunking a flattened representation cannot reliably reconstruct all of this information afterward.

## Paragraph 3 — WRITE
Introduce the existing design choices at a high level:

One common response is simple deterministic parsing followed by fixed-size splitting. Another is to use learned or semantic models to decide where document structure or topic boundaries lie. Explain the trade-off without turning this into a long related-work discussion.

## Paragraph 4 — WRITE
Introduce DocStruct as the response:

DocStruct treats layout as information that should be recovered before chunking. It uses PDF geometry as a deterministic source of evidence and optionally uses visual layout detection as a second, independent source. These observations are fused before the document is ordered, extracted, organized into a hierarchy, and converted into chunks.

## Paragraph 5 — WRITE
Explain the design philosophy:

The system deliberately avoids language-model calls on the parse path. This makes the output locally reproducible and auditable and allows changes to downstream chunking parameters without rerunning earlier expensive stages through the cache architecture.

Do not overclaim complete determinism for the hybrid GPU path; state the verified scope carefully.

## Paragraph 6 — WRITE
Introduce the evaluation motivation naturally:

Once the complete pipeline exists, the next question is whether the resulting chunks are actually useful. Explain that the work evaluates retrieval quality, structural fidelity, and determinism rather than relying on a single parse-quality claim.

## Paragraph 7 — WRITE
Introduce the relevance-rule finding:

While evaluating retrieval, the same chunks produced different rankings depending on whether relevance was defined using answer spans, evidence regions, or pages. Explain that this reveals a methodological issue in chunker comparisons: the evaluation rule itself can favor different chunk sizes.

## Contributions
Use a concise numbered list based on the contribution structure above.

---

# 19. Related Work

Keep the same section.

Do not make Related Work dominate the paper.

Organize it around three areas:

1. PDF/document parsing and structure extraction.
2. Chunking strategies for RAG.
3. Retrieval/chunking evaluation.

The key purpose of Related Work is to establish where DocStruct sits.

Do not turn every cited paper into a mini-summary.

Preserve the existing references and claims unless the current source clearly supports a correction.

---

# 20. Method

## 3.1 Problem Setting

WRITE a short paragraph defining:

PDF → ordered list of structured chunks.

Each chunk contains:
- text
- page information
- reading order
- section path
- structural metadata

Explain that downstream retrieval sees these chunks.

---

## 3.2 Pipeline

WRITE the main pipeline explanation as a connected sequence.

The paragraph should introduce:

PDF
→ geometry and optional vision
→ fusion
→ reading order
→ extraction
→ hierarchy
→ chunk assembly.

Then explain that each stage passes a richer representation to the next.

Include the new main pipeline diagram here.

---

## 3.3 Layout Detection

Write two connected paragraphs.

### Paragraph A — Geometry detector — WRITE
Explain what PDF primitives are used and what kinds of structure they can reveal.

### Paragraph B — Vision detector — WRITE
Explain why a second visual view exists, what it can see that PDF geometry cannot, and that it is optional. State that experiments later test whether this additional complexity actually helps.

Do not claim it helps before the results section.

---

## 3.4 Fusion and Reading Order

### Paragraph A — Fusion — WRITE
Explain proposals → IoU matching → arbitration → structured Blocks.

### Paragraph B — Reading order — WRITE
Explain why reading order matters and how the column-aware ordering works.

### Paragraph C — Alternatives/limitations — WRITE
Explain XY-cut was implemented and evaluated but not selected as default, and mention the full-width/multi-column limitation.

---

## 3.5 Content Extraction

### Paragraph A — WRITE
Explain detection versus extraction.

### Paragraph B — WRITE
Explain text extraction and font-scaled spacing tolerance.

### Paragraph C — WRITE
Explain table extraction and the raw-text coverage guard.

### Paragraph D — PROMPT CLAUDE
Discuss gated cleanup features and document furniture handling only to the extent that the current paper/source provides enough detail. Do not invent additional extraction behavior.

---

## 3.6 Section Hierarchy and Chunk Assembly

### Paragraph A — WRITE
Explain heading detection and SectionPath.

### Paragraph B — WRITE
Explain chunk assembly and the structural boundary + size floor interaction.

### Paragraph C — WRITE
Explain why the minimum chunk floor matters and how the development experiments motivated it.

### Paragraph D — WRITE
Explain atomic tables, captions, metadata, and section attribution.

Do not turn this section into a parameter table.

---

# 21. Evaluation Setup

## 4.1 Protocol — WRITE

Explain:
- all systems run through one evaluation harness
- chunker is the main variable
- embedder/retriever settings are held fixed
- retrieval depth is fixed
- metrics include MRR, nDCG@5, Recall@5, and Precision@1
- uncertainty is assessed using the existing bootstrap procedure

Explain this in prose first, then provide the formal details.

## 4.2 Corpora — WRITE

Explain the role of each corpus:

- OHR-Bench = main human-authored retrieval evaluation
- PMC = structure evaluation using publisher-authored JATS
- FinanceBench = negative/diagnostic result rather than a reliable leaderboard under the tested protocol

## 4.3 Relevance Rules — WRITE

Explain span, region, and page relevance simply.

Then explain the conceptual issue:

Each rule asks a slightly different question and therefore has different relationships with chunk size.

Do not dump all correlations here; reserve the important evidence for Results.

---

# 22. Results

Organize the Results around questions, not around tables.

## 5.1 Retrieval evaluation and relevance rules

### Paragraph 1 — WRITE
State the main observation: the ranking changes depending on the relevance definition even though the chunks are identical.

### Paragraph 2 — WRITE
Explain the DocStruct/unstructured inversion.

### Paragraph 3 — WRITE
Explain why chunk size helps interpret this result.

### Paragraph 4 — WRITE
Explain the region threshold sweep and what it establishes.

### Paragraph 5 — WRITE
Give the scope limitation: one retrieval configuration, so do not claim the inversion is universal across all embedders/retrievers.

Include the relevant table/figure.

---

## 5.2 Structural fidelity

### Paragraph 1 — WRITE
Explain the PMC/JATS experiment as a retrieval-independent test of structural recovery.

### Paragraph 2 — WRITE
Explain the result: geometry-only leads the hybrid on Pk and WindowDiff and is faster.

### Paragraph 3 — WRITE
Carefully explain what this does and does not prove.

Do not claim that publisher section boundaries are automatically optimal RAG boundaries.

---

## 5.3 Determinism

### Paragraph 1 — WRITE
Explain why determinism needs to be tested across processes.

### Paragraph 2 — WRITE
Report the 95/95 byte-identical geometry-only result.

### Paragraph 3 — WRITE
Explain the scope limitation: version pinning and unverified CUDA/hybrid determinism.

---

## 5.4 Limitations and negative results

This section should be honest and useful.

### Paragraph 1 — WRITE
Vision detector:
- no significant OHR retrieval improvement
- geometry-only wins PMC structural metric
- hybrid is slower
- therefore vision is not claimed as a general contribution

### Paragraph 2 — WRITE
Context efficiency:
- DocStruct retrieves substantially more context than leaner baselines
- explain that this is the cost of larger structural units
- avoid making the paragraph a statistics dump

### Paragraph 3 — WRITE
Coverage/duplication:
- coverage is not perfect
- duplication is high
- explain what design decisions cause this

### Paragraph 4 — WRITE
FinanceBench:
Explain why it was not usable as a retrieval leaderboard under the tested fixed-embedder protocol, despite relevant chunks existing.

### Paragraph 5 — WRITE
Speed:
Explain that parsing dense financial PDFs is currently expensive and that table/geometry materialization dominates some cases.

---

# 23. Conclusion

## Paragraph 1 — WRITE
Return to the main system contribution:

DocStruct provides a deterministic, structure-aware route from PDF layout to retrieval chunks.

## Paragraph 2 — WRITE
Summarize the evidence:
- competitive retrieval under some relevance definitions
- structural recovery evidence
- verified geometry-only determinism
- clear limitations

## Paragraph 3 — WRITE
State the broader methodological finding:

Chunker evaluation should not rely blindly on one relevance definition because the rule can favor different chunk sizes and change the apparent ranking.

End with a restrained statement of what remains open:
- stronger embedder/retriever settings
- broader corpora
- direct parse-fidelity benchmarks
- calibrated fusion confidence
- further investigation of vision usefulness.

Do not end with a generic "future work will..." paragraph full of promises.

---

# 24. Limitations

Preserve a dedicated Limitations section if required by the current submission format.

Organize it clearly:

1. Scope: born-digital/text PDFs; no OCR; English-heavy corpus.
2. Speed: expensive on dense financial documents.
3. Direct parse fidelity: not evaluated with OmniDocBench/READoc metrics.
4. Layout detector calibration: fusion constants remain partly unvalidated.
5. Domain coverage: academic/finance results are weaker and academic sample is small.
6. Hybrid determinism: GPU path not fully verified.

Do not apologize excessively. State limitations as technical boundaries of the evidence.

---

# 25. Ethics Statement

Keep the substance of the current paper:
- public corpora
- no corpus redistribution
- no human subjects/personal data beyond public author information
- no language model on parse path
- negative results are reported rather than selectively omitted

Rewrite for natural flow.

---

# 26. Figures/diagrams

The current figures should be reconsidered.

The most important figure should be a **full end-to-end architecture diagram**, not merely a compact list of stages.

Create a figure that visually communicates:

```text
PDF
  ↓
┌──────────────────────────────┐
│ Layout Evidence              │
│                              │
│ Geometry       Vision        │
│ pdfplumber     DocLayNet     │
└──────────────┬───────────────┘
               ↓
            Fusion
               ↓
       Structured Blocks
               ↓
        Reading Order
               ↓
        Content Extraction
               ↓
        Section Hierarchy
               ↓
         Chunk Assembly
               ↓
     Retrieval-oriented Chunks
```

Possible second figure:

Show one two-column PDF page and visually demonstrate:

```text
raw PDF layout
      ↓
detected blocks
      ↓
reading order
      ↓
section path
      ↓
final chunks
```

This would give the reader a concrete example of what the system actually does.

Possible third figure:

A simple conceptual evaluation diagram:

```text
Same chunks
   │
   ├── Span relevance → ranking A
   ├── Region relevance → ranking B
   └── Page relevance → ranking C
```

This should make the relevance-rule finding immediately understandable.

Do not overcrowd figures with formulas or dozens of metrics.

---

# 27. Important factual guardrails

You MUST NOT invent:

- new datasets
- new experiments
- new metrics
- new improvements
- new architectural components
- new numerical results
- claims that vision improves retrieval generally
- claims that DocStruct is universally superior
- claims that hybrid GPU execution is fully deterministic
- claims that structural fidelity automatically implies better RAG

If the current source does not contain enough information to write a paragraph safely, write:

> `[CLAUDE: PROMPT NEEDED — explain what evidence/details are required here rather than inventing content.]`

The user specifically wants some paragraphs written and other paragraphs left as prompts when there is insufficient certainty.

---

# 28. What to preserve from the current paper

Preserve:
- citations
- datasets
- experimental setup
- tables where useful
- supported numerical results
- negative results
- limitations
- terminology
- technical correctness

But reorganize their presentation around the new story.

Do not mechanically preserve the current paragraph order.

Do not preserve awkward sentences merely because they are technically correct.

---

# 29. What to de-emphasize

Reduce emphasis on:
- leaderboard rhetoric
- long sequences of statistical comparisons
- clever one-line claims
- excessive discussion of relevance-rule mechanics before the reader understands DocStruct
- treating chunking as if it were the entire system
- repeated caveats embedded in every sentence

The relevance-rule finding remains important, but it should feel like something the evaluation **revealed**, not something the entire system was built solely to prove.

---

# 30. Two-pass workflow

This is **Pass 1**.

Do NOT optimize heavily for word count yet.

The goal is to establish:
- correct narrative
- complete pipeline explanation
- coherent paragraph sequence
- correct contribution framing
- natural prose
- technically supported claims

After Pass 1, the paper will be edited again for:
- abstract length
- conference/page limits
- paragraph compression
- redundancy
- figure placement
- citation density
- final wording.

Therefore, for this pass, prioritize clarity and complete logical flow over extreme concision.

---

# 31. Final instruction to Claude

Rewrite the complete paper according to this plan.

Do not merely "polish" the current text.

Think of the paper as a researcher explaining:

1. What problem we encountered.
2. Why ordinary PDF-to-text processing is insufficient.
3. What DocStruct does.
4. How its complete pipeline works.
5. Why each stage exists.
6. What we measured.
7. What the measurements tell us.
8. What failed.
9. What contribution the work actually supports.

The final manuscript should feel like **one research story from beginning to end**.

The reader should understand DocStruct's complete pipeline before encountering the detailed retrieval leaderboard analysis.

Use the existing paper as the factual source of truth. Use the architecture and project details in this prompt to restore the intended system-level framing.

Do not invent missing information.

When uncertain, explicitly leave a Claude prompt instead of fabricating a paragraph.

Do not optimize for sounding "academic" at the expense of sounding natural.

Do not try to hide weaknesses. The negative results are part of the research contribution.

The desired result is a paper that sounds like a technically competent researcher carefully explaining what they built, why they built it, what happened when they tested it, and what they learned.
