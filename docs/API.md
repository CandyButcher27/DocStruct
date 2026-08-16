# DocStruct API

`pip install docstruct-rag` → `import docstruct`

The distribution is **`docstruct-rag`** because `docstruct` was already taken on PyPI
(an unrelated document-tree package by smrt-co, last released 2023-06-22). The import
name is unchanged. Distribution name and import name need not match — the same way
`pip install scikit-learn` gives you `import sklearn`.

---

## Install

```bash
pip install docstruct-rag                    # core: geometry-only, no model, no network
pip install "docstruct-rag[model]"           # + YOLOv8/DocLayNet vision detector
pip install "docstruct-rag[langchain]"       # + Document.to_langchain()
pip install "docstruct-rag[llamaindex]"      # + Document.to_llamaindex()
pip install "docstruct-rag[all]"             # everything, including benchmark tooling
```

The core install has two dependencies: `pdfplumber` and `numpy`. That is deliberate —
the contract is *fully local, no network at parse time*, and every optional path is
behind an extra so the base case stays small.

---

## The 30-second version

```python
import docstruct

doc = docstruct.parse("paper.pdf")

doc.text                    # whole document in reading order
doc.markdown                # headings, tables and captions preserved
doc.chunks                  # retrieval-ready units, each with a section path

for chunk in doc:           # Document is iterable over its chunks
    print(chunk.section_path.h1, "|", chunk.content[:60])
```

Same PDF in, same chunks out. No LLM is called anywhere on this path.

---

## Entry points

### `parse(pdf_path, **options) -> Document`

The main entry point.

```python
doc = docstruct.parse(
    "paper.pdf",
    weights="yolov8m-doclaynet.pt",   # enable the vision detector (optional)
    cache_dir=".cache",               # cache detector output by content hash
    password="secret",                # encrypted PDF
    config={"MIN_CHUNK_TOKENS": 300}, # per-call config override, thread-safe
    on_page=lambda i, n: print(i, n), # progress callback
)
```

`config` overrides anything in `docstruct/config.py` for this call only — no global
mutation, so two threads can parse with different settings.

### `parse_bytes(data, *, name="<bytes>", **options) -> Document`

For when you have bytes and not a path — an HTTP upload, an S3 object, a database blob.

```python
@app.post("/parse")
def handler(upload):
    doc = docstruct.parse_bytes(upload.read(), name=upload.filename)
    return doc.to_jsonl()
```

Raises `InvalidPDFError` immediately if the bytes lack a `%PDF` header, rather than
failing deeper in the pipeline with a less useful message.

> **Honest note:** this writes one temporary file and deletes it. The pipeline is
> path-oriented throughout — the cache keys on the file, the detector rasterises by
> path — so it is not zero-copy. It saves you the temp-file dance, not the disk write.

### `parse_many(paths, *, workers=None, on_error="raise", **options)`

Batch parsing across **processes** (not threads — parsing is CPU-bound and would
serialise on the GIL). Yields `(path, Document)` pairs in completion order.

```python
for path, doc in docstruct.parse_many(glob.glob("corpus/*.pdf"), workers=8,
                                      on_error="return"):
    if isinstance(doc, Exception):
        log.warning("%s failed: %s", path, doc)
        continue
    index(doc.to_langchain())
```

`on_error="return"` yields the exception in place of the Document. Use it for real
corpora: one malformed PDF in 500 should not lose the other 499.

Results arrive out of order. Each *document's* chunks are still deterministic; it is
only the batch iteration order that varies.

---

## `Document`

### Views over the text

| Member | Type | What it gives you |
|---|---|---|
| `.text` | `str` | Whole document, blocks in reading order |
| `.markdown` | `str` | Headings as `#`, tables as pipe tables, captions kept |
| `.chunks` | `list[Chunk]` | The retrieval units |
| `.blocks` | `list[Block]` | Pre-chunking layout regions, with bboxes and provenance |
| `.pages()` | `dict[int, str]` | Text keyed by page number |
| `.sections()` | `list[str]` | Section paths in document order |
| `.tables` | `list[(rows, page, section)]` | Extracted tables as cell grids |
| `.figures` | `list[(page, bbox)]` | Figure regions |
| `.chunks_of_type(t)` | `list[Chunk]` | Filter by `text`/`table`/`figure_caption`/`abstract`/`references` |
| `.diagnostics` | `dict` | Fusion counts, timings, what each detector contributed |

`len(doc)` is the chunk count and `iter(doc)` iterates chunks, so `for c in doc:` works.

### Export

| Method | Output |
|---|---|
| `.to_dict()` | Plain dict — path, diagnostics, chunks |
| `.to_json(path=None, indent=2)` | JSON string, optionally written to disk |
| `.to_jsonl(path=None)` | One `{id, text, metadata}` object per line |
| `.to_markdown(path=None)` | Markdown string, optionally written |
| `.stats()` | Counts: chunks, pages, word totals, chunks by type |

### Framework hand-off

```python
docs  = doc.to_langchain()    # list[langchain_core.documents.Document]
nodes = doc.to_llamaindex()   # list[llama_index.core.schema.TextNode]
```

Both carry the **section path in metadata**, which is the entire point: a retriever can
then filter by section, or show the user which section an answer came from. A
fixed-size splitter cannot tell it that, because it never knew.

Neither framework is a hard dependency. Calling these without the package installed
raises `ImportError` naming exactly what to `pip install`.

### Chunk metadata

Every exported chunk carries this, flat:

```python
{
  "source":        "paper.pdf",
  "chunk_type":    "text",
  "page":          4,
  "reading_order": 12,
  "section_h1":    "Method",
  "section_h2":    "Architecture Overview",
  "section_h3":    None,
  "section_path":  "Method > Architecture Overview",
}
```

**Flat on purpose.** Chroma, Pinecone and most vector stores reject nested metadata
values, and a nested dict here would fail at the user's ingest call rather than
anywhere near our code. There is a test pinning this.

---

## CLI

```bash
docstruct run paper.pdf                  # chunk and print
docstruct index corpus/ --db ./chroma    # build a vector index
docstruct query "how does X work?"       # retrieve
docstruct visualize paper.pdf --out png  # render detected blocks
docstruct benchmark --pdfs-dir data/ --qa gold.json   # leaderboard
```

`python -m docstruct.cli` always works; the `docstruct` shim can go stale if the
project directory moves.

---

## Exceptions

All inherit `DocStructError`, so one `except` catches everything:

| Exception | Raised when |
|---|---|
| `InvalidPDFError` | Not a PDF, or structurally corrupt |
| `EncryptedPDFError` | Password-protected and no/wrong password |
| `EmptyDocumentError` | Parsed fine, no extractable text — usually a scan |

`EmptyDocumentError` is the one to handle explicitly: DocStruct is **born-digital
only**. A scanned page has no text layer, and we do not run OCR. Route those to an OCR
pipeline and come back.

---

## Determinism, precisely

The guarantee is: **the same PDF, the same version, the same config → the same chunks**,
including chunk ids, boundaries, ordering and section paths.

Verify it yourself:

```bash
python scripts/verify_determinism.py --pdfs-dir data/ohrbench --runs 2
```

That parses every document twice in **separate processes** and compares content
hashes. A single-process check cannot see anything that varies across process
boundaries.

What the guarantee does *not* cover:
- **Across versions.** A release that changes chunking changes chunk boundaries. Pin
  the version if you keep an index around.
- **The vision path on GPU.** CUDA kernel selection is not guaranteed bit-reproducible.
  Geometry-only (`weights=None`) is pure Python and NumPy and has no such caveat.

---

## Design notes

**Why `weights` is a parameter and not a setting.** The vision detector is optional in
the strong sense: with no weights the pipeline runs geometry-only and is still the
system the paper measures (`docstruct_geo`). On three of four corpora the vision
detector shows no significant gain, so geometry-only is a legitimate default, not a
degraded mode.

**Why config is a dict and not a `ParseConfig` object.** A typed config was designed and
deliberately not built — see `memory/decisions.md`. Threading a frozen dataclass through
every function is a large refactor whose only payoff is autocomplete, and the per-call
override dict already gives thread safety. Revisit if the option count grows.

**Why chunks can exceed a size limit.** Chunk assembly uses a size *floor*
(`MIN_CHUNK_TOKENS`), not a ceiling. Structural boundaries are crossed when the content
so far is too small to stand alone. That single decision is worth +0.043 MRR against
flushing on every boundary — the largest measured gain in the system.

---

## What is deliberately not here

- **No OCR.** Born-digital only. Out of scope, not a missing feature.
- **No generation.** DocStruct stops at chunks. It is not a RAG framework.
- **No LLM anywhere on the parse path.** This is the contract. A proposal that puts a
  model call in the pipeline is rejected rather than implemented.
- **No async API.** Parsing is CPU-bound; `parse_many` with processes is the right
  parallelism. An `async def` wrapper around blocking CPU work would be theatre.
