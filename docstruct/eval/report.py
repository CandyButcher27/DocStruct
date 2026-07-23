"""Render benchmark results into a Markdown baseline report + raw JSON."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List

from docstruct import config
from docstruct.eval.benchmark import ToolResult


def _ci(result: ToolResult, metric: str) -> str:
    bounds = result.ci.get(metric)
    return f"[{bounds[0]}, {bounds[1]}]" if bounds else "—"


def _table(results: List[ToolResult]) -> List[str]:
    k = config.BENCHMARK_TOP_K
    rows = [
        f"| Rank | Tool | MRR (hybrid) | MRR 95% CI | NDCG@{k} | Recall@{k} | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Context words | MRR/1k words | Chunk s | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        star = " **(ours)**" if r.name == "docstruct" else ""
        lift = round(r.mrr - r.vec_mrr, 4)
        rows.append(
            f"| {i} | {r.name}{star} | **{r.mrr}** | {_ci(r, 'mrr')} | {r.ndcg} | {r.recall} | {r.hit1} | "
            f"{r.vec_mrr} | {lift:+} | {r.n_chunks} | {r.mean_chunk_words} | "
            f"{r.context_words} | {r.mrr_per_kword} | {r.chunk_seconds} | {r.errors} |"
        )
    return rows


def _significance_section(results: List[ToolResult], reference: str) -> List[str]:
    """Paired bootstrap of the reference tool against each baseline."""
    compared = [r for r in results if r.vs_reference]
    if not compared:
        return []
    rows = [
        f"## Is the gap real? Paired bootstrap vs `{reference}`",
        "",
        f"Every tool answers the **same** questions, so the comparison is paired: one "
        f"resample of question indices is applied to both tools, which cancels the "
        f"between-question variance and isolates the difference between the chunkers. "
        f"Positive Δ means `{reference}` is ahead. 10,000 resamples, seeded.",
        "",
        "Reading two overlapping marginal CIs as \"not significant\" is the standard "
        "way to miss a consistent per-question difference — that is what this table "
        "exists to prevent, in both directions.",
        "",
        f"| vs | Metric | Δ ({reference} − tool) | 95% CI of Δ | p | n paired | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in compared:
        for metric in ("mrr", "ndcg", "recall", "hit1"):
            stat = r.vs_reference.get(metric)
            if not stat:
                continue
            significant = stat["ci_low"] > 0 or stat["ci_high"] < 0
            verdict = ("**significant**" if significant else "not significant")
            rows.append(
                f"| {r.name} | {metric.upper()} | {stat['diff']:+} | "
                f"[{stat['ci_low']}, {stat['ci_high']}] | {stat['p_value']} | "
                f"{stat['n']} | {verdict} |"
            )
    rows.append("")
    return rows


def _extraction_table(results: List[ToolResult]) -> List[str]:
    """Extraction fidelity — the one quality axis here that is not about retrieval."""
    if not any(r.coverage for r in results):
        return []
    rows = [
        "## Extraction fidelity (no gold, no LLM)",
        "",
        "Measured against each PDF's own raw pdfplumber text, so the document is its "
        "own ground truth. This is the only cross-tool quality signal in the report "
        "that measures **extraction** rather than retrieval, and the only one "
        "available for the whole corpus — hand-annotated detection boxes exist for "
        "two documents.",
        "",
        "| Tool | Coverage | Duplication |",
        "|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.coverage, reverse=True):
        star = " **(ours)**" if r.name.startswith("docstruct") else ""
        rows += [f"| {r.name}{star} | {r.coverage} | {r.duplication} |"]
    rows += [
        "",
        "- **Coverage** — fraction of the document's word *instances* that appear in "
        "some chunk. This is where silent loss shows up and nowhere else: dropped "
        "table rows, headings that end up in no chunk, skipped figures. Counted as a "
        "multiset, so dropping every repeat of a term is not scored as covered.",
        "- **Duplication** — chunk words divided by document words. Above 1.0 means "
        "content is emitted more than once, inflating the index and letting two "
        "chunks split the evidence for one query. Overlap raises it deliberately, so "
        "read it as a cost next to coverage, not as a defect.",
        "",
    ]
    return rows


def _per_doc_table(result: ToolResult) -> List[str]:
    """Per-doc breakdown for one tool, sorted worst MRR first."""
    if not result.per_doc:
        return []
    rows = [
        f"### Per-doc breakdown: {result.name} (worst first)",
        "",
        "| Doc | Q | Chunks | Avg words | MRR | Recall@5 | Hit@1 | Hits |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(result.per_doc, key=lambda x: x["mrr"]):
        flag = " ⚠" if d["mrr"] == 0.0 else ""
        rows.append(
            f"| {d['doc']}{flag} | {d['n_questions']} | {d['n_chunks']} | "
            f"{d['avg_words_per_chunk']} | {d['mrr']} | {d['recall']} | "
            f"{d['hit1']} | {d['hits']}/{d['n_questions']} |"
        )
    return rows


def config_snapshot() -> Dict[str, object]:
    """Every public config value, so a run's settings travel with its numbers.

    Without this, two reports with different MRR give no way to tell *why* short of
    diffing ``config.py`` at the commit each was generated from.
    """
    return {
        key: value
        for key, value in sorted(vars(config).items())
        if key.isupper() and not key.startswith("_")
    }


def _config_section(meta: Dict) -> List[str]:
    cfg = meta.get("config") or {}
    if not cfg:
        return []
    chunking = ("MAX_CHUNK_TOKENS", "MIN_CHUNK_TOKENS", "CHUNK_OVERLAP_TOKENS",
                "OVERLAP_ON_BOUNDARY", "BREAK_TEXT_ON_TABLE", "BREAK_TEXT_ON_CAPTION",
                "INLINE_HEADER_TEXT", "HEADER_LEVELS")
    rows = [
        "## Run configuration",
        "",
        "Chunking settings active for this run (full snapshot in the JSON sidecar "
        "under `meta.config`):",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    rows += [f"| `{k}` | `{cfg[k]}` |" for k in chunking if k in cfg]
    rows.append("")
    return rows


def render_markdown(results: List[ToolResult], meta: Dict) -> str:
    k = config.BENCHMARK_TOP_K
    lines = [
        "# DocStruct retrieval baseline report",
        "",
        f"_Generated {meta['timestamp']}_",
        "",
        "## Setup",
        "",
        f"- **Documents:** {meta['n_docs']} born-digital PDFs",
        f"- **Questions:** {meta['n_questions']} LLM-generated (model `{meta['llm_model']}`), "
        "each with a verbatim answer span validated against the source",
        f"- **Embedder (constant):** `{config.EMBEDDING_MODEL}`  |  **Retrievers:** dense cosine and "
        f"hybrid (dense + BM25 fused by RRF, k={config.RRF_K}), top-{config.BENCHMARK_TOP_K}, per-document index",
        "- **Relevance:** a retrieved chunk counts as relevant if it contains the answer span "
        "(normalized substring, token-overlap fallback) — a deterministic proxy for RAGAS context precision/recall",
        "- **Fair-comparison principle:** embedder + retrievers are identical for every tool; "
        "**only the chunker varies**, so the table measures chunking quality. The hybrid retriever is "
        "the `RAG_Fundamentals` two-indexes-plus-RRF recipe; the **Hybrid lift** column is its MRR gain over vector-only.",
        "",
        f"Tools benchmarked: {', '.join(r.name for r in results)}."
        + (f" Skipped (deps not installed): {', '.join(meta['skipped'])}." if meta.get("skipped") else ""),
        "",
        "## Leaderboard (ranked by MRR)",
        "",
        *_table(results),
        "",
        *_extraction_table(results),
        *_significance_section(results, meta.get("reference", "docstruct")),
        "## How to read this",
        "",
        "- **MRR** — average reciprocal rank of the first chunk that contains the answer (higher = answers surface earlier).",
        "- **MRR 95% CI** — percentile bootstrap over the per-question scores (10,000 "
        "resamples, seeded). It is the uncertainty on *this tool's* number in isolation; "
        "for comparing two tools use the paired table above, not CI overlap.",
        "- **Recall@k** — fraction of questions where the answer appears in the top-k.",
        "- **Hit@1** — fraction where the very first chunk already contains the answer.",
        "- **Avg words/chunk** — chunk granularity; huge chunks can inflate recall while hurting precision/precision-of-context.",
        f"- **Context words** — words actually handed to the generator per query (summed over the "
        f"top-{k} retrieved chunks). MRR can always be bought by making chunks bigger; this is what "
        "that costs downstream in context window, latency and token spend.",
        "- **MRR/1k words** — MRR per 1000 words of retrieved context. Rewards finding the answer "
        "*and* being cheap to feed to an LLM, which raw MRR alone does not.",
        "",
        *_config_section(meta),
        "## DocStruct's unique axis (not in any competitor)",
        "",
        "Every chunk carries a section path, enabling **filtered retrieval** "
        "(`where={\"h1\": \"4. Experiments\"}`). No competitor here exposes section-hierarchy "
        "metadata, so this capability is qualitative, not in the table.",
        "",
        "## Honest caveats",
        "",
        "- Questions are **LLM-generated**: this measures retrieval consistency and the "
        "DocStruct-vs-tools delta well, but is weaker than human-judged relevance as an absolute claim.",
        "- Containment relevance can miss paraphrased answers — but it is applied **identically** to "
        "every tool, so the ranking stays fair.",
        "- Dataset is arXiv-heavy (born-digital prose); broader domains (legal/financial/manuals) are future work.",
        "- **Chunk s is not a fair speed comparison when `--cache-dir` is set.** Only the DocStruct "
        "adapter uses that cache (detector proposals and populated blocks, keyed by PDF hash), so on a "
        "warm cache its column reports cache-hit time while every other tool is measured cold. Compare "
        "wall-clock only from a run with no `--cache-dir`, or against `meta.docstruct_cold_chunk_seconds` "
        "if present.",
        "- **MRR/1k words is a tradeoff axis, not a ranking.** It necessarily favours tools that emit "
        "very small chunks and therefore retrieve very little text, regardless of whether they rank well. "
        "Read it next to MRR, not instead of it.",
        "- This is a **signal/baseline**, not the Phase-2 public benchmark (50 PDFs, 200 human-checked Q&A).",
        "",
    ]
    for r in results:
        if r.per_doc:
            lines += ["", *_per_doc_table(r), ""]
    return "\n".join(lines)


def write_report(results: List[ToolResult], meta: Dict, out_md: str, out_json: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(results, meta))
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {"meta": meta, "results": [asdict(r) for r in results]},
            fh,
            indent=2,
            ensure_ascii=False,
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
