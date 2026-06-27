"""Render benchmark results into a Markdown baseline report + raw JSON."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List

from docstruct import config
from docstruct.eval.benchmark import ToolResult


def _table(results: List[ToolResult]) -> List[str]:
    k = config.BENCHMARK_TOP_K
    rows = [
        f"| Rank | Tool | MRR (hybrid) | NDCG@{k} | Recall@{k} | Hit@1 | MRR (vector) | Hybrid lift | Chunks | Avg words/chunk | Chunk s | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        star = " **(ours)**" if r.name == "docstruct" else ""
        lift = round(r.mrr - r.vec_mrr, 4)
        rows.append(
            f"| {i} | {r.name}{star} | **{r.mrr}** | {r.ndcg} | {r.recall} | {r.hit1} | "
            f"{r.vec_mrr} | {lift:+} | {r.n_chunks} | {r.mean_chunk_words} | "
            f"{r.chunk_seconds} | {r.errors} |"
        )
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


def render_markdown(results: List[ToolResult], meta: Dict) -> str:
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
        "## How to read this",
        "",
        "- **MRR** — average reciprocal rank of the first chunk that contains the answer (higher = answers surface earlier).",
        "- **Recall@k** — fraction of questions where the answer appears in the top-k.",
        "- **Hit@1** — fraction where the very first chunk already contains the answer.",
        "- **Avg words/chunk** — chunk granularity; huge chunks can inflate recall while hurting precision/precision-of-context.",
        "",
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
