"""Command-line interface for DocStruct."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Optional, Sequence

from docstruct.pipeline import run_pipeline


def _section_label(section) -> str:
    parts = [p for p in (section.h1, section.h2, section.h3) if p]
    return " > ".join(parts) if parts else "(root)"


def _cmd_run(args) -> int:
    result = run_pipeline(args.pdf, weights=args.weights, cache_dir=args.cache_dir)
    diag = result.diagnostics
    print(
        f"[{diag['mode']}] pages={diag['pages']} blocks={diag['n_blocks']} "
        f"chunks={diag['n_chunks']} "
        f"(matched={diag['matched']}, uni_model={diag['unmatched_model']}, "
        f"uni_geo={diag['unmatched_geometry']})"
    )
    for chunk in result.chunks[: args.limit]:
        preview = chunk.content.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:90] + "..."
        print(f"  [{chunk.chunk_type}] {_section_label(chunk.section_path)} "
              f"(p{chunk.page_num}): {preview}")

    if args.json:
        payload = [dataclasses.asdict(c) for c in result.chunks]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"wrote {len(payload)} chunks -> {args.json}")
    return 0


def _cmd_index(args) -> int:
    from docstruct.indexing.vector_store import VectorStore

    store = VectorStore(persist_dir=args.db)
    total = 0
    for pdf in args.pdfs:
        result = run_pipeline(pdf, weights=args.weights, cache_dir=args.cache_dir)
        n = store.index(result.chunks, doc_id=pdf)
        total += n
        print(f"indexed {n} chunks from {pdf}")
    print(f"collection now holds {store.count()} chunks")
    return 0


def _cmd_query(args) -> int:
    from docstruct.indexing.vector_store import VectorStore
    from docstruct.query.retriever import Retriever

    where = {"h1": args.h1} if args.h1 else None
    retriever = Retriever(VectorStore(persist_dir=args.db), hybrid=args.hybrid)
    results = retriever.retrieve(args.text, top_k=args.top_k, where=where)
    if not results:
        print("no results")
        return 0
    for r in results:
        preview = r.content.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"{r.citation()}\n  {preview}\n")
    return 0


def _cmd_visualize(args) -> int:
    from docstruct.visualize import render_annotated

    result = run_pipeline(args.pdf, weights=args.weights, cache_dir=args.cache_dir)
    out = render_annotated(args.pdf, result.blocks, args.out)
    print(f"wrote annotated PDF -> {out}")
    return 0


def _cmd_export_annotations(args) -> int:
    from docstruct.eval.annotate import export_annotation_session

    out, n_boxes, n_pages = export_annotation_session(
        args.pdf, args.out, weights=args.weights, dpi=args.dpi
    )
    print(f"wrote draft session -> {out} ({n_boxes} boxes, {n_pages} pages)")
    print("Open tools/annotate.html in a browser, load this file, correct the "
          "boxes, and download the ground-truth JSON to data/annotations/.")
    return 0


def _cmd_gen_qa(args) -> int:
    from docstruct.llm.client import LLMClient
    from docstruct.eval.qa_generator import generate_for_pdf, save_qa, load_qa

    client = LLMClient(model=args.model) if args.model else LLMClient()
    print(f"generating Q&A with model '{client.model}' ...", flush=True)

    # Resume: keep questions already saved, skip documents already done.
    all_items = load_qa(args.out) if os.path.exists(args.out) else []
    done = {it.source_doc for it in all_items}
    if done:
        print(f"resuming: {len(all_items)} items for {len(done)} docs already on disk", flush=True)

    for pdf in args.pdfs:
        if os.path.basename(pdf) in done:
            print(f"  {pdf}: skip (already done)", flush=True)
            continue
        items = generate_for_pdf(pdf, client, weights=args.weights, n=args.per_doc, cache_dir=args.cache_dir)
        all_items.extend(items)
        save_qa(all_items, args.out)  # incremental: survive interruptions
        print(f"  {pdf}: {len(items)} questions (total {len(all_items)})", flush=True)

    print(f"wrote {len(all_items)} Q&A items -> {args.out}", flush=True)
    return 0


def _cmd_benchmark(args) -> int:
    import glob as _glob

    from docstruct.eval.adapters import get_adapters, _ALL
    from docstruct.eval.benchmark import run_benchmark
    from docstruct.eval.qa_generator import load_qa
    from docstruct.eval.report import now_iso, write_report

    qa = load_qa(args.qa)
    pdfs = sorted(_glob.glob(os.path.join(args.pdfs_dir, "*.pdf")))
    names = args.tools.split(",") if args.tools else None
    adapters = get_adapters(names, weights=args.weights, cache_dir=args.cache_dir)
    skipped = [n for n in (names or _ALL) if n not in adapters]
    print(f"tools: {list(adapters)}  skipped: {skipped}")

    results = run_benchmark(adapters, pdfs, qa, top_k=args.top_k, cache_dir=args.cache_dir, rrf_k=args.rrf_k)
    meta = {
        "timestamp": now_iso(),
        "n_docs": len({q.source_doc for q in qa}),
        "n_questions": len(qa),
        "llm_model": args.model or "gpt-oss:120b",
        "skipped": skipped,
    }
    write_report(results, meta, args.report_md, args.report_json)
    print(f"\nwrote report -> {args.report_md}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.name:14} MRR={r.mrr}  NDCG={r.ndcg}  Recall={r.recall}  Hit@1={r.hit1}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docstruct",
        description="Local, deterministic, structure-aware PDF chunking for RAG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline on a PDF and print chunks")
    run_p.add_argument("pdf", help="path to a born-digital PDF")
    run_p.add_argument("--weights", default=None, help="YOLOv8 weights to enable hybrid mode")
    run_p.add_argument("--cache-dir", default=None, help="cache detector proposals here")
    run_p.add_argument("--json", default=None, help="write all chunks to this JSON file")
    run_p.add_argument("--limit", type=int, default=15, help="chunks to preview")
    run_p.set_defaults(func=_cmd_run)

    idx_p = sub.add_parser("index", help="Index one or more PDFs into a vector store")
    idx_p.add_argument("pdfs", nargs="+", help="PDF paths to index")
    idx_p.add_argument("--db", required=True, help="Chroma persist directory")
    idx_p.add_argument("--weights", default=None)
    idx_p.add_argument("--cache-dir", default=None)
    idx_p.set_defaults(func=_cmd_index)

    q_p = sub.add_parser("query", help="Query an indexed vector store")
    q_p.add_argument("text", help="query string")
    q_p.add_argument("--db", required=True, help="Chroma persist directory")
    q_p.add_argument("--top-k", type=int, default=5)
    q_p.add_argument("--h1", default=None, help="filter to a top-level section (dense mode)")
    q_p.add_argument("--hybrid", action="store_true", help="dense + BM25 fused by RRF")
    q_p.set_defaults(func=_cmd_query)

    v_p = sub.add_parser("visualize", help="Render an annotated PDF of detected blocks")
    v_p.add_argument("pdf", help="path to a born-digital PDF")
    v_p.add_argument("--out", required=True, help="output annotated PDF path")
    v_p.add_argument("--weights", default=None)
    v_p.add_argument("--cache-dir", default=None)
    v_p.set_defaults(func=_cmd_visualize)

    e_p = sub.add_parser(
        "export-annotations",
        help="Export a model-assisted annotation session for tools/annotate.html",
    )
    e_p.add_argument("pdf", help="path to a born-digital PDF")
    e_p.add_argument("--out", required=True, help="session JSON output path")
    e_p.add_argument("--weights", default=None, help="weights for hybrid pre-fill")
    e_p.add_argument("--dpi", type=int, default=110, help="page render DPI for the UI")
    e_p.set_defaults(func=_cmd_export_annotations)

    g_p = sub.add_parser("gen-qa", help="Generate LLM Q&A gold for benchmarking")
    g_p.add_argument("pdfs", nargs="+", help="PDF paths")
    g_p.add_argument("--out", required=True, help="output Q&A JSON path")
    g_p.add_argument("--weights", default=None, help="weights for hybrid chunking")
    g_p.add_argument("--per-doc", type=int, default=5, help="questions per document")
    g_p.add_argument("--model", default=None, help="LLM model id (default gpt-oss:120b)")
    g_p.add_argument("--cache-dir", default=None, help="cache detector proposals (reused by benchmark)")
    g_p.set_defaults(func=_cmd_gen_qa)

    b_p = sub.add_parser("benchmark", help="Run the cross-tool retrieval benchmark")
    b_p.add_argument("--pdfs-dir", required=True, help="directory of PDFs")
    b_p.add_argument("--qa", required=True, help="Q&A JSON from gen-qa")
    b_p.add_argument("--report-md", default="reports/baseline_report.md")
    b_p.add_argument("--report-json", default="reports/baseline_results.json")
    b_p.add_argument("--tools", default=None, help="comma list (default: all available)")
    b_p.add_argument("--weights", default=None, help="weights for the DocStruct adapter")
    b_p.add_argument("--top-k", type=int, default=5)
    b_p.add_argument("--model", default=None, help="LLM model id used (recorded in report)")
    b_p.add_argument("--cache-dir", default=None, help="reuse cached detector proposals from gen-qa")
    b_p.add_argument("--rrf-k", type=int, default=60, help="RRF k constant (default 60; lower weights top results more)")
    b_p.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
