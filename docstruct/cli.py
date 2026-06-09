"""Command-line interface for DocStruct."""

from __future__ import annotations

import argparse
import dataclasses
import json
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
    retriever = Retriever(VectorStore(persist_dir=args.db))
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
    q_p.add_argument("--h1", default=None, help="filter to a top-level section")
    q_p.set_defaults(func=_cmd_query)

    v_p = sub.add_parser("visualize", help="Render an annotated PDF of detected blocks")
    v_p.add_argument("pdf", help="path to a born-digital PDF")
    v_p.add_argument("--out", required=True, help="output annotated PDF path")
    v_p.add_argument("--weights", default=None)
    v_p.add_argument("--cache-dir", default=None)
    v_p.set_defaults(func=_cmd_visualize)

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
