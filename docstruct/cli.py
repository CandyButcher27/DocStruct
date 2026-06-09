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
    result = run_pipeline(args.pdf, weights=args.weights)
    diag = result.diagnostics
    print(
        f"[{diag['mode']}] pages={diag['pages']} blocks={diag['n_blocks']} "
        f"chunks={diag['n_chunks']} "
        f"(confirmed/disputed via matched={diag['matched']}, "
        f"uni_model={diag['unmatched_model']}, uni_geo={diag['unmatched_geometry']})"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docstruct",
        description="Local, deterministic, structure-aware PDF chunking for RAG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline on a PDF and print chunks")
    run_p.add_argument("pdf", help="path to a born-digital PDF")
    run_p.add_argument("--weights", default=None, help="YOLOv8 weights to enable hybrid mode")
    run_p.add_argument("--json", default=None, help="write all chunks to this JSON file")
    run_p.add_argument("--limit", type=int, default=15, help="chunks to preview")
    run_p.set_defaults(func=_cmd_run)

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
