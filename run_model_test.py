#!/usr/bin/env python3
"""Quick model pipeline smoke test for DocStruct.

Runs main.py with model-capable detector and verifies output JSON files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _variant_path(base_output: Path, variant: str) -> Path:
    return base_output.with_name(f"{base_output.stem}_{variant}{base_output.suffix}")


def _summarize(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    counts: dict[str, int] = {}
    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            t = block.get("block_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-based DocStruct smoke test.")
    parser.add_argument("--pdf", type=Path, default=Path("sample.pdf"), help="Input PDF path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model_test.json"),
        help="Base output JSON path",
    )
    parser.add_argument(
        "--detector",
        choices=["doclaynet", "combined", "table_transformer"],
        default="combined",
        help="Model-capable detector backend",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs in main.py")
    parser.add_argument(
        "--fail-on-detector-error",
        action="store_true",
        help="Fail fast if detector model cannot initialize",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {args.pdf}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        str(args.pdf),
        str(args.output),
        "--detector",
        args.detector,
        "--output-variants",
        "all",
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.fail_on_detector_error:
        cmd.append("--fail-on-detector-error")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    expected = [_variant_path(args.output, v) for v in ("geometry", "model", "hybrid")]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing output files: {', '.join(str(p) for p in missing)}")

    print("\nModel smoke test outputs:")
    for p in expected:
        counts = _summarize(p)
        print(f"- {p}: {counts}")


if __name__ == "__main__":
    main()

