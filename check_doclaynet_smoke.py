#!/usr/bin/env python3
"""Smoke test for local deformable-detr-doclaynet model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from models.doclaynet_detector import LocalDocLayNetDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether local deformable-detr-doclaynet can load and run inference."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("hf_models/deformable-detr-doclaynet"),
        help="Path to local model directory.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("hf_models/deformable-detr-doclaynet/examples/doclaynet_example_1.png"),
        help="Path to a test image (PNG/JPG).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection threshold passed to the detector.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many detections to print.",
    )
    parser.add_argument(
        "--require-detections",
        action="store_true",
        help="Exit with code 1 if no detections are produced.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model_dir.exists():
        print(f"FAIL: model directory not found: {args.model_dir}")
        return 1
    if not args.image.exists():
        print(f"FAIL: test image not found: {args.image}")
        return 1

    print(f"Loading model from: {args.model_dir}")
    detector = LocalDocLayNetDetector(model_path=args.model_dir, confidence_threshold=args.threshold)
    if not detector.is_ready():
        print("FAIL: detector is not ready.")
        print(f"Reason: {detector.get_last_error() or 'unknown error'}")
        return 1

    image_bytes = args.image.read_bytes()
    with Image.open(args.image) as img:
        width, height = img.size

    try:
        detections = detector.detect(image_bytes, float(width), float(height))
    except Exception as exc:  # defensive fallback
        print(f"FAIL: inference raised an exception: {exc}")
        return 1

    print(f"PASS: inference ran successfully on {args.image}")
    print(f"Detections: {len(detections)}")

    for idx, det in enumerate(detections[: max(0, args.top_k)], start=1):
        bbox = det.bbox
        print(
            f"{idx:02d}. type={det.detection_type:<7} conf={det.confidence:.3f} "
            f"bbox=({bbox.x0:.1f}, {bbox.y0:.1f}, {bbox.x1:.1f}, {bbox.y1:.1f})"
        )

    if args.require_detections and not detections:
        print("FAIL: no detections found and --require-detections was set.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
