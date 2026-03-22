#!/usr/bin/env python3
"""DocStruct main pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from models.detector import ClasswiseThresholdConfig, create_detector
from pipeline.classification import classify_blocks
from pipeline.confidence import ensure_confidence_scores
from pipeline.decomposition import decompose_pdf
from pipeline.layout import process_page_layout
from pipeline.reading_order import assign_reading_order
from pipeline.table_candidates import detect_geometry_table_candidates, fuse_table_candidates, model_table_candidates
from pipeline.tables_figures import refine_blocks
from pipeline.validator import validate_and_build_document
from utils.logging import log_pipeline_stage, setup_logger
from utils.rendering import render_page_as_png

logger = setup_logger(__name__)

IMAGE_BASED_DETECTORS = {"table_transformer", "doclaynet", "combined"}
VALID_VARIANTS = ["geometry", "model", "hybrid"]


def _variant_output_path(base_output: str, variant: str) -> str:
    path = Path(base_output)
    return str(path.with_name(f"{path.stem}_{variant}{path.suffix}"))


def process_pdf(
    pdf_path: str,
    output_path: str,
    detector_type: str = "stub",
    variant: str = "hybrid",
    output_variants: str = "single",
    fail_on_detector_error: bool = False,
    geometry_table_threshold: float = 0.45,
    table_overlap_threshold: float = 0.35,
    fusion_acceptance_threshold: float = 0.45,
    threshold_table: float = 0.7,
    threshold_figure: float = 0.7,
    threshold_text: float = 0.5,
    threshold_header: float = 0.6,
    threshold_caption: float = 0.6,
    doclaynet_confidence_threshold: float = 0.5,
    table_match_threshold: float = 0.4,
) -> None:
    logger.info(f"Processing PDF: {pdf_path}")

    if variant not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{variant}'. Valid: {', '.join(VALID_VARIANTS)}")
    if output_variants not in {"single", "all"}:
        raise ValueError("output_variants must be 'single' or 'all'")
    selected_variants = [variant] if output_variants == "single" else list(VALID_VARIANTS)

    # Build class-wise threshold configuration
    threshold_config = ClasswiseThresholdConfig(
        table=threshold_table,
        figure=threshold_figure,
        text=threshold_text,
        header=threshold_header,
        caption=threshold_caption,
    )

    detector = create_detector(
        detector_type=detector_type,
        threshold_config=threshold_config,
        doclaynet_confidence_threshold=doclaynet_confidence_threshold,
    )
    logger.info(f"Using detector: {detector.get_model_name()}")

    if detector_type in IMAGE_BASED_DETECTORS and not detector.is_ready():
        err = detector.get_last_error() or "Unknown detector initialization error"
        message = f"Detector '{detector_type}' is not ready: {err}"
        if fail_on_detector_error:
            raise RuntimeError(message)
        logger.warning(f"{message}. Continuing with geometry-only fallback where possible.")

    log_pipeline_stage(logger, "Decomposition")
    pages_data = decompose_pdf(pdf_path)
    results_by_variant = {v: [] for v in selected_variants}

    for page_data in pages_data:
        page_num = page_data.page_num
        logger.info(f"Processing page {page_num}")

        log_pipeline_stage(logger, "Layout Formation", page_num)
        layout_blocks, image_blocks = process_page_layout(page_data)

        log_pipeline_stage(logger, "Classification", page_num)
        page_image_bytes = b""
        if detector_type in IMAGE_BASED_DETECTORS:
            page_image_bytes = render_page_as_png(pdf_path, page_num) or b""

        detections = detector.detect(page_image_bytes, page_data.width, page_data.height)
        geometry_candidates, geo_diag = detect_geometry_table_candidates(
            layout_blocks,
            page_data.lines,
            score_threshold=geometry_table_threshold,
        )
        model_candidates, model_diag = model_table_candidates(
            detections,
            score_threshold=threshold_table,
        )
        fused_candidates, fusion_diag = fuse_table_candidates(
            geometry_candidates,
            model_candidates,
            overlap_threshold=table_overlap_threshold,
            fusion_acceptance_threshold=fusion_acceptance_threshold,
            model_confidence_threshold=threshold_table,
        )
        logger.info(
            "Page %s table diagnostics: geometry_candidates=%s model_candidates=%s fused_tables=%s dropped_candidates=%s",
            page_num,
            geo_diag.get("geometry_candidates_kept", 0),
            model_diag.get("model_candidates", 0),
            fusion_diag.get("fused_tables", 0),
            fusion_diag.get("dropped_candidates", 0),
        )

        table_candidates_by_variant = {
            "geometry": geometry_candidates,
            "model": model_candidates,
            "hybrid": fused_candidates,
        }

        for out_variant in selected_variants:
            classified_blocks = classify_blocks(
                layout_blocks,
                image_blocks,
                detections,
                page_data.width,
                page_data.height,
                page_data.lines,
                variant_mode=out_variant,
                table_candidates=table_candidates_by_variant.get(out_variant, []),
                table_match_threshold=table_match_threshold,
            )
            ordered_blocks = assign_reading_order(classified_blocks, page_data.width)
            refined_blocks = refine_blocks(ordered_blocks, page_data)
            validated_blocks = ensure_confidence_scores(refined_blocks)
            results_by_variant[out_variant].append((page_data, validated_blocks))

    for out_variant in selected_variants:
        log_pipeline_stage(logger, f"Schema Validation ({out_variant})")
        document = validate_and_build_document(results_by_variant[out_variant], Path(pdf_path).name)
        final_output_path = output_path if output_variants == "single" else _variant_output_path(output_path, out_variant)
        Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing {out_variant} output to {final_output_path}")
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

        total_blocks = len(document.get_all_blocks())
        block_types = {}
        for block in document.get_all_blocks():
            block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
        logger.info(f"[{out_variant}] Summary: {total_blocks} blocks extracted")
        for block_type, count in sorted(block_types.items()):
            logger.info(f"  {block_type}: {count}")

    logger.info("Pipeline complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocStruct - Layout-first PDF understanding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.pdf output.json
  python main.py --detector combined input.pdf output.json --output-variants all
  python main.py --detector doclaynet input.pdf output.json --variant model
        """,
    )
    parser.add_argument("input_pdf", type=str, help="Path to input PDF file")
    parser.add_argument("output_json", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--detector",
        type=str,
        default="stub",
        choices=["stub", "table_transformer", "doclaynet", "combined"],
        help="Detector backend to use",
    )
    parser.add_argument("--variant", type=str, default="hybrid", choices=VALID_VARIANTS)
    parser.add_argument("--output-variants", type=str, default="single", choices=["single", "all"])
    parser.add_argument("--fail-on-detector-error", action="store_true")
    parser.add_argument("--geometry-table-threshold", type=float, default=0.45)
    parser.add_argument("--table-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--fusion-acceptance-threshold", type=float, default=0.45)
    parser.add_argument("--model-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--doclaynet-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--table-match-threshold", type=float, default=0.4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging

        logger.setLevel(logging.DEBUG)

    if not Path(args.input_pdf).exists():
        logger.error(f"Input file not found: {args.input_pdf}")
        sys.exit(1)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        process_pdf(
            args.input_pdf,
            args.output_json,
            detector_type=args.detector,
            variant=args.variant,
            output_variants=args.output_variants,
            fail_on_detector_error=args.fail_on_detector_error,
            geometry_table_threshold=args.geometry_table_threshold,
            table_overlap_threshold=args.table_overlap_threshold,
            fusion_acceptance_threshold=args.fusion_acceptance_threshold,
            threshold_table=args.model_confidence_threshold,
            threshold_figure=args.model_confidence_threshold,
            threshold_text=args.model_confidence_threshold,
            threshold_header=args.model_confidence_threshold,
            threshold_caption=args.model_confidence_threshold,
            doclaynet_confidence_threshold=args.doclaynet_confidence_threshold,
            table_match_threshold=args.table_match_threshold,
        )
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
