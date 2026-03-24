#!/usr/bin/env python3
"""DocStruct main pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from models.detector import ClasswiseThresholdConfig, Detection, create_detector
from pipeline.classification import classify_blocks, classify_fused_proposal
from pipeline.confidence import ensure_confidence_scores
from pipeline.decomposition import decompose_pdf
from pipeline.hybrid_proposals import detections_to_proposals, generate_geometry_proposals
from pipeline.layout import process_page_layout
from pipeline.proposal_fusion import deduplicate_proposals, get_fusion_summary, match_and_fuse_proposals
from pipeline.reading_order import assign_reading_order
from pipeline.table_candidates import detect_geometry_table_candidates, fuse_table_candidates, model_table_candidates
from pipeline.tables_figures import refine_blocks
from pipeline.validator import validate_and_build_document
from schemas.block import BoundingBox
from utils.config import (
    get_detector_thresholds,
    get_doclaynet_confidence,
    get_table_candidate_params,
    load_config,
)
from utils.geometry import bbox_overlap, extract_text_from_bbox, refine_bbox_with_lines
from utils.logging import log_pipeline_stage, setup_logger
from utils.ocr import is_scanned_page, ocr_page
from utils.rendering import render_page_as_png

logger = setup_logger(__name__)

IMAGE_BASED_DETECTORS = {"table_transformer", "doclaynet", "combined"}
VALID_VARIANTS = ["geometry", "model", "hybrid"]
VALID_MODES = ["standard", "model-first", "true-hybrid"]

# Detection type to block type mapping for model-first pipeline
DETECTION_TO_BLOCK_TYPE = {
    "table": "table",
    "figure": "figure",
    "text": "text",
    "header": "header",
    "caption": "caption",
}


def _variant_output_path(base_output: str, variant: str) -> str:
    path = Path(base_output)
    return str(path.with_name(f"{path.stem}_{variant}{path.suffix}"))


def _convert_detections_to_blocks(
    detections: list[Detection],
    page_data,
    pdf_path: str,
) -> list[dict]:
    """
    Convert model detections directly to blocks with geometry refinement and OCR fallback.

    This is the core of the model-first pipeline where model detections drive
    block identification, and geometry/OCR are used for refinement.
    """
    blocks = []
    lines = page_data.lines

    # OCR fallback for scanned pages
    if is_scanned_page(page_data):
        logger.info(f"Page {page_data.page_num} detected as scanned -> running OCR")
        image_bytes = render_page_as_png(pdf_path, page_data.page_num)
        ocr_spans = ocr_page(
            image_bytes,
            page_data.page_num,
            page_data.width,
            page_data.height,
        )
        if ocr_spans:
            lines = ocr_spans

    for idx, det in enumerate(detections):
        block_type = DETECTION_TO_BLOCK_TYPE.get(det.detection_type)
        if block_type is None:
            logger.warning(f"Unknown detection type: {det.detection_type}, skipping")
            continue

        # Geometry refinement: expand bbox to include overlapping lines
        refined_bbox = refine_bbox_with_lines(det.bbox, lines)

        # Text extraction from refined bbox
        text = extract_text_from_bbox(lines, refined_bbox)

        block = {
            "block_id": str(uuid.uuid4()),
            "block_type": block_type,
            "page_num": page_data.page_num,
            "image_block": {
                "bbox": refined_bbox,
                "page_num": page_data.page_num,
                "source": "model",
                "confidence": det.confidence,
            },
            "bbox": refined_bbox,
            "text": text if text else None,
            "confidence": {
                "rule_score": 0.0,
                "model_score": det.confidence,
                "geometric_score": 0.3,  # Geometry used for refinement
                "final_confidence": det.confidence,
            },
            "reading_order": idx,
            "table_data": None,
            "image_metadata": None,
            "caption_id": None,
            "caption_target_id": None,
            "parent_id": None,
        }
        blocks.append(block)

    return blocks


def process_pdf_model_first(
    pdf_path: str,
    output_path: str,
    detector_type: str = "doclaynet",
    doclaynet_confidence_threshold: float | None = None,
    config_path: str | None = None,
) -> None:
    """
    Model-first hybrid pipeline: model detections drive block identification.

    This pipeline:
    1. Renders each page and runs model detection
    2. Converts detections directly to blocks with geometry refinement
    3. Falls back to OCR for scanned pages
    4. Applies reading order and final refinement
    """
    logger.info(f"[MODEL-FIRST] Processing PDF: {pdf_path}")

    cfg = load_config(config_path)
    cfg_doclaynet = get_doclaynet_confidence(cfg)
    _doclaynet_conf = doclaynet_confidence_threshold if doclaynet_confidence_threshold is not None else cfg_doclaynet

    detector = create_detector(
        detector_type=detector_type,
        doclaynet_confidence_threshold=_doclaynet_conf,
    )
    logger.info(f"Using detector: {detector.get_model_name()}")

    log_pipeline_stage(logger, "Decomposition")
    pages_data = decompose_pdf(pdf_path)
    results = []

    for page_data in pages_data:
        page_num = page_data.page_num
        logger.info(f"Processing page {page_num}")

        log_pipeline_stage(logger, "Rendering", page_num)
        image_bytes = render_page_as_png(pdf_path, page_num)

        log_pipeline_stage(logger, "Detection", page_num)
        detections = detector.detect(
            image_bytes,
            page_data.width,
            page_data.height,
        )
        logger.info(f"[Page {page_num}] Detections: {len(detections)}")

        log_pipeline_stage(logger, "Convert -> Blocks", page_num)
        blocks = _convert_detections_to_blocks(detections, page_data, pdf_path)

        log_pipeline_stage(logger, "Reading Order", page_num)
        ordered = assign_reading_order(blocks, page_data.width)

        log_pipeline_stage(logger, "Refinement", page_num)
        refined = refine_blocks(ordered, page_data)

        log_pipeline_stage(logger, "Confidence", page_num)
        validated = ensure_confidence_scores(refined)

        results.append((page_data, validated))

    log_pipeline_stage(logger, "Schema Validation")
    document = validate_and_build_document(results, Path(pdf_path).name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

    total_blocks = len(document.get_all_blocks())
    block_types = {}
    for block in document.get_all_blocks():
        block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
    logger.info(f"[model-first] Summary: {total_blocks} blocks extracted")
    for block_type, count in sorted(block_types.items()):
        logger.info(f"  {block_type}: {count}")

    logger.info("Pipeline complete!")


def _find_matching_layout_block(bbox: BoundingBox, layout_blocks: list, threshold: float = 0.3):
    """Find the LayoutBlock that best matches a bounding box."""
    best_match = None
    best_iou = 0.0

    for block in layout_blocks:
        block_bbox = getattr(block, "bbox", None)
        if block_bbox is None:
            continue

        iou = bbox_overlap(bbox, block_bbox)
        if iou > best_iou and iou >= threshold:
            best_iou = iou
            best_match = block

    return best_match


def _calculate_page_avg_font_size(layout_blocks: list) -> float:
    """Calculate average font size across all layout blocks on a page."""
    total = 0.0
    count = 0
    for block in layout_blocks:
        if hasattr(block, "avg_font_size") and block.avg_font_size > 0:
            total += block.avg_font_size
            count += 1
    return total / count if count > 0 else 12.0


def process_pdf_true_hybrid(
    pdf_path: str,
    output_path: str,
    detector_type: str = "doclaynet",
    doclaynet_confidence_threshold: float | None = None,
    proposal_iou_threshold: float = 0.35,
    nms_iou_threshold: float = 0.50,
    config_path: str | None = None,
) -> None:
    """
    TRUE HYBRID pipeline: Model detects + Geometry confirms/fills gaps.

    This pipeline:
    1. Decomposes PDF and generates geometry proposals (independent)
    2. Runs model detection and converts to proposals
    3. Matches and fuses proposals (confirmed, model-only, geometry-only)
    4. Applies source-appropriate classification weights
    5. Deduplicates with priority-based NMS
    6. Applies reading order and refinement

    Args:
        pdf_path: Path to input PDF
        output_path: Path for output JSON
        detector_type: Model detector to use
        doclaynet_confidence_threshold: Confidence threshold for DocLayNet
        proposal_iou_threshold: IoU threshold for matching proposals
        nms_iou_threshold: IoU threshold for NMS deduplication
        config_path: Optional config file path
    """
    logger.info(f"[TRUE-HYBRID] Processing PDF: {pdf_path}")

    cfg = load_config(config_path)
    cfg_doclaynet = get_doclaynet_confidence(cfg)
    _doclaynet_conf = (
        doclaynet_confidence_threshold
        if doclaynet_confidence_threshold is not None
        else cfg_doclaynet
    )

    detector = create_detector(
        detector_type=detector_type,
        doclaynet_confidence_threshold=_doclaynet_conf,
    )
    logger.info(f"Using detector: {detector.get_model_name()}")

    log_pipeline_stage(logger, "Decomposition")
    pages_data = decompose_pdf(pdf_path)
    results = []

    total_confirmed = 0
    total_model_only = 0
    total_geo_only = 0

    for page_data in pages_data:
        page_num = page_data.page_num
        logger.info(f"Processing page {page_num}")

        # Stage 1: Generate geometry proposals (INDEPENDENT of model)
        log_pipeline_stage(logger, "Layout & Geometry Proposals", page_num)
        layout_blocks, image_blocks = process_page_layout(page_data)
        geo_proposals = generate_geometry_proposals(
            page_data, layout_blocks, page_data.lines
        )
        logger.info(f"[Page {page_num}] Geometry proposals: {len(geo_proposals)}")

        # Stage 2: Model detection
        log_pipeline_stage(logger, "Model Detection", page_num)
        image_bytes = render_page_as_png(pdf_path, page_num)
        detections = detector.detect(image_bytes, page_data.width, page_data.height)
        logger.info(f"[Page {page_num}] Model detections: {len(detections)}")

        # Convert detections to proposals
        model_proposals = detections_to_proposals(detections)

        # Stage 3: Match & Fuse
        log_pipeline_stage(logger, "Proposal Fusion", page_num)
        fused_proposals, diag = match_and_fuse_proposals(
            model_proposals, geo_proposals, iou_threshold=proposal_iou_threshold
        )
        logger.info(f"[Page {page_num}] Fusion: {get_fusion_summary(diag)}")

        total_confirmed += diag["confirmed"]
        total_model_only += diag["model_only"]
        total_geo_only += diag["geometry_only"]

        # Stage 4: NMS Deduplication
        log_pipeline_stage(logger, "Deduplication", page_num)
        deduped = deduplicate_proposals(fused_proposals, iou_threshold=nms_iou_threshold)
        logger.info(f"[Page {page_num}] After dedup: {len(deduped)} blocks")

        # Stage 5: Classification
        log_pipeline_stage(logger, "Classification", page_num)
        avg_font_size = _calculate_page_avg_font_size(layout_blocks)
        classified_blocks = []

        for proposal in deduped:
            # Find matching LayoutBlock if any (for rule scoring)
            matching_block = _find_matching_layout_block(proposal.bbox, layout_blocks)

            classification = classify_fused_proposal(
                proposal,
                matching_block,
                page_data.width,
                page_data.height,
                avg_font_size,
                cfg=cfg,
            )

            # Build block dict
            text = ""
            if matching_block is not None:
                text = getattr(matching_block, "text", "") or ""
            elif proposal.source != "geometry_only":
                # Try to extract text from bbox for model-detected blocks
                text = extract_text_from_bbox(page_data.lines, proposal.bbox)

            block_data = {
                "block_id": str(uuid.uuid4()),
                "block_type": classification["block_type"],
                "page_num": page_num,
                "bbox": proposal.bbox,
                "text": text if text else None,
                "confidence": classification["confidence"],
                "has_model_support": classification["has_model_support"],
                "proposal_source": classification["proposal_source"],
                "agreement_score": classification["agreement_score"],
                "reading_order": 0,  # Will be set by assign_reading_order
                "table_data": None,
                "image_metadata": None,
                "caption_id": None,
                "caption_target_id": None,
                "parent_id": None,
            }

            # Add layout_block or image_block for downstream processing
            if matching_block is not None:
                block_data["layout_block"] = matching_block
            else:
                block_data["image_block"] = {
                    "bbox": proposal.bbox,
                    "page_num": page_num,
                    "source": proposal.source,
                    "confidence": proposal.final_confidence,
                }

            classified_blocks.append(block_data)

        # Stage 6: Reading order & Refinement
        log_pipeline_stage(logger, "Reading Order", page_num)
        ordered = assign_reading_order(classified_blocks, page_data.width)
        refined = refine_blocks(ordered, page_data)
        validated = ensure_confidence_scores(refined)

        results.append((page_data, validated))

    # Final output
    log_pipeline_stage(logger, "Schema Validation")
    document = validate_and_build_document(results, Path(pdf_path).name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

    # Summary
    total_blocks = len(document.get_all_blocks())
    block_types = {}
    for block in document.get_all_blocks():
        block_types[block.block_type] = block_types.get(block.block_type, 0) + 1

    logger.info(f"[true-hybrid] Summary: {total_blocks} blocks extracted")
    logger.info(f"  Sources: confirmed={total_confirmed}, model_only={total_model_only}, geo_only={total_geo_only}")
    for block_type, count in sorted(block_types.items()):
        logger.info(f"  {block_type}: {count}")

    logger.info("Pipeline complete!")


def process_pdf(
    pdf_path: str,
    output_path: str,
    detector_type: str = "stub",
    variant: str = "hybrid",
    output_variants: str = "single",
    fail_on_detector_error: bool = False,
    # All numeric params are None by default so they can be overridden by config
    geometry_table_threshold: float | None = None,
    table_overlap_threshold: float | None = None,
    fusion_acceptance_threshold: float | None = None,
    threshold_table: float | None = None,
    threshold_figure: float | None = None,
    threshold_text: float | None = None,
    threshold_header: float | None = None,
    threshold_caption: float | None = None,
    doclaynet_confidence_threshold: float | None = None,
    table_match_threshold: float | None = None,
    config_path: str | None = None,
) -> None:
    logger.info(f"Processing PDF: {pdf_path}")

    # Load config (cached after first call)
    cfg = load_config(config_path)
    tc_params = get_table_candidate_params(cfg)
    cfg_detector = get_detector_thresholds(cfg)
    cfg_doclaynet = get_doclaynet_confidence(cfg)

    # CLI overrides config; fall back to config value when CLI arg is None
    _geo_threshold = geometry_table_threshold if geometry_table_threshold is not None else tc_params["geometry_threshold"]
    _overlap_threshold = table_overlap_threshold if table_overlap_threshold is not None else tc_params["overlap_threshold"]
    _fusion_threshold = fusion_acceptance_threshold if fusion_acceptance_threshold is not None else tc_params["fusion_acceptance"]
    _table_match = table_match_threshold if table_match_threshold is not None else tc_params["table_match_threshold"]
    _doclaynet_conf = doclaynet_confidence_threshold if doclaynet_confidence_threshold is not None else cfg_doclaynet

    # Build class-wise threshold configuration (CLI wins, else use config values)
    threshold_config = ClasswiseThresholdConfig(
        table=threshold_table if threshold_table is not None else cfg_detector.table,
        figure=threshold_figure if threshold_figure is not None else cfg_detector.figure,
        text=threshold_text if threshold_text is not None else cfg_detector.text,
        header=threshold_header if threshold_header is not None else cfg_detector.header,
        caption=threshold_caption if threshold_caption is not None else cfg_detector.caption,
    )

    if variant not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{variant}'. Valid: {', '.join(VALID_VARIANTS)}")
    if output_variants not in {"single", "all"}:
        raise ValueError("output_variants must be 'single' or 'all'")
    selected_variants = [variant] if output_variants == "single" else list(VALID_VARIANTS)

    detector = create_detector(
        detector_type=detector_type,
        threshold_config=threshold_config,
        doclaynet_confidence_threshold=_doclaynet_conf,
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
        logger.info(f"[Page {page_num}] Detections total: {len(detections)}")

        if detections:
            label_counts = {}
            for d in detections:
                label_counts[d.detection_type] = label_counts.get(d.detection_type, 0) + 1
            logger.info(f"[Page {page_num}] Detection label distribution: {dict(label_counts)}")

            confidences = [d.confidence for d in detections]
            logger.info(
                f"[Page {page_num}] Detection confidence stats: "
                f"min={min(confidences):.3f}, max={max(confidences):.3f}, avg={sum(confidences)/len(confidences):.3f}"
            )
        else:
            logger.warning(f"[Page {page_num}] No detections from model")

        geometry_candidates, geo_diag = detect_geometry_table_candidates(
            layout_blocks,
            page_data.lines,
            score_threshold=_geo_threshold,
        )
        model_candidates, model_diag = model_table_candidates(
            detections,
            score_threshold=threshold_config.table,
        )
        fused_candidates, fusion_diag = fuse_table_candidates(
            geometry_candidates,
            model_candidates,
            overlap_threshold=_overlap_threshold,
            fusion_acceptance_threshold=_fusion_threshold,
            model_confidence_threshold=threshold_config.table,
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
                table_match_threshold=_table_match,
                cfg=cfg,
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
    logger.info(f"Threshold config used: {threshold_config}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocStruct - Layout-first PDF understanding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.pdf output.json
  python main.py --detector combined input.pdf output.json --output-variants all
  python main.py --detector doclaynet input.pdf output.json --variant model
  python main.py --mode model-first --detector doclaynet input.pdf output.json
  python main.py --mode true-hybrid --detector doclaynet input.pdf output.json
        """,
    )
    parser.add_argument("input_pdf", type=str, help="Path to input PDF file")
    parser.add_argument("output_json", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a custom YAML config file (default: config/defaults.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=VALID_MODES,
        help="Pipeline mode: 'standard' (layout-first), 'model-first' (detection-driven), or 'true-hybrid' (model+geometry fusion)",
    )
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
    parser.add_argument("--geometry-table-threshold", type=float, default=None)
    parser.add_argument("--table-overlap-threshold", type=float, default=None)
    parser.add_argument("--fusion-acceptance-threshold", type=float, default=None)
    parser.add_argument("--model-confidence-threshold", type=float, default=None)
    parser.add_argument("--doclaynet-confidence-threshold", type=float, default=None)
    parser.add_argument("--table-match-threshold", type=float, default=None)
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
        if args.mode == "model-first":
            process_pdf_model_first(
                args.input_pdf,
                args.output_json,
                detector_type=args.detector,
                doclaynet_confidence_threshold=args.doclaynet_confidence_threshold,
                config_path=args.config,
            )
        elif args.mode == "true-hybrid":
            process_pdf_true_hybrid(
                args.input_pdf,
                args.output_json,
                detector_type=args.detector,
                doclaynet_confidence_threshold=args.doclaynet_confidence_threshold,
                config_path=args.config,
            )
        else:
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
                config_path=args.config,
            )
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
