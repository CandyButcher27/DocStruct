"""Evaluation runner for local PDF, local DocLayNet, and HF page-image modes."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.ground_truth import (
    HF_DOCLAYNET_DATASET_ID,
    HF_PUBLAYNET_DATASET_ID,
    load_doclaynet_hf,
    load_doclaynet_local,
    load_hf_publaynet,
    load_publaynet_sample,
)
from evaluation.metrics import compute_map_at_thresholds
from main import process_pdf
from pipeline.decomposition import PageData
from pipeline.layout import process_page_layout
from utils.logging import setup_logger
from utils.ocr import ocr_page

logger = setup_logger(__name__)
VARIANTS = ["geometry", "model", "hybrid"]


def _variant_output_path(base_output: Path, variant: str) -> Path:
    return base_output.with_name(f"{base_output.stem}_{variant}{base_output.suffix}")


def write_csv_report(results: List[Dict[str, Any]], output_path: str) -> None:
    if not results:
        raise RuntimeError("No evaluation results were produced")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id",
        "variant",
        "mAP@0.50",
        "mAP@0.75",
        "macro_f1@0.50",
        "text_f1",
        "header_f1",
        "table_f1",
        "figure_f1",
        "caption_f1",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "doc_id": row.get("doc_id", ""),
                    "variant": row.get("variant", ""),
                    "mAP@0.50": metrics.get("mAP@0.50", ""),
                    "mAP@0.75": metrics.get("mAP@0.75", ""),
                    "macro_f1@0.50": metrics.get("macro_f1@0.50", ""),
                    "text_f1": metrics.get("per_class_f1", {}).get("text", ""),
                    "header_f1": metrics.get("per_class_f1", {}).get("header", ""),
                    "table_f1": metrics.get("per_class_f1", {}).get("table", ""),
                    "figure_f1": metrics.get("per_class_f1", {}).get("figure", ""),
                    "caption_f1": metrics.get("per_class_f1", {}).get("caption", ""),
                }
            )
    logger.info(f"Report written to {output_path}")


def _normalize_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for block in page_blocks:
        bbox_obj = None
        if "layout_block" in block:
            bbox_obj = block["layout_block"].bbox
        elif "image_block" in block:
            bbox_obj = block["image_block"].get("bbox")
        elif "bbox" in block:
            bbox_obj = block["bbox"]
        
        if bbox_obj:
            # Handle Pydantic BoundingBox objects vs dicts
            if hasattr(bbox_obj, "model_dump"):
                bbox_dict = bbox_obj.model_dump()
            elif hasattr(bbox_obj, "dict"):
                bbox_dict = bbox_obj.dict()
            else:
                bbox_dict = bbox_obj
                
            normalized.append({
                "block_type": block.get("block_type", "text"),
                "bbox": bbox_dict,
                "confidence": block.get("confidence", {}).get("final_confidence", 0.0),
            })
    return normalized


def _evaluate_local_pdfs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    docs = load_publaynet_sample(args.data_dir, args.max_docs)
    if not docs:
        raise RuntimeError("No local annotation documents were loaded")

    logger.info(f"Running local PDF evaluation with variant comparison: docs={len(docs)}")
    results = []
    tmp_dir = Path(args.output).parent / "_eval_predictions"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        pdf_path = doc.get("pdf_path")
        if not pdf_path:
            raise RuntimeError("Local evaluation requires matching PDFs under <data-dir>/pdfs/")

        base_output = tmp_dir / f"{doc['image_id']}.json"
        process_pdf(
            pdf_path,
            str(base_output),
            detector_type=args.detector,
            output_variants="all",
            fail_on_detector_error=args.fail_on_detector_error,
            threshold_table=args.model_confidence_threshold,
            threshold_figure=args.model_confidence_threshold,
            threshold_text=args.model_confidence_threshold,
            threshold_header=args.model_confidence_threshold,
            threshold_caption=args.model_confidence_threshold,
            doclaynet_confidence_threshold=args.doclaynet_confidence_threshold,
        )

        for variant in VARIANTS:
            output_path = _variant_output_path(base_output, variant)
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            predictions = []
            for page in data.get("pages", []):
                predictions.extend(_normalize_blocks(page.get("blocks", [])))
            metrics = compute_map_at_thresholds(predictions, doc["ground_truths"])
            results.append({"doc_id": doc.get("image_id", ""), "variant": variant, "metrics": metrics})
    return results


def _evaluate_local_doclaynet(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Evaluate using locally downloaded DocLayNet page images + annotations.

    This mode:
    - Reads images and ground-truths from ``data/doclaynet/``.
    - Runs the detector on each page image directly (no PDF decomposition).
    - Computes mAP / F1 against the ground-truth bboxes.
    """
    docs = load_doclaynet_local(args.data_dir, args.max_docs)
    if not docs:
        raise RuntimeError(
            f"No DocLayNet pages found in {args.data_dir}. "
            "Run: python scripts/download_doclaynet.py first."
        )
    logger.info(f"Running local DocLayNet evaluation: pages={len(docs)}")

    # Lazy-load detector (avoids heavy imports if not needed)
    from models.detector import ClasswiseThresholdConfig, create_detector
    from utils.config import load_config, get_classification_thresholds, get_ensemble_weights

    cfg = load_config(getattr(args, 'config', None))
    threshold_config = ClasswiseThresholdConfig(
        table=args.model_confidence_threshold,
        figure=args.model_confidence_threshold,
        text=args.model_confidence_threshold,
        header=args.model_confidence_threshold,
        caption=args.model_confidence_threshold,
    )
    detector = create_detector(
        detector_type=args.detector,
        threshold_config=threshold_config,
        doclaynet_confidence_threshold=args.doclaynet_confidence_threshold,
    )
    if not detector.is_ready():
        err = detector.get_last_error() or "unknown"
        if args.fail_on_detector_error:
            raise RuntimeError(f"Detector not ready: {err}")
        logger.warning(f"Detector not ready ({err}); detections will be empty.")

    from pipeline.classification import classify_blocks
    from pipeline.confidence import ensure_confidence_scores

    results = []
    for doc in docs:
        img_path = doc.get("image_path", "")
        img_w = float(doc.get("image_width", 1025))
        img_h = float(doc.get("image_height", 1025))
        ground_truths = doc.get("ground_truths", [])

        # Read image bytes for detector
        try:
            page_image_bytes = Path(img_path).read_bytes()
        except (FileNotFoundError, OSError) as exc:
            logger.warning(f"Cannot read image {img_path}: {exc}")
            page_image_bytes = b""

        detections = detector.detect(page_image_bytes, img_w, img_h)

        # Phase 5: OCR Fallback for image-only evaluation (DocLayNet PNGs)
        # We must generate text blocks so the classifier has something to work with.
        page_data = PageData(page_num=0, width=img_w, height=img_h)
        ocr_spans = ocr_page(page_image_bytes, 0, img_w, img_h)
        for span in ocr_spans:
            page_data.add_span(span)
        
        layout_blocks, image_blocks = process_page_layout(page_data)

        # Run classification stage
        classified_blocks = classify_blocks(
            layout_blocks=layout_blocks,
            image_blocks=image_blocks,
            detections=detections,
            page_width=img_w,
            page_height=img_h,
            variant_mode="hybrid",  # default for the combined list
            cfg=cfg,
        )
        validated = ensure_confidence_scores(classified_blocks)
        predictions = _normalize_blocks(validated)

        for variant in VARIANTS:
            # Re-classify for each specific variant mode to get variant-specific results
            variant_blocks = classify_blocks(
                layout_blocks=layout_blocks,
                image_blocks=image_blocks,
                detections=detections,
                page_width=img_w,
                page_height=img_h,
                variant_mode=variant,
                cfg=cfg,
            )
            val_variant = ensure_confidence_scores(variant_blocks)
            predictions = _normalize_blocks(val_variant)
            
            metrics = compute_map_at_thresholds(predictions, ground_truths)
            results.append({
                "doc_id": doc.get("image_id", ""),
                "variant": variant,
                "metrics": metrics,
            })

    return results


def _evaluate_hf_images(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset == "doclaynet":
        docs = load_doclaynet_hf(args.max_docs)
        ds_id = HF_DOCLAYNET_DATASET_ID
    else:
        docs = load_hf_publaynet(args.max_docs)
        ds_id = HF_PUBLAYNET_DATASET_ID

    if not docs:
        raise RuntimeError(f"No documents loaded from {ds_id}")
    logger.info(f"Running HF image evaluation on {ds_id}: docs={len(docs)}")
    results = []
    for doc in docs:
        image = doc.get("image")
        if image is None:
            raise RuntimeError("HF dataset item is missing the page image payload")

        tmp_pdf = Path(args.output).parent / "_hf_tmp" / f"{doc['image_id']}.pdf"
        tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
        # This mode remains a placeholder until PDF-backed evaluation is added.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        for variant in VARIANTS:
            metrics = compute_map_at_thresholds([], doc["ground_truths"])
            results.append({"doc_id": doc.get("image_id", ""), "variant": variant, "metrics": metrics})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DocStruct Evaluation Runner")
    parser.add_argument("--eval-mode", choices=["local_pdf", "local_doclaynet", "hf_image"], default="local_pdf")
    parser.add_argument("--data-dir", default="./data", help="Local dataset directory")
    parser.add_argument(
        "--detector",
        default="combined",
        choices=["stub", "table_transformer", "doclaynet", "combined"],
        help="Detector backend to evaluate",
    )
    parser.add_argument(
        "--dataset",
        choices=["publaynet", "doclaynet"],
        default="publaynet",
        help="HF dataset to use in hf_image mode",
    )
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--fail-on-detector-error", action="store_true")
    parser.add_argument("--model-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--doclaynet-confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.eval_mode == "local_pdf":
        all_results = _evaluate_local_pdfs(args)
    elif args.eval_mode == "local_doclaynet":
        all_results = _evaluate_local_doclaynet(args)
    else:
        all_results = _evaluate_hf_images(args)
    write_csv_report(all_results, args.output)
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
