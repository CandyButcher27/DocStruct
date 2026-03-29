"""Evaluation runner for local DocLayNet and HF page-image modes."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Any, Dict, List

from evaluation.ground_truth import (
    HF_DOCLAYNET_DATASET_ID,
    load_doclaynet_hf,
    load_doclaynet_local,
)
from evaluation.metrics import compute_map_at_thresholds
from pipeline.decomposition import PageData
from pipeline.hybrid_proposals import detections_to_proposals, generate_geometry_proposals
from pipeline.layout import process_page_layout
from pipeline.proposal_fusion import deduplicate_proposals, match_and_fuse_proposals
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
        "image_path",
        "matched_gt_image_id",
        "match_score",
        "match_status",
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
                    "image_path": row.get("image_path", ""),
                    "matched_gt_image_id": row.get("matched_gt_image_id", ""),
                    "match_score": row.get("match_score", ""),
                    "match_status": row.get("match_status", ""),
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


def _bbox_to_dict(bbox_obj: Any) -> Dict[str, float] | None:
    if bbox_obj is None:
        return None
    if hasattr(bbox_obj, "model_dump"):
        return bbox_obj.model_dump()
    if hasattr(bbox_obj, "dict"):
        return bbox_obj.dict()
    if isinstance(bbox_obj, dict):
        return bbox_obj
    return None


def _prediction(block_type: str | None, bbox_obj: Any, confidence: float) -> Dict[str, Any] | None:
    if not block_type:
        return None
    if block_type not in {"text", "header", "table", "figure", "caption"}:
        return None
    bbox = _bbox_to_dict(bbox_obj)
    if not bbox:
        return None
    return {
        "block_type": block_type,
        "bbox": bbox,
        "confidence": float(confidence),
    }


def _predictions_from_detections(detections: List[Any]) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    for det in detections:
        p = _prediction(
            getattr(det, "detection_type", None),
            getattr(det, "bbox", None),
            getattr(det, "confidence", 0.0),
        )
        if p is not None:
            preds.append(p)
    return preds


def _predictions_from_region_proposals(proposals: List[Any]) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    for prop in proposals:
        p = _prediction(
            getattr(prop, "proposed_type", None),
            getattr(prop, "bbox", None),
            getattr(prop, "confidence", 0.0),
        )
        if p is not None:
            preds.append(p)
    return preds


def _predictions_from_fused_proposals(fused: List[Any]) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    for prop in fused:
        p = _prediction(
            getattr(prop, "block_type", None),
            getattr(prop, "bbox", None),
            getattr(prop, "final_confidence", 0.0),
        )
        if p is not None:
            preds.append(p)
    return preds


def _run_all_variants(
    page_image_bytes: bytes,
    img_w: float,
    img_h: float,
    detections: List[Any],
    ground_truths: List[Dict[str, Any]],
    doc_id: str,
    image_path: str,
) -> List[Dict[str, Any]]:
    """Run geometry, model, and hybrid variants and return result rows."""
    model_predictions = _predictions_from_detections(detections)

    pred_class_counts: Dict[str, int] = {}
    for d in detections:
        t = getattr(d, "detection_type", "")
        pred_class_counts[t] = pred_class_counts.get(t, 0) + 1
    gt_class_counts: Dict[str, int] = {}
    for g in ground_truths:
        t = g.get("block_type", "")
        gt_class_counts[t] = gt_class_counts.get(t, 0) + 1
    logger.debug("[eval] %s  det=%s  gt=%s", doc_id, pred_class_counts, gt_class_counts)

    page_data = PageData(page_num=0, width=img_w, height=img_h)
    ocr_spans = ocr_page(page_image_bytes, 0, img_w, img_h)
    logger.info("[eval] image_id=%s OCR_spans=%d", doc_id, len(ocr_spans))
    for span in ocr_spans:
        page_data.add_span(span)

    layout_blocks, _ = process_page_layout(page_data)
    geo_proposals = generate_geometry_proposals(
        page_data=page_data,
        layout_blocks=layout_blocks,
        page_lines=[],
    )
    model_proposals = detections_to_proposals(detections)
    fused_proposals, _ = match_and_fuse_proposals(
        model_proposals=model_proposals,
        geometry_proposals=geo_proposals,
        iou_threshold=0.35,
    )
    hybrid_predictions = _predictions_from_fused_proposals(
        deduplicate_proposals(fused_proposals, iou_threshold=0.5)
    )
    geometry_predictions = _predictions_from_region_proposals(geo_proposals)

    rows = []
    for variant in VARIANTS:
        predictions = {
            "geometry": geometry_predictions,
            "model": model_predictions,
        }.get(variant, hybrid_predictions)
        metrics = compute_map_at_thresholds(predictions, ground_truths)
        rows.append({
            "doc_id": doc_id,
            "variant": variant,
            "metrics": metrics,
            "image_path": image_path,
            "matched_gt_image_id": doc_id,
            "match_score": 1.0,
            "match_status": "direct",
        })
    return rows



def _evaluate_local_doclaynet(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Evaluate using locally downloaded DocLayNet page images + annotations.

    Each record from load_doclaynet_local already contains image_path, dimensions,
    and ground_truths together — no image-to-GT pairing is needed.
    """
    docs = load_doclaynet_local(args.data_dir, args.max_docs)
    if not docs:
        raise RuntimeError(
            f"No DocLayNet pages found in {args.data_dir}. "
            "Run: python scripts/download_doclaynet.py first."
        )
    logger.info(f"Running local DocLayNet evaluation: pages={len(docs)}")

    from models.detector import ClasswiseThresholdConfig, create_detector
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

    results: List[Dict[str, Any]] = []
    for doc in docs:
        image_path = doc.get("image_path", "")
        img_w = float(doc.get("image_width", 1025))
        img_h = float(doc.get("image_height", 1025))
        ground_truths = doc.get("ground_truths", [])
        doc_id = doc.get("image_id", "")

        try:
            page_image_bytes = Path(image_path).read_bytes()
        except (FileNotFoundError, OSError) as exc:
            logger.warning(f"Cannot read image {image_path}: {exc}")
            continue

        detections = detector.detect(page_image_bytes, img_w, img_h)
        results.extend(
            _run_all_variants(page_image_bytes, img_w, img_h, detections, ground_truths, doc_id, image_path)
        )

    return results


def _evaluate_hf_images(args: argparse.Namespace) -> List[Dict[str, Any]]:
    docs = load_doclaynet_hf(args.max_docs)
    if not docs:
        raise RuntimeError(f"No documents loaded from {HF_DOCLAYNET_DATASET_ID}")
    logger.info(f"Running HF image evaluation on {HF_DOCLAYNET_DATASET_ID}: docs={len(docs)}")

    from models.detector import ClasswiseThresholdConfig, create_detector
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

    results = []
    for doc in docs:
        image = doc.get("image")
        if image is None:
            raise RuntimeError("HF dataset item is missing the page image payload")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        page_image_bytes = buf.getvalue()
        img_w, img_h = float(image.size[0]), float(image.size[1])
        doc_id = doc.get("image_id", "")
        ground_truths = doc.get("ground_truths", [])

        detections = detector.detect(page_image_bytes, img_w, img_h)
        results.extend(
            _run_all_variants(page_image_bytes, img_w, img_h, detections, ground_truths, doc_id, "")
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DocStruct Evaluation Runner")
    parser.add_argument("--eval-mode", choices=["local_doclaynet", "hf_image"], default="local_doclaynet")
    parser.add_argument("--data-dir", default="./data", help="Local dataset directory")
    parser.add_argument(
        "--detector",
        default="combined",
        choices=["stub", "table_transformer", "doclaynet", "combined"],
        help="Detector backend to evaluate",
    )
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--fail-on-detector-error", action="store_true")
    parser.add_argument("--model-confidence-threshold", type=float, default=0.3)
    parser.add_argument("--doclaynet-confidence-threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.eval_mode == "local_doclaynet":
        all_results = _evaluate_local_doclaynet(args)
    else:
        all_results = _evaluate_hf_images(args)
    write_csv_report(all_results, args.output)
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
