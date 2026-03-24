"""
Evaluation metrics for DocStruct benchmarking.

Computes IoU, precision, recall, F1, and mAP@0.5 / mAP@0.75 per block type.
All functions operate on lists of predicted and ground-truth BoundingBox dicts.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# Block types tracked in all metrics reports
BLOCK_TYPES = ["text", "header", "table", "figure", "caption"]


# ── Bounding-box helpers ────────────────────────────────────────────────────


def _iou(pred: Dict, gt: Dict) -> float:
    """Compute IoU between two bbox dicts with keys x0,y0,x1,y1."""
    x_left   = max(pred["x0"], gt["x0"])
    y_top    = max(pred["y0"], gt["y0"])
    x_right  = min(pred["x1"], gt["x1"])
    y_bottom = min(pred["y1"], gt["y1"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_pred = (pred["x1"] - pred["x0"]) * (pred["y1"] - pred["y0"])
    area_gt   = (gt["x1"]   - gt["x0"])   * (gt["y1"]   - gt["y0"])
    union = area_pred + area_gt - inter
    return inter / union if union > 0 else 0.0


# ── Per-class metrics ────────────────────────────────────────────────────────


def _compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float,
) -> float:
    """
    Compute Average Precision for a single class at a given IoU threshold.

    predictions: list of {'bbox': dict, 'confidence': float}
    ground_truths: list of {'bbox': dict}
    """
    if not ground_truths:
        return float("nan")

    preds = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = set()

    for i, pred in enumerate(preds):
        best_iou = 0.0
        best_idx = -1
        for j, gt in enumerate(ground_truths):
            if j in matched:
                continue
            iou = _iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= iou_threshold:
            tp[i] = 1
            matched.add(best_idx)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / len(ground_truths)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    # Interpolated AP (11-point)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_recall = precision[recall >= thr]
        ap += (prec_at_recall.max() if len(prec_at_recall) > 0 else 0.0)
    return ap / 11


# ── Public API ───────────────────────────────────────────────────────────────


def compute_block_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1, and AP per block type.

    Args:
        predictions:   List of {'block_type': str, 'bbox': dict, 'confidence': float}
        ground_truths: List of {'block_type': str, 'bbox': dict}
        iou_threshold: IoU threshold for a prediction to count as TP

    Returns:
        Dict with keys: 'per_class', 'macro_f1', 'mAP'
    """
    per_class: Dict[str, Dict] = {}

    for cls in BLOCK_TYPES:
        cls_preds = [p for p in predictions if p.get("block_type") == cls]
        cls_gts   = [g for g in ground_truths if g.get("block_type") == cls]

        if not cls_gts:
            per_class[cls] = {
                "precision": float("nan"),
                "recall":    float("nan"),
                "f1":        float("nan"),
                "AP":        float("nan"),
                "num_gt":    0,
                "num_pred":  len(cls_preds),
            }
            continue

        # Match predictions to GT at this threshold
        matched_gt = set()
        tp = 0
        for pred in cls_preds:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(cls_gts):
                if j in matched_gt:
                    continue
                iou = _iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_j)

        precision = tp / len(cls_preds) if cls_preds else 0.0
        recall    = tp / len(cls_gts)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        ap = _compute_ap(cls_preds, cls_gts, iou_threshold)

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "AP":        round(ap, 4),
            "num_gt":    len(cls_gts),
            "num_pred":  len(cls_preds),
        }

    # Macro averages (ignoring classes with no GT)
    valid = [v for v in per_class.values() if not np.isnan(v["f1"])]
    macro_f1 = float(np.mean([v["f1"] for v in valid])) if valid else float("nan")
    valid_ap = [v for v in per_class.values() if not np.isnan(v["AP"])]
    mean_ap  = float(np.mean([v["AP"] for v in valid_ap])) if valid_ap else float("nan")

    return {
        "per_class": per_class,
        "macro_f1":  round(macro_f1, 4),
        "mAP":       round(mean_ap, 4),
        "iou_threshold": iou_threshold,
    }


def compute_map_at_thresholds(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute mAP at common IoU thresholds (0.5, 0.75)."""
    r50 = compute_block_metrics(predictions, ground_truths, iou_threshold=0.5)
    r75 = compute_block_metrics(predictions, ground_truths, iou_threshold=0.75)
    
    per_class_f1 = {cls: data["f1"] for cls, data in r50.get("per_class", {}).items() if "f1" in data}
    
    return {
        "mAP@0.50": r50["mAP"],
        "mAP@0.75": r75["mAP"],
        "macro_f1@0.50": r50["macro_f1"],
        "per_class_f1": per_class_f1,
    }
