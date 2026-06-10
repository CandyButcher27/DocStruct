"""Evaluation metrics for both pipeline layers.

Layer 1 (detection): per-class precision / recall / F1 at a fixed IoU, plus
mean Average Precision (mAP) using confidence-ranked all-point interpolation.
Layer 2 (retrieval): Mean Reciprocal Rank and NDCG@k.

Detection inputs are any objects exposing ``label``, ``bbox`` (with a
top-left :class:`~docstruct.schema.BoundingBox`), ``page_num`` and
``confidence`` -- e.g. :class:`~docstruct.schema.Proposal`. Matching is done
per page so detections never match ground truth on another page.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from docstruct.utils.geometry import bbox_overlap


# --------------------------------------------------------------------------- #
# Detection metrics
# --------------------------------------------------------------------------- #
def _group(items, label) -> Dict[int, list]:
    out: Dict[int, list] = defaultdict(list)
    for it in items:
        if it.label == label:
            out[it.page_num].append(it)
    return out


def _labels(preds, gts) -> List[str]:
    return sorted({it.label for it in list(preds) + list(gts)})


def detection_prf(preds, gts, iou_threshold: float = 0.5) -> Dict[str, dict]:
    """Per-class and macro precision/recall/F1 at a single IoU threshold."""
    report: Dict[str, dict] = {}
    f1s, precisions, recalls = [], [], []

    for label in _labels(preds, gts):
        pred_pages = _group(preds, label)
        gt_pages = _group(gts, label)
        tp = fp = fn = 0

        for page in set(pred_pages) | set(gt_pages):
            page_preds = sorted(
                pred_pages.get(page, []), key=lambda p: p.confidence, reverse=True
            )
            page_gts = gt_pages.get(page, [])
            used = set()
            for pred in page_preds:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(page_gts):
                    if j in used:
                        continue
                    iou = bbox_overlap(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou >= iou_threshold:
                    used.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += len(page_gts) - len(used)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    report["macro"] = {
        "precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
    }
    return report


def _ap_from_pr(recalls: List[float], precisions: List[float]) -> float:
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))


def average_precision(preds, gts, label: str, iou_threshold: float = 0.5) -> float:
    """All-point-interpolated AP for one class at a given IoU."""
    gt_pages = _group(gts, label)
    n_gt = sum(len(v) for v in gt_pages.values())
    if n_gt == 0:
        return 0.0

    ranked = sorted(
        (p for p in preds if p.label == label),
        key=lambda p: p.confidence,
        reverse=True,
    )
    used: Dict[int, set] = defaultdict(set)
    cum_tp = cum_fp = 0
    recalls, precisions = [], []

    for pred in ranked:
        page_gts = gt_pages.get(pred.page_num, [])
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(page_gts):
            if j in used[pred.page_num]:
                continue
            iou = bbox_overlap(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            used[pred.page_num].add(best_j)
            cum_tp += 1
        else:
            cum_fp += 1
        recalls.append(cum_tp / n_gt)
        precisions.append(cum_tp / max(cum_tp + cum_fp, 1))

    return _ap_from_pr(recalls, precisions)


def mean_average_precision(preds, gts, iou_threshold: float = 0.5) -> Dict[str, float]:
    """Per-class AP and overall mAP at a given IoU."""
    per_class = {
        label: round(average_precision(preds, gts, label, iou_threshold), 4)
        for label in sorted({g.label for g in gts})
    }
    m = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {"per_class": per_class, "mAP": round(m, 4)}


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(cases: Iterable[Tuple[Sequence[str], Iterable[str]]]) -> float:
    cases = list(cases)
    if not cases:
        return 0.0
    return round(sum(reciprocal_rank(r, rel) for r, rel in cases) / len(cases), 4)


def dcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    return sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant_set
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    ideal = dcg_at_k(list(relevant_set), relevant_set, k)
    if ideal == 0:
        return 0.0
    return round(dcg_at_k(ranked, relevant_set, k) / ideal, 4)


def mean_ndcg_at_k(
    cases: Iterable[Tuple[Sequence[str], Iterable[str]]], k: int = 5
) -> float:
    cases = list(cases)
    if not cases:
        return 0.0
    return round(sum(ndcg_at_k(r, rel, k) for r, rel in cases) / len(cases), 4)
