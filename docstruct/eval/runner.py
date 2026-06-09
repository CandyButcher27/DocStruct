"""Evaluation harness tying the pipeline to the metrics.

Layer 1 (detection) compares fused blocks against ground-truth boxes loaded from
a simple JSON format. Layer 2 (retrieval) runs question/answer cases against an
index and reports MRR / NDCG@k. A convenience routine indexes both DocStruct and
naive baseline chunks to quantify the structure-aware advantage.

Ground-truth JSON format (one file per document)::

    {"boxes": [{"label": "text", "page_num": 0,
                "bbox": [x0, y0, x1, y1, page_width, page_height]}, ...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from docstruct import config
from docstruct.schema import BoundingBox, Chunk
from docstruct.eval import metrics


@dataclass
class GTBox:
    label: str
    bbox: BoundingBox
    page_num: int
    confidence: float = 1.0


def load_ground_truth(path: str) -> List[GTBox]:
    """Load ground-truth detection boxes from the JSON format above."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    boxes = []
    for entry in data.get("boxes", []):
        boxes.append(
            GTBox(
                label=entry["label"],
                bbox=BoundingBox(*entry["bbox"]),
                page_num=entry["page_num"],
            )
        )
    return boxes


def _as_detection_item(item) -> GTBox:
    confidence = item.confidence
    if hasattr(confidence, "final"):  # ConfidenceBreakdown -> scalar
        confidence = confidence.final
    return GTBox(item.label, item.bbox, item.page_num, confidence)


def evaluate_detection(blocks, ground_truth, iou_threshold: float = 0.5) -> Dict[str, object]:
    """Per-class P/R/F1 and mAP of fused blocks against ground truth."""
    preds = [_as_detection_item(b) for b in blocks]
    return {
        "prf": metrics.detection_prf(preds, ground_truth, iou_threshold),
        "map": metrics.mean_average_precision(preds, ground_truth, iou_threshold),
    }


def evaluate_retrieval(
    retriever,
    cases: Sequence[Tuple[str, Sequence[str]]],
    top_k: int = config.RETRIEVAL_TOP_K,
) -> Dict[str, float]:
    """Run (question, relevant_ids) cases and report MRR + NDCG@k."""
    ranked_cases = []
    for question, relevant in cases:
        results = retriever.retrieve(question, top_k=top_k)
        ranked_cases.append(([r.chunk_id for r in results], list(relevant)))
    return {
        "mrr": metrics.mean_reciprocal_rank(ranked_cases),
        f"ndcg@{top_k}": metrics.mean_ndcg_at_k(ranked_cases, top_k),
        "n_cases": len(ranked_cases),
    }


def compare_chunking(
    docstruct_chunks: List[Chunk],
    naive_chunks: List[Chunk],
    cases: Sequence[Tuple[str, Sequence[str]]],
    top_k: int = config.RETRIEVAL_TOP_K,
) -> Dict[str, Dict[str, float]]:
    """Index both chunk sets in fresh collections and compare retrieval quality.

    Relevant ids in ``cases`` must reference DocStruct chunk ids; the naive run
    is scored on whether retrieved naive chunks textually contain the relevant
    DocStruct chunk content (mapping happens in the caller's cases for naive).
    """
    from docstruct.indexing.vector_store import VectorStore
    from docstruct.query.retriever import Retriever

    out: Dict[str, Dict[str, float]] = {}
    for name, chunks in (("docstruct", docstruct_chunks), ("naive", naive_chunks)):
        store = VectorStore(collection_name=f"eval_{name}")
        store.index(chunks)
        out[name] = evaluate_retrieval(Retriever(store), cases, top_k=top_k)
    return out
