import math
from dataclasses import dataclass

from docstruct.schema import BoundingBox
from docstruct.eval.metrics import (
    average_precision,
    detection_prf,
    dcg_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank,
)


@dataclass
class Det:
    label: str
    bbox: BoundingBox
    page_num: int
    confidence: float = 1.0


def _b(x0, y0, x1, y1):
    return BoundingBox(x0, y0, x1, y1, 600, 800)


def test_perfect_detection():
    gts = [Det("text", _b(0, 0, 10, 10), 0), Det("table", _b(20, 20, 40, 40), 0)]
    preds = [Det("text", _b(0, 0, 10, 10), 0, 0.9), Det("table", _b(20, 20, 40, 40), 0, 0.8)]
    report = detection_prf(preds, gts)
    assert report["text"]["f1"] == 1.0
    assert report["macro"]["f1"] == 1.0


def test_false_positive_and_negative():
    gts = [Det("text", _b(0, 0, 10, 10), 0)]
    preds = [
        Det("text", _b(0, 0, 10, 10), 0, 0.9),     # TP
        Det("text", _b(100, 100, 110, 110), 0, 0.5),  # FP (no GT)
    ]
    report = detection_prf(preds, gts)
    assert report["text"]["tp"] == 1
    assert report["text"]["fp"] == 1
    assert report["text"]["fn"] == 0
    assert math.isclose(report["text"]["precision"], 0.5)


def test_iou_below_threshold_is_miss():
    gts = [Det("text", _b(0, 0, 10, 10), 0)]
    preds = [Det("text", _b(8, 8, 18, 18), 0, 0.9)]  # small overlap
    report = detection_prf(preds, gts, iou_threshold=0.5)
    assert report["text"]["tp"] == 0
    assert report["text"]["fp"] == 1
    assert report["text"]["fn"] == 1


def test_page_isolation():
    gts = [Det("text", _b(0, 0, 10, 10), 0)]
    preds = [Det("text", _b(0, 0, 10, 10), 1, 0.9)]  # right box, wrong page
    report = detection_prf(preds, gts)
    assert report["text"]["tp"] == 0
    assert report["text"]["fn"] == 1


def test_average_precision_perfect_is_one():
    gts = [Det("text", _b(0, 0, 10, 10), 0), Det("text", _b(20, 0, 30, 10), 0)]
    preds = [
        Det("text", _b(0, 0, 10, 10), 0, 0.9),
        Det("text", _b(20, 0, 30, 10), 0, 0.8),
    ]
    assert average_precision(preds, gts, "text") == 1.0


def test_map_aggregates_classes():
    gts = [Det("text", _b(0, 0, 10, 10), 0), Det("table", _b(20, 20, 40, 40), 0)]
    preds = [
        Det("text", _b(0, 0, 10, 10), 0, 0.9),
        Det("table", _b(20, 20, 40, 40), 0, 0.8),
    ]
    result = mean_average_precision(preds, gts)
    assert result["mAP"] == 1.0
    assert set(result["per_class"]) == {"text", "table"}


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_mrr_average():
    cases = [(["a", "b"], {"a"}), (["a", "b"], {"b"})]
    assert mean_reciprocal_rank(cases) == 0.75  # (1 + 0.5)/2


def test_ndcg_perfect_and_partial():
    # single relevant at top -> ndcg 1.0
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    # relevant at rank 2: dcg = 1/log2(3); ideal = 1/log2(2)=1
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["x", "a", "y"], {"a"}, 3), round(expected, 4), abs_tol=1e-4)


def test_dcg_counts_only_top_k():
    assert dcg_at_k(["x", "x", "a"], {"a"}, 2) == 0.0
