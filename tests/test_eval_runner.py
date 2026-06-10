import json

from docstruct.schema import Block, ConfidenceBreakdown, Source
from docstruct.eval.baselines import naive_chunk
from docstruct.eval.runner import evaluate_detection, load_ground_truth, GTBox
from tests.conftest import make_bbox


def _blk(bid, label, text, ro, final=0.9):
    return Block(
        bbox=make_bbox(0, ro * 50, 100, ro * 50 + 20),
        label=label,
        confidence=ConfidenceBreakdown(0.0, 0.0, final),
        source=Source.CONFIRMED,
        page_num=0,
        block_id=bid,
        reading_order=ro,
        text=text,
    )


def test_naive_chunk_fixed_windows():
    blocks = [
        _blk("a", "text", "one two three four", 0),
        _blk("b", "text", "five six seven eight", 1),
    ]
    chunks = naive_chunk(blocks, max_tokens=3)
    assert len(chunks) == 3  # 8 words / 3 -> 3,3,2
    assert chunks[0].content == "one two three"
    assert all(c.chunk_type == "text" for c in chunks)
    assert all(c.section_path.h1 is None for c in chunks)  # no structure


def test_evaluate_detection_adapts_block_confidence():
    blocks = [_blk("a", "text", "x", 0, final=0.9)]
    gt = [GTBox("text", make_bbox(0, 0, 100, 20), 0)]
    report = evaluate_detection(blocks, gt)
    assert report["prf"]["text"]["tp"] == 1
    assert report["map"]["mAP"] == 1.0


def test_load_ground_truth(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps(
            {"boxes": [{"label": "table", "page_num": 1, "bbox": [0, 0, 10, 10, 600, 800]}]}
        ),
        encoding="utf-8",
    )
    boxes = load_ground_truth(str(path))
    assert len(boxes) == 1
    assert boxes[0].label == "table"
    assert boxes[0].page_num == 1
    assert boxes[0].bbox.page_width == 600
