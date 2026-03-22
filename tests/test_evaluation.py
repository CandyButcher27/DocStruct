"""Tests for evaluation loaders, mapping, and conflict-aware classification."""

import sys
from types import SimpleNamespace

import pytest

from evaluation.ground_truth import HF_PUBLAYNET_DATASET_ID, load_hf_publaynet
from models.detector import Detection
from models.doclaynet_detector import map_doclaynet_label
from pipeline.classification import classify_block
from pipeline.decomposition import TextSpan
from pipeline.layout import LayoutBlock
from schemas.block import BoundingBox


def test_load_hf_publaynet_uses_canonical_dataset_id(monkeypatch):
    def fake_load_dataset(dataset_id, split):
        assert dataset_id == HF_PUBLAYNET_DATASET_ID
        assert split == "validation"
        return [{"block_types": ["table"], "bboxes": [{"x0": 1, "y0": 2, "x1": 3, "y1": 4}], "image": object()}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    docs = load_hf_publaynet(max_docs=1)
    assert len(docs) == 1
    assert docs[0]["dataset_id"] == HF_PUBLAYNET_DATASET_ID


def test_load_hf_publaynet_fails_loudly(monkeypatch):
    def fake_load_dataset(dataset_id, split):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    with pytest.raises(RuntimeError, match="boom"):
        load_hf_publaynet(max_docs=1)


def test_doclaynet_label_mapping_to_five_classes():
    assert map_doclaynet_label("Title") == "header"
    assert map_doclaynet_label("Section-header") == "header"
    assert map_doclaynet_label("Text") == "text"
    assert map_doclaynet_label("Picture") == "figure"
    assert map_doclaynet_label("Caption") == "caption"


def test_classify_block_uses_page_lines_for_table_conflict_resolution():
    block = LayoutBlock(page_num=0)
    bbox = BoundingBox(x0=10, y0=100, x1=160, y1=160)
    block.add_span(TextSpan("1 2 3 4 5 6", bbox, "Arial", 12.0, 0))

    detection = Detection(
        bbox=BoundingBox(x0=10, y0=100, x1=160, y1=160),
        detection_type="table",
        confidence=0.95,
    )
    page_lines = [
        {"x0": 20, "y0": 110, "x1": 150, "y1": 110, "orientation": "horizontal"},
        {"x0": 20, "y0": 130, "x1": 150, "y1": 130, "orientation": "horizontal"},
        {"x0": 20, "y0": 150, "x1": 150, "y1": 150, "orientation": "horizontal"},
    ]

    result = classify_block(
        block,
        detections=[detection],
        page_width=612,
        page_height=792,
        avg_font_size=12.0,
        page_lines=page_lines,
        variant_mode="hybrid",
    )
    assert result["block_type"] == "table"
