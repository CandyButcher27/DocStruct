import pytest

from docstruct import config
from docstruct.fusion.containment import suppress_text_in_tables
from docstruct.schema import Block, ConfidenceBreakdown, Source
from tests.conftest import make_bbox


@pytest.fixture
def label_aware(monkeypatch):
    monkeypatch.setattr(config, "LABEL_AWARE_CONTAINMENT", True)


def _blk(bid, label, box, text):
    return Block(
        bbox=box,
        label=label,
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.5),
        source=Source.UNILATERAL_MODEL,
        page_num=0,
        block_id=bid,
        text=text,
    )


def test_text_inside_covering_table_is_dropped(label_aware):
    table = _blk("t", "table", make_bbox(0, 0, 200, 200), "alpha beta gamma delta")
    inside = _blk("x", "text", make_bbox(10, 10, 90, 90), "alpha beta gamma")
    kept = {b.block_id for b in suppress_text_in_tables([table, inside])}
    assert kept == {"t"}


def test_text_not_covered_by_table_is_kept(label_aware):
    table = _blk("t", "table", make_bbox(0, 0, 200, 200), "unrelated cells here")
    inside = _blk("x", "text", make_bbox(10, 10, 90, 90), "completely different words entirely")
    kept = {b.block_id for b in suppress_text_in_tables([table, inside])}
    assert kept == {"t", "x"}


def test_text_outside_table_is_kept(label_aware):
    table = _blk("t", "table", make_bbox(0, 0, 100, 100), "alpha beta gamma")
    outside = _blk("x", "text", make_bbox(300, 300, 400, 400), "alpha beta gamma")
    kept = {b.block_id for b in suppress_text_in_tables([table, outside])}
    assert kept == {"t", "x"}


def test_default_off_is_noop():
    table = _blk("t", "table", make_bbox(0, 0, 200, 200), "alpha beta gamma")
    inside = _blk("x", "text", make_bbox(10, 10, 90, 90), "alpha beta gamma")
    assert len(suppress_text_in_tables([table, inside])) == 2
