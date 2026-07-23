import pytest

from docstruct import config
from docstruct.extraction.furniture import _normalize, strip_page_furniture
from docstruct.schema import Block, ConfidenceBreakdown, Source
from tests.conftest import make_bbox


@pytest.fixture
def strip_on(monkeypatch):
    monkeypatch.setattr(config, "STRIP_PAGE_FURNITURE", True)
    monkeypatch.setattr(config, "FURNITURE_MIN_PAGES", 3)


def _blk(bid, text, y0, page):
    return Block(
        bbox=make_bbox(50, y0, 550, y0 + 15),
        label="text",
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.5),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=page,
        block_id=bid,
        text=text,
    )


def test_normalize_maps_digits_to_hash():
    assert _normalize("Page 14 of 20") == "page # of #"


def test_repeated_footer_across_pages_is_dropped(strip_on):
    blocks = []
    for page in range(3):
        blocks.append(_blk(f"body{page}", f"real content about topic {chr(88 + page)}", 100, page))
        blocks.append(_blk(f"foot{page}", f"J. Smith et al.  {page + 1}", 780, page))
    kept = {b.block_id for b in strip_page_furniture(blocks)}
    assert all(f"body{p}" in kept for p in range(3))
    assert not any(f"foot{p}" in kept for p in range(3))


def test_content_appearing_on_few_pages_is_kept(strip_on):
    blocks = [
        _blk("a", "unique footer", 780, 0),
        _blk("b", "another footer", 780, 1),
        _blk("body", "content", 100, 0),
    ]
    kept = {b.block_id for b in strip_page_furniture(blocks)}
    assert kept == {"a", "b", "body"}


def test_default_off_is_noop():
    blocks = [_blk(f"f{p}", "same footer 1", 780, p) for p in range(5)]
    assert len(strip_page_furniture(blocks)) == 5
