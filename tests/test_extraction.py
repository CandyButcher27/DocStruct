import pytest

from docstruct import config
from docstruct.extraction.table_extractor import (
    _grid_covers_region,
    serialize_table,
    table_to_keyvalue,
    table_to_plaintext,
)
from docstruct.extraction.text_extractor import _clean_text


def test_table_to_keyvalue_pairs_headers_with_cells():
    grid = [["Quarter", "Revenue"], ["Q3", "4.5"], ["Q4", "5.1"]]
    kv = table_to_keyvalue(grid).splitlines()
    assert kv == ["Quarter: Q3; Revenue: 4.5", "Quarter: Q4; Revenue: 5.1"]


def test_table_to_keyvalue_falls_back_when_no_header():
    grid = [["1", "2"], ["3", "4"]]  # numeric row 0 -> not a header
    assert table_to_keyvalue(grid) == table_to_plaintext(grid)


def test_serialize_table_honors_config(monkeypatch):
    from docstruct import config
    grid = [["Quarter", "Revenue"], ["Q3", "4.5"]]
    assert serialize_table(grid) == table_to_plaintext(grid)
    monkeypatch.setattr(config, "TABLE_SERIALIZATION", "keyvalue")
    assert serialize_table(grid) == table_to_keyvalue(grid)


@pytest.fixture
def _restore_config():
    saved = (config.DEHYPHENATE, config.NORMALIZE_TEXT)
    yield
    config.DEHYPHENATE, config.NORMALIZE_TEXT = saved


def test_clean_text_is_noop_by_default():
    assert _clean_text("trans-\nfer ﬁle") == "trans-\nfer ﬁle"


def test_clean_text_dehyphenates_when_enabled(_restore_config):
    config.DEHYPHENATE = True
    assert _clean_text("trans-\nfer") == "transfer"


def test_clean_text_normalizes_ligatures_and_soft_hyphens(_restore_config):
    config.NORMALIZE_TEXT = True
    assert _clean_text("ﬁle") == "file"          # fi ligature -> "fi"
    assert _clean_text("soft" + chr(0x00AD) + "hyphen") == "softhyphen"


def test_table_to_plaintext_joins_cells_per_row():
    grid = [["a", "b"], ["1", "2"], ["3", "4"]]
    assert table_to_plaintext(grid).splitlines() == ["a  b", "1  2", "3  4"]


def test_table_to_plaintext_skips_empty_cells():
    assert table_to_plaintext([["a", "", "c"]]) == "a  c"


def test_table_to_plaintext_empty():
    assert table_to_plaintext([]) == ""


def test_grid_covering_most_of_the_region_is_accepted():
    raw = "a b c d e f g h i j"
    assert _grid_covers_region("a b c d e f g h i j", raw)


def test_partial_grid_is_rejected_so_raw_text_wins():
    """A half-ruled table must not silently drop its unruled rows."""
    raw = "header row\n" + "\n".join(f"row {i} value" for i in range(10))
    assert not _grid_covers_region("header row", raw)


def test_empty_region_never_rejects_the_grid():
    assert _grid_covers_region("", "")


def _tbl_block(bid, page, ro, grid, x0=50, x1=550):
    from docstruct.schema import Block, ConfidenceBreakdown, Source
    from tests.conftest import make_bbox
    return Block(
        bbox=make_bbox(x0, 0, x1, 200),
        label="table",
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.8),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=page, block_id=bid, reading_order=ro,
        text=serialize_table(grid), table_data=grid,
    )


def test_multipage_table_merge_joins_and_drops_header(monkeypatch):
    from docstruct import config
    from docstruct.extraction.table_extractor import merge_multipage_tables
    monkeypatch.setattr(config, "MERGE_MULTIPAGE_TABLES", True)
    header = ["Quarter", "Revenue"]
    top = _tbl_block("t0", 0, 0, [header, ["Q1", "1.0"], ["Q2", "2.0"]])
    bot = _tbl_block("t1", 1, 1, [header, ["Q3", "3.0"]])  # header repeats
    out = merge_multipage_tables([top, bot])
    assert [b.block_id for b in out] == ["t0"]
    assert out[0].table_data == [header, ["Q1", "1.0"], ["Q2", "2.0"], ["Q3", "3.0"]]


def test_multipage_table_not_merged_when_columns_differ(monkeypatch):
    from docstruct import config
    from docstruct.extraction.table_extractor import merge_multipage_tables
    monkeypatch.setattr(config, "MERGE_MULTIPAGE_TABLES", True)
    top = _tbl_block("t0", 0, 0, [["A", "B"], ["1", "2"]])
    bot = _tbl_block("t1", 1, 1, [["A", "B", "C"], ["1", "2", "3"]])
    assert len(merge_multipage_tables([top, bot])) == 2


def test_multipage_table_not_merged_when_x_misaligned(monkeypatch):
    from docstruct import config
    from docstruct.extraction.table_extractor import merge_multipage_tables
    monkeypatch.setattr(config, "MERGE_MULTIPAGE_TABLES", True)
    top = _tbl_block("t0", 0, 0, [["A", "B"], ["1", "2"]], x0=50, x1=250)
    bot = _tbl_block("t1", 1, 1, [["A", "B"], ["3", "4"]], x0=350, x1=550)
    assert len(merge_multipage_tables([top, bot])) == 2


def test_multipage_table_merge_off_by_default():
    from docstruct.extraction.table_extractor import merge_multipage_tables
    top = _tbl_block("t0", 0, 0, [["A", "B"], ["1", "2"]])
    bot = _tbl_block("t1", 1, 1, [["A", "B"], ["3", "4"]])
    assert len(merge_multipage_tables([top, bot])) == 2
