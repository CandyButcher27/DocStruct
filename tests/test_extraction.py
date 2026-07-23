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
