from docstruct.extraction.table_extractor import _grid_covers_region, table_to_plaintext


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
