from docstruct.extraction.table_extractor import table_to_markdown


def test_table_to_markdown_basic():
    grid = [["a", "b"], ["1", "2"], ["3", "4"]]
    md = table_to_markdown(grid)
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"


def test_table_to_markdown_ragged_rows_padded():
    grid = [["a", "b", "c"], ["1"]]
    md = table_to_markdown(grid)
    lines = md.splitlines()
    assert lines[0].count("|") == 4  # 3 columns
    assert lines[2] == "| 1 |  |  |"


def test_table_to_markdown_empty():
    assert table_to_markdown([]) == ""
