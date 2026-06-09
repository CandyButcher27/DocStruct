from docstruct.geometry.detector import (
    _CAPTION_RE,
    _Line,
    _cluster_graphics,
    _group_words_into_lines,
    _line_kind,
    _split_columns,
)


def _word(text, x0, top, x1, bottom, size=10.0, font="Times-Roman"):
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "size": size,
        "fontname": font,
    }


def _line(x0, x1, top=0.0, bottom=12.0, size=10.0, bold=False, text="x"):
    return _Line([], x0, x1, top, bottom, size, bold, text)


def test_group_words_into_lines_clusters_by_top():
    words = [
        _word("Hello", 0, 100, 40, 110),
        _word("world", 45, 100, 90, 110),   # same line
        _word("next", 0, 130, 40, 140),     # new line
    ]
    lines = _group_words_into_lines(words)
    assert len(lines) == 2
    assert lines[0].text == "Hello world"
    assert lines[1].text == "next"


def test_group_words_detects_bold_majority():
    words = [
        _word("A", 0, 0, 10, 10, font="Arial-BoldMT"),
        _word("B", 12, 0, 22, 10, font="Arial-BoldMT"),
        _word("c", 24, 0, 34, 10, font="Arial"),
    ]
    line = _group_words_into_lines(words)[0]
    assert line.bold is True


def test_line_kind_header_by_size():
    assert _line_kind(_line(0, 100, size=18.0), body_median=10.0) == "header"
    assert _line_kind(_line(0, 100, size=10.5), body_median=10.0) == "body"


def test_split_columns_single_when_no_gap():
    lines = [_line(0, 100), _line(10, 110), _line(20, 120)]
    cols = _split_columns(lines, page_width=600)
    assert len(cols) == 1


def test_split_columns_two_when_wide_gap():
    left = [_line(20, 120), _line(30, 130)]
    right = [_line(400, 500), _line(410, 510)]
    cols = _split_columns(left + right, page_width=600)
    assert len(cols) == 2
    # leftmost column first
    assert all(l.x0 < 200 for l in cols[0])
    assert all(l.x0 >= 200 for l in cols[1])


def test_cluster_graphics_merges_overlap_and_keeps_separate():
    boxes = [
        (0, 0, 50, 50),
        (40, 40, 90, 90),     # overlaps first
        (300, 300, 350, 350), # separate
    ]
    clusters = _cluster_graphics(boxes, gap=5.0)
    assert len(clusters) == 2


def test_caption_regex():
    assert _CAPTION_RE.match("Figure 1: results")
    assert _CAPTION_RE.match("Table 2 Summary")
    assert _CAPTION_RE.match("Fig. 3a")
    assert not _CAPTION_RE.match("The figure shows the trend")
    assert not _CAPTION_RE.match("In Table form we see")
