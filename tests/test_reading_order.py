from docstruct.schema import Block, ConfidenceBreakdown, Source
from docstruct.reading_order import (
    assign_reading_order,
    detect_columns,
    sort_reading_order,
)
from tests.conftest import make_bbox


def _block(bid, label, box):
    return Block(
        bbox=box,
        label=label,
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.5),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=0,
        block_id=bid,
    )


def test_single_column_sorts_top_to_bottom_ascending_y0():
    blocks = [
        _block("b_low", "text", make_bbox(50, 400, 250, 450)),
        _block("b_top", "header", make_bbox(50, 20, 250, 45)),
        _block("b_mid", "text", make_bbox(50, 100, 250, 300)),
    ]
    order = sort_reading_order(blocks, page_width=600)
    ids_in_order = [blocks[i].block_id for i in order]
    assert ids_in_order == ["b_top", "b_mid", "b_low"]


def test_two_column_detected_and_left_before_right():
    # left column x~100, right column x~450, big gap
    blocks = [
        _block("L_top", "text", make_bbox(50, 20, 150, 60)),
        _block("R_top", "text", make_bbox(400, 20, 500, 60)),
        _block("L_bot", "text", make_bbox(50, 200, 150, 260)),
        _block("R_bot", "text", make_bbox(400, 200, 500, 260)),
    ]
    columns = detect_columns(blocks, page_width=600)
    assert len(columns) == 2
    order = sort_reading_order(blocks, page_width=600)
    ids = [blocks[i].block_id for i in order]
    assert ids == ["L_top", "L_bot", "R_top", "R_bot"]


def test_caption_attaches_one_directionally_to_figure_below():
    figure = _block("fig0", "figure", make_bbox(300, 100, 500, 300))
    caption = _block("cap0", "caption", make_bbox(300, 305, 500, 330))
    blocks = [figure, caption]
    assign_reading_order(blocks, page_width=600)
    assert caption.caption_target_id == "fig0"
    # one-directional: figure carries no back-reference field set
    assert figure.caption_target_id is None


def test_caption_too_far_not_attached():
    figure = _block("fig0", "figure", make_bbox(0, 0, 30, 30))
    caption = _block("cap0", "caption", make_bbox(560, 770, 590, 790))
    blocks = [figure, caption]
    assign_reading_order(blocks, page_width=600)
    assert caption.caption_target_id is None


def test_reading_order_assigned_sequentially():
    blocks = [
        _block("a", "text", make_bbox(50, 300, 250, 350)),
        _block("b", "text", make_bbox(50, 50, 250, 100)),
    ]
    assign_reading_order(blocks, page_width=600)
    assert blocks[1].reading_order == 0  # higher on page
    assert blocks[0].reading_order == 1


# --- recursive XY-cut ---

def _b(bid, x0, y0, x1, y1):
    from docstruct.schema import Block, ConfidenceBreakdown, Source
    from tests.conftest import make_bbox

    return Block(
        bbox=make_bbox(x0, y0, x1, y1),
        label="text",
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.8),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=0,
        block_id=bid,
    )


def _order(blocks, page_width=600.0):
    from docstruct.utils.xy_cut import xy_cut_order

    return [blocks[i].block_id for i in xy_cut_order(blocks, page_width)]


def test_xy_cut_reads_two_columns_in_order():
    blocks = [
        _b("r1", 320, 100, 560, 200),
        _b("l1", 40, 100, 280, 200),
        _b("l2", 40, 220, 280, 320),
        _b("r2", 320, 220, 560, 320),
    ]
    assert _order(blocks) == ["l1", "l2", "r1", "r2"]


def test_full_width_title_precedes_both_columns():
    """The case the legacy centre-gap split gets wrong."""
    blocks = [
        _b("left", 40, 120, 280, 400),
        _b("right", 320, 120, 560, 400),
        _b("title", 40, 20, 560, 90),
    ]
    assert _order(blocks) == ["title", "left", "right"]


def test_full_width_block_splits_the_flow_around_it():
    blocks = [
        _b("l_top", 40, 40, 280, 140),
        _b("r_top", 320, 40, 560, 140),
        _b("wide", 40, 170, 560, 260),
        _b("l_bot", 40, 300, 280, 400),
        _b("r_bot", 320, 300, 560, 400),
    ]
    assert _order(blocks) == ["l_top", "r_top", "wide", "l_bot", "r_bot"]


def test_single_column_is_top_to_bottom():
    blocks = [_b("c", 40, 300, 560, 380), _b("a", 40, 40, 560, 120), _b("b", 40, 160, 560, 240)]
    assert _order(blocks) == ["a", "b", "c"]


def test_narrow_indent_is_not_treated_as_a_column_gutter():
    blocks = [_b("a", 40, 40, 300, 120), _b("b", 306, 40, 560, 120)]
    # 6pt gap on a 600pt page is 1%, below XY_CUT_MIN_COLUMN_GAP_RATIO
    assert _order(blocks) == ["a", "b"]


def test_empty_and_single_block():
    from docstruct.utils.xy_cut import xy_cut_order

    assert xy_cut_order([], 600.0) == []
    assert xy_cut_order([_b("only", 0, 0, 10, 10)], 600.0) == [0]


def test_multi_column_noop_on_two_columns(monkeypatch):
    from docstruct import config
    monkeypatch.setattr(config, "MULTI_COLUMN", True)
    blocks = [
        _block("L", "text", make_bbox(50, 20, 150, 60)),
        _block("R", "text", make_bbox(400, 20, 500, 60)),
    ]
    # byte-identical to the legacy 2-column split
    assert detect_columns(blocks, page_width=600) == [[0], [1]]


def test_multi_column_splits_three_columns(monkeypatch):
    from docstruct import config
    monkeypatch.setattr(config, "MULTI_COLUMN", True)
    blocks = [
        _block("A", "text", make_bbox(20, 20, 120, 60)),
        _block("B", "text", make_bbox(250, 20, 350, 60)),
        _block("C", "text", make_bbox(480, 20, 580, 60)),
    ]
    cols = detect_columns(blocks, page_width=600)
    assert len(cols) == 3
    # legacy single-largest-gap split would give only 2
    monkeypatch.setattr(config, "MULTI_COLUMN", False)
    assert len(detect_columns(blocks, page_width=600)) == 2


def test_band_split_orders_full_width_then_columns(monkeypatch):
    from docstruct import config
    monkeypatch.setattr(config, "BAND_SPLIT", True)
    # full-width title on top, then a 2-column body below it
    blocks = [
        _block("title", "header", make_bbox(20, 10, 580, 40)),  # full width
        _block("L", "text", make_bbox(20, 60, 250, 200)),
        _block("R", "text", make_bbox(350, 60, 580, 200)),
    ]
    order = sort_reading_order(blocks, page_width=600)
    ids = [blocks[i].block_id for i in order]
    assert ids == ["title", "L", "R"]
