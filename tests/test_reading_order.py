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
