import math

from docstruct.utils.geometry import bbox_overlap, bbox_center, merge_bboxes
from tests.conftest import make_bbox


def test_area_width_height():
    box = make_bbox(0, 0, 10, 20)
    assert box.area == 200
    assert box.width == 10
    assert box.height == 20


def test_iou_identical_is_one():
    a = make_bbox(0, 0, 10, 10)
    assert bbox_overlap(a, a) == 1.0


def test_iou_disjoint_is_zero():
    a = make_bbox(0, 0, 10, 10)
    b = make_bbox(20, 20, 30, 30)
    assert bbox_overlap(a, b) == 0.0


def test_iou_half_overlap():
    a = make_bbox(0, 0, 10, 10)
    b = make_bbox(5, 0, 15, 10)
    # intersection 50, union 150 -> 1/3
    assert math.isclose(bbox_overlap(a, b), 1 / 3, rel_tol=1e-9)


def test_merge_preserves_page_dims():
    a = make_bbox(0, 0, 10, 10, page_width=595, page_height=842)
    b = make_bbox(5, 5, 20, 30, page_width=595, page_height=842)
    merged = merge_bboxes([a, b])
    assert (merged.x0, merged.y0, merged.x1, merged.y1) == (0, 0, 20, 30)
    assert merged.page_width == 595
    assert merged.page_height == 842


def test_center():
    assert bbox_center(make_bbox(0, 0, 10, 20)) == (5.0, 10.0)
