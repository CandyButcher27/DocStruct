import pytest

from docstruct.schema import BoundingBox, Proposal

PAGE_W = 600.0
PAGE_H = 800.0


def make_bbox(x0, y0, x1, y1, page_width=PAGE_W, page_height=PAGE_H):
    return BoundingBox(x0, y0, x1, y1, page_width, page_height)


def make_proposal(label, conf, box, source, pid, page=0):
    return Proposal(label, conf, box, source, page, pid)


@pytest.fixture
def bb():
    return make_bbox


@pytest.fixture
def proposal():
    return make_proposal
