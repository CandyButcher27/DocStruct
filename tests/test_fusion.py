import math

from docstruct import config
from docstruct.schema import Source
from docstruct.fusion.matcher import match_proposals
from docstruct.fusion.arbiter import arbitrate
from docstruct.fusion.fusion import fuse
from tests.conftest import make_bbox, make_proposal


def test_matcher_confirmed_and_unilateral():
    model = [
        make_proposal("text", 0.9, make_bbox(0, 0, 100, 100), "model", "m0"),
        make_proposal("figure", 0.7, make_bbox(200, 200, 300, 300), "model", "m1"),
    ]
    geometry = [
        make_proposal("text", 0.6, make_bbox(2, 2, 98, 98), "geometry", "g0"),
        make_proposal("caption", 0.5, make_bbox(400, 400, 480, 430), "geometry", "g1"),
    ]
    result = match_proposals(model, geometry)
    assert result.diagnostics == {
        "matched": 1,
        "unmatched_model": 1,
        "unmatched_geometry": 1,
    }


def test_matcher_prefers_higher_confidence_model_for_shared_geometry():
    # A low-conf model box is listed first, a high-conf one second; both fully
    # overlap the single geometry box. Confidence-ordered greedy must let the
    # high-conf box claim the match, not the one that happened to be listed first.
    model = [
        make_proposal("text", 0.50, make_bbox(0, 0, 100, 100), "model", "m_low"),
        make_proposal("text", 0.95, make_bbox(0, 0, 100, 100), "model", "m_high"),
    ]
    geometry = [make_proposal("text", 0.6, make_bbox(0, 0, 100, 100), "geometry", "g0")]
    result = match_proposals(model, geometry)
    assert result.matched[0].model.proposal_id == "m_high"


def test_matcher_respects_iou_threshold():
    model = [make_proposal("text", 0.9, make_bbox(0, 0, 100, 100), "model", "m0")]
    # tiny overlap, below 0.35
    geometry = [make_proposal("text", 0.6, make_bbox(90, 90, 190, 190), "geometry", "g0")]
    result = match_proposals(model, geometry)
    assert result.diagnostics["matched"] == 0
    assert result.diagnostics["unmatched_model"] == 1
    assert result.diagnostics["unmatched_geometry"] == 1


def test_nms_suppresses_lower_priority_overlap():
    # confirmed pair and a stray geometry box on top of it -> geometry suppressed
    model = [make_proposal("text", 0.9, make_bbox(0, 0, 100, 100), "model", "m0")]
    geometry = [
        make_proposal("text", 0.6, make_bbox(1, 1, 99, 99), "geometry", "g0"),
        make_proposal("text", 0.4, make_bbox(3, 3, 97, 97), "geometry", "g1"),
    ]
    result = match_proposals(model, geometry)
    assert result.diagnostics["matched"] == 1
    # g1 overlaps the confirmed block heavily -> suppressed by NMS
    assert result.diagnostics["unmatched_geometry"] == 0


def test_arbiter_confirmed_formula():
    model = [make_proposal("text", 0.8, make_bbox(0, 0, 100, 100), "model", "m0")]
    geometry = [make_proposal("text", 0.6, make_bbox(0, 0, 100, 100), "geometry", "g0")]
    result = match_proposals(model, geometry)
    items = arbitrate(result.matched)
    assert len(items) == 1
    item = items[0]
    assert item.source is Source.CONFIRMED
    expected = min(1.0, 0.85 + 0.10 * 0.8 + 0.05 * item.iou)
    assert math.isclose(item.final, round(expected, 4), rel_tol=1e-9)


def test_arbiter_disputed_winner_times_085():
    model = [make_proposal("header", 0.81, make_bbox(0, 0, 100, 40), "model", "m0")]
    geometry = [make_proposal("text", 0.55, make_bbox(0, 0, 100, 40), "geometry", "g0")]
    result = match_proposals(model, geometry)
    items = arbitrate(result.matched)
    item = items[0]
    assert item.source is Source.DISPUTED
    assert item.label == "header"  # model wins (higher conf)
    assert math.isclose(item.final, round(0.81 * config.DISPUTED_MULTIPLIER, 4))


def test_fuse_emits_all_four_sources():
    model = [
        make_proposal("text", 0.9, make_bbox(0, 0, 100, 100), "model", "m0"),     # confirmed
        make_proposal("header", 0.8, make_bbox(0, 200, 100, 240), "model", "m1"), # disputed
        make_proposal("figure", 0.7, make_bbox(300, 0, 400, 100), "model", "m2"), # uni model
    ]
    geometry = [
        make_proposal("text", 0.6, make_bbox(1, 1, 99, 99), "geometry", "g0"),
        make_proposal("text", 0.5, make_bbox(0, 200, 100, 240), "geometry", "g1"),
        make_proposal("caption", 0.5, make_bbox(300, 500, 400, 530), "geometry", "g2"),  # uni geo
    ]
    result = match_proposals(model, geometry)
    blocks = fuse(result)
    sources = {b.source for b in blocks}
    assert sources == {
        Source.CONFIRMED,
        Source.DISPUTED,
        Source.UNILATERAL_MODEL,
        Source.UNILATERAL_GEOMETRY,
    }
    assert len({b.block_id for b in blocks}) == len(blocks)  # unique ids


def test_unilateral_confidence_clamped_to_bounds():
    model = [make_proposal("text", 1.0, make_bbox(0, 0, 100, 100), "model", "m0")]
    result = match_proposals(model, [])
    block = fuse(result)[0]
    bounds = config.CONFIDENCE_BOUNDS["unilateral_model"]
    assert bounds["floor"] <= block.confidence.final <= bounds["ceiling"]
    assert block.confidence.model_score == 1.0
    assert block.confidence.geometry_score == 0.0
