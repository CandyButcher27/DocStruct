from docstruct.schema import Proposal
from docstruct.cache.pdf_cache import ProposalCache, file_hash, proposals_from_json, proposals_to_json
from docstruct.cache.model_cache import ModelProposalCache
from tests.conftest import make_bbox


def _props():
    return [
        Proposal("text", 0.7, make_bbox(0, 0, 10, 10), "geometry", 0, "geo_0_0"),
        Proposal("table", 0.8, make_bbox(20, 20, 40, 40), "geometry", 1, "geo_1_0"),
    ]


def test_proposal_json_roundtrip():
    props = _props()
    restored = proposals_from_json(proposals_to_json(props))
    assert len(restored) == 2
    assert restored[0].label == "text"
    assert restored[1].bbox.x1 == 40
    assert restored[0].source == "geometry"


def test_file_hash_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    assert file_hash(str(a)) == file_hash(str(b))
    b.write_bytes(b"world")
    assert file_hash(str(a)) != file_hash(str(b))


def test_cache_set_get_roundtrip(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cache = ProposalCache(str(tmp_path / "cache"))
    assert cache.get(str(pdf)) is None
    cache.set(str(pdf), _props())
    restored = cache.get(str(pdf))
    assert restored is not None
    assert [p.label for p in restored] == ["text", "table"]


def test_model_cache_namespace_differs_by_weights(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    c1 = ModelProposalCache(str(tmp_path / "c"), "weights/yolov8m-doclaynet.pt")
    c2 = ModelProposalCache(str(tmp_path / "c"), "weights/yolov8n-doclaynet.pt")
    c1.set(str(pdf), _props())
    # different weights namespace -> miss
    assert c2.get(str(pdf)) is None
    assert c1.get(str(pdf)) is not None
