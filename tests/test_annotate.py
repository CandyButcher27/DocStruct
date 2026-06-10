import glob
import importlib.util
import json
import os

import pytest

_HAS_FITZ = importlib.util.find_spec("fitz") is not None
_PDFS = sorted(glob.glob(os.path.join("data", "raw-pdfs", "*.pdf")), key=os.path.getsize)

pytestmark = pytest.mark.skipif(
    not (_HAS_FITZ and _PDFS), reason="pymupdf or sample PDFs unavailable"
)


def test_export_session_is_gt_compatible(tmp_path):
    from docstruct.eval.annotate import export_annotation_session
    from docstruct.eval.runner import load_ground_truth

    out = tmp_path / "session.json"
    path, n_boxes, n_pages = export_annotation_session(_PDFS[0], str(out))  # geometry-only

    session = json.loads(out.read_text(encoding="utf-8"))
    assert set(session) >= {"doc", "dpi", "pages", "boxes"}
    assert n_pages == len(session["pages"]) > 0
    assert n_boxes == len(session["boxes"])
    assert session["pages"][0]["image"].startswith("data:image/png;base64,")

    # The session boxes drop straight into the ground-truth loader.
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"boxes": session["boxes"]}), encoding="utf-8")
    loaded = load_ground_truth(str(gt))
    assert len(loaded) == n_boxes
    if loaded:
        assert loaded[0].bbox.page_width > 0
        assert loaded[0].label in {"text", "header", "table", "figure", "caption"}
