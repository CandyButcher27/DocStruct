import glob
import os

import pytest

from docstruct.pipeline import run_pipeline

_PDFS = sorted(glob.glob(os.path.join("data", "raw-pdfs", "*.pdf")), key=os.path.getsize)


@pytest.fixture(scope="module")
def result():
    if not _PDFS:
        pytest.skip("no sample PDFs available")
    return run_pipeline(_PDFS[0])  # smallest PDF for speed


def test_pipeline_runs_geometry_only(result):
    assert result.diagnostics["mode"] == "geometry-only"
    assert result.diagnostics["n_blocks"] > 0
    assert len(result.blocks) == result.diagnostics["n_blocks"]


def test_block_ids_unique_and_reading_order_contiguous(result):
    assert len({b.block_id for b in result.blocks}) == len(result.blocks)
    orders = sorted(b.reading_order for b in result.blocks)
    assert orders[0] == 0
    assert orders[-1] == len(result.blocks) - 1


def test_pipeline_chunks_have_section_metadata(result):
    for chunk in result.chunks:
        assert chunk.chunk_type in {"text", "table", "figure_caption", "abstract", "references"}
        assert "h1" in chunk.metadata
        assert chunk.content.strip()


def test_geometry_only_mode_ignores_weights():
    """The ablation path must skip the model even when weights are available.

    Passed a weights path it cannot load, `geometry-only` must never reach the
    model at all — if it did, this would raise instead of returning blocks.
    """
    if not _PDFS:
        pytest.skip("no sample PDFs available")
    out = run_pipeline(_PDFS[0], weights="no/such/weights.pt", pipeline_mode="geometry-only")
    assert out.diagnostics["mode"] == "geometry-only"
    assert out.diagnostics["unmatched_model"] == 0
    assert out.diagnostics["n_blocks"] > 0


def test_model_only_mode_runs_no_geometry_pass():
    """Without weights, model-only has no detector at all: zero blocks, no crash."""
    if not _PDFS:
        pytest.skip("no sample PDFs available")
    out = run_pipeline(_PDFS[0], pipeline_mode="model-only")
    assert out.diagnostics["unmatched_geometry"] == 0
    assert out.diagnostics["n_blocks"] == 0


def test_unknown_pipeline_mode_is_rejected():
    with pytest.raises(ValueError):
        run_pipeline("ignored.pdf", pipeline_mode="both")
