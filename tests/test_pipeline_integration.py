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
