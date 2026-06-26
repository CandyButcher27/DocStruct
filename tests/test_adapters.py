import glob
import os

import pytest

from docstruct.eval.adapters import build_adapter, get_adapters
from docstruct.eval.adapters.base import EvalChunk

_PDFS = sorted(glob.glob(os.path.join("data", "raw-pdfs", "*.pdf")), key=os.path.getsize)


def test_evalchunk_defaults():
    c = EvalChunk(id="x", text="hello")
    assert c.metadata == {}


def test_build_adapter_unknown_raises():
    with pytest.raises(ValueError):
        build_adapter("does-not-exist")


def test_get_adapters_returns_available_subset():
    ad = get_adapters(["docstruct", "langchain"])
    assert "docstruct" in ad  # always available (core)
    for name, adapter in ad.items():
        assert adapter.name == name


@pytest.mark.skipif(not _PDFS, reason="no sample PDFs")
def test_langchain_adapter_chunks_a_pdf():
    ad = get_adapters(["langchain"])
    if "langchain" not in ad:
        pytest.skip("langchain-text-splitters not installed")
    chunks = ad["langchain"].chunk(_PDFS[0])
    assert chunks and all(isinstance(c, EvalChunk) for c in chunks)
    assert all(c.text.strip() for c in chunks)
