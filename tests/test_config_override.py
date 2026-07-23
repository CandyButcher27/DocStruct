import os

import pytest

import docstruct
from docstruct import config


def test_override_sets_and_restores():
    original = config.MIN_CHUNK_TOKENS
    with config.override(MIN_CHUNK_TOKENS=999):
        assert config.MIN_CHUNK_TOKENS == 999
    assert config.MIN_CHUNK_TOKENS == original


def test_override_restores_on_exception():
    original = config.MIN_CHUNK_TOKENS
    with pytest.raises(RuntimeError):
        with config.override(MIN_CHUNK_TOKENS=1):
            raise RuntimeError("boom")
    assert config.MIN_CHUNK_TOKENS == original


def test_override_rejects_unknown_key():
    with pytest.raises(AttributeError):
        with config.override(NOT_A_REAL_KEY=1):
            pass


_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "raw-pdfs", "doc11.pdf")


@pytest.mark.skipif(not os.path.exists(_PDF), reason="corpus PDF not present")
def test_parse_config_applies_and_does_not_leak():
    before = config.MAX_CHUNK_TOKENS
    default = docstruct.parse(_PDF)
    tiny = docstruct.parse(_PDF, config={"MAX_CHUNK_TOKENS": 20})
    # a tiny token ceiling forces more, smaller chunks
    assert len(tiny.chunks) > len(default.chunks)
    # the override did not permanently mutate the module global
    assert config.MAX_CHUNK_TOKENS == before
