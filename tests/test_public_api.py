import json
import os

import pytest

import docstruct
from docstruct.document import parse_bytes
from docstruct.errors import InvalidPDFError

_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "raw-pdfs", "doc1.pdf")
pytestmark = pytest.mark.skipif(not os.path.exists(_PDF), reason="corpus not on disk")


@pytest.fixture(scope="module")
def doc():
    return docstruct.parse(_PDF)


def test_parse_bytes_matches_parse_from_path(doc):
    with open(_PDF, "rb") as fh:
        from_bytes = parse_bytes(fh.read(), name="doc1.pdf")
    assert from_bytes.path == "doc1.pdf"
    assert [c.content for c in from_bytes.chunks] == [c.content for c in doc.chunks]


def test_parse_bytes_rejects_non_pdf():
    with pytest.raises(InvalidPDFError):
        parse_bytes(b"this is not a pdf", name="bad.pdf")


def test_parse_bytes_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    before = set(os.listdir(tmp_path))
    with open(_PDF, "rb") as fh:
        parse_bytes(fh.read())
    assert not {f for f in os.listdir(tmp_path)} - before


def test_to_jsonl_is_one_valid_json_object_per_chunk(doc):
    lines = doc.to_jsonl().split("\n")
    assert len(lines) == len(doc.chunks)
    first = json.loads(lines[0])
    assert set(first) == {"id", "text", "metadata"}
    assert first["metadata"]["source"] == doc.path


def test_metadata_is_flat_and_json_safe(doc):
    # Vector stores reject nested metadata values; a regression here breaks ingest
    # for every downstream user rather than raising anywhere near this code.
    meta = doc._metadata(doc.chunks[0])
    for key, value in meta.items():
        assert isinstance(value, (str, int, float, type(None))), f"{key} is {type(value)}"
    json.dumps(meta)


def test_metadata_section_path_joins_present_levels_only(doc):
    for chunk in doc.chunks:
        meta = doc._metadata(chunk)
        levels = [meta["section_h1"], meta["section_h2"], meta["section_h3"]]
        assert meta["section_path"] == " > ".join(x for x in levels if x)


def test_stats_totals_agree_with_the_chunks(doc):
    st = doc.stats()
    assert st["n_chunks"] == len(doc.chunks)
    assert st["chunk_words_total"] == sum(len(c.content.split()) for c in doc.chunks)
    assert sum(st["chunks_by_type"].values()) == len(doc.chunks)
    assert st["chunk_words_min"] <= st["chunk_words_mean"] <= st["chunk_words_max"]


def test_framework_exports_carry_text_and_metadata(doc):
    for name, export in (("langchain", doc.to_langchain), ("llamaindex", doc.to_llamaindex)):
        try:
            items = export()
        except ImportError:
            pytest.skip(f"{name} not installed")
        assert len(items) == len(doc.chunks)
        text = items[0].page_content if name == "langchain" else items[0].text
        assert text == doc.chunks[0].content
        assert items[0].metadata["section_path"] == doc._metadata(doc.chunks[0])["section_path"]


def test_parse_many_yields_a_document_per_path():
    from docstruct.document import parse_many

    paths = [_PDF]
    out = dict(parse_many(paths))
    assert set(out) == {os.path.abspath(_PDF)} or set(out) == {str(_PDF)}
    assert all(hasattr(v, "chunks") for v in out.values())


def test_parse_many_rejects_a_bad_on_error_value():
    from docstruct.document import parse_many

    with pytest.raises(ValueError):
        list(parse_many([_PDF], on_error="explode"))
