import json

import docstruct
from docstruct.document import Document
from docstruct.schema import Block, ConfidenceBreakdown, Source
from tests.conftest import make_bbox


def _blk(bid, label, text, ro, font_size=None, table_data=None, page=0):
    return Block(
        bbox=make_bbox(0, ro * 50, 100, ro * 50 + 12),
        label=label,
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.8),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=page,
        block_id=bid,
        reading_order=ro,
        font_size=font_size,
        text=text,
        table_data=table_data,
    )


def _doc():
    blocks = [
        _blk("h1", "header", "Introduction", 0, font_size=18),
        _blk("t1", "text", "we present a system", 1),
        _blk("h2", "header", "Setup", 2, font_size=13),
        _blk("tb", "table", "a  b\n1  2", 3, table_data=[["a", "b"], ["1", "2"]]),
        _blk("cap", "caption", "Figure 1: a plot", 4, page=1),
    ]
    from docstruct.chunking.assembler import build_chunks

    return Document("paper.pdf", blocks, build_chunks(blocks), {"mode": "geometry-only", "pages": 2})


def test_public_api_exports_parse():
    assert callable(docstruct.parse)
    assert docstruct.Document is Document


def test_text_is_reading_ordered_and_complete():
    text = _doc().text
    assert text.index("Introduction") < text.index("we present a system") < text.index("Setup")
    assert "Figure 1: a plot" in text


def test_markdown_uses_heading_levels_and_table_syntax():
    md = _doc().markdown
    assert "# Introduction" in md
    assert "## Setup" in md
    assert "| a | b |" in md
    assert "*Figure 1: a plot*" in md


def test_pages_splits_by_page_number():
    pages = _doc().pages()
    assert set(pages) == {0, 1}
    assert "Figure 1: a plot" in pages[1]
    assert "Figure 1" not in pages[0]


def test_sections_lists_distinct_paths_in_order():
    assert _doc().sections()[0] == "Introduction"


def test_len_and_iter_walk_chunks():
    doc = _doc()
    assert len(doc) == len(doc.chunks)
    assert [c.chunk_id for c in doc] == [c.chunk_id for c in doc.chunks]


def test_to_json_roundtrips(tmp_path):
    doc = _doc()
    out = tmp_path / "chunks.json"
    doc.to_json(str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["path"] == "paper.pdf"
    assert len(payload["chunks"]) == len(doc.chunks)


def test_chunks_of_type_filters():
    doc = _doc()
    assert all(c.chunk_type == "table" for c in doc.chunks_of_type("table"))
    assert doc.chunks_of_type("table")


def test_tables_accessor_returns_grid_page_section():
    grid, page, section = _doc().tables[0]
    assert grid == [["a", "b"], ["1", "2"]]
    assert page == 0
    assert section.h2 == "Setup"


def test_figures_accessor_lists_figure_blocks():
    blocks = [
        _blk("f", "figure", "", 0),
        _blk("t", "text", "body", 1),
    ]
    from docstruct.chunking.assembler import build_chunks

    doc = Document("p.pdf", blocks, build_chunks(blocks), {})
    assert doc.figures == [(0, blocks[0].bbox)]


def test_markdown_escapes_control_chars_and_renders_figure_placeholder():
    blocks = [
        _blk("t", "text", "use pipe | and star * literally", 0),
        _blk("f", "figure", "", 1, page=1),
    ]
    from docstruct.chunking.assembler import build_chunks

    md = Document("p.pdf", blocks, build_chunks(blocks), {}).markdown
    assert r"\|" in md and r"\*" in md
    assert "![figure](page=1" in md


def test_to_markdown_writes_file(tmp_path):
    out = tmp_path / "doc.md"
    md = _doc().to_markdown(str(out))
    assert out.read_text(encoding="utf-8") == md
    assert "# Introduction" in md
