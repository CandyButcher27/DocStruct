from docstruct.schema import Block, ConfidenceBreakdown, Source
from docstruct.chunking.hierarchy_builder import assign_header_levels
from docstruct.chunking.assembler import build_chunks
from tests.conftest import make_bbox


def _blk(bid, label, text, ro, font_size=None, height=12, caption_target=None):
    box = make_bbox(0, ro * 50, 100, ro * 50 + height)
    return Block(
        bbox=box,
        label=label,
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.8),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=0,
        block_id=bid,
        reading_order=ro,
        font_size=font_size,
        text=text,
        caption_target_id=caption_target,
    )


def test_header_levels_ranked_by_font_size():
    blocks = [
        _blk("h1", "header", "Title", 0, font_size=20),
        _blk("h2", "header", "Section", 1, font_size=14),
        _blk("h3", "header", "Subsection", 2, font_size=12),
        _blk("h4", "header", "Another section", 3, font_size=14),
    ]
    levels = assign_header_levels(blocks)
    assert levels["h1"] == 1
    assert levels["h2"] == 2
    assert levels["h4"] == 2  # same size as h2
    assert levels["h3"] == 3


def test_header_updates_section_path_and_clears_deeper():
    blocks = [
        _blk("h1", "header", "Methods", 0, font_size=16),
        _blk("t1", "text", "intro to methods", 1),
        _blk("h2", "header", "Setup", 2, font_size=12),
        _blk("t2", "text", "setup details", 3),
        _blk("h1b", "header", "Results", 4, font_size=16),
        _blk("t3", "text", "results text", 5),
    ]
    chunks = build_chunks(blocks)
    paths = {c.content: c.section_path for c in chunks}
    assert paths["intro to methods"].h1 == "Methods"
    assert paths["setup details"].h2 == "Setup"
    # new h1 clears h2
    assert paths["results text"].h1 == "Results"
    assert paths["results text"].h2 is None


def test_text_accumulates_until_token_limit(monkeypatch):
    from docstruct import config

    monkeypatch.setattr(config, "MAX_CHUNK_TOKENS", 5)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 0)
    blocks = [
        _blk("t1", "text", "one two three", 0),
        _blk("t2", "text", "four five six", 1),  # crosses limit -> flush
        _blk("t3", "text", "seven", 2),
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 2


def test_chunk_overlap_carries_tail_into_next(monkeypatch):
    from docstruct import config

    monkeypatch.setattr(config, "MAX_CHUNK_TOKENS", 6)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 3)
    blocks = [
        _blk("t1", "text", "a b c", 0),
        _blk("t2", "text", "d e f", 1),  # total 6 -> flush, t2 kept as overlap
        _blk("t3", "text", "g h i", 2),
    ]
    chunks = build_chunks(blocks)
    all_content = " ".join(c.content for c in chunks)
    # overlap tail (t2) must appear in more than one chunk
    assert sum("d e f" in c.content for c in chunks) >= 2
    # all content from all blocks must appear somewhere
    assert "a b c" in all_content
    assert "g h i" in all_content


def test_table_is_atomic_chunk():
    blocks = [
        _blk("h", "header", "Data", 0, font_size=16),
        _blk("tb", "table", "| a | b |\n| --- | --- |\n| 1 | 2 |", 1),
    ]
    chunks = build_chunks(blocks)
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].source_block_ids == ["tb"]


def test_caption_becomes_figure_caption_linked_to_target():
    blocks = [
        _blk("fig", "figure", "", 0),
        _blk("cap", "caption", "Figure 1: a plot", 1, caption_target="fig"),
    ]
    chunks = build_chunks(blocks)
    fc = [c for c in chunks if c.chunk_type == "figure_caption"]
    assert len(fc) == 1
    assert "fig" in fc[0].source_block_ids
    assert "cap" in fc[0].source_block_ids


def test_references_section_skipped():
    blocks = [
        _blk("h", "header", "References", 0, font_size=14),
        _blk("r1", "text", "[1] Some citation 2020", 1),
    ]
    chunks = build_chunks(blocks)
    assert chunks == []


def test_abstract_chunk_type():
    blocks = [
        _blk("h", "header", "Abstract", 0, font_size=14),
        _blk("a1", "text", "we present a system", 1),
    ]
    chunks = build_chunks(blocks)
    assert any(c.chunk_type == "abstract" for c in chunks)
