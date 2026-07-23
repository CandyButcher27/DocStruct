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


def test_header_updates_section_path_and_clears_deeper(monkeypatch):
    from docstruct import config

    # cut at every boundary so each section lands in its own chunk
    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 0)
    monkeypatch.setattr(config, "INLINE_HEADER_TEXT", False)
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


def test_header_text_is_retrievable_in_chunk_body():
    """Headings must live in some chunk body, not only in section metadata."""
    blocks = [
        _blk("h1", "header", "Attention Is All You Need", 0, font_size=16),
        _blk("t1", "text", "we propose the Transformer", 1),
    ]
    chunks = build_chunks(blocks)
    assert any("Attention Is All You Need" in c.content for c in chunks)


def test_boundary_below_floor_does_not_split(monkeypatch):
    from docstruct import config

    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 50)
    monkeypatch.setattr(config, "INLINE_HEADER_TEXT", False)
    blocks = [
        _blk("t1", "text", "alpha beta gamma", 0),
        _blk("h", "header", "Section Two", 1, font_size=16),
        _blk("t2", "text", "delta epsilon zeta", 2),
    ]
    chunks = build_chunks(blocks)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) == 1
    assert "alpha beta gamma" in text_chunks[0].content
    assert "delta epsilon zeta" in text_chunks[0].content


def test_boundary_above_floor_splits(monkeypatch):
    from docstruct import config

    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 2)
    monkeypatch.setattr(config, "INLINE_HEADER_TEXT", False)
    blocks = [
        _blk("t1", "text", "alpha beta gamma", 0),
        _blk("h", "header", "Section Two", 1, font_size=16),
        _blk("t2", "text", "delta epsilon zeta", 2),
    ]
    chunks = build_chunks(blocks)
    assert [c.content for c in chunks] == ["alpha beta gamma", "delta epsilon zeta"]


def test_chunk_keeps_the_section_it_started_in(monkeypatch):
    """Crossing a header to reach the floor must not relabel the earlier text."""
    from docstruct import config

    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 50)
    monkeypatch.setattr(config, "INLINE_HEADER_TEXT", False)
    blocks = [
        _blk("h1", "header", "Methods", 0, font_size=16),
        _blk("t1", "text", "short methods line", 1),
        _blk("h2", "header", "Results", 2, font_size=16),
        _blk("t2", "text", "short results line", 3),
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 1
    assert chunks[0].section_path.h1 == "Methods"


def test_table_does_not_split_surrounding_prose(monkeypatch):
    from docstruct import config

    monkeypatch.setattr(config, "MIN_CHUNK_TOKENS", 0)
    monkeypatch.setattr(config, "BREAK_TEXT_ON_TABLE", False)
    blocks = [
        _blk("t1", "text", "before the table", 0),
        _blk("tb", "table", "a b\n1 2", 1),
        _blk("t2", "text", "after the table", 2),
    ]
    chunks = build_chunks(blocks)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) == 1
    assert "before the table" in text_chunks[0].content
    assert "after the table" in text_chunks[0].content
    assert any(c.chunk_type == "table" for c in chunks)


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


def test_section_number_overrides_font_rank():
    """A document that marks depth by numbering, not by size.

    All four headings share one font size, so the font-rank signal alone would
    call every one of them level 1 and flatten the hierarchy completely.
    """
    blocks = [
        _blk("a", "header", "3 Method", 0, font_size=12),
        _blk("b", "header", "3.2 Setup", 1, font_size=12),
        _blk("c", "header", "3.2.1 Ablations", 2, font_size=12),
        _blk("d", "header", "4 Results", 3, font_size=12),
    ]
    levels = assign_header_levels(blocks)
    assert [levels["a"], levels["b"], levels["c"], levels["d"]] == [1, 2, 3, 1]


def test_font_rank_still_applies_to_unnumbered_headers():
    blocks = [
        _blk("a", "header", "Introduction", 0, font_size=18),
        _blk("b", "header", "2.1 Related work", 1, font_size=18),
        _blk("c", "header", "Background", 2, font_size=12),
    ]
    levels = assign_header_levels(blocks)
    assert levels["a"] == 1      # unnumbered, largest size
    assert levels["b"] == 2      # numbering wins over its (level-1) size
    assert levels["c"] == 2      # unnumbered, second size rank


def test_numbering_depth_ignores_non_headings():
    from docstruct.chunking.hierarchy_builder import numbering_depth

    assert numbering_depth("4 Experiments") == 1
    assert numbering_depth("4.2. Setup") == 2
    assert numbering_depth("  4.2.1 Ablation") == 3
    assert numbering_depth("4.2.1.7.3 Very deep") == 3      # clamped to HEADER_LEVELS
    assert numbering_depth("12") is None                    # a bare page number
    assert numbering_depth("Introduction") is None
    assert numbering_depth("") is None
    assert numbering_depth(None) is None
    assert numbering_depth("Section 3 covers this") is None  # not anchored


def test_numbering_depth_appendix_and_roman():
    from docstruct.chunking.hierarchy_builder import numbering_depth

    assert numbering_depth("A. Notation") == 1
    assert numbering_depth("A.1 Proof") == 2
    assert numbering_depth("B.2.1 Details") == 3
    assert numbering_depth("IV. Results") == 1
    assert numbering_depth("IX.1 Corollary") == 2
    # guards against false positives
    assert numbering_depth("A survey of methods") is None
    assert numbering_depth("ABSTRACT") is None
    assert numbering_depth("We propose") is None


def test_numbering_signal_can_be_switched_off(monkeypatch):
    from docstruct import config

    monkeypatch.setattr(config, "HEADER_NUMBERING_LEVELS", False)
    blocks = [
        _blk("a", "header", "3 Method", 0, font_size=12),
        _blk("b", "header", "3.2 Setup", 1, font_size=12),
    ]
    levels = assign_header_levels(blocks)
    assert levels["a"] == levels["b"] == 1


def test_table_split_rows_emits_multiple_chunks_with_header(monkeypatch):
    from docstruct import config
    from docstruct.extraction.table_extractor import serialize_table
    monkeypatch.setattr(config, "TABLE_SPLIT_ROWS", True)
    monkeypatch.setattr(config, "MAX_CHUNK_TOKENS", 6)
    grid = [["Quarter", "Revenue"]] + [[f"Q{i}", f"{i}.0"] for i in range(1, 9)]
    tbl = Block(
        bbox=make_bbox(0, 0, 100, 200),
        label="table",
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.8),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=0,
        block_id="tb",
        reading_order=0,
        text=serialize_table(grid),
        table_data=grid,
    )
    chunks = [c for c in build_chunks([tbl]) if c.chunk_type == "table"]
    assert len(chunks) > 1
    assert all(c.content.startswith("Quarter") for c in chunks)


def test_table_not_split_when_disabled():
    from docstruct.extraction.table_extractor import serialize_table
    grid = [["Quarter", "Revenue"]] + [[f"Q{i}", f"{i}.0"] for i in range(1, 30)]
    tbl = Block(
        bbox=make_bbox(0, 0, 100, 200),
        label="table",
        confidence=ConfidenceBreakdown(0.0, 0.0, 0.8),
        source=Source.UNILATERAL_GEOMETRY,
        page_num=0,
        block_id="tb",
        reading_order=0,
        text=serialize_table(grid),
        table_data=grid,
    )
    chunks = [c for c in build_chunks([tbl]) if c.chunk_type == "table"]
    assert len(chunks) == 1


def test_header_rank_by_weight_separates_bold_from_regular(monkeypatch):
    from docstruct import config
    from docstruct.chunking.hierarchy_builder import assign_header_levels
    monkeypatch.setattr(config, "HEADER_RANK_BY_WEIGHT", True)
    monkeypatch.setattr(config, "HEADER_NUMBERING_LEVELS", False)
    bold = _blk("b", "header", "Bold Heading", 0, font_size=12)
    reg = _blk("r", "header", "Regular Heading", 1, font_size=12)
    bold.is_bold, reg.is_bold = True, False
    levels = assign_header_levels([bold, reg])
    assert levels["b"] < levels["r"]  # bold ranks above regular at equal size


def test_header_rank_by_weight_off_collapses_equal_size(monkeypatch):
    from docstruct import config
    from docstruct.chunking.hierarchy_builder import assign_header_levels
    monkeypatch.setattr(config, "HEADER_NUMBERING_LEVELS", False)
    bold = _blk("b", "header", "Bold", 0, font_size=12)
    reg = _blk("r", "header", "Regular", 1, font_size=12)
    bold.is_bold, reg.is_bold = True, False
    levels = assign_header_levels([bold, reg])
    assert levels["b"] == levels["r"]  # default: size only


def test_references_kept_as_chunks_when_enabled(monkeypatch):
    from docstruct import config
    monkeypatch.setattr(config, "KEEP_REFERENCES", True)
    blocks = [
        _blk("h", "header", "References", 0, font_size=14),
        _blk("r1", "text", "[1] Some citation 2020", 1),
        _blk("r2", "text", "[2] Another citation 2021", 2),
    ]
    chunks = build_chunks(blocks)
    refs = [c for c in chunks if c.chunk_type == "references"]
    assert len(refs) == 1
    assert "[1] Some citation 2020" in refs[0].content
    assert "[2] Another citation 2021" in refs[0].content
