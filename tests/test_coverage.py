from docstruct.eval.coverage import text_coverage

_DOC = "The system reaches an F1 score of 0.82. The system is fast."


def test_perfect_extraction_scores_one_on_both_axes():
    out = text_coverage([_DOC], _DOC)
    assert out["coverage"] == 1.0
    assert out["duplication"] == 1.0


def test_nothing_produced_is_zero_coverage():
    assert text_coverage([], _DOC)["coverage"] == 0.0
    assert text_coverage(["", "   "], _DOC)["coverage"] == 0.0


def test_empty_reference_does_not_divide_by_zero():
    out = text_coverage(["anything"], "")
    assert out["coverage"] == 0.0
    assert out["reference_words"] == 0


def test_coverage_and_duplication_move_independently():
    """Emitting the document twice is full coverage at double the index cost."""
    out = text_coverage([_DOC, _DOC], _DOC)
    assert out["coverage"] == 1.0
    assert out["duplication"] == 2.0


def test_counting_is_multiset_not_set():
    """Dropping the repeat of a phrase is real loss and must be scored as loss."""
    out = text_coverage(["The system reaches an F1 score of 0.82."], _DOC)
    assert 0.0 < out["coverage"] < 1.0


def test_case_is_ignored_but_lost_word_breaks_are_not_free():
    assert text_coverage(["the SYSTEM is Fast"], "the system is fast")["coverage"] == 1.0
    glued = text_coverage(["IreneAmerini wrote it"], "Irene Amerini wrote it")
    assert glued["coverage"] < 1.0


def test_invented_text_costs_duplication_not_coverage():
    out = text_coverage([_DOC + " entirely new words here"], _DOC)
    assert out["coverage"] == 1.0
    assert out["duplication"] > 1.0


def test_dropped_table_row_is_visible():
    """The failure this metric exists to catch."""
    doc = "Header row\nRaw-GRPO 903 82.5 95.9\nOther-Method 811 79.1 92.4"
    kept_only_ruled_part = text_coverage(["Header row\nOther-Method 811 79.1 92.4"], doc)
    assert kept_only_ruled_part["coverage"] < 0.75
