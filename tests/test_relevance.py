import pytest

from docstruct.eval.relevance import (
    contains_verbatim,
    get_relevance,
    is_relevant,
    is_relevant_region,
    normalize_text,
)


def test_normalize_collapses_whitespace_and_case():
    assert normalize_text("  Hello   WORLD\n\t") == "hello world"


def test_contains_verbatim_substring():
    src = "The model achieves an F1 of 0.82 on the test set."
    assert contains_verbatim(src, "F1 of 0.82")
    assert contains_verbatim(src, "f1 of 0.82")  # case-insensitive
    assert not contains_verbatim(src, "F1 of 0.95")


def test_contains_verbatim_handles_whitespace_diff():
    assert contains_verbatim("two   distinct\ntones", "two distinct tones")


def test_is_relevant_exact_then_overlap():
    chunk = "We compare against three baselines including BM25 and dense retrieval."
    assert is_relevant(chunk, "three baselines")                  # substring
    assert is_relevant(chunk, "three baselines including BM25", 0.6)  # full substring
    # token-overlap fallback: span words mostly present though not contiguous
    assert is_relevant("baselines three compared", "three baselines", min_overlap=0.6)


def test_is_relevant_false_when_absent():
    assert not is_relevant("totally unrelated content here", "three baselines", min_overlap=0.6)


def test_spacing_differences_do_not_change_relevance():
    """Word breaks in a PDF are inferred, so extractors disagree on them."""
    from docstruct.eval.relevance import is_relevant

    gold = "IreneAmerini1,ElenaBalashova2,SaynaEbrahimi3"
    better = "Irene Amerini1, Elena Balashova2, Sayna Ebrahimi3 organised the workshop"
    assert is_relevant(better, gold)
    assert is_relevant(gold, gold)


def test_despacing_does_not_match_unrelated_text():
    from docstruct.eval.relevance import is_relevant

    assert not is_relevant("a completely different sentence about turbines", "IreneAmerini1")


def test_unicode_dash_variants_are_not_a_miss():
    """A PDF's non-breaking hyphen and a model's plain one are the same word."""
    from docstruct.eval.relevance import contains_verbatim, is_relevant

    doc = "the FA-ISS index stores another copy of the mean"
    quoted = "the FA‑ISS index stores another copy"      # U+2011
    assert contains_verbatim(doc, quoted)
    assert is_relevant(doc, quoted)
    assert contains_verbatim("range 43–45 of seeds", "range 43-45 of seeds")


def test_curly_and_straight_quotes_match():
    from docstruct.eval.relevance import contains_verbatim

    assert contains_verbatim("the model’s output layer", "the model's output layer")


def test_genuinely_different_text_still_misses():
    """The folding must not turn the check into a fuzzy match."""
    from docstruct.eval.relevance import contains_verbatim

    assert not contains_verbatim("the FA-ISS index stores a copy", "the FA-ISS index deletes a copy")


def test_region_relevance_credits_partial_coverage_of_a_block():
    # FinanceBench-shaped gold: the evidence is a whole table block, so no chunk
    # can contain it. A chunk carrying most of the block's terms is still the
    # chunk that answers the question.
    # A 400-word evidence block; the chunk sits inside it and answers the question.
    filler = " ".join(f"lineitem{i} {i * 37}" for i in range(200))
    region = (
        "Consolidated Statement of Cash Flows Years ended December 31 "
        "Purchases of property plant and equipment PP&E 1,577 1,373 1,420 " + filler
    )
    chunk = "Purchases of property plant and equipment PP&E 1,577 1,373 1,420"

    # Containment cannot fire, and the span fallback is capped by the size ratio:
    # the chunk holds a small fraction of the region's tokens however good it is.
    assert not is_relevant(chunk, region)
    # The overlap coefficient normalises by the smaller side, so it fires.
    assert is_relevant_region(chunk, region)


def test_region_relevance_rejects_an_unrelated_chunk():
    region = "Purchases of property plant and equipment PP&E 1,577 1,373 1,420"
    chunk = "Item 3. Legal Proceedings. Discussion of respirator mask litigation."
    assert not is_relevant_region(chunk, region)


def test_get_relevance_rejects_an_unknown_mode():
    assert get_relevance("span") is is_relevant
    assert get_relevance("region") is is_relevant_region
    with pytest.raises(ValueError):
        get_relevance("paragraph")


def test_page_relevance_matches_any_page_the_chunk_drew_from():
    from docstruct.eval.relevance import is_relevant_page

    assert is_relevant_page([4], 4)
    assert is_relevant_page([3, 4], 4)          # chunk straddling a page break
    assert is_relevant_page(4, 4)               # scalar tolerated
    assert not is_relevant_page([3, 5], 4)
    # No page metadata must never count as a hit; the benchmark refuses the run.
    assert not is_relevant_page([], 4)
    assert not is_relevant_page(None, 4)
