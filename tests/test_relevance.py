from docstruct.eval.relevance import contains_verbatim, is_relevant, normalize_text


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
