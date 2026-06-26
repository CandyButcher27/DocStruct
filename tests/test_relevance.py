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
