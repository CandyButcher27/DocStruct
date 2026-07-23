from docstruct.eval.qa_generator import QAItem, _generate_from_text, save_qa, load_qa


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, messages, **kwargs):
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r


_SPAN = "reaches an F1 score of 0.82 on the benchmark dataset"


def test_generate_accepts_verbatim_rejects_hallucination():
    text = f"The system {_SPAN}. " * 5
    client = FakeClient([
        {"items": [
            {"question": "What F1 score?", "answer_span": _SPAN},
            {"question": "Q2?", "answer_span": "a span that is not in the source at all xyz"},
        ]}
    ])
    items = _generate_from_text(text, "doc.pdf", "fulldoc", client, n=2)
    assert len(items) == 1
    assert items[0].answer_span == _SPAN
    assert items[0].source_doc == "doc.pdf"
    assert items[0].source_chunk_id == "fulldoc"


def test_generate_rejects_spans_below_the_length_floor():
    """A 4-word span is verbatim, and still useless: almost any chunk contains it."""
    text = f"The system {_SPAN}. " * 5
    client = FakeClient([{"items": [
        {"question": "What F1 score?", "answer_span": "F1 score of 0.82"},
    ]}])
    assert _generate_from_text(text, "doc.pdf", "fulldoc", client, n=1) == []


def test_long_documents_are_segmented_and_the_budget_is_spread():
    """Questions must come from the whole document, not only the part that fits."""
    from docstruct.eval.qa_generator import _split_evenly

    assert _split_evenly(5, 1) == [5]
    assert _split_evenly(5, 2) == [3, 2]
    assert _split_evenly(5, 4) == [2, 1, 1, 1]
    assert sum(_split_evenly(5, 7)) == 5


def test_generate_empty_on_llm_error():
    class FailClient:
        def chat_json(self, *a, **kw):
            raise RuntimeError("timeout")

    items = _generate_from_text("some text here " * 20, "doc.pdf", "fulldoc", FailClient(), n=2)
    assert items == []


def test_qa_roundtrip(tmp_path):
    items = [QAItem("q", "span", "doc.pdf", "fulldoc", -1, "")]
    p = tmp_path / "qa.json"
    save_qa(items, str(p))
    loaded = load_qa(str(p))
    assert loaded == items


def test_sampled_segments_cover_the_whole_document():
    """Questions must not all come from a long paper's introduction."""
    from docstruct.eval.qa_generator import _spread

    assert _spread(3, 5) == [0, 1, 2]        # fewer segments than wanted: take all
    assert _spread(9, 5) == [0, 2, 4, 6, 8]  # spread, both ends included
    assert _spread(9, 2) == [0, 8]
    assert _spread(9, 1) == [0]
    assert _spread(1, 5) == [0]
    for total in range(1, 30):
        picked = _spread(total, 5)
        assert picked == sorted(set(picked))
        assert all(0 <= i < total for i in picked)
        assert len(picked) <= min(total, 5)
