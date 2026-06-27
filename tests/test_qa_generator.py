from docstruct.eval.qa_generator import QAItem, _generate_from_text, save_qa, load_qa


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, messages, **kwargs):
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r


def test_generate_accepts_verbatim_rejects_hallucination():
    text = "The system reaches an F1 score of 0.82 on the benchmark dataset. " * 5
    client = FakeClient([
        {"items": [
            {"question": "What F1 score?", "answer_span": "F1 score of 0.82"},
            {"question": "Q2?", "answer_span": "not in source at all xyz"},
        ]}
    ])
    items = _generate_from_text(text, "doc.pdf", "fulldoc", client, n=2)
    assert len(items) == 1
    assert items[0].answer_span == "F1 score of 0.82"
    assert items[0].source_doc == "doc.pdf"
    assert items[0].source_chunk_id == "fulldoc"


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
