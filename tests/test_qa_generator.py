from docstruct.schema import Chunk, SectionPath
from docstruct.eval.qa_generator import QAItem, _sample_chunks, generate_for_chunks, save_qa, load_qa


def _chunk(cid, text, ctype="text", h1="S"):
    return Chunk(
        chunk_id=cid, chunk_type=ctype, content=text,
        section_path=SectionPath(h1=h1), page_num=0, reading_order=0,
        source_block_ids=[cid], metadata={},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, messages, **kwargs):
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r


def test_sample_picks_rich_text_chunks():
    chunks = [
        _chunk("a", "word " * 50),
        _chunk("b", "short", ctype="text"),
        _chunk("c", "lots of content here " * 20, ctype="abstract"),
        _chunk("t", "table stuff " * 50, ctype="table"),  # excluded type
    ]
    picked = _sample_chunks(chunks, 5)
    ids = {c.chunk_id for c in picked}
    assert "a" in ids and "c" in ids
    assert "b" not in ids  # too short
    assert "t" not in ids  # wrong type


def test_generate_accepts_verbatim_rejects_hallucination():
    text = "The system reaches an F1 score of 0.82 on the benchmark dataset."
    chunks = [_chunk("a", text + " " + "padding " * 40)]
    client = FakeClient([
        {"question": "What F1 score?", "answer_span": "F1 score of 0.82"},   # valid
    ])
    items = generate_for_chunks(chunks, "doc.pdf", client, n=1)
    assert len(items) == 1
    assert items[0].answer_span == "F1 score of 0.82"
    assert items[0].source_doc == "doc.pdf"

    client2 = FakeClient([{"question": "Q?", "answer_span": "F1 score of 0.99"}])  # not in source
    assert generate_for_chunks(chunks, "doc.pdf", client2, n=1) == []


def test_qa_roundtrip(tmp_path):
    items = [QAItem("q", "span", "doc.pdf", "chunk_0", 0, "S1 > S2")]
    p = tmp_path / "qa.json"
    save_qa(items, str(p))
    loaded = load_qa(str(p))
    assert loaded == items
