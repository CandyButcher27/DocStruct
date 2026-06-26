import math

from docstruct.eval.benchmark import _score, _qa_by_doc, _rrf, ToolResult
from docstruct.eval.qa_generator import QAItem


def _qa(doc, q="q", span="s"):
    return QAItem(q, span, doc, "c0", 0, "S")


def test_score_first_relevant():
    rr, hit1, recall, ndcg = _score(["the answer span here", "no", "no"], "answer span", k=3)
    assert rr == 1.0
    assert hit1 == 1.0
    assert recall == 1.0
    assert ndcg == 1.0


def test_score_relevant_at_rank_2():
    rr, hit1, recall, ndcg = _score(["nope", "contains answer span", "no"], "answer span", k=3)
    assert rr == 0.5
    assert hit1 == 0.0
    assert recall == 1.0
    # binary single-relevant ndcg = (1/log2(3)) / (1/log2(2))
    assert math.isclose(ndcg, (1 / math.log2(3)), rel_tol=1e-6)


def test_score_no_relevant():
    rr, hit1, recall, ndcg = _score(["a", "b", "c"], "missing span", k=3)
    assert (rr, hit1, recall, ndcg) == (0.0, 0.0, 0.0, 0.0)


def test_score_respects_k():
    # relevant only at position 4, but k=3 -> miss
    rr, hit1, recall, _ = _score(["x", "y", "z", "answer span"], "answer span", k=3)
    assert rr == 0.0 and recall == 0.0


def test_qa_by_doc_groups():
    qa = [_qa("a.pdf"), _qa("a.pdf"), _qa("b.pdf")]
    by = _qa_by_doc(qa)
    assert set(by) == {"a.pdf", "b.pdf"}
    assert len(by["a.pdf"]) == 2


def test_toolresult_defaults():
    r = ToolResult(name="x")
    assert r.mrr == 0.0 and r.vec_mrr == 0.0 and r.n_questions == 0


def test_rrf_rewards_agreement():
    # index 2 is top in both lists -> fused winner; 0 and 1 trail
    fused = _rrf([[2, 0, 1], [2, 1, 0]])
    assert fused[0] == 2
    assert set(fused) == {0, 1, 2}


def test_rrf_single_list_preserves_order():
    assert _rrf([[3, 1, 2]]) == [3, 1, 2]
