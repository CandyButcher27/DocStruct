from docstruct.eval.benchmark import ToolResult
from docstruct.eval.report import render_markdown, now_iso


def test_render_markdown_has_table_and_tools():
    results = [
        ToolResult(name="docstruct", mrr=0.8, ndcg=0.82, recall=0.9, hit1=0.6, n_chunks=120),
        ToolResult(name="langchain", mrr=0.4, ndcg=0.45, recall=0.6, hit1=0.2, n_chunks=90),
    ]
    meta = {"timestamp": now_iso(), "n_docs": 5, "n_questions": 25,
            "llm_model": "gpt-oss:120b", "skipped": ["docling"]}
    md = render_markdown(results, meta)
    assert "# DocStruct retrieval baseline report" in md
    assert "docstruct **(ours)**" in md
    assert "langchain" in md
    assert "Leaderboard" in md
    assert "gpt-oss:120b" in md
    assert "docling" in md  # listed as skipped
