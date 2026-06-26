import importlib.util

import pytest

from docstruct.schema import Chunk, SectionPath

_HAS_DEPS = all(
    importlib.util.find_spec(m) for m in ("chromadb", "sentence_transformers")
)
_HAS_BM25 = importlib.util.find_spec("rank_bm25") is not None

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="retrieval extras not installed")


def _chunk(cid, content, h1=None):
    return Chunk(
        chunk_id=cid,
        chunk_type="text",
        content=content,
        section_path=SectionPath(h1=h1),
        page_num=0,
        reading_order=0,
        source_block_ids=[cid],
        metadata={},
    )


@pytest.fixture(scope="module")
def store():
    from docstruct.indexing.vector_store import VectorStore

    chunks = [
        _chunk("c0", "The corpus was prepared by manually typing Tangkhul words.", h1="Corpus"),
        _chunk("c1", "Neural networks predict air pollutant concentration levels.", h1="Methods"),
        _chunk("c2", "Word formation includes compounding and reduplication.", h1="Morphology"),
        _chunk("c3", "The configuration identifier is BM25-XR-7 for the ablation run.", h1="Setup"),
    ]
    s = VectorStore(collection_name="test_retrieval")
    s.index(chunks)
    return s


@pytest.fixture(scope="module")
def retriever(store):
    from docstruct.query.retriever import Retriever

    return Retriever(store)


def test_semantic_retrieval_ranks_relevant_first(retriever):
    results = retriever.retrieve("how was the dataset collected?", top_k=3)
    assert results
    assert results[0].chunk_id == "c0"
    assert results[0].citation()


def test_section_filtered_retrieval(retriever):
    results = retriever.retrieve("anything", top_k=3, where={"h1": "Methods"})
    assert results
    assert all(r.section_path == "Methods" for r in results)


@pytest.mark.skipif(not _HAS_BM25, reason="rank-bm25 not installed")
def test_hybrid_retrieval_finds_exact_code(store):
    from docstruct.query.retriever import Retriever

    hybrid = Retriever(store, hybrid=True)
    # an exact identifier that dense embeddings handle poorly but BM25 nails
    results = hybrid.retrieve("BM25-XR-7", top_k=3)
    assert results
    assert results[0].chunk_id == "c3"
