"""ChromaDB-backed vector store with local sentence-transformer embeddings.

Heavy dependencies (``chromadb``, ``sentence_transformers``) are imported lazily.
Chunk section paths are flattened into scalar metadata so retrieval can filter
on ``h1``/``h2``/``h3`` (e.g. ``where={"h1": "3. Methodology"}``).
"""

from __future__ import annotations

from typing import List, Optional

from docstruct import config
from docstruct.schema import Chunk


def _chunk_metadata(chunk: Chunk, doc_id: Optional[str]) -> dict:
    meta = {
        "chunk_type": chunk.chunk_type,
        "page_num": chunk.page_num,
        "reading_order": chunk.reading_order,
    }
    for level in ("h1", "h2", "h3"):
        value = getattr(chunk.section_path, level)
        if value:
            meta[level] = value
    if doc_id:
        meta["doc_id"] = doc_id
    return meta


class VectorStore:
    """Embeds chunks and stores them in a Chroma collection."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = config.COLLECTION_NAME,
        embedding_model: str = config.EMBEDDING_MODEL,
        embedder=None,
    ) -> None:
        import chromadb

        self.client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        if embedder is not None:  # reuse a shared model (e.g. across a benchmark)
            self.embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(embedding_model)

    def index(self, chunks: List[Chunk], doc_id: Optional[str] = None,
              *, include_references: bool = False) -> int:
        """Embed and add chunks; returns the number indexed.

        Reference chunks (only present when ``config.KEEP_REFERENCES`` is set) are
        excluded by default — they are for citation analysis, not retrieval.
        """
        if not include_references:
            chunks = [c for c in chunks if c.chunk_type != "references"]
        if not chunks:
            return 0
        ids = [f"{doc_id}:{c.chunk_id}" if doc_id else c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_chunk_metadata(c, doc_id) for c in chunks]
        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()
        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        return len(ids)

    def query(self, text: str, top_k: int = config.RETRIEVAL_TOP_K, where: Optional[dict] = None):
        """Return the raw Chroma query response for a text query."""
        embedding = self.embedder.encode([text], show_progress_bar=False).tolist()
        return self.collection.query(
            query_embeddings=embedding, n_results=top_k, where=where
        )

    def count(self) -> int:
        return self.collection.count()
