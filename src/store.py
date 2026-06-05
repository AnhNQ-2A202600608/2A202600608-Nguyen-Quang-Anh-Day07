from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self._client = chromadb.Client()
            self._collection = self._client.create_collection(name=collection_name, get_or_create=True)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        record_id = doc.id
        if not record_id:
            record_id = f"doc_{self._next_index}"
            self._next_index += 1
        return {
            "id": record_id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": doc.metadata or {}
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        scored_records = []
        for record in records:
            score = compute_similarity(query_emb, record["embedding"])
            rec_copy = dict(record)
            rec_copy["score"] = score
            scored_records.append(rec_copy)
        
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                doc_id = doc.id or f"doc_{self._next_index}"
                self._next_index += 1
                ids.append(doc_id)
                documents.append(doc.content)
                embeddings.append(self._embedding_fn(doc.content))
                metadatas.append(doc.metadata or {})
            if ids:
                self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            records = [self._make_record(d) for d in docs]
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
            ret = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    ret.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": results["distances"][0][i] if "distances" in results and results["distances"] else 0.0,
                    })
            return ret
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        metadata_filter = metadata_filter or {}
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb], 
                n_results=top_k, 
                where=metadata_filter
            )
            ret = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    ret.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })
            return ret
        else:
            filtered_records = []
            for record in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if record["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            before_count = self._collection.count()
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
            return self._collection.count() < before_count
        else:
            initial_len = len(self._store)
            self._store = [r for r in self._store if r.get("id") != doc_id and r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_len
