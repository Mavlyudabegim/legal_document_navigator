"""
store.py  —  save chunks + vectors in ChromaDB and search them

ChromaDB runs inside your Python process and saves to a folder: no server to install.
We compute vectors ourselves (embedder.py) and hand them to Chroma, so you always know
which model produced them.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from app.ingestion.chunker import Chunk


class VectorStore:
    def __init__(self, path: str = "./chroma_db", collection: str = "legal_chunks",
                 embedding_model: str = "unknown"):
        # chromadb is imported here, not at the top, so the app can run without it
        # when it uses Supabase (e.g. on Streamlit Cloud). Locally: pip install chromadb
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "Local storage needs chromadb (pip install chromadb). On Streamlit Cloud this "
                "means SUPABASE_URL / SUPABASE_KEY are missing from the app's Secrets.") from e
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name=collection,
            # cosine distance; we store the model name so you never mix vectors of two models
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
            embedding_function=None,
        )
        stored = (self.col.metadata or {}).get("embedding_model")
        if stored and stored != embedding_model:
            raise ValueError(f"Collection was built with {stored!r}, not {embedding_model!r}. "
                             "Use a new collection name or re-ingest.")

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        metas = []
        for c in chunks:
            d = asdict(c)
            # Chroma metadata only accepts str / int / float / bool, never None
            metas.append({k: ("" if v is None else v) for k, v in d.items()
                          if k not in ("text", "id")})
        self.col.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                        embeddings=vectors.tolist(), metadatas=metas)

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    # --- the 4 methods below are what the retriever and the app use. SupabaseVectorStore
    # --- (app/storage/supabase_store.py) has the same ones, so either store can be plugged in.
    @property
    def name(self) -> str:
        return f"chroma:{self.col.name}"

    def count(self, doc_id: str | None = None) -> int:
        """Number of chunks (of one document). The retriever rebuilds its index when it changes."""
        if doc_id is None:
            return self.col.count()
        return len(self.col.get(where={"doc_id": doc_id}, include=[])["ids"])

    def get_all(self, doc_id: str | None = None) -> dict:
        """All chunks of one document: {'ids', 'documents', 'metadatas', 'embeddings'}."""
        got = self.col.get(where={"doc_id": doc_id} if doc_id else None,
                           include=["documents", "metadatas", "embeddings"])
        return {"ids": got["ids"], "documents": got["documents"] or [],
                "metadatas": got["metadatas"] or [],
                "embeddings": got["embeddings"] if got["embeddings"] is not None else []}

    def list_doc_ids(self) -> list[str]:
        metas = self.col.get(include=["metadatas"])["metadatas"] or []
        return sorted({m["doc_id"] for m in metas})

    def search(self, query_vector: np.ndarray, k: int = 5, doc_id: str | None = None) -> list[dict]:
        res = self.col.query(query_embeddings=[query_vector.tolist()], n_results=k,
                             where={"doc_id": doc_id} if doc_id else None)
        hits = []
        for i in range(len(res["ids"][0])):
            m = res["metadatas"][0][i]
            hits.append({
                "id": res["ids"][0][i],
                "score": round(1 - res["distances"][0][i], 3),   # cosine similarity
                "text": res["documents"][0][i],
                "section": m["section_path"],
                "section_number": m["section_number"] or None,
                "pages": (m["page_start"], m["page_end"]),
            })
        return hits