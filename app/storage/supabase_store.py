"""
supabase_store.py  —  keep everything in the cloud (Supabase) instead of on this computer

    pip install supabase
    .env:  SUPABASE_URL=https://<project>.supabase.co
           SUPABASE_KEY=<secret key>          (never commit this)
    tables + bucket: run supabase_setup.sql once in the Supabase SQL Editor

Where things go:
    documents table   one row per PDF: doc_id + original file name
    chunks table      clause text + its vector (pgvector) + metadata (section, pages, ...)
    chats table       chat history of each document (JSON)
    'uploads' bucket  the PDF files themselves

Two classes with the same methods as the local versions:
    SupabaseVectorStore  ~ VectorStore (Chroma)   -> used by ingest() and the Retriever
    SupabaseFiles        ~ LocalFiles             -> used by the Streamlit app

Search still happens in memory (see retriever.py): we download one document's vectors once,
then compare in numpy. A contract has at most a few hundred chunks, so that is fast and exact.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

BUCKET = "uploads"
BATCH = 100                         # rows per insert request
PAGE = 1000                         # Supabase returns at most 1000 rows per request


@lru_cache(maxsize=None)
def get_client(url: str, key: str):
    """One connection per URL/key, shared by both classes."""
    from supabase import create_client       # pip install supabase
    return create_client(url, key)


class SupabaseVectorStore:
    def __init__(self, url: str, key: str):
        self.db = get_client(url, key)

    name = "Supabase (chunks table)"

    def add(self, chunks, vectors: np.ndarray) -> None:
        if not chunks:
            return
        # a chunk must belong to a row in 'documents' (foreign key). Create it if missing,
        # but never overwrite the real file name the app may have saved already.
        doc_ids = sorted({c.doc_id for c in chunks})
        self.db.table("documents").upsert([{"doc_id": d, "name": d} for d in doc_ids],
                                          ignore_duplicates=True).execute()
        rows = []
        for c, v in zip(chunks, vectors):
            meta = {k: ("" if val is None else val) for k, val in asdict(c).items()
                    if k not in ("text", "id")}
            rows.append({"id": c.id, "doc_id": c.doc_id, "text": c.text,
                         "embedding": [float(x) for x in v], "metadata": meta})
        for i in range(0, len(rows), BATCH):
            self.db.table("chunks").upsert(rows[i:i + BATCH]).execute()

    def delete_document(self, doc_id: str) -> None:
        """Delete the chunks only (re-ingesting a PDF calls this). The document row, its PDF
        and its chat stay; SupabaseFiles.delete() removes those."""
        self.db.table("chunks").delete().eq("doc_id", doc_id).execute()

    def count(self, doc_id: str | None = None) -> int:
        q = self.db.table("chunks").select("id", count="exact", head=True)
        if doc_id:
            q = q.eq("doc_id", doc_id)
        return q.execute().count or 0

    def get_all(self, doc_id: str | None = None) -> dict:
        out = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        start = 0
        while True:                                       # page through, 1000 rows at a time
            q = self.db.table("chunks").select("id,text,embedding,metadata").order("id")
            if doc_id:
                q = q.eq("doc_id", doc_id)
            rows = q.range(start, start + PAGE - 1).execute().data
            for r in rows:
                out["ids"].append(r["id"])
                out["documents"].append(r["text"])
                out["metadatas"].append(r["metadata"])
                emb = r["embedding"]                      # pgvector comes back as "[0.1,0.2,...]"
                out["embeddings"].append(json.loads(emb) if isinstance(emb, str) else emb)
            if len(rows) < PAGE:
                return out
            start += PAGE

    def list_doc_ids(self) -> list[str]:
        rows = self.db.table("documents").select("doc_id").execute().data
        return sorted(r["doc_id"] for r in rows)


class SupabaseFiles:
    def __init__(self, url: str, key: str):
        self.db = get_client(url, key)
        # PDFs are downloaded once into a temporary folder so PyMuPDF can open them.
        # This is only a cache: the real copy lives in Supabase Storage.
        self.cache = Path(tempfile.gettempdir()) / "legal_navigator_pdfs"

    name = "Supabase (storage + tables)"

    # ---------------------------------------------------------------- file names
    def names(self) -> dict[str, str]:
        rows = self.db.table("documents").select("doc_id,name").execute().data
        return {r["doc_id"]: r["name"] for r in rows}

    def set_name(self, doc_id: str, filename: str) -> None:
        self.db.table("documents").upsert({"doc_id": doc_id, "name": filename}).execute()

    # ---------------------------------------------------------------- PDFs
    def save_pdf(self, doc_id: str, data: bytes) -> Path:
        self.db.storage.from_(BUCKET).upload(
            f"{doc_id}.pdf", data, {"content-type": "application/pdf", "upsert": "true"})
        return self._cache_write(doc_id, data)

    def pdf_path(self, doc_id: str) -> Path | None:
        local = self.cache / f"{doc_id}.pdf"
        if local.exists():
            return local
        try:
            data = self.db.storage.from_(BUCKET).download(f"{doc_id}.pdf")
        except Exception:                                  # not in the bucket
            return None
        return self._cache_write(doc_id, data)

    def _cache_write(self, doc_id: str, data: bytes) -> Path:
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"{doc_id}.pdf"
        path.write_bytes(data)
        return path

    # ---------------------------------------------------------------- chats
    def load_chat(self, doc_id: str) -> list[dict]:
        rows = self.db.table("chats").select("messages").eq("doc_id", doc_id).execute().data
        return rows[0]["messages"] if rows else []

    def save_chat(self, doc_id: str, messages: list[dict]) -> None:
        self.db.table("chats").upsert({
            "doc_id": doc_id, "messages": messages,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

    # ---------------------------------------------------------------- delete
    def delete(self, doc_id: str) -> None:
        """Delete the document row -> its chunks and chat go too ('on delete cascade'),
        then the PDF in storage and the local cached copy."""
        self.db.table("documents").delete().eq("doc_id", doc_id).execute()
        try:
            self.db.storage.from_(BUCKET).remove([f"{doc_id}.pdf"])
        except Exception:
            pass
        (self.cache / f"{doc_id}.pdf").unlink(missing_ok=True)