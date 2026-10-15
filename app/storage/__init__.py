"""
Pick where the app keeps its data.

    SUPABASE_URL and SUPABASE_KEY set in .env  ->  cloud: Supabase (documents, chunks, chats, PDFs)
    not set                                    ->  this computer: Chroma + data/ folders
"""
from __future__ import annotations

import os


def get_storage(chroma_path: str = "./chroma_db", embedding_model: str = "unknown"):
    """Returns (vector_store, files)."""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if url and key:
        from app.storage.supabase_store import SupabaseFiles, SupabaseVectorStore
        return SupabaseVectorStore(url, key), SupabaseFiles(url, key)
    from app.ingestion.store import VectorStore
    from app.storage.local_files import LocalFiles
    return VectorStore(chroma_path, embedding_model=embedding_model), LocalFiles()