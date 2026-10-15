"""
Pick where the app keeps its data.

    SUPABASE_URL and SUPABASE_KEY set  ->  cloud: Supabase (documents, chunks, chats, PDFs)
    not set                            ->  this computer: Chroma + data/ folders

The two values are read from environment variables (.env locally) or, on Streamlit Cloud,
from the app's Secrets.
"""
from __future__ import annotations

import os


def _setting(name: str) -> str | None:
    """Environment variable first, then Streamlit secrets (Streamlit Cloud)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:            # not running in Streamlit, or no secrets file
        return None


def get_storage(chroma_path: str = "./chroma_db", embedding_model: str = "unknown"):
    """Returns (vector_store, files)."""
    url, key = _setting("SUPABASE_URL"), _setting("SUPABASE_KEY")
    if url and key:
        from app.storage.supabase_store import SupabaseFiles, SupabaseVectorStore
        return SupabaseVectorStore(url, key), SupabaseFiles(url, key)
    from app.ingestion.store import VectorStore
    from app.storage.local_files import LocalFiles
    return VectorStore(chroma_path, embedding_model=embedding_model), LocalFiles()