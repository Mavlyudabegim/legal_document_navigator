"""
config.py  —  all settings in one place

Values come from environment variables (.env locally, "Secrets" on Streamlit Cloud),
with the defaults below = the best setup from the ContractNLI evaluation.
"""
from __future__ import annotations

import os

# embeddings: must be the same model the documents were indexed with
EMB_BACKEND = os.environ.get("EMB_BACKEND", "local")
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")

# local storage (only used when SUPABASE_URL / SUPABASE_KEY are not set)
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")

# answering: Groq gpt-oss-20b, hybrid search, top-5 clauses, no reranker, no minimum score
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
TOP_K = 5
SEARCH_MODE = "hybrid"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# DEV_MODE=1 shows a "Developer settings" panel in the sidebar
DEV_MODE = os.environ.get("DEV_MODE") == "1"