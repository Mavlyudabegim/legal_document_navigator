"""
local_files.py  —  keep PDFs, file names and chats in folders on this computer

Used when SUPABASE_URL / SUPABASE_KEY are NOT set (offline mode):
    data/uploads/<doc_id>.pdf      the uploaded PDFs
    data/uploads/names.json        doc_id -> original file name
    data/chats/<doc_id>.json       chat history per document
SupabaseFiles (supabase_store.py) has exactly the same methods, so the app doesn't care which one it gets.
"""
from __future__ import annotations

import json
from pathlib import Path


class LocalFiles:
    def __init__(self, upload_dir: str = "data/uploads", chat_dir: str = "data/chats"):
        self.upload_dir, self.chat_dir = Path(upload_dir), Path(chat_dir)
        self.names_file = self.upload_dir / "names.json"

    name = "local folders (data/)"

    # ---------------------------------------------------------------- file names
    def names(self) -> dict[str, str]:
        try:
            return json.loads(self.names_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def set_name(self, doc_id: str, filename: str) -> None:
        names = self.names()
        names[doc_id] = filename
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.names_file.write_text(json.dumps(names, ensure_ascii=False, indent=1))

    # ---------------------------------------------------------------- PDFs
    def save_pdf(self, doc_id: str, data: bytes) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.upload_dir / f"{doc_id}.pdf"
        path.write_bytes(data)
        return path

    def pdf_path(self, doc_id: str) -> Path | None:
        p = self.upload_dir / f"{doc_id}.pdf"
        return p if p.exists() else None

    # ---------------------------------------------------------------- chats
    def load_chat(self, doc_id: str) -> list[dict]:
        try:
            return json.loads((self.chat_dir / f"{doc_id}.json").read_text())
        except (OSError, json.JSONDecodeError):
            return []

    def save_chat(self, doc_id: str, messages: list[dict]) -> None:
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        (self.chat_dir / f"{doc_id}.json").write_text(json.dumps(messages, ensure_ascii=False, indent=1))

    # ---------------------------------------------------------------- delete
    def delete(self, doc_id: str) -> None:
        """Remove the PDF, the chat and the name (the chunks are removed by the vector store)."""
        for p in (self.upload_dir / f"{doc_id}.pdf", self.chat_dir / f"{doc_id}.json"):
            p.unlink(missing_ok=True)
        names = self.names()
        if names.pop(doc_id, None) is not None:
            self.names_file.write_text(json.dumps(names, ensure_ascii=False, indent=1))