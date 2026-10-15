from __future__ import annotations
 
import hashlib
import os
import re
import sqlite3
 
import numpy as np
from  openai import OpenAI
 
PREFIXES = {  # model name contains -> (query prefix, document prefix)
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
    "e5-": ("query: ", "passage: "),
}

class Embedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", backend: str = "local",
                 batch_size: int = 32, cache_path: str | None = ".embedding_cache.sqlite"):
        self.model_name, self.backend, self.batch_size = model, backend, batch_size
        self.q_prefix, self.d_prefix = next(
            (p for key, p in PREFIXES.items() if key in model.lower()), ("", ""))
        self._model = None
        self._tokenizer = None
        self._cache = None
        if cache_path:
            self._cache = sqlite3.connect(cache_path)
            self._cache.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")


    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.d_prefix + t for t in texts])

 
    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.q_prefix + text])[0]

 
    @property
    def dim(self) -> int:
        return len(self.embed_query("dimension check"))

 
    def count_tokens(self, text: str) -> int:
        """Exact token count with the model's own tokenizer (local backend),
        otherwise the 1 word ≈ 1.3 tokens estimate."""
        if self.backend == "local":
            self._load()
            return len(self._model.tokenizer.encode(text, add_special_tokens=False))
        return int(len(text.split()) * 1.3) + 1


    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.backend}|{self.model_name}|{text}".encode()).hexdigest()

 
    def _embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        if self._cache:
            for k in keys:
                row = self._cache.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
                if row:
                    found[k] = np.frombuffer(row[0], dtype=np.float32)
        todo = [(k, t) for k, t in zip(keys, texts) if k not in found]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            vecs = self._compute([t for _, t in batch])
            for (k, _), v in zip(batch, vecs):
                found[k] = v
                if self._cache:
                    self._cache.execute("INSERT OR REPLACE INTO emb VALUES (?,?)", (k, v.tobytes()))
        if self._cache:
            self._cache.commit()
        return np.vstack([found[k] for k in keys])

    def _load(self):
        if self._model is None and self.backend == "local":
            from sentence_transformers import SentenceTransformer  
            self._model = SentenceTransformer(self.model_name)

    def _compute(self, texts: list[str]) -> np.ndarray:
        if self.backend == "local":
            self._load()
            v = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elif self.backend == "openai":
            from openai import OpenAI  
            resp = OpenAI().embeddings.create(model=self.model_name, input=texts)
            v = np.array([d.embedding for d in resp.data])
        elif self.backend == "hash":
            v = np.vstack([_hash_vector(t) for t in texts])
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)  


def _hash_vector(text: str, dim: int = 512) -> np.ndarray:
    """Test-only: bag of words hashed into a fixed-size vector. Matches words, not meaning."""
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return v

if __name__ == "__main__":
    backend = os.environ.get("EMB_BACKEND", "local")
    e = Embedder(backend=backend, model="hash" if backend == "hash" else "BAAI/bge-small-en-v1.5",
                 cache_path=None)
    docs = ["Tenant may terminate this Agreement early by giving sixty days' written notice.",
            "No animals shall be kept on the Premises without consent."]
    q = e.embed_query("Can I end my lease before it finishes?")
    print("similarities:", (e.embed_documents(docs) @ q).round(3))