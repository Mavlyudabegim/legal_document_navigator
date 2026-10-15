"""
reranker.py  —  second, more careful look at the best candidates

How it differs from the embedder:
  * Embedder (bi-encoder): turns the question and each clause into vectors SEPARATELY,
    then compares the vectors. Fast, but it never sees question and clause together.
  * Reranker (cross-encoder): reads the question AND one clause TOGETHER and outputs
    "how well does this clause answer this question" (0 = not at all, 1 = perfectly).
    Much more accurate, but slower, so we only run it on ~20 candidates, not the whole document.

Its score is also much better for the "not found" decision than cosine similarity:
an unrelated clause gets ~0.00, a relevant one usually > 0.3.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2  (~90 MB, fast on a laptop CPU).
More accurate but slower: BAAI/bge-reranker-base  (~1.1 GB).
"""
from __future__ import annotations

import numpy as np


class Reranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
        from sentence_transformers import CrossEncoder  # pip install sentence-transformers
        self.model_name = model
        self.model = CrossEncoder(model, max_length=max_length)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        s = np.asarray(self.model.predict([(query, t) for t in texts], show_progress_bar=False),
                       dtype=float).reshape(-1)
        # Some models return raw "logits" (any number), others probabilities (0..1).
        # Squash logits into 0..1 so min_score always means the same thing.
        if s.min() < 0 or s.max() > 1:
            s = 1 / (1 + np.exp(-s))
        return s