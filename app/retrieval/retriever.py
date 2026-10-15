"""
retriever.py  —  question  ->  the few clauses that can answer it

Three modes, so the evaluation can compare them:

  "dense"   : embedding similarity only (what we had before).
              Good at meaning ("end the contract" ~ "terminate"), weak at exact words.
  "hybrid"  : embeddings + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF).
              BM25 catches exact legal terms the embedding misses: "Effective Date",
              "Receiving Party", "§4", "three (3) years".
  + reranker: (optional, works with both modes) a cross-encoder re-reads the top ~20
              candidates together with the question and re-orders them. Its score also
              decides "found / not found" much more reliably than cosine similarity.

Why we search in memory instead of asking Chroma:
  One contract has tens to a few hundred chunks. Comparing the question with all of them
  takes well under a millisecond with numpy, and the result is exact. Chroma stays the
  permanent storage; its fast approximate search only matters at 100,000+ chunks.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi  # pip install rank-bm25

from app.ingestion.embedder import Embedder

if TYPE_CHECKING:                     # only for type hints: Chroma isn't needed with Supabase
    from app.ingestion.store import VectorStore

try:
    from nltk.stem import PorterStemmer  # pip install nltk  ("parties" and "party" -> "parti")
    _stem = PorterStemmer().stem
except ImportError:
    def _stem(w: str) -> str:
        return w

DEFAULT_MIN_COSINE = 0.5     # used without a reranker  (cosine similarity, -1..1)
DEFAULT_MIN_RERANK = 0.1     # used with a reranker     (relevance probability, 0..1)

_STOP = set("""a an and are as at be been but by can could did do does for from had has have
how i if in into is it its may me might my no not of on or our shall should so than that the
their them then there these they this to under until upon was we were what when where which
who whom will with would you your""".split())


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


@dataclass
class Source:
    label: str                  # "S1", "S2", ... the ID the LLM uses to cite this chunk
    chunk_id: str
    text: str
    section: str                # "MUTUAL NON-DISCLOSURE AGREEMENT > 4. Term and Termination."
    section_number: str | None  # "4"
    page_start: int
    page_end: int
    score: float                # the score used for found / not found (rerank or cosine)
    dense_score: float = 0.0    # cosine similarity, always filled (for debugging)
    rerank_score: float | None = None

    @property
    def short_ref(self) -> str:
        """Human-readable reference: '§4 Term and Termination, p. 2'"""
        title = self.section.split(" > ")[-1].rstrip(".")
        if self.section_number:
            title = re.sub(rf"^{re.escape(self.section_number)}\.?\s*", "", title)
            title = f"§{self.section_number} {title}"
        pages = f"p. {self.page_start}" if self.page_start == self.page_end \
            else f"pp. {self.page_start}–{self.page_end}"
        return f"{title}, {pages}"


@dataclass
class _DocIndex:
    ids: list[str]
    texts: list[str]
    search_texts: list[str]      # text WITH heading path: better for BM25 and the reranker
    metas: list[dict]
    emb: np.ndarray              # one row per chunk, already normalised
    bm25: BM25Okapi | None = field(default=None)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 5,
                 min_score: float | None = None, mode: str = "hybrid", reranker=None,
                 candidates: int = 20, rrf_k: int = 60):
        if mode not in ("dense", "hybrid"):
            raise ValueError("mode must be 'dense' or 'hybrid'")
        self.embedder, self.store, self.k = embedder, store, k
        self.mode, self.reranker, self.candidates, self.rrf_k = mode, reranker, candidates, rrf_k
        self.min_score = min_score if min_score is not None else (
            DEFAULT_MIN_RERANK if reranker else DEFAULT_MIN_COSINE)
        self._cache: dict = {}

    # ------------------------------------------------------------ index per document
    def _index(self, doc_id: str | None) -> _DocIndex:
        key = (doc_id, self.store.count(doc_id))        # count changes -> rebuild
        if key not in self._cache:
            # works with both stores: Chroma (local) and Supabase (cloud)
            got = self.store.get_all(doc_id)
            metas = got["metadatas"]
            texts = got["documents"]
            search_texts = [m.get("embed_text") or t for m, t in zip(metas, texts)]
            emb = np.asarray(got["embeddings"], dtype=np.float32)
            idx = _DocIndex(got["ids"], texts, search_texts, metas, emb)
            if idx.ids:
                idx.bm25 = BM25Okapi([tokenize(t) for t in search_texts])
            self._cache = {key: idx}                      # keep only the current document
        return self._cache[key]

    # ------------------------------------------------------------ search
    def _rrf(self, *rankings: list[int]) -> list[int]:
        """Reciprocal Rank Fusion: a chunk ranked high in ANY list ends up high.
        score = sum over lists of 1 / (rrf_k + rank). Uses ranks, not raw scores, so we
        don't have to make cosine and BM25 scores comparable."""
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] += 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=fused.get, reverse=True)

    def search(self, question: str, doc_id: str | None = None) -> list[Source]:
        """Top-k candidates with scores, including weak ones (for debugging)."""
        idx = self._index(doc_id)
        if not idx.ids:
            return []
        q = self.embedder.embed_query(question)
        cos = idx.emb @ q
        dense_rank = list(np.argsort(-cos)[: self.candidates])

        if self.mode == "dense":
            order = dense_rank
        else:
            bm = idx.bm25.get_scores(tokenize(question))
            sparse_rank = [i for i in np.argsort(-bm)[: self.candidates] if bm[i] > 0]
            order = self._rrf(dense_rank, sparse_rank)[: self.candidates]

        rerank = None
        if self.reranker is not None:
            rerank = dict(zip(order, self.reranker.score(question, [idx.search_texts[i] for i in order])))
            order = sorted(order, key=lambda i: rerank[i], reverse=True)

        out = []
        for n, i in enumerate(order[: self.k], start=1):
            m = idx.metas[i]
            out.append(Source(
                label=f"S{n}", chunk_id=idx.ids[i], text=idx.texts[i],
                section=m.get("section_path", ""), section_number=m.get("section_number") or None,
                page_start=int(m.get("page_start", 1)), page_end=int(m.get("page_end", 1)),
                score=float(rerank[i]) if rerank else float(cos[i]),
                dense_score=float(cos[i]),
                rerank_score=float(rerank[i]) if rerank else None,
            ))
        return out

    def filter(self, hits: list[Source]) -> list[Source]:
        """Keep only hits that pass min_score, relabelled S1..Sn (copies, originals untouched)."""
        good = []
        for s in hits:
            if s.score >= self.min_score:
                good.append(Source(**{**s.__dict__, "label": f"S{len(good) + 1}"}))
        return good

    def retrieve(self, question: str, doc_id: str | None = None) -> list[Source]:
        return self.filter(self.search(question, doc_id))

    def all_chunks(self, doc_id: str) -> list[Source]:
        """Every chunk of one document, in reading order (for 'whole contract' mode).
        Chunks are stored in the order the chunker made them; sorting by page keeps that order."""
        idx = self._index(doc_id)
        order = sorted(range(len(idx.ids)), key=lambda i: int(idx.metas[i].get("page_start", 1)))
        out = []
        for n, i in enumerate(order, start=1):
            m = idx.metas[i]
            out.append(Source(
                label=f"S{n}", chunk_id=idx.ids[i], text=idx.texts[i],
                section=m.get("section_path", ""), section_number=m.get("section_number") or None,
                page_start=int(m.get("page_start", 1)), page_end=int(m.get("page_end", 1)),
                score=1.0))
        return out