"""
highlight.py  —  show a PDF page and highlight the sentences an answer was based on

    page_count(pdf)                       how many pages
    render_page(pdf, page, phrases)       page as a PNG, phrases highlighted in yellow
    key_sentences(query, clause, emb)     the 1-2 sentences of a cited clause worth highlighting
    citation_query(question, answer, n)   what to look for inside citation [n]
"""
from __future__ import annotations

import re

import numpy as np
import pymupdf
import streamlit as st
from rank_bm25 import BM25Okapi

from app.ingestion.chunker import split_sentences
from app.retrieval.retriever import tokenize


@st.cache_data(show_spinner=False)
def page_count(pdf_path: str) -> int:
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


@st.cache_data(show_spinner=False)
def render_page(pdf_path: str, page_no: int, phrases: tuple[str, ...] = (), zoom: float = 1.6) -> bytes:
    """Render one PDF page as an image, with the given phrases highlighted in yellow.
    phrases = ()  -> plain page (just uploaded, nothing asked yet)

    For each phrase we first look for the whole sentence on the page. If the parser changed
    the text a little (line breaks, spaces), we fall back to 6-word pieces, but only use a
    piece that appears ONCE on the page; common pieces like "of the Receiving Party" would
    otherwise light up the wrong lines."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        for phrase in phrases:
            quads = page.search_for(phrase, quads=True)
            if not quads:
                words = phrase.split()
                for i in range(0, max(len(words) - 2, 1), 6):
                    hits = page.search_for(" ".join(words[i:i + 6]), quads=True)
                    if len(hits) == 1:
                        quads += hits
            for quad in quads:
                page.add_highlight_annot(quad)
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def _pieces(clause: str) -> list[str]:
    """Clause -> sentences. Very long sentences (legal lists) are cut further at ';', ':'
    and list markers like (a), (ii), so we can highlight just the relevant item."""
    out = []
    bullets = [b for b in re.split(r"\s*[•▪●◦]\s*", clause) if b.strip()]   # one bullet = one item
    for sent in (x for b in bullets for x in split_sentences(b)):
        if len(sent.split()) > 40:
            out += [p.strip() for p in re.split(r"(?<=[;:])\s+|\s+(?=\([a-z0-9]{1,4}\)\s)", sent)]
        else:
            out.append(sent.strip())
    return [p for p in out if len(p.split()) >= 4]


@st.cache_data(show_spinner=False)
def key_sentences(query: str, clause: str, _embedder, max_n: int = 2) -> tuple[str, ...]:
    """The 1-2 sentences of a clause that best match the question + the answer sentence
    that cited it. Score = meaning (embedding cosine) + shared rare words (BM25).
    (_embedder starts with "_" so Streamlit's cache doesn't try to hash the model.)"""
    sents = _pieces(clause)
    if not query.strip():
        return (clause,)                                   # nothing to compare with
    if len(sents) <= 1:
        return tuple(sents) or (clause,)
    cos = _embedder.embed_documents(sents) @ _embedder.embed_query(query)
    # BM25 inside this one clause: words that appear in every sentence ("Party",
    # "Confidential Information") count little, rare shared words ("employees") count a lot
    bm = BM25Okapi([tokenize(t) or ["_"] for t in sents]).get_scores(tokenize(query))
    score = cos + 0.5 * (bm / bm.max() if bm.max() > 0 else bm)
    best = score.max()
    keep = [i for i in np.argsort(-score)[:max_n] if score[i] >= best - 0.05]
    return tuple(sents[i] for i in sorted(keep))          # keep reading order


def citation_query(question: str, answer: str, n: int) -> str:
    """What to look for inside cited clause [n]: the question plus the answer sentence(s)
    that carry the [n] marker (they say what the clause was used for)."""
    parts = re.split(r"(?<=[.!?])\s+", answer)
    citing = [re.sub(r"\[\d+\]", "", p) for p in parts if f"[{n}]" in p]
    return " ".join([question] + citing)