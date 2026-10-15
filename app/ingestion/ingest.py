"""
ingest.py  —  Pipeline A: PDF -> parse -> chunk -> embed -> store

    python ingest.py contract.pdf
    python ingest.py contract.pdf --doc-id lease-2027 --backend openai --model text-embedding-3-small
"""
import argparse
import os
import time

from app.ingestion.chunker import chunk_elements
from app.ingestion.embedder import Embedder
from app.ingestion.parser import parse_pdf
from app.ingestion.store import VectorStore


def ingest(pdf: str, doc_id: str, embedder: Embedder, store: VectorStore, verbose=True):
    t0 = time.time()
    elements = parse_pdf(pdf)
    title = elements[0].text if elements and elements[0].level == 0 else os.path.basename(pdf)
    chunks = chunk_elements(elements, doc_id=doc_id, doc_title=title,
                            count_tokens=embedder.count_tokens)
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store.delete_document(doc_id)            # re-ingesting replaces the old version
    store.add(chunks, vectors)
    if verbose:
        n_head = sum(e.kind == "heading" for e in elements)
        print(f"parsed   {len(elements)} elements ({n_head} headings)")
        print(f"chunked  {len(chunks)} chunks, "
              f"{min(c.n_tokens for c in chunks)}–{max(c.n_tokens for c in chunks)} tokens each")
        print(f"embedded {vectors.shape[0]} vectors × {vectors.shape[1]} dims")
        print(f"stored   in {store.name}  ({time.time() - t0:.1f}s)")
    return chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--doc-id")
    ap.add_argument("--backend", default="local", choices=["local", "openai", "hash"])
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--db", default="./chroma_db")
    a = ap.parse_args()
    emb = Embedder(model=a.model, backend=a.backend)
    st = VectorStore(a.db, embedding_model=f"{a.backend}:{a.model}")
    ingest(a.pdf, a.doc_id or os.path.splitext(os.path.basename(a.pdf))[0], emb, st)