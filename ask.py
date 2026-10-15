"""
ask.py  —  chat with your document in the terminal

    python ask.py                                   # interactive, Claude (needs ANTHROPIC_API_KEY)
    python ask.py --provider ollama                 # free local model
    python ask.py -q "Can I leave early?"           # one question, then exit
    python ask.py --debug                           # also show scores and what was sent to the LLM

Run `python ingest.py your.pdf` first, with the same embedding settings.
"""
import argparse
import sys
import textwrap

from app.generation.llm import LLM
from app.generation.qa import Answer, answer_question
from app.ingestion.embedder import Embedder
from app.ingestion.store import VectorStore
from app.retrieval.retriever import Retriever

from dotenv import load_dotenv
load_dotenv()  

# simple terminal colours (turned off when output is not a terminal)
_TTY = sys.stdout.isatty()
def c(text, code): return f"\033[{code}m{text}\033[0m" if _TTY else text
BOLD, DIM, CYAN, YELLOW, GREEN = "1", "2", "36", "33", "32"

WIDTH = 88


def show(ans: Answer, debug: bool, retriever: Retriever, doc_id: str | None) -> None:
    if debug:
        print(c("\n── retrieval (all top-k, before the min-score filter) ──", DIM))
        for s in retriever.search(ans.question, doc_id):
            mark = "✓" if s.score >= retriever.min_score else "✗"
            print(c(f"  {mark} {s.score:.3f}  {s.short_ref}", DIM))
        if ans.raw_llm_output:
            print(c("── raw LLM output ──", DIM))
            print(c(textwrap.indent(textwrap.fill(ans.raw_llm_output, WIDTH - 2), "  "), DIM))

    print()
    colour = GREEN if ans.found else YELLOW
    for para in ans.text.split("\n"):
        print(c(textwrap.fill(para, WIDTH), colour) if para.strip() else "")

    if ans.citations:
        print(c("\nSources", BOLD))
        for i, s in enumerate(ans.citations, start=1):
            print(f"  [{i}] {c(s.short_ref, CYAN)}  {c(f'(match {s.score:.2f})', DIM)}")
            quote = s.text.replace("\n", " ")
            quote = quote if len(quote) <= 220 else quote[:217] + "…"
            print(c(textwrap.indent(textwrap.fill(f'"{quote}"', WIDTH - 6), "      "), DIM))
    for w in ans.warnings:
        print(c(f"  ⚠ {w}", YELLOW))
    print()


def main():
    ap = argparse.ArgumentParser(description="Ask questions about an ingested legal document.")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--doc-id", help="only search this document")
    ap.add_argument("--provider", default="groq", choices=["groq", "openai", "ollama"])
    ap.add_argument("--model", help="LLM model name (default depends on provider)")
    ap.add_argument("-k", type=int, default=5, help="how many chunks to retrieve")
    ap.add_argument("--min-score", type=float, default=0.5, help="similarity cut-off")
    ap.add_argument("--emb-backend", default="local", choices=["local", "openai", "hash"])
    ap.add_argument("--emb-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--db", default="./chroma_db")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    embedder = Embedder(model=a.emb_model, backend=a.emb_backend)
    store = VectorStore(a.db, embedding_model=f"{a.emb_backend}:{a.emb_model}")
    if store.col.count() == 0:
        sys.exit("The vector database is empty. Run `python ingest.py your.pdf` first.")
    retriever = Retriever(embedder, store, k=a.k, min_score=a.min_score)
    llm = LLM(provider=a.provider, model=a.model)

    def run(q: str):
        try:
            show(answer_question(q, retriever, llm, a.doc_id), a.debug, retriever, a.doc_id)
        except Exception as e:                       # e.g. missing API key, Ollama not running
            print(c(f"\nError: {type(e).__name__}: {e}\n", YELLOW))

    if a.question:
        run(a.question)
        return

    print(c(f"Legal Document Navigator  ·  {llm.provider}:{llm.model}  ·  "
            f"{store.col.count()} chunks", BOLD))
    print(c("Ask a question about your document. Type 'exit' to quit.\n", DIM))
    while True:
        try:
            q = input(c("You: ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()