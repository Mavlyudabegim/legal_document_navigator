"""
streamlit_app.py  —  web interface for Legal Document Navigator

    pip install streamlit python-dotenv
    streamlit run streamlit_app.py

Layout
    sidebar : upload a PDF, pick the active document, settings
    left    : chat — answers with clickable citations
    right   : source viewer — the cited PDF page with the clause highlighted

Nothing about RAG is new here: upload calls ingest(), each question calls answer_question().
The UI only displays what the pipeline returns.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi
import streamlit as st

try:                                    # reads GROQ_API_KEY etc. from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.generation.llm import LLM, PROVIDERS
from app.generation.qa import answer_question
from app.ingestion.chunker import split_sentences
from app.ingestion.embedder import Embedder
from app.ingestion.ingest import ingest
from app.storage import get_storage
from app.retrieval.retriever import Retriever, Source, tokenize

DB_PATH = "./chroma_db"
EMB_BACKEND = os.environ.get("EMB_BACKEND", "local")          # must match how you ingested
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")

st.set_page_config(page_title="Legal Document Navigator", page_icon="⚖️", layout="wide")

# The page itself must not scroll, otherwise the PDF moves with it. So the chat and the PDF
# each get their own scroll box, as tall as the window minus the title and the input bar.
# (st.container(key="chat-box") gets the CSS class "st-key-chat-box".)
st.markdown("""
<style>
:root {
    --chat-top: 22rem;      /* above + below the chat box: title, caption, input bar */
    --viewer-top: 19.5rem;    /* above + below the PDF box: page buttons, hint line, input bar */
}
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; padding-bottom: 0; }
h1 { font-size: 1.9rem !important; padding-top: 0 !important; }
div[data-testid="stLayoutWrapper"]:has(> .st-key-chat-box), .st-key-chat-box {
    flex: 0 0 calc(100vh - var(--chat-top)) !important;
    height: calc(100vh - var(--chat-top)) !important;
}
div[data-testid="stLayoutWrapper"]:has(> .st-key-pdf-viewer), .st-key-pdf-viewer {
    flex: 0 0 calc(100vh - var(--viewer-top)) !important;
    height: calc(100vh - var(--viewer-top)) !important;
}
/* whole PDF page visible: as tall as the box allows, never wider than the column */
.st-key-pdf-viewer img {
    max-height: calc(100vh - var(--viewer-top) - 0.5rem);
    max-width: 100%;
    width: auto !important;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ shared objects
# st.cache_resource = create once, reuse on every rerun (loading a model is slow)
@st.cache_resource
def get_embedder() -> Embedder:
    # cache_path=None: SQLite connections can't be shared between Streamlit's threads
    return Embedder(model=EMB_MODEL, backend=EMB_BACKEND, cache_path=None)


@st.cache_resource
def get_store_and_files():
    """Supabase if SUPABASE_URL + SUPABASE_KEY are in .env, otherwise Chroma + data/ folders.
    store = chunks + vectors, files = PDFs, file names and chats (see app/storage/)."""
    return get_storage(DB_PATH, embedding_model=f"{EMB_BACKEND}:{EMB_MODEL}")


@st.cache_resource
def get_reranker(model: str):
    from app.retrieval.reranker import Reranker
    return Reranker(model)


@st.cache_resource
def get_llm(provider: str, model: str, thinking: str) -> LLM:
    effort = None if thinking == "default" else thinking
    return LLM(provider=provider, model=model or None, reasoning_effort=effort)


# ------------------------------------------------------------------ helpers
def list_documents(store) -> list[str]:
    """Your uploaded documents (the ContractNLI evaluation NDAs, 'cnli-…', are hidden)."""
    return [d for d in store.list_doc_ids() if not d.startswith("cnli-")]


def make_doc_id(filename: str, data: bytes) -> str:
    """'My Lease (final).pdf' -> 'my-lease-final-3fa9c1'. The hash part means the same
    file is never ingested twice, and two different files with one name don't clash."""
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:40] or "doc"
    return f"{stem}-{hashlib.sha1(data).hexdigest()[:6]}"


def display_name(doc_id: str, names: dict) -> str:
    """Original file name if we know it, else the id without the hash: 'legal-document-sample'."""
    return names.get(doc_id) or re.sub(r"-[0-9a-f]{6}$", "", doc_id)


def delete_document(doc_id: str) -> None:
    """Remove a document everywhere: its chunks, the PDF, its chat, its name."""
    store.delete_document(doc_id)
    files.delete(doc_id)
    state["history"].pop(doc_id, None)


def pdf_path_for(doc_id: str) -> Path | None:
    """A local path PyMuPDF can open (with Supabase: downloaded once into a temp folder)."""
    return files.pdf_path(doc_id)


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
def key_sentences(query: str, clause: str, max_n: int = 2) -> tuple[str, ...]:
    """The 1-2 sentences of a clause that best match the question + the answer sentence
    that cited it. Score = meaning (embedding cosine) + shared rare words (BM25)."""
    sents = _pieces(clause)
    if not query.strip():
        return (clause,)                                   # nothing to compare with
    if len(sents) <= 1:
        return tuple(sents) or (clause,)
    cos = embedder.embed_documents(sents) @ embedder.embed_query(query)
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


def show_citation(doc_id: str, src, query: str = "") -> None:
    """Remember the clicked citation and jump the viewer to its page."""
    state["selected"] = {"doc_id": doc_id, "source": src, "query": query}
    state[f"page-{doc_id}"] = src.page_start


_LABEL = re.compile(r"^\s*[*_]*\s*(yes|no|partly)\s*[.,:!]?\s*[*_]*\s*", re.IGNORECASE)


def strip_label(text: str) -> str:
    """Remove the 'Yes.' / 'No.' / 'Partly.' the prompt puts at the start of every answer.
    The model still writes it (the evaluation grades answers by it, see prompts.py rule 10);
    we just don't show it. 'No' as a real first word ("No party may...") is kept, because
    we only strip it when it is followed by punctuation."""
    m = _LABEL.match(text)
    if not m or not re.match(r"^\s*[*_]*\s*(yes|no|partly)\s*[*_]*\s*[.,:!]", text, re.IGNORECASE):
        return text
    rest = text[m.end():]
    return rest[:1].upper() + rest[1:] if rest else text


def md_safe(text: str) -> str:
    """Streamlit treats $...$ as maths; escape dollar signs in contract text."""
    return text.replace("$", r"\$")


def _src_to_dict(s: Source) -> dict:
    return dict(s.__dict__)


def save_chat(doc_id: str, messages: list[dict]) -> None:
    """Save one document's chat (Source objects -> plain dicts, so it can be stored as JSON)."""
    out = []
    for m in messages:
        m = dict(m)
        m["citations"] = [_src_to_dict(c) for c in m.get("citations", [])]
        if m.get("debug"):
            m["debug"] = {**m["debug"], "hits": [_src_to_dict(c) for c in m["debug"]["hits"]]}
        out.append(m)
    files.save_chat(doc_id, out)


def load_chat(doc_id: str) -> list[dict]:
    try:
        messages = files.load_chat(doc_id)
    except Exception:                       # e.g. no internet: start with an empty chat
        return []
    for m in messages:
        m["citations"] = [Source(**c) for c in m.get("citations", [])]
        if m.get("debug"):
            m["debug"]["hits"] = [Source(**c) for c in m["debug"]["hits"]]
    return messages


def ingest_upload(up) -> str | None:
    """Save + index one uploaded PDF. Returns its doc_id, or None if it could not be read."""
    data = up.getvalue()
    doc_id = make_doc_id(up.name, data)
    if doc_id in state["failed"]:
        return None
    if store.count(doc_id) == 0:                     # not indexed yet
        files.set_name(doc_id, up.name)
        path = files.save_pdf(doc_id, data)          # local folder or Supabase Storage
        with st.status(f"Reading {up.name}…", expanded=True) as status:
            st.write("Parsing, splitting into clauses, embedding…")
            try:
                chunks = ingest(str(path), doc_id, embedder, store, verbose=False)
                status.update(label=f"{up.name}: {len(chunks)} clauses indexed", state="complete")
            except Exception as e:
                status.update(label=f"Could not read {up.name}", state="error")
                st.error(str(e))
                state["failed"].add(doc_id)
                files.delete(doc_id)                 # don't keep a half-saved document
                return None
    return doc_id


# ------------------------------------------------------------------ sidebar
store, files = get_store_and_files()
embedder = get_embedder()
state = st.session_state
state.setdefault("history", {})          # doc_id -> list of messages
state.setdefault("selected", None)       # the citation shown in the viewer

with st.sidebar:
    st.header("Documents")
    state.setdefault("failed", set())      # PDFs that could not be read (don't retry each rerun)
    state.setdefault("uploader_key", 0)

    # The upload box is ONLY for adding files. After a file is read we empty the box
    # (new key = new empty widget), so files appear once: in the list below, where
    # clicking a name opens it.
    uploads = st.file_uploader("➕ Add contracts (PDF)", type="pdf", accept_multiple_files=True,
                               key=f"uploader-{state['uploader_key']}")
    if uploads:
        added = [d for d in (ingest_upload(up) for up in uploads) if d]
        if added:                                    # show the newest one: page 1, no highlight
            state["doc_picker"] = added[-1]
            state["selected"] = None
            state[f"page-{added[-1]}"] = 1
        state["uploader_key"] += 1
        st.rerun()

    docs = list_documents(store)
    if "doc_picker" not in state:                    # first run / after a browser refresh:
        state["doc_picker"] = st.query_params.get("doc")   # reopen the document from the URL
    if state.get("doc_picker") not in docs:
        state["doc_picker"] = None                   # None = nothing open, nothing shown

    # Your documents: click a name to open it, the bin to delete it
    names = files.names()
    for d in docs:
        name_col, del_col = st.columns([5, 1], gap="small", vertical_alignment="center")
        is_open = d == state["doc_picker"]
        if name_col.button(("📖 " if is_open else "📄 ") + display_name(d, names),
                           key=f"open-{d}", width="stretch",
                           type="primary" if is_open else "secondary",
                           help="Open this document"):
            if not is_open:
                state["doc_picker"] = d
                state["selected"] = None
                state.setdefault(f"page-{d}", 1)
            st.rerun()
        if del_col.button("", icon=":material/delete:", key=f"bin-{d}", width="stretch",
                          help="Delete this document"):
            state["confirm_delete"] = d              # ask first, deleting can't be undone
            st.rerun()
        if state.get("confirm_delete") == d:         # small confirm box under that document
            with st.container(border=True):
                st.markdown(f"Delete **{display_name(d, names)}** and its chat?")
                yes_col, no_col = st.columns(2)
                if yes_col.button("Delete", key=f"del-{d}", type="primary", width="stretch"):
                    delete_document(d)
                    if state["doc_picker"] == d:
                        state["doc_picker"] = None
                        state["selected"] = None
                    state["confirm_delete"] = None
                    st.rerun()
                if no_col.button("Cancel", key=f"cancel-{d}", width="stretch"):
                    state["confirm_delete"] = None
                    st.rerun()
    if not docs:
        st.caption("No documents yet.")

    doc_id = state["doc_picker"]
    if doc_id:                                       # remember it in the URL (survives refresh)
        st.query_params["doc"] = doc_id
    elif "doc" in st.query_params:
        del st.query_params["doc"]

    # No settings for users: the app always uses the best setup from our evaluation
    # (Groq gpt-oss-20b, hybrid search, top-5 clauses, no reranker, no minimum score).
    # Developers can still change things: start the app with  DEV_MODE=1 streamlit run streamlit_app.py
    provider, model, thinking = "groq", "", "default"
    k, mode, use_rerank, min_score, show_debug = 5, "hybrid", False, 0.0, False
    if os.environ.get("DEV_MODE") == "1":
        with st.expander("Developer settings"):
            providers = list(PROVIDERS)
            provider = st.selectbox("LLM provider", providers, index=providers.index("groq"))
            model = st.text_input("Model", placeholder=PROVIDERS[provider][0])
            thinking = st.selectbox("Thinking", ["default", "low", "medium", "high"])
            k = st.slider("Clauses to retrieve (k)", 1, 10, 5)
            mode = st.selectbox("Search", ["hybrid", "dense"])
            use_rerank = st.toggle("Reranker", value=False)
            min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)
            show_debug = st.toggle("Show under the hood", value=False)
    if doc_id and st.button("Clear chat", icon=":material/delete_sweep:"):
        state["history"][doc_id] = []
        save_chat(doc_id, [])
        state["selected"] = None
        st.rerun()


# ------------------------------------------------------------------ main area
TITLE = "⚖️ Legal Document Navigator"
SUBTITLE = "Ask questions about your contract in plain language. Every answer shows the clause it came from."

if not doc_id:
    st.title(TITLE)
    st.caption(SUBTITLE)
    st.info("Add a PDF in the sidebar, or click one of your documents to open it.")
    st.stop()

# cache_resource keeps the retriever (and its BM25 index) between reruns
@st.cache_resource
def get_retriever(k: int, min_score: float, mode: str, use_rerank: bool) -> Retriever:
    reranker = get_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_rerank else None
    return Retriever(embedder, store, k=k, min_score=min_score, mode=mode, reranker=reranker)


retriever = get_retriever(k, min_score, mode, use_rerank)
if doc_id not in state["history"]:                # after a refresh: read the saved chat
    state["history"][doc_id] = load_chat(doc_id)
history = state["history"][doc_id]
chat_col, view_col = st.columns([1, 1], gap="large")

# ---------- left: chat history
with chat_col:
    st.title(TITLE)          # title sits above the chat only, so the PDF can start at the top
    st.caption("Plain-language answers with sources · not legal advice")   # one short line
    # the chat scrolls inside this box; the page itself does not scroll, so the PDF never moves
    with st.container(height=520, border=False, key="chat-box"):
        if not history:
            st.markdown("**Try asking:**  \n"
                        "• Who are the parties?  \n"
                        "• How long does this agreement last?  \n"
                        "• Can I end it early?")
        for mi, msg in enumerate(history):
            with st.chat_message(msg["role"]):
                if msg.get("error"):
                    st.error(msg["text"])
                    continue
                shown = strip_label(msg["text"]) if msg["role"] == "assistant" else msg["text"]
                st.markdown(md_safe(shown))
                for ci, src in enumerate(msg.get("citations", []), start=1):
                    if st.button(f"[{ci}] {src.short_ref}", key=f"cite-{doc_id}-{mi}-{ci}"):
                        asked = history[mi - 1]["text"] if mi > 0 else ""
                        show_citation(doc_id, src, citation_query(asked, msg["text"], ci))
                for w in msg.get("warnings", []):
                    st.caption(f"⚠️ {w}")
                if show_debug and msg.get("debug"):
                    with st.expander("Under the hood"):
                        st.markdown("**Retrieval** (all top-k, ✓ = above minimum similarity)")
                        for s in msg["debug"]["hits"]:
                            mark = "✓" if s.score >= msg["debug"]["min_score"] else "✗"
                            extra = f"  (cosine {s.dense_score:.2f})" if s.rerank_score is not None else ""
                            st.text(f"{mark} {s.score:.3f}  {s.short_ref}{extra}")
                        t = msg["debug"].get("timings", {})
                        st.caption(f"Time: retrieval {t.get('retrieval', 0):.2f}s · LLM {t.get('llm', 0):.2f}s")
                        if msg["debug"]["raw"]:
                            st.markdown("**Raw LLM output**")
                            st.text(msg["debug"]["raw"])

# ---------- new question (chat_input is pinned to the bottom of the page)
question = st.chat_input("Ask about this document…")
if question:
    history.append({"role": "user", "text": question})
    try:
        llm = get_llm(provider, model.strip(), thinking)
        with chat_col, st.spinner("Reading the relevant clauses…"):
            ans = answer_question(question, retriever, llm, doc_id=doc_id)
        history.append({
            "role": "assistant", "text": ans.text, "citations": ans.citations,
            "warnings": ans.warnings,
            "debug": {"hits": ans.candidates, "raw": ans.raw_llm_output,
                      "min_score": min_score, "timings": ans.timings},
        })
        if ans.citations:                       # open the first source automatically
            show_citation(doc_id, ans.citations[0], citation_query(question, ans.text, 1))
    except Exception as e:                      # missing API key, Ollama not running, ...
        history.append({"role": "assistant", "error": True, "text": f"{type(e).__name__}: {e}"})
    save_chat(doc_id, history)
    st.rerun()

# ---------- right: document viewer
# Always shows the PDF. Before any question: plain pages you can flip through.
# After a question (or a click on [1], [2] ...): jumps to the cited page and highlights the clause.
with view_col:
    pdf = pdf_path_for(doc_id)
    sel = state.get("selected")
    src = sel["source"] if sel and sel["doc_id"] == doc_id else None
    # only the most relevant 1-2 sentences of the cited clause get highlighted
    highlights = key_sentences(sel.get("query") or "", src.text) if src is not None else ()
    on_page = False

    if pdf:
        n_pages = page_count(str(pdf))
        page_key = f"page-{doc_id}"
        state.setdefault(page_key, 1)
        prev_col, label_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
        if prev_col.button("◀ Prev", key="prev-page", width="stretch",
                           disabled=state[page_key] <= 1):
            state[page_key] -= 1
            st.rerun()                                # redraw so the buttons update too
        if next_col.button("Next ▶", key="next-page", width="stretch",
                           disabled=state[page_key] >= n_pages):
            state[page_key] += 1
            st.rerun()
        state[page_key] = min(max(int(state[page_key]), 1), n_pages)   # stay inside the PDF
        page_no = state[page_key]
        label_col.markdown(f"<div style='text-align:center'>Page {page_no} of {n_pages}</div>",
                           unsafe_allow_html=True)

        # highlight only if the selected clause is on the page being shown
        on_page = src is not None and src.page_start <= page_no <= src.page_end
        if src is not None:
            col_a, col_b = st.columns([4, 1], vertical_alignment="center")
            col_a.markdown(f"🟨 **{src.short_ref}**"
                           + ("" if on_page else f"  —  see page {src.page_start}"))
            if col_b.button("Clear", key="clear-highlight", width="stretch"):
                state["selected"] = None
                st.rerun()
        else:
            st.caption("Ask a question to highlight the clause it comes from.")

    # the page image is shrunk to fit this box, so the whole page is visible (see CSS at the top)
    with st.container(height=520, border=False, key="pdf-viewer"):
        if pdf:
            st.image(render_page(str(pdf), int(page_no), highlights if on_page else ()))
        else:
            st.caption("Original PDF not stored (this document was ingested from the terminal). "
                       "Upload it here to see the pages.")
        if src is not None:
            with st.expander("Clause text", expanded=pdf is None):
                shown = src.text
                for h in highlights:                     # the highlighted sentences in bold
                    shown = shown.replace(h, f"**{h}**")
                st.markdown(md_safe(shown))