"""
streamlit_app.py  —  web interface for Legal Document Navigator

    streamlit run streamlit_app.py

Layout
    sidebar : add PDFs, open or delete your documents
    left    : chat — answers with clickable citations
    right   : the PDF, with the sentences the answer used highlighted

Nothing about RAG happens here: an upload calls ingest(), each question calls answer_question().
Helpers for the page live in app/ui/, settings in app/config.py, storage in app/storage/.
"""
from __future__ import annotations

import os

import streamlit as st

# 1) secrets first: .env locally, st.secrets on Streamlit Cloud -> environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:                       # no secrets.toml: fine when running locally
    pass

from app import config  # noqa: E402  (must come after the secrets are loaded)
from app.generation.llm import LLM, PROVIDERS  # noqa: E402
from app.generation.qa import answer_question  # noqa: E402
from app.ingestion.embedder import Embedder  # noqa: E402
from app.ingestion.ingest import ingest  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.storage import get_storage  # noqa: E402
from app.ui import chat_io  # noqa: E402
from app.ui.highlight import citation_query, key_sentences, page_count, render_page  # noqa: E402
from app.ui.identity import browser_id  # noqa: E402
from app.ui.styles import CSS  # noqa: E402
from app.ui.text import display_name, make_doc_id, md_safe, strip_label  # noqa: E402

st.set_page_config(page_title="Legal Document Navigator", page_icon="⚖️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ shared objects
# st.cache_resource = create once, reuse on every rerun (loading a model is slow)
@st.cache_resource
def get_embedder() -> Embedder:
    # cache_path=None: SQLite connections can't be shared between Streamlit's threads
    return Embedder(model=config.EMB_MODEL, backend=config.EMB_BACKEND, cache_path=None)


@st.cache_resource
def get_store_and_files():
    """Supabase if SUPABASE_URL + SUPABASE_KEY are set, otherwise Chroma + data/ folders.
    store = chunks + vectors, files = PDFs, file names and chats (see app/storage/)."""
    return get_storage(config.CHROMA_PATH, embedding_model=f"{config.EMB_BACKEND}:{config.EMB_MODEL}")


@st.cache_resource
def get_reranker(model: str):
    from app.retrieval.reranker import Reranker
    return Reranker(model)


@st.cache_resource
def get_llm(provider: str, model: str, thinking: str) -> LLM:
    effort = None if thinking == "default" else thinking
    return LLM(provider=provider, model=model or None, reasoning_effort=effort)


# ------------------------------------------------------------------ helpers
def list_documents() -> list[str]:
    """Documents uploaded from THIS browser (evaluation NDAs, 'cnli-…', are hidden)."""
    return [d for d in files.doc_ids(owner) if not d.startswith("cnli-")]


def delete_document(doc_id: str) -> None:
    """Remove a document everywhere: its chunks, the PDF, its chat, its name.
    files.delete() checks the owner first, so a browser can only delete its own documents."""
    if files.delete(doc_id, owner):
        store.delete_document(doc_id)
    state["history"].pop(doc_id, None)


def show_citation(doc_id: str, src, query: str = "") -> None:
    """Remember the clicked citation and jump the viewer to its page."""
    state["selected"] = {"doc_id": doc_id, "source": src, "query": query}
    state[f"page-{doc_id}"] = src.page_start


def save_chat(doc_id: str, messages: list[dict]) -> None:
    files.save_chat(doc_id, chat_io.to_json(messages))


def load_chat(doc_id: str) -> list[dict]:
    try:
        return chat_io.from_json(files.load_chat(doc_id))
    except Exception:                       # e.g. no internet: start with an empty chat
        return []


def ingest_upload(up) -> str | None:
    """Save + index one uploaded PDF. Returns its doc_id, or None if it could not be read."""
    data = up.getvalue()
    doc_id = make_doc_id(up.name, data, owner)
    if doc_id in state["failed"]:
        return None
    if store.count(doc_id) == 0:                     # not indexed yet
        files.set_name(doc_id, up.name, owner)
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
                files.delete(doc_id, owner)          # don't keep a half-saved document
                return None
    return doc_id


# ------------------------------------------------------------------ sidebar
store, files = get_store_and_files()
embedder = get_embedder()
state = st.session_state
owner = browser_id()                     # each browser has its own private list of documents
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

    docs = list_documents()
    if "doc_picker" not in state:                    # first run / after a browser refresh:
        state["doc_picker"] = st.query_params.get("doc")   # reopen the document from the URL
    if state.get("doc_picker") not in docs:
        state["doc_picker"] = None                   # None = nothing open, nothing shown

    # Your documents: click a name to open it, the bin to delete it
    names = files.names(owner)
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

    # No settings for users: the app always uses the best setup from the evaluation (app/config.py).
    # Developers can still change things: start the app with  DEV_MODE=1 streamlit run streamlit_app.py
    provider, model, thinking = config.LLM_PROVIDER, "", "default"
    k, mode, use_rerank, min_score, show_debug = config.TOP_K, config.SEARCH_MODE, False, 0.0, False
    if config.DEV_MODE:
        with st.expander("Developer settings"):
            providers = list(PROVIDERS)
            provider = st.selectbox("LLM provider", providers, index=providers.index(provider))
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
    reranker = get_reranker(config.RERANKER_MODEL) if use_rerank else None
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
    pdf = files.pdf_path(doc_id)          # local path (Supabase: downloaded once to a temp folder)
    sel = state.get("selected")
    src = sel["source"] if sel and sel["doc_id"] == doc_id else None
    # only the most relevant 1-2 sentences of the cited clause get highlighted
    highlights = key_sentences(sel.get("query") or "", src.text, embedder) if src is not None else ()
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