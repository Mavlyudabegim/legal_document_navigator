"""
styles.py  —  the CSS that makes the layout work

The page itself must not scroll, otherwise the PDF moves with it. So the chat and the PDF
each get their own scroll box, as tall as the window minus what is above and below them.
st.container(key="chat-box") gets the CSS class "st-key-chat-box", which is how we find it.
"""

CSS = """
<style>
:root {
    --chat-top: 22rem;      /* above + below the chat box: title, caption, input bar */
    --viewer-top: 19.5rem;  /* above + below the PDF box: page buttons, hint line, input bar */
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
"""