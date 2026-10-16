"""
identity.py  —  a private space per browser, without any login

The first time someone opens the app, their browser gets a random ID (like 3f9a…c21),
saved in a cookie for one year. Every document is stored together with that ID, and the
app only lists documents with the same ID. So each browser sees only its own files.

Limits (no login means no account):
  - another browser or device = a new, empty space
  - clearing the browser's cookies = the old documents can't be reached any more
"""
from __future__ import annotations

import inspect
import re
import uuid

import streamlit as st
import streamlit.components.v1 as components

COOKIE = "lnav_uid"
_VALID = re.compile(r"[0-9a-f]{32}")


def browser_id() -> str:
    """This browser's ID: from the cookie if it has one, otherwise a new one (and set the cookie)."""
    try:
        uid = st.context.cookies.get(COOKIE)
    except Exception:                        # no browser (e.g. automated tests)
        uid = None
    if isinstance(uid, str) and _VALID.fullmatch(uid):
        return uid
    uid = st.session_state.get("browser_id") or uuid.uuid4().hex
    st.session_state["browser_id"] = uid
    # Streamlit can read cookies but not write them, so a tiny invisible script writes it.
    # Next visit, st.context.cookies has it. (https: SameSite=None so it also works inside iframes)
    script = f"""<script>
        const extra = location.protocol === "https:" ? "; SameSite=None; Secure" : "; SameSite=Lax";
        document.cookie = "{COOKIE}={uid}; max-age=31536000; path=/" + extra;
    </script>"""
    with st.sidebar:
        if "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
            st.html(script, unsafe_allow_javascript=True)      # newer Streamlit
        else:
            components.html(script, height=0)                  # older Streamlit
    return uid