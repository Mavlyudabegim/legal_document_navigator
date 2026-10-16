"""text.py  —  small text helpers for the interface"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def make_doc_id(filename: str, data: bytes, owner: str = "") -> str:
    """'My Lease (final).pdf' -> 'my-lease-final-3fa9c1'. The hash covers the owner + the file,
    so one person never ingests the same file twice, and two people uploading the same file
    get two separate copies (one can't delete the other's)."""
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:40] or "doc"
    return f"{stem}-{hashlib.sha1(owner.encode() + data).hexdigest()[:6]}"


def display_name(doc_id: str, names: dict) -> str:
    """Original file name if we know it, else the id without the hash: 'legal-document-sample'."""
    return names.get(doc_id) or re.sub(r"-[0-9a-f]{6}$", "", doc_id)


_LABEL = re.compile(r"^\s*[*_]*\s*(yes|no|partly)\s*[.,:!]?\s*[*_]*\s*", re.IGNORECASE)
_LABEL_WITH_PUNCT = re.compile(r"^\s*[*_]*\s*(yes|no|partly)\s*[*_]*\s*[.,:!]", re.IGNORECASE)


def strip_label(text: str) -> str:
    """Remove the 'Yes.' / 'No.' / 'Partly.' the prompt puts at the start of every answer.
    The model still writes it (the evaluation grades answers by it, see prompts.py rule 10);
    we just don't show it. 'No' as a real first word ("No party may...") is kept, because
    we only strip it when it is followed by punctuation."""
    m = _LABEL.match(text)
    if not m or not _LABEL_WITH_PUNCT.match(text):
        return text
    rest = text[m.end():]
    return rest[:1].upper() + rest[1:] if rest else text


def md_safe(text: str) -> str:
    """Streamlit treats $...$ as maths; escape dollar signs in contract text."""
    return text.replace("$", r"\$")