"""
chat_io.py  —  turn a chat into plain JSON and back

Chat messages hold Source objects (the cited clauses). JSON can't store objects, so we
save them as dicts and rebuild the Source objects when the chat is loaded again.
"""
from __future__ import annotations

from app.retrieval.retriever import Source


def to_json(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = dict(m)
        m["citations"] = [dict(c.__dict__) for c in m.get("citations", [])]
        if m.get("debug"):
            m["debug"] = {**m["debug"], "hits": [dict(c.__dict__) for c in m["debug"]["hits"]]}
        out.append(m)
    return out


def from_json(messages: list[dict]) -> list[dict]:
    for m in messages:
        m["citations"] = [Source(**c) for c in m.get("citations", [])]
        if m.get("debug"):
            m["debug"]["hits"] = [Source(**c) for c in m["debug"]["hits"]]
    return messages