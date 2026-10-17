"""
qa.py  —  Pipeline B: question -> retrieve -> prompt -> LLM -> checked answer with citations

    retrieve  : find the clauses that match the question (retriever.py)
    guard     : nothing relevant found?  -> answer "not found", skip the LLM (saves money,
                and the model can't make something up from irrelevant text)
    generate  : send rules + clauses + question to the LLM (prompts.py, llm.py)
    verify    : read the [S1] labels in the answer, drop labels that were never given to
                the model, renumber the rest [1], [2] ... and attach the real section + page

Whole-contract mode (whole_doc_tokens > 0): if the whole document is shorter than that many
tokens, the LLM gets EVERY chunk instead of the top-k. Nothing can be missed by search, and
exceptions written in other clauses are visible too. Long documents still use search.

Follow-up for "not mentioned" answers (followup="more" or "whole"):
  "The top-5 clauses don't mention it" is not the same as "the contract doesn't mention it".
  When the first answer says the document doesn't address the question (or is empty), we look
  again before giving up:
    more  : the NEXT clauses in the ranking (6..followup_k) are sent in a second call
    whole : the rest of the contract (every clause not shown yet) is sent in a second call
  Answer.not_found_scope then says how much was checked: "top_k", "more_clauses" or "document".
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.generation.llm import LLM
from app.generation.prompts import NOT_FOUND, build_messages
from app.retrieval.retriever import Retriever, Source

# matches [S1], [S2, S3], [S1][S4] and also the styles bigger gpt-oss models like to use:
# 【S1】, 【S1†L3-L5】, [S1†source], [S1; S2]
_CITE = re.compile(r"[\[【]\s*(S\d+(?:\s*[,;]\s*S\d+)*)(?:\s*[†:][^\]】]*)?\s*[\]】]")


def cited_labels(text: str) -> list[str]:
    """All source labels cited in a raw answer, in order of first appearance: ['S2', 'S1']."""
    order: list[str] = []
    for m in _CITE.finditer(text or ""):
        for label in re.split(r"\s*[,;]\s*", m.group(1)):
            if label not in order:
                order.append(label)
    return order


@dataclass
class Answer:
    question: str
    text: str                                         # answer with [1], [2] markers
    found: bool                                       # False = "not found in the document"
    citations: list[Source] = field(default_factory=list)   # [1] -> citations[0], ...
    retrieved: list[Source] = field(default_factory=list)   # everything sent to the LLM
    warnings: list[str] = field(default_factory=list)
    raw_llm_output: str = ""
    candidates: list[Source] = field(default_factory=list)  # all top-k, before the cut-off
    timings: dict = field(default_factory=dict)             # seconds per step
    followup: str | None = None          # None, or "more"/"whole" if a second look was taken
    first_text: str | None = None        # the answer before the follow-up
    not_found_scope: str | None = None   # for "not mentioned": "top_k", "more_clauses", "document"


def _process_citations(raw: str, sources: list[Source]) -> tuple[str, list[Source], list[str]]:
    by_label = {s.label: s for s in sources}
    order: list[str] = []          # labels in order of first appearance
    invalid: set[str] = set()

    def replace(m: re.Match) -> str:
        out = []
        for label in re.split(r"\s*[,;]\s*", m.group(1)):
            if label not in by_label:
                invalid.add(label)            # the model cited something it was never given
                continue
            if label not in order:
                order.append(label)
            n = order.index(label) + 1
            if f"[{n}]" not in out:
                out.append(f"[{n}]")
        return "".join(out)

    text = _CITE.sub(replace, raw)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)          # tidy "word [S9]." -> "word."
    warnings = [f"Removed citation to unknown source {l}." for l in sorted(invalid)]
    return text, [by_label[l] for l in order], warnings


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


_NOT_MENTIONED_OPENINGS = ("the document doesn't say", "the document does not say",
                           "i couldn't find", "i could not find")


def says_not_mentioned(answer: Answer) -> bool:
    """True if the answer is a 'not in the document' answer (or empty)."""
    if not answer.found:
        return True
    t = re.sub(r"^[\s*_#>\-]+", "", answer.text or "").lower().replace("’", "'")
    return t.startswith(_NOT_MENTIONED_OPENINGS)


def _relabel(sources: list[Source]) -> list[Source]:
    return [Source(**{**s.__dict__, "label": f"S{i}"}) for i, s in enumerate(sources, start=1)]


def answer_question(question: str, retriever: Retriever, llm: LLM, doc_id: str | None = None,
                    whole_doc_tokens: int = 0, followup: str = "none", followup_k: int = 15) -> Answer:
    first = _answer_once(question, retriever, llm, doc_id, whole_doc_tokens)
    if followup == "none" or not doc_id or not says_not_mentioned(first):
        if says_not_mentioned(first):
            first.not_found_scope = "document" if first.timings.get("whole_doc") else "top_k"
        return first

    # second look: clauses the model has NOT seen yet
    t0 = time.perf_counter()
    seen = {s.chunk_id for s in first.retrieved}
    if followup == "whole":
        extra = [s for s in retriever.all_chunks(doc_id) if s.chunk_id not in seen]
        scope = "document"
    else:
        extra = [s for s in retriever.search(question, doc_id=doc_id, k=followup_k) if s.chunk_id not in seen]
        scope = "more_clauses"
    if not extra:
        first.not_found_scope = scope
        return first
    second = _answer_once(question, retriever, llm, doc_id, 0, sources=_relabel(extra))
    second.timings["retrieval"] = first.timings.get("retrieval", 0) + second.timings.get("retrieval", 0)
    second.timings["llm"] = first.timings.get("llm", 0) + second.timings.get("llm", 0)
    second.timings["followup"] = time.perf_counter() - t0
    second.followup, second.first_text = followup, first.text
    second.candidates = first.candidates
    if says_not_mentioned(second):              # still nothing: keep the first answer, but now
        first.followup, first.first_text = followup, first.text      # we know the wider scope
        first.not_found_scope, first.timings = scope, second.timings
        return first
    return second                               # the second look found an answer


def _answer_once(question: str, retriever: Retriever, llm: LLM, doc_id: str | None,
                 whole_doc_tokens: int, sources: list[Source] | None = None) -> Answer:
    question = question.strip()
    t0 = time.perf_counter()
    if sources is not None:                                    # follow-up: sources given
        candidates, whole = sources, False
    else:
        candidates = retriever.search(question, doc_id=doc_id)    # search ONCE ...
        sources = retriever.filter(candidates)                     # ... then apply the cut-off
        whole = False
    if whole_doc_tokens and doc_id:
        everything = retriever.all_chunks(doc_id)
        if everything and sum(_approx_tokens(s.text) for s in everything) <= whole_doc_tokens:
            sources, whole = everything, True                  # short contract: send it all
    timings = {"retrieval": time.perf_counter() - t0, "whole_doc": whole}

    if not sources:                                           # guard: skip the LLM
        return Answer(question, NOT_FOUND, found=False, candidates=candidates, timings=timings,
                      warnings=["No clause scored above the minimum similarity."])

    system, user = build_messages(question, sources)
    t1 = time.perf_counter()
    raw = llm.complete(system, user)
    if not raw.strip():                        # reasoning models can run out of tokens
        raw = llm.complete(system, user)       # try once more
    timings["llm"] = time.perf_counter() - t1
    if not raw.strip():
        return Answer(question, "The model returned an empty answer. Please try again.",
                      found=False, candidates=candidates, timings=timings,
                      warnings=["Empty response from the LLM."])

    common = dict(retrieved=sources, raw_llm_output=raw, candidates=candidates, timings=timings)
    if NOT_FOUND.lower().rstrip(".") in raw.lower():
        return Answer(question, NOT_FOUND, found=False, **common)

    text, cited, warnings = _process_citations(raw, sources)
    if not cited:
        warnings.append("The answer has no citations, so it can't be checked against the document.")
    return Answer(question, text, found=True, citations=cited, warnings=warnings, **common)