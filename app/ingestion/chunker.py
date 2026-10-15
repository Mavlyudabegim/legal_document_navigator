"""
chunker.py  —  Elements (from parser.py)  ->  Chunks ready for embedding

Strategy: "structure-aware chunking"
  1. Group paragraphs under their heading -> one Section per clause (e.g. "4. RENT").
     Keep the full heading path ("RESIDENTIAL LEASE > 4. RENT") for context and citations.
  2. Merge sections that are too small to stand alone into their neighbour.
  3. Split sections that are too big: first at paragraph (sub-clause) borders,
     then at sentence borders, never in the middle of a sentence. Add a small overlap.
  4. Prefix every chunk with its heading path before embedding.

Sizes are measured in TOKENS (the unit embedding models count in), not characters.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from app.ingestion.parser import Element


@dataclass
class Section:
    path: list[str]                 # ["RESIDENTIAL LEASE AGREEMENT", "4. RENT"]
    number: str | None              # "4"
    paragraphs: list[tuple[str, int, int]] = field(default_factory=list)  # (text, page_start, page_end)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str                       # original text, shown to the user
    embed_text: str                 # text that is actually embedded (with heading path)
    section_path: str               # "RESIDENTIAL LEASE AGREEMENT > 4. RENT"
    section_number: str | None      # "4"  -> lets the UI show "§4"
    page_start: int
    page_end: int
    n_tokens: int
    part: int = 1                   # 1, 2, ... if a long section was split


def approx_tokens(text: str) -> int:
    """Rough rule of thumb for English: 1 word ≈ 1.3 tokens.
    For exact numbers pass the embedding model's tokenizer (see Embedder.count_tokens)."""
    return int(len(text.split()) * 1.3) + 1


# Sentence splitting. Legal text is full of dots that are NOT sentence ends:
# "Ltd.", "No.", "Sec.", "e.g.", "U.S.", "EUR 950.00", "4.1". We protect those first.
_ABBREV = ["Ltd", "Inc", "Co", "Corp", "No", "Nos", "Sec", "Art", "Para", "Cl", "Mr", "Mrs", "Ms",
           "Dr", "St", "Apt", "vs", "e.g", "i.e", "etc", "U.S", "U.K", "approx", "Jan", "Feb",
           "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"]
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.", re.IGNORECASE)
_SENT_END = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\"“])")


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + "<DOT>", text)
    parts = _SENT_END.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------- 1. group into sections
def build_sections(elements: list[Element]) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []          # (level, heading text)
    current = Section(path=[], number=None)

    for el in elements:
        if el.kind == "heading":
            if current.paragraphs:
                sections.append(current)
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            stack.append((el.level, el.text))
            current = Section(path=[h for _, h in stack], number=el.number)
        else:
            current.paragraphs.append((el.text, el.page_start, el.page_end))
    if current.paragraphs:
        sections.append(current)
    return sections


# ---------------------------------------------------------------- 2 + 3. size control
def _pack(units: list[str], max_tokens: int, count: Callable[[str], int],
          overlap: int) -> list[list[str]]:
    """Greedily put units (paragraphs or sentences) into groups that fit max_tokens.
    The last `overlap` units of a group are repeated at the start of the next one."""
    groups, cur = [], []
    for u in units:
        if cur and count(" ".join(cur + [u])) > max_tokens:
            groups.append(cur)
            cur = list(cur[-overlap:]) if overlap else []      # always a LIST of units
            if cur and count(" ".join(cur + [u])) > max_tokens:
                cur = []                                       # overlap doesn't fit: skip it
        cur.append(u)
    if cur:
        groups.append(cur)
    return groups


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last resort for ONE 'sentence' longer than max_tokens (long lists or tables without
    full stops, common in contracts converted from SEC filings): cut it by word count.
    ~1.5 words per token budget keeps every piece safely under the limit."""
    words = text.split()
    step = max(20, int(max_tokens / 1.5))
    return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]


def _split_section_text(paras: list[str], max_tokens: int, count: Callable[[str], int],
                        overlap_sentences: int) -> list[str]:
    text = "\n".join(paras)
    if count(text) <= max_tokens:
        return [text]
    # try paragraph (sub-clause) borders first: "4.1 ...", "4.2 ..."
    pieces = []
    for group in _pack(paras, max_tokens, count, overlap=0):
        g = "\n".join(group)
        if count(g) <= max_tokens:
            pieces.append(g)
        else:  # one single paragraph is too long -> sentence level
            sentences = []
            for sent in split_sentences(g):
                sentences += _hard_split(sent, max_tokens) if count(sent) > max_tokens else [sent]
            for sg in _pack(sentences, max_tokens, count, overlap_sentences):
                pieces.append(" ".join(sg))
    return pieces


def chunk_elements(
    elements: list[Element],
    doc_id: str,
    doc_title: str | None = None,
    max_tokens: int = 350,          # bge-small / MiniLM read max 512 tokens; leave room for the prefix
    min_tokens: int = 15,           # only merge near-empty sections: in contracts every clause
                                    # should keep its own chunk so citations ("§2") stay exact
    overlap_sentences: int = 1,
    count_tokens: Callable[[str], int] = approx_tokens,
) -> list[Chunk]:
    sections = build_sections(elements)

    # 2. merge tiny sections into the previous section with the same parent
    merged: list[Section] = []
    for s in sections:
        text = " ".join(p[0] for p in s.paragraphs)
        prev = merged[-1] if merged else None
        if (prev and count_tokens(text) < min_tokens
                and prev.path[:-1] == s.path[:-1]
                and count_tokens(" ".join(p[0] for p in prev.paragraphs) + text) <= max_tokens):
            # keep the small section's heading inside the text so nothing is lost
            prev.paragraphs.append((s.path[-1] + "\n" + text if s.path else text,
                                    s.paragraphs[0][1], s.paragraphs[-1][2]))
        else:
            merged.append(s)

    # 3 + 4. split big sections, build chunk objects
    chunks: list[Chunk] = []
    for s in merged:
        path = " > ".join(s.path) if s.path else (doc_title or "Document")
        if doc_title and (not s.path or s.path[0] != doc_title):
            path = f"{doc_title} > {path}"
        paras = [p[0] for p in s.paragraphs]
        pages = (s.paragraphs[0][1], s.paragraphs[-1][2])
        for part, piece in enumerate(_split_section_text(paras, max_tokens, count_tokens,
                                                         overlap_sentences), start=1):
            uid = hashlib.sha1(f"{doc_id}|{path}|{part}|{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                id=uid, doc_id=doc_id, text=piece,
                embed_text=f"{path}\n{piece}",
                section_path=path, section_number=s.number,
                page_start=pages[0], page_end=pages[1],
                n_tokens=count_tokens(piece), part=part,
            ))
    return chunks


if __name__ == "__main__":
    import sys
    from parser import parse_pdf
    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_lease.pdf"
    for c in chunk_elements(parse_pdf(pdf), doc_id="demo"):
        print(f"§{c.section_number or '-':<3} p{c.page_start}-{c.page_end}  ~{c.n_tokens:>3} tok  "
              f"part {c.part}  {c.section_path}")