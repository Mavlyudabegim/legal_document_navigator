"""
text_parser.py  —  plain text  ->  list of Elements (same output as parser.py, but no PDF)

Used for datasets that ship contracts as text (ContractNLI, CUAD), and later for .txt uploads.
Without fonts we can't see bold or size, so a heading is detected from the text alone:
  * a SHORT line (≤ 12 words) that starts with numbering ("5.", "Section 5", "ARTICLE IV")
    and reads like a title (mostly Capitalised words, no full stop in the middle), or
  * a short line in ALL CAPS ("CONFIDENTIALITY", "TERM AND TERMINATION").
Lines broken in the middle of a sentence are joined back into paragraphs.
"""
from __future__ import annotations

import re

from app.ingestion.parser import NUMBERING, Element

_SMALL = {"of", "and", "or", "the", "to", "in", "on", "for", "a", "an", "by", "with", "&"}


def _is_heading(line: str) -> bool:
    words = line.split()
    if not words or len(words) > 12 or line.endswith((",", ";")):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):          # ALL CAPS
        return len(words) <= 10
    if NUMBERING.match(line):
        rest = NUMBERING.sub("", line, count=1).strip(" .:-")
        rw = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", rest)]
        if not rw:
            return False
        titled = sum(w[0].isupper() or w.lower() in _SMALL for w in rw) / len(rw)
        return titled >= 0.8 and "." not in rest.rstrip(".")
    return False


def _number(line: str) -> tuple[str | None, int]:
    m = NUMBERING.match(line)
    if not m:
        return None, 1
    num = m.group("num")
    return num, (num.count(".") + 1 if num[0].isdigit() else 1)


def parse_text(text: str, title: str | None = None) -> list[Element]:
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    elements: list[Element] = []
    if title:
        elements.append(Element("heading", title, 1, 1, level=0))

    cur: Element | None = None
    for line in lines:
        if not line:
            cur = None                                   # blank line = paragraph break
            continue
        if _is_heading(line):
            num, level = _number(line)
            elements.append(Element("heading", line, 1, 1, level=level, number=num))
            cur = None
            continue
        starts_item = bool(re.match(r"^(\(?[a-z0-9]{1,4}[.)]|\d+(\.\d+)+|[•\-–])\s", line))
        if cur is not None and not starts_item and not cur.text.endswith((".", ":", ";")):
            # the sentence continues on this line
            if cur.text.endswith("-") and line[:1].islower():
                cur.text = cur.text[:-1] + line
            else:
                cur.text += " " + line
        elif cur is not None and not starts_item and line[:1].islower():
            cur.text += " " + line
        else:
            cur = Element("paragraph", line, 1, 1)
            elements.append(cur)

    # headings without numbering get level 1 so they don't nest under each other
    for e in elements:
        if e.kind == "heading" and e.level != 0 and e.number is None:
            e.level = 1
    return elements