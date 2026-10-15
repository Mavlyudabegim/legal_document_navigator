from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

@dataclass
class Line:
    text: str 
    page: int # page number
    size: float # average font size
    bold: bool # most characters are bold?
    y0: float # top position on the page
    y1: float # bottom position on the page
    block: int # PyMuPDF block number (lines in one block belong together)


@dataclass
class Element:
    kind: str #"heading" or "paragraph"
    text:str
    page_start: int
    page_end: int
    level:int=0 # heading level: 1="4.", 2="4.1",...
    number: str|None = None #"4", "4.1", "IV"
    meta:dict= field(default_factory=dict)


# "4.", "4.1", "Section 4", "ARTICLE IV", "§ 4", "Clause 4" at the start of a line

NUMBERING = re.compile(
    r"^\s*(?:(?:section|article|clause|§)\s*)?"
    r"(?P<num>\d+(?:\.\d+)*|[IVXLC]+)\.?(?=\s|$)",
    re.IGNORECASE,
)

def _read_lines(doc: pymupdf.Document)-> tuple[list[Line], list[float]]:
    lines, page_heights = [],[]
    for pno, page in enumerate(doc, start=1):
        page_heights.append(page.rect.height)
        data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)

        for b in data["blocks"]:
            for ln in b.get("lines", []):
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue

                text = "".join(s["text"] for s in spans).strip()
                chars = sum(len(s["text"]) for s in spans)
                size = sum(s["size"]*len(s["text"]) for s in spans)/chars
                bold_chars = sum(
                    len(s["text"]) for s in spans
                    if s["flags"] & 16 or "bold" in s["font"].lower()
                )
                lines.append(Line(text, pno, round(size, 1), bold_chars > chars / 2,
                                  ln["bbox"][1], ln["bbox"][3], b["number"]))


    return lines, page_heights

# ------------------------------------------ headers / footers
def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _remove_headers_footers(lines: list[Line], page_heights: list[float],
                            margin: float = 0.12, min_share: float =0.5) -> list[Line]:
    n_pages=len(page_heights)
    if n_pages < 2:
        return lines


    def in_margin(l: Line)-> bool:
        h = page_heights[l.page -1]
        return l.y1 < h * margin or l.y0 > h * (1-margin)


    pages_per_text = Counter()
    for key in {(l.page, _norm(l.text)) for l in lines if in_margin(l)}:
        pages_per_text[key[1]]+=1

    repeating = {t for t, c in pages_per_text.items() if c >= max(2, min_share * n_pages)}
    return [l for l in lines if not (in_margin(l) and _norm(l.text) in repeating)]

# ------------------------------------------ headings

def _body_font_size(lines: list[Line]) -> float:
    """The most common font size = normal body text"""

    c = Counter()
    for l in lines:
        c[l.size]+=len(l.text)

    return c.most_common(1)[0][0]


def _is_heading(l: Line, body_size: float, max_words: int=12) -> bool:
    words = len(l.text.split())
    if words > max_words or l.text.endswith((",", ";")):
        return False
    looks_different = l.size>=body_size+0.8 or l.bold
    shouting = l.text.isupper() and words<=8 and len(l.text)>3
    numbered = bool(NUMBERING.match(l.text))

    # A heading must look like a heading. Numbering alone is not enough
    return looks_different or (shouting and numbered)


def _heading_number(text:str)->tuple[str | None, int]:
    m=NUMBERING.match(text)
    if not m:
        return None, 1
    num =m.group("num")
    return num, num.count(".")+1 if num[0].isdigit() else 1

# ------------------------------------------ lines -> paragraphs

def _join(prev: str, nxt:str)->str:
    if re.search(r"[A-Za-z]-$", prev) and nxt[:1].islower():
        return prev[:-1]+nxt
    return prev+" "+nxt


def _starts_new_paragraph(text: str) -> bool:
    return bool(re.match(r"^\s*(\d+(\.\d+)+|\([a-z0-9]{1,4}\))\s", text))

def parse_pdf(path:str) -> list[Element]:
    doc =pymupdf.open(path)
    lines, heights = _read_lines(doc)

    if not lines:
        raise ValueError(
            f"No text found in {path}. It is probably a scanned PDF: "
            "run OCR first (e.g. `ocrmypdf input.pdf output.pdf`)."
        )

    lines = _remove_headers_footers(lines, heights)
    body_size= _body_font_size(lines)

    elements: list[Element]= []
    cur: Element|None = None
    prev_line: Line|None=None

    for l in lines:
        if _is_heading(l, body_size):
            if(cur and cur.kind=="heading" and prev_line and prev_line.block == l.block
               and not NUMBERING.match(l.text)):
                cur.text+=" "+l.text
                continue
            num, level = _heading_number(l.text)
            cur = Element("heading", l.text, l.page, l.page, level=level, number=num,
                      meta={"size":l.size, "bold":l.bold})

            elements.append(cur)

        else:
            prev_open = prev_line is not None and not prev_line.text.endswith((".",":",";"))
            same_paragraph = (
                cur is not None and cur.kind == "paragraph" and prev_line is not None
                and (prev_open or not _starts_new_paragraph(l.text))
                and (prev_line.block == l.block or (l.page!=prev_line.page and prev_open))
            )

            if same_paragraph:
                cur.text = _join(cur.text, l.text)
                cur.page_end=l.page

            else:
                cur = Element("paragraph", l.text, l.page, l.page)
                elements.append(cur)

        prev_line=l

    # Heading levels: if the document has no numbering, fall back to font size ranks
    sizes = sorted({e.meta["size"] for e in elements if e.kind=="heading"}, reverse=True)

    for e in elements:
        if e.kind == "heading" and e.number is None:
            e.level = sizes.index(e.meta["size"])+1


    # The document title (first, biggest heading) is level 0 so it doesn't nest everything
    if elements and elements[0].kind=="heading" and elements[0].number is None:
        elements[0].level=0
    return elements


if __name__ == "__main__":
    import sys
    for e in parse_pdf(sys.argv[1] if len(sys.argv) > 1 else "data/legal_document_sample.pdf"):
        tag = f"H{e.level}" if e.kind == "heading" else "  P"
        print(f"[p{e.page_start}-{e.page_end}] {tag}  {e.text[:90]}")

        





