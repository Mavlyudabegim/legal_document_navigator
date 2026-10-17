"""
fewshot.py  —  ContractNLI-specific guidance for the answer prompt

Most remaining mistakes are not retrieval mistakes: the model reads the right clause but
picks a different label than the ContractNLI lawyers would (e.g. "marked confidential OR
otherwise understood to be confidential" counts as FALSE for "Explicit identification").
Those conventions are hard to guess from the question alone, so we show the model:

  1. the original ContractNLI hypothesis behind the question (more precise than our paraphrase)
  2. a few labelled examples for the SAME hypothesis, taken from OTHER contracts in a
     different split (never the split we evaluate on, so there is no leakage)

This is only used by the benchmark (eval/contractnli.py --fewshot N / --with-hypothesis);
the app itself never sees it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ANSWER = {"Entailment": "Yes.", "Contradiction": "No."}
MAX_CLAUSE_CHARS = 700


def _clause(doc: dict, span_ids: list[int]) -> str:
    text = " ".join(doc["text"][doc["spans"][i][0]:doc["spans"][i][1]].strip() for i in span_ids)
    text = " ".join(text.split())
    return text if len(text) <= MAX_CLAUSE_CHARS else text[:MAX_CLAUSE_CHARS].rsplit(" ", 1)[0] + " …"


def load_examples(data_dir: str | Path, split: str, eval_split: str, n: int, seed: int = 0) -> dict:
    """{hypothesis_key: [(clause, answer), ...]} with up to n examples per hypothesis,
    mixing "Yes." and "No." examples when both exist (the more common answer first)."""
    if split == eval_split:
        raise SystemExit(f"--fewshot-split {split} is the split being evaluated; pick another one")
    path = Path(data_dir) / f"{split}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found (needed for --fewshot)")
    data = json.loads(path.read_text())
    pool: dict[str, dict[str, list[str]]] = {}
    for doc in data["documents"]:
        for key, ann in doc["annotation_sets"][0]["annotations"].items():
            if ann["choice"] in ANSWER and ann["spans"]:
                pool.setdefault(key, {}).setdefault(ann["choice"], []).append(_clause(doc, ann["spans"]))
    rng = random.Random(seed)
    out = {}
    for key, by_choice in pool.items():
        # the more common answer goes first, so a rare "No." doesn't dominate the examples
        order = sorted(by_choice, key=lambda c: -len(by_choice[c]))
        for lst in by_choice.values():
            rng.shuffle(lst)
            lst.sort(key=len)                     # prefer short, focused clauses
            del lst[max(n, 1) * 3:]
        picked = []
        i = 0
        while len(picked) < n and any(by_choice.get(c) for c in order):
            c = order[i % len(order)]
            if by_choice.get(c):
                picked.append((by_choice[c].pop(0), ANSWER[c]))
            i += 1
        out[key] = picked
    return out


def guidance_for(q: dict, examples: dict | None, with_hypothesis: bool) -> str | None:
    parts = []
    if with_hypothesis or examples:
        parts.append(
            f'This question checks one statement about the contract:\n"{q["hypothesis"]}"\n'
            'Answer "Yes." if the contract makes this statement true, "No." if the contract '
            'says the opposite, and "The document doesn\'t say this." if no clause addresses it.')
    ex = (examples or {}).get(q["hypothesis_key"], [])
    if ex:
        lines = ["How lawyers labelled this same statement in OTHER contracts "
                 "(for calibration only; do not cite these):"]
        for i, (clause, ans) in enumerate(ex, 1):
            lines.append(f'Example {i}. Clause: "{clause}"\nCorrect answer: {ans}')
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts) or None
