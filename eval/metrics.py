"""
metrics.py  —  the numbers behind the ContractNLI evaluation

    class_report(rows)         precision / recall / F1 per label, macro-F1, confusion matrix
    wilson(k, n)               95% confidence interval for a proportion (accuracy, recall, ...)
    bootstrap_ci(rows, fn)     95% interval for any metric (e.g. F1) by resampling questions
    weighted_accuracy(rows, w) accuracy re-weighted to the real label mix (for stratified samples)
    citation_quality(row)      do the cited clauses contain the lawyers' evidence?
    mcnemar(a, b)              is setup B really better than setup A on the SAME questions?
    phenomena(text)            exception / negation / condition words in a clause

A "row" is one answered question: {"gold": "TRUE"|"FALSE"|"NOT_MENTIONED", "pred": ..., ...}.
"""
from __future__ import annotations

import math
import random
import re

LABELS = ["TRUE", "FALSE", "NOT_MENTIONED"]
NAMES = {"TRUE": "Entailment", "FALSE": "Contradiction", "NOT_MENTIONED": "NotMentioned"}


# ------------------------------------------------------------------ per-class metrics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion k/n. Better than ±1.96·SE when n is small."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def prf(rows: list[dict], label: str) -> dict:
    tp = sum(r["pred"] == label and r["gold"] == label for r in rows)
    n_pred = sum(r["pred"] == label for r in rows)
    n_gold = sum(r["gold"] == label for r in rows)
    p = tp / n_pred if n_pred else 0.0
    rc = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": p, "recall": rc, "f1": f1, "support": n_gold, "tp": tp, "n_pred": n_pred}


def class_report(rows: list[dict]) -> dict:
    per = {lab: prf(rows, lab) for lab in LABELS}
    n = len(rows)
    correct = sum(r["pred"] == r["gold"] for r in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "accuracy_ci": wilson(correct, n),
        "macro_f1": sum(per[lab]["f1"] for lab in LABELS) / len(LABELS),
        "per_class": per,
        "confusion": {g: {p: sum(r["gold"] == g and r["pred"] == p for r in rows) for p in LABELS}
                      for g in LABELS},
    }


def bootstrap_ci(rows: list[dict], fn, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Resample the questions with replacement 1000 times and recompute the metric:
    the middle 95% of those values is the interval. Works for F1, where Wilson doesn't."""
    if not rows:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    vals = sorted(fn([rows[rnd.randrange(len(rows))] for _ in rows]) for _ in range(n_boot))
    return (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1])


def weighted_accuracy(rows: list[dict], weights: dict[str, float]) -> float:
    """Accuracy as if the sample had the real label mix. A stratified sample over-represents
    the rare FALSE class (so its F1 is measured on enough examples); this undoes that for the
    headline accuracy. weights = share of each label in the full question set."""
    acc = {}
    for lab in LABELS:
        g = [r for r in rows if r["gold"] == lab]
        if g:
            acc[lab] = sum(r["pred"] == lab for r in g) / len(g)
    total = sum(weights[lab] for lab in acc)
    return sum(weights[lab] * acc[lab] for lab in acc) / total if total else 0.0


def weighted_prf(rows: list[dict], weights: dict[str, float]) -> dict[str, dict]:
    """Precision / recall / F1 per label as if the sample had the real label mix.
    Why: precision depends on how common a label is. A sample with extra Contradictions
    makes Contradiction precision (and F1) look better than it would be on real contracts.
    Recall doesn't change (it only looks at questions of that label); precision is rebuilt
    from each gold label's answer rates, weighted by the real label shares."""
    rate = {}                                     # rate[g][p] = share of gold-g questions answered p
    for g in LABELS:
        gs = [r for r in rows if r["gold"] == g]
        rate[g] = {p: (sum(r["pred"] == p for r in gs) / len(gs)) if gs else 0.0 for p in LABELS}
    out = {}
    for lab in LABELS:
        tp = weights[lab] * rate[lab][lab]
        predicted = sum(weights[g] * rate[g][lab] for g in LABELS)
        p = tp / predicted if predicted else 0.0
        rc = rate[lab][lab]
        out[lab] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0}
    return out


# ------------------------------------------------------------------ citations
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def evidence_covered(chunk_text: str, evidence: str, n: int = 6, need: float = 0.5) -> bool:
    """Is this evidence sentence inside the chunk? Compared on normalised text with word
    6-grams, so evidence split across two chunks still counts if half of it is here."""
    c, e = _norm(chunk_text), _norm(evidence)
    if not e:
        return False
    if e in c:
        return True
    words = e.split()
    if len(words) <= n:
        return False
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return sum(g in c for g in grams) / len(grams) >= need


def citation_quality(cited_texts: list[str], evidence: list[str]) -> dict:
    """For one answer:
        precision  share of cited clauses that contain at least one evidence sentence
        recall     share of evidence sentences that appear in at least one cited clause
        supported  at least one cited clause contains evidence
    Evidence = the sentences the lawyers marked as proving the label."""
    if not cited_texts or not evidence:
        return {"n_cited": len(cited_texts), "precision": None, "recall": None, "supported": False}
    good = [any(evidence_covered(c, e) for e in evidence) for c in cited_texts]
    found = [any(evidence_covered(c, e) for c in cited_texts) for e in evidence]
    return {"n_cited": len(cited_texts), "precision": sum(good) / len(good),
            "recall": sum(found) / len(found), "supported": any(good)}


def citation_summary(rows: list[dict]) -> dict:
    """Over all rows that have saved citations (see contractnli.py):
        answers_with_citations  TRUE/FALSE answers that cite something
        supported               ... and at least one citation contains the gold evidence
        precision / recall      averaged over those answers
        right_label_unsupported correct TRUE/FALSE label, but no citation contains evidence
                                ("right answer, wrong reason")
        strict_accuracy         label correct AND (for TRUE/FALSE) supported by a citation
        unsupported_claims      gold NOT_MENTIONED, but the model said yes/no and cited something
    """
    have = [r for r in rows if "citations" in r]
    if not have:
        return {}
    ans = [r for r in have if r["gold"] in ("TRUE", "FALSE") and r["pred"] in ("TRUE", "FALSE")]
    cq = [citation_quality(r["citations"], r.get("evidence", [])) for r in ans]
    with_cites = [q for q in cq if q["n_cited"]]
    correct_tf = [(r, q) for r, q in zip(ans, cq) if r["pred"] == r["gold"]]
    strict = sum(
        (r["pred"] == r["gold"]) and (r["gold"] == "NOT_MENTIONED" or
                                      citation_quality(r["citations"], r.get("evidence", []))["supported"])
        for r in have)
    return {
        "n_rows": len(have),
        "answers": len(ans),
        "answers_with_citations": len(with_cites),
        "supported": sum(q["supported"] for q in with_cites) / len(with_cites) if with_cites else None,
        "precision": (sum(q["precision"] for q in with_cites) / len(with_cites)) if with_cites else None,
        "recall": (sum(q["recall"] for q in with_cites) / len(with_cites)) if with_cites else None,
        "right_label_unsupported": (sum(not q["supported"] for _, q in correct_tf) / len(correct_tf))
                                   if correct_tf else None,
        "strict_accuracy": strict / len(have),
        "unsupported_claims": sum(r["gold"] == "NOT_MENTIONED" and r["pred"] in ("TRUE", "FALSE")
                                  and bool(r["citations"]) for r in have),
    }


# ------------------------------------------------------------------ paired comparison
def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Compare two setups on the SAME questions. Only questions where exactly one setup is
    right carry information: b = A right & B wrong, c = A wrong & B right.
    Exact two-sided binomial test on b vs c (fine for small numbers)."""
    key = lambda r: (r["doc_id"], r["hypothesis_key"])  # noqa: E731
    a = {key(r): r["pred"] == r["gold"] for r in rows_a}
    b_ = {key(r): r["pred"] == r["gold"] for r in rows_b}
    common = sorted(set(a) & set(b_))
    b = sum(a[k] and not b_[k] for k in common)
    c = sum(b_[k] and not a[k] for k in common)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)
    return {"common": len(common), "a_only": b, "b_only": c, "both_right": sum(a[k] and b_[k] for k in common),
            "both_wrong": sum(not a[k] and not b_[k] for k in common), "p_value": p,
            "acc_a": sum(a[k] for k in common) / len(common) if common else None,
            "acc_b": sum(b_[k] for k in common) / len(common) if common else None}


# ------------------------------------------------------------------ linguistic phenomena
PHENOMENA = {
    # words that carve something out of a rule: "... except as required by law"
    "exception": r"\b(except|exception|unless|other than|excluding|exclude[sd]?|save (?:as|for)|"
                 r"notwithstanding|but not|apart from|with the exception of)\b",
    # words that flip the meaning: "shall not", "no license", "without"
    "negation": r"\b(not|no|never|neither|nor|none|without|cannot|can't|won't|shall not|may not|"
                r"must not|prohibited|forbidden|in no event|under no circumstances)\b",
    # words that make the rule depend on something: "if required by law", "subject to"
    "condition": r"\b(if|in the event|subject to|only if|provided that|provided,? however|upon|"
                 r"whenever|to the extent|in case|on condition)\b",
}
_PH = {k: re.compile(v, re.IGNORECASE) for k, v in PHENOMENA.items()}


# boilerplate that contains "not"/"without" but doesn't negate anything
_BOILERPLATE = re.compile(r"\b(?:including,? )?(?:but )?not limited to\b|\bwithout limitation\b|"
                          r"\bwhether or not\b", re.IGNORECASE)


def phenomena(text: str) -> list[str]:
    """Which of exception / negation / condition appear in this text.
    Phrases like "including but not limited to" are removed first: they contain "not" but
    widen a definition rather than negate it."""
    text = _BOILERPLATE.sub(" ", text or "")
    return [k for k, rx in _PH.items() if rx.search(text)]


# ------------------------------------------------------------------ questions from the same contract
# The 300 questions come from 61 contracts. Questions about the same contract are related
# (one hard-to-read contract makes several answers wrong), so treating them as independent
# makes intervals too narrow. The cluster bootstrap resamples whole CONTRACTS instead.
def _by_doc(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["doc_id"], []).append(r)
    return groups


def cluster_bootstrap_ci(rows: list[dict], fn, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for fn(rows), resampling contracts (with all their questions) with replacement."""
    groups = list(_by_doc(rows).values())
    if not groups:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for _ in groups for r in groups[rnd.randrange(len(groups))]]
        vals.append(fn(sample))
    vals.sort()
    return (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1])


def accuracy(rows: list[dict]) -> float:
    return sum(r["pred"] == r["gold"] for r in rows) / len(rows) if rows else 0.0


def macro_f1(rows: list[dict]) -> float:
    return sum(prf(rows, lab)["f1"] for lab in LABELS) / len(LABELS)


def paired_cluster_diff(rows_a: list[dict], rows_b: list[dict], fn=accuracy,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference fn(B) - fn(A) on the questions both runs answered, with a 95% interval from
    resampling contracts (keeps the pairing AND the within-contract correlation).
    p is the two-sided bootstrap p-value: how often the difference crosses zero."""
    key = lambda r: (r["doc_id"], r["hypothesis_key"])  # noqa: E731
    b_by = {key(r): r for r in rows_b}
    pairs = [(r, b_by[key(r)]) for r in rows_a if key(r) in b_by]
    groups: dict[str, list] = {}
    for pa, pb in pairs:
        groups.setdefault(pa["doc_id"], []).append((pa, pb))
    ids = list(groups.values())
    if not ids:
        return {"diff": float("nan"), "ci": (float("nan"), float("nan")), "p": float("nan"), "n": 0}
    diff = fn([p[1] for p in pairs]) - fn([p[0] for p in pairs])
    rnd = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [p for _ in ids for p in ids[rnd.randrange(len(ids))]]
        vals.append(fn([p[1] for p in s]) - fn([p[0] for p in s]))
    vals.sort()
    p = 2 * min(sum(v <= 0 for v in vals), sum(v >= 0 for v in vals)) / n_boot
    return {"diff": diff, "ci": (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]),
            "p": min(1.0, p), "n": len(pairs), "n_docs": len(ids)}


def doc_level(rows: list[dict]) -> list[dict]:
    """Accuracy per contract, worst first."""
    out = [{"doc_id": d, "n": len(rs), "accuracy": accuracy(rs)} for d, rs in _by_doc(rows).items()]
    return sorted(out, key=lambda x: (x["accuracy"], -x["n"]))


# ------------------------------------------------------------------ latency and cost
def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def latency_cost(rows: list[dict], price_in: float = 0.0, price_out: float = 0.0) -> dict:
    """Seconds per question (mean / median / 90th percentile), tokens per question (answer +
    judge calls; thinking tokens count as output) and the cost of 1,000 questions at the given
    prices in USD per 1M tokens (0 for a local Ollama model)."""
    secs = [r["seconds"] for r in rows if "seconds" in r]
    have = [r for r in rows if "usage" in r]
    tin = [r["usage"].get("prompt_tokens", 0) + r.get("judge_usage", {}).get("prompt_tokens", 0) for r in have]
    tout = [r["usage"].get("completion_tokens", 0) + r.get("judge_usage", {}).get("completion_tokens", 0) for r in have]
    calls = [r["usage"].get("calls", 0) + r.get("judge_usage", {}).get("calls", 0) for r in have]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    out = {"seconds_mean": mean(secs), "seconds_p50": _quantile(secs, 0.5), "seconds_p90": _quantile(secs, 0.9),
           "tokens_in": mean(tin), "tokens_out": mean(tout), "calls": mean(calls),
           "empty": sum(r.get("empty", False) or "returned an empty answer" in (r.get("answer") or "") for r in rows),
           # questions where at least one answer call stopped at max_tokens (thinking used them all)
           "hit_token_limit": sum((r.get("usage", {}).get("length_hits", 0) > 0) or r.get("finish_reason") == "length"
                                  for r in rows)}
    if have and (price_in or price_out):
        out["usd_per_1000"] = 1000 * ((out["tokens_in"] or 0) * price_in + (out["tokens_out"] or 0) * price_out) / 1e6
    return out


# ------------------------------------------------------------------ citation failures
def evidence_overlap(chunk_text: str, evidence: str, n: int = 6) -> float:
    """Share of the evidence's word 6-grams found in the chunk (0..1). evidence_covered()
    needs >= 0.5; a value between 0.2 and 0.5 means the citation holds PART of the evidence."""
    c, e = _norm(chunk_text), _norm(evidence)
    if not e:
        return 0.0
    if e in c:
        return 1.0
    words = e.split()
    if len(words) <= n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return sum(g in c for g in grams) / len(grams)


def citation_failure_kind(row: dict, evidence_in_top_k: bool | None) -> str:
    """Why a correct yes/no answer is not supported by its citations:
        no_citation           the answer cites nothing
        evidence_not_retrieved the evidence was not among the clauses the model saw
                               (correct label from other clauses, or a lucky guess)
        partial_evidence      a cited clause holds part of the evidence (20–50% of it):
                               often the evidence is split across two chunks
        cited_other_clause    the evidence WAS retrieved, but the model cited a different clause
                               (a clause that may also support the answer: check by hand)"""
    cites, ev = row.get("citations") or [], row.get("evidence") or []
    if not cites:
        return "no_citation"
    retrieved = row.get("retrieved")
    if retrieved is not None:
        in_top_k = any(evidence_covered(t, e) for t in retrieved for e in ev)
    else:
        in_top_k = bool(evidence_in_top_k)
    if not in_top_k:
        return "evidence_not_retrieved"
    if max((evidence_overlap(c, e) for c in cites for e in ev), default=0.0) >= 0.2:
        return "partial_evidence"
    return "cited_other_clause"