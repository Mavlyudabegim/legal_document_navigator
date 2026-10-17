"""
contractnli.py  —  evaluate the app on ContractNLI: hundreds of real NDAs with expert labels

ContractNLI (Koreeda & Manning, 2021; CC BY 4.0) = ~600 real NDAs. For every NDA, lawyers
checked the same 17 statements ("hypotheses"), e.g. "Receiving Party may share some
Confidential Information with some of Receiving Party's employees", and labelled each:
    Entailment     the NDA says it is TRUE        -> the app should answer "yes"
    Contradiction  the NDA says it is FALSE       -> the app should answer "no"
    NotMentioned   the NDA doesn't say            -> the app should say "not found"
For TRUE/FALSE they also marked the exact sentences that prove it ("evidence spans").
That gives us, for free, a ground truth for search (did we retrieve the evidence?), for
refusals (NotMentioned) and for the answer itself (yes / no).

SETUP (once)
    1. Download https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip
    2. Unzip it into  data/contract-nli/   (you should have data/contract-nli/dev.json, test.json)

USE
    # 1. ingest 20 NDAs from the dev split and write the question file
    python -m eval.contractnli prepare --split dev --n-docs 20

    # 2. retrieval only (fast, free): compare setups
    python -m eval.contractnli run --configs dense,hybrid,hybrid+rerank

    # 3. full pipeline on a FIXED, stratified sample (same questions for every setup)
    python -m eval.contractnli prepare --split dev --n-docs 61          # all 61 dev NDAs
    python -m eval.contractnli run --configs hybrid --llm --sample 300 --tag my-setup
    #    Stopped by a rate limit? Progress is saved: run the SAME command later to continue.

    # 4. reports
    python -m eval.contractnli analyze                 # per-label F1, citations, contradiction errors
    python -m eval.contractnli compare --a OLD.json --b NEW.json   # paired test on the same questions

Tune on the "dev" split. When you're done, run once on "test" for the numbers in your README.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.generation.llm import LLM
from app.generation.qa import answer_question
from app.ingestion.chunker import chunk_elements
from app.ingestion.embedder import Embedder
from app.ingestion.store import VectorStore
from app.ingestion.text_parser import parse_text
from app.retrieval.retriever import Retriever
from eval.analysis import build_report
from eval.metrics import LABELS, NAMES, citation_summary, class_report, evidence_covered, mcnemar

ROOT = Path(__file__).parent
DEFAULT_DATA = Path("data/contract-nli")
QUESTIONS_FILE = ROOT / "contractnli_questions.json"
DOC_PREFIX = "cnli-"

# One question per hypothesis, phrased as "Does the agreement say / allow / prohibit ...?".
# Version 2 of the questions. Version 1 asked "Does the receiving party have to notify ...?".
# For a contract that is SILENT, the honest plain-English answer to that is "No, it doesn't
# have to", but the lawyers' label is NotMentioned. Asking what the AGREEMENT says removes
# that mismatch: silent -> "The document doesn't say this", explicit opposite -> "No."
QUESTIONS_VERSION = 2
QUESTIONS = {
    "nda-1":  "Does the agreement say that the disclosing party must explicitly mark or identify information as confidential for it to be protected?",
    "nda-2":  "Does the agreement say that confidential information includes only technical information?",
    "nda-3":  "Does the agreement say that information shared verbally (orally) can be confidential information?",
    "nda-4":  "Does the agreement say that the receiving party may use confidential information only for the purpose of the agreement?",
    "nda-5":  "Does the agreement allow the receiving party to share confidential information with its own employees?",
    "nda-7":  "Does the agreement allow the receiving party to share confidential information with third parties such as consultants, agents or professional advisors?",
    "nda-8":  "Does the agreement say that the receiving party must notify the disclosing party if it is legally required to disclose confidential information?",
    "nda-10": "Does the agreement say that the receiving party must keep the agreement itself, or the fact that the parties negotiated it, confidential?",
    "nda-11": "Does the agreement prohibit reverse engineering of objects that contain confidential information?",
    "nda-12": "Does the agreement allow the receiving party to independently develop information similar to the confidential information?",
    "nda-13": "Does the agreement allow the receiving party to acquire similar information from a third party?",
    "nda-15": "Does the agreement say that the receiving party gets no license or rights to the confidential information?",
    "nda-16": "Does the agreement say that the receiving party must return or destroy confidential information when the agreement ends?",
    "nda-17": "Does the agreement allow the receiving party to make copies of confidential information in some cases?",
    "nda-18": "Does the agreement prohibit the receiving party from soliciting or hiring the disclosing party's employees or representatives?",
    "nda-19": "Does the agreement say that some obligations continue after the agreement ends?",
    "nda-20": "Does the agreement allow the receiving party to keep some confidential information even after it returns or destroys it?",
}
GOLD = {"Entailment": "TRUE", "Contradiction": "FALSE", "NotMentioned": "NOT_MENTIONED"}


# evidence matching (evidence_covered) lives in eval/metrics.py


# ------------------------------------------------------------------ prepare
def prepare(a) -> None:
    path = Path(a.data) / f"{a.split}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Download contract-nli.zip and unzip it into {a.data}/")
    data = json.loads(path.read_text())
    labels = data["labels"]
    docs = data["documents"]
    random.Random(a.seed).shuffle(docs)
    docs = docs[: a.n_docs]

    embedder = Embedder(model=a.emb_model, backend=a.emb_backend)
    store = VectorStore(a.db, embedding_model=f"{a.emb_backend}:{a.emb_model}")
    questions, n_chunks = [], []

    for d in docs:
        doc_id = f"{DOC_PREFIX}{d['id']}"
        elements = parse_text(d["text"], title=Path(d["file_name"]).stem)
        chunks = chunk_elements(elements, doc_id=doc_id, doc_title=Path(d["file_name"]).stem,
                                count_tokens=embedder.count_tokens, max_tokens=a.max_tokens)
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        store.delete_document(doc_id)
        store.add(chunks, vectors)
        n_chunks.append(len(chunks))

        ann = d["annotation_sets"][0]["annotations"]
        for key, label in labels.items():
            if key not in ann:
                continue
            choice = ann[key]["choice"]
            evidence = [d["text"][d["spans"][i][0]:d["spans"][i][1]] for i in ann[key]["spans"]]
            questions.append({
                "doc_id": doc_id, "hypothesis_key": key,
                "short": label.get("short_description", key),
                "hypothesis": label["hypothesis"],
                "question": QUESTIONS.get(key, f"According to the agreement, is this true: {label['hypothesis']}"),
                "gold": GOLD[choice], "answerable": choice != "NotMentioned",
                "evidence": evidence,
            })
        print(f"  {doc_id:<16} {len(chunks):>3} chunks   {Path(d['file_name']).name[:50]}")

    QUESTIONS_FILE.write_text(json.dumps({"split": a.split, "seed": a.seed, "max_tokens": a.max_tokens,
                                          "questions_version": QUESTIONS_VERSION,
                                          "questions": questions}, indent=1))
    g = Counter(q["gold"] for q in questions)
    print(f"\n{len(docs)} NDAs ingested, {statistics.mean(n_chunks):.0f} chunks per NDA on average")
    print(f"{len(questions)} questions  ·  TRUE {g['TRUE']} · FALSE {g['FALSE']} · "
          f"NOT_MENTIONED {g['NOT_MENTIONED']}")
    print(f"Saved {QUESTIONS_FILE}")


# ------------------------------------------------------------------ run: retrieval
def eval_retrieval(name, retriever, questions):
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], q["doc_id"])
        ms = 1000 * (time.perf_counter() - t0)
        rank = next((r for r, h in enumerate(hits, 1)
                     if any(evidence_covered(h.text, e) for e in q["evidence"])), None)
        covered = sum(any(evidence_covered(h.text, e) for h in hits) for e in q["evidence"])
        rows.append({"doc_id": q["doc_id"], "key": q["hypothesis_key"], "short": q["short"],
                     "rank": rank, "ms": ms,
                     "evidence_recall": covered / len(q["evidence"]) if q["evidence"] else 0.0})
    n = len(rows)
    return {
        "config": name, "n": n,
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "hit@3": sum(bool(r["rank"]) and r["rank"] <= 3 for r in rows) / n,
        "hit@k": sum(r["rank"] is not None for r in rows) / n,
        "mrr": statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in rows),
        "evidence_recall": statistics.mean(r["evidence_recall"] for r in rows),
        "ms": statistics.mean(r["ms"] for r in rows[1:] or rows),
    }, rows


# ------------------------------------------------------------------ run: end-to-end
JUDGE_SYSTEM = "You classify answers about a contract. Reply with exactly one word."
JUDGE_USER = """Statement: {hyp}
An assistant was asked a question about this statement and answered:
\"\"\"{ans}\"\"\"
According to the assistant's answer, is the statement TRUE, FALSE, or NOT_MENTIONED in the contract?
- TRUE: the answer says the contract states or clearly allows/requires it.
- FALSE: the answer says the contract explicitly contradicts it (forbids, excludes, or says the opposite).
- NOT_MENTIONED: the answer says the contract does not address it, is silent, or only covers it indirectly.
Reply with exactly one word: TRUE, FALSE or NOT_MENTIONED."""


def label_from_opening(answer: str) -> str | None:
    """The prompt makes every answer start with "Yes.", "No.", "Partly." or
    "The document doesn't say this directly." Reading that opening is exact and free,
    so the LLM judge is only needed for "Partly." (and for answers in another format)."""
    t = re.sub(r"^[\s*_#>\-]+", "", answer or "").lower().replace("’", "'")
    if not t or t.startswith(("i couldn't find", "i could not find", "the document doesn't say",
                              "the document does not say")):
        return "NOT_MENTIONED"
    if re.match(r"yes\b", t):
        return "TRUE"
    if re.match(r"no\b(?!t)", t):
        return "FALSE"
    return None                       # "Partly." or free text -> ask the judge


PARTLY_RULES = {"true": "TRUE", "not-mentioned": "NOT_MENTIONED"}


def grade(answer: str, hypothesis: str, judge, mode: str, partly: str = "judge") -> tuple[str, str]:
    """Returns (label, how). mode 'auto' = opening phrase first, judge as fallback.
    partly: what a "Partly." answer counts as. 'judge' (default) lets the LLM judge decide;
    'true' / 'not-mentioned' use a fixed rule (see the contradiction analysis: the judge
    often turns "Partly." into FALSE, which creates false contradictions)."""
    if mode == "auto":
        lab = label_from_opening(answer)
        if lab:
            return lab, "opening"
        if partly in PARTLY_RULES and re.match(r"^[\s*_#>\-]*partly\b", answer or "", re.IGNORECASE):
            return PARTLY_RULES[partly], "partly-rule"
    v = judge.complete(JUDGE_SYSTEM, JUDGE_USER.format(hyp=hypothesis, ans=answer))
    v = v.strip().upper().replace(" ", "_")
    return next((x for x in ("NOT_MENTIONED", "FALSE", "TRUE") if x in v), "NOT_MENTIONED"), "judge"


class DailyLimitReached(Exception):
    """The API wants us to wait longer than --max-wait (usually the per-DAY quota)."""


def _wait_seconds(msg: str) -> float | None:
    """Groq says e.g. 'Please try again in 7.5s' or 'try again in 2m30.1s' or '1h2m'."""
    m = re.search(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", msg)
    if not m or not any(m.groups()):
        return None
    h, mnt, sec = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mnt * 60 + sec


class RetryLLM:
    """Wraps the LLM for free-tier APIs:
      * pauses between calls (--pause) so we stay under the per-minute limits
      * on 'too many requests' (HTTP 429) waits exactly as long as the API asks, then retries
      * if the API asks for a very long wait (daily quota used up), stops cleanly instead of
        hanging; progress is saved, so running the same command later continues."""
    def __init__(self, llm: LLM, pause: float, max_wait: float):
        self.llm, self.pause, self.max_wait = llm, pause, max_wait
        self.provider, self.model = llm.provider, llm.model

    def usage(self) -> dict:
        return self.llm.usage() if hasattr(self.llm, "usage") else {}

    @property
    def last_finish_reason(self):
        return getattr(self.llm, "last_finish_reason", None)

    def complete(self, system: str, user: str) -> str:
        for attempt in range(8):
            try:
                out = self.llm.complete(system, user)
                time.sleep(self.pause)
                return out
            except Exception as e:
                msg = str(e)
                if "429" not in msg and "rate" not in msg.lower():
                    raise
                wait = _wait_seconds(msg) or min(60, 2 ** attempt * 2)
                if wait > self.max_wait:
                    raise DailyLimitReached(f"{self.model}: API asks to wait {wait / 60:.0f} min") from e
                print(f"      rate limit on {self.model}, waiting {wait:.0f}s …")
                time.sleep(wait + 1)
        raise DailyLimitReached("still rate-limited after 8 retries")


def _cache_key(q: dict, setup: str) -> str:
    return f"{setup}|{q['doc_id']}|{q['hypothesis_key']}"


def _usage_diff(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0)
            for k in ("calls", "prompt_tokens", "completion_tokens", "length_hits")}


def eval_end_to_end(retriever, llm, judge, questions, setup: str, cache_path: Path,
                    grading: str = "auto", whole_doc_tokens: int = 0, votes: int = 1,
                    partly: str = "judge", followup: str = "none", followup_k: int = 15):
    """Answers are saved to cache_path one by one, so a rate-limit stop loses nothing:
    run the same command again and it continues where it stopped."""
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    todo = [q for q in questions if _cache_key(q, setup) not in done]
    if len(todo) < len(questions):
        print(f"  {len(questions) - len(todo)} questions already answered earlier, {len(todo)} to go")

    stopped = None
    with cache_path.open("a") as f:
        for i, q in enumerate(todo, 1):
            t0 = time.perf_counter()
            u_llm, u_judge = llm.usage(), judge.usage()
            try:
                tries = []
                for _ in range(max(1, votes)):
                    kw = {"followup": followup, "followup_k": followup_k} if followup != "none" else {}
                    a = answer_question(q["question"], retriever, llm, q["doc_id"],
                                        whole_doc_tokens=whole_doc_tokens, **kw)
                    if not a.found:
                        lab, how_ = "NOT_MENTIONED", "not-found"
                    else:
                        lab, how_ = grade(a.text, q["hypothesis"], judge, grading, partly)
                    tries.append((lab, how_, a))
                # majority vote (with 1 vote this is just the single answer)
                pred = Counter(t[0] for t in tries).most_common(1)[0][0]
                _, how, a = next(t for t in tries if t[0] == pred)
            except DailyLimitReached as e:
                stopped = str(e)
                break
            row = {"key": _cache_key(q, setup),
                   **{k: q[k] for k in ("doc_id", "hypothesis_key", "short", "question", "gold")},
                   "pred": pred, "graded_by": how, "answer": a.text,
                   "seconds": time.perf_counter() - t0,
                   "cited_right": any(evidence_covered(c.text, e) for c in a.citations for e in q["evidence"]),
                   # saved so citation quality can be measured later (eval/metrics.py)
                   "citations": [c.text for c in a.citations],
                   "retrieved": [s.text for s in a.retrieved],
                   "evidence": q["evidence"],
                   # latency / cost / failure diagnostics
                   "usage": _usage_diff(u_llm, llm.usage()),
                   "judge_usage": _usage_diff(u_judge, judge.usage()),
                   "finish_reason": llm.last_finish_reason,
                   "empty": "returned an empty answer" in (a.text or ""),
                   # follow-up search for "not mentioned" answers (qa.py)
                   "followup": getattr(a, "followup", None),
                   "first_label": label_from_opening(a.first_text) if getattr(a, "first_text", None) else None,
                   "not_found_scope": getattr(a, "not_found_scope", None)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[row["key"]] = row
            mark = "✓" if pred == q["gold"] else "✗"
            print(f"  {i:>4}/{len(todo)} {mark} gold={q['gold']:<13} pred={pred:<13} {q['short'][:40]}")

    rows = [done[_cache_key(q, setup)] for q in questions if _cache_key(q, setup) in done]
    if not rows:
        return None, rows, stopped
    acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
    ans = [r for r in rows if r["gold"] != "NOT_MENTIONED"]
    una = [r for r in rows if r["gold"] == "NOT_MENTIONED"]
    summary = {
        "n": len(rows),
        "label_accuracy": acc,
        "answered": (sum(r["pred"] != "NOT_MENTIONED" for r in ans) / len(ans)) if ans else None,
        "refused_right": (sum(r["pred"] == "NOT_MENTIONED" for r in una) / len(una)) if una else None,
        "cited_right": (sum(r["cited_right"] for r in ans) / len(ans)) if ans else None,
        "seconds": statistics.mean(r["seconds"] for r in rows),
        "confusion": Counter(f"{r['gold']}->{r['pred']}" for r in rows),
    }
    return summary, rows, stopped


def paper_metrics(rows: list[dict]) -> dict:
    """The numbers the ContractNLI paper reports (Table 4): accuracy, F1 for Contradiction
    (FALSE) and F1 for Entailment (TRUE), plus a 95% confidence interval for accuracy
    (Wilson interval: with few questions the true accuracy could be anywhere in this range)."""
    import math
    n = len(rows)
    acc = sum(r["pred"] == r["gold"] for r in rows) / n
    z = 1.96
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)

    def f1(label):
        tp = sum(r["pred"] == label and r["gold"] == label for r in rows)
        p = tp / max(1, sum(r["pred"] == label for r in rows))
        rc = tp / max(1, sum(r["gold"] == label for r in rows))
        return 2 * p * rc / (p + rc) if p + rc else 0.0

    return {"accuracy": acc, "ci_low": centre - half, "ci_high": centre + half,
            "f1_contradiction": f1("FALSE"), "f1_entailment": f1("TRUE"), "n": n}


def print_paper_comparison(rows: list[dict], weights: dict | None = None) -> None:
    m = paper_metrics(rows)
    if weights:              # stratified sample: compare at the real label mix, like the paper
        from eval.metrics import weighted_accuracy, weighted_prf
        w = weighted_prf(rows, weights)
        m.update(accuracy=weighted_accuracy(rows, weights), f1_contradiction=w["FALSE"]["f1"],
                 f1_entailment=w["TRUE"]["f1"])
        print("\n  (stratified sample: numbers below are re-weighted to the real label mix)")
    print(f"\n  Compared with the ContractNLI paper (Span NLI BERT-large, fine-tuned, full test set):")
    print(f"  {'':<22}{'yours':>10}{'paper':>10}")
    print(f"  {'Accuracy':<22}{m['accuracy']:>10.3f}{0.875:>10.3f}   "
          f"(95% range for yours: {m['ci_low']:.3f}–{m['ci_high']:.3f}, n={m['n']})")
    print(f"  {'F1 Contradiction':<22}{m['f1_contradiction']:>10.3f}{0.357:>10.3f}")
    print(f"  {'F1 Entailment':<22}{m['f1_entailment']:>10.3f}{0.834:>10.3f}")


def pct(x) -> str:
    return "  n/a " if x is None else f"{100 * x:5.1f}%"


# ------------------------------------------------------------------ fixed, stratified sample
SAMPLES_DIR = ROOT / "samples"


def label_shares(questions: list[dict]) -> dict[str, float]:
    c = Counter(q["gold"] for q in questions)
    return {lab: c[lab] / len(questions) for lab in LABELS}


def stratified_sample(questions: list[dict], n: int, seed: int, strategy: str) -> list[dict]:
    """proportional : the same label mix as the full question set
    boost-false     : ALL Contradiction questions (they are only ~10% of ContractNLI, so a
                      proportional sample has too few to measure their F1), the rest split
                      proportionally between Entailment and NotMentioned. The headline accuracy
                      is then re-weighted to the real mix (metrics.weighted_accuracy)."""
    rnd = random.Random(seed)
    groups = {lab: sorted((q for q in questions if q["gold"] == lab),
                          key=lambda q: (q["doc_id"], q["hypothesis_key"])) for lab in LABELS}
    for g in groups.values():
        rnd.shuffle(g)
    n = min(n, len(questions))
    if strategy == "boost-false":
        take = {"FALSE": min(len(groups["FALSE"]), n // 2)}
        rest = n - take["FALSE"]
        t, nm = len(groups["TRUE"]), len(groups["NOT_MENTIONED"])
        take["TRUE"] = min(t, round(rest * t / (t + nm)))
        take["NOT_MENTIONED"] = min(nm, rest - take["TRUE"])
    else:
        shares = label_shares(questions)
        take = {lab: round(n * shares[lab]) for lab in LABELS}
    return [q for lab in LABELS for q in groups[lab][: take[lab]]]


def fixed_sample(questions: list[dict], qfile: dict, n: int, seed: int, strategy: str) -> list[dict]:
    """The sample is saved to eval/samples/ the first time, then always re-used, so every
    setup you test answers exactly the same questions (a fair, paired comparison)."""
    n_docs = len({q["doc_id"] for q in questions})
    path = SAMPLES_DIR / f"{qfile.get('split', 'dev')}-{n_docs}docs-{n}-{strategy}-seed{seed}.json"
    by_key = {(q["doc_id"], q["hypothesis_key"]): q for q in questions}
    if path.exists():
        keys = [tuple(k) for k in json.loads(path.read_text())["keys"]]
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(f"{path.name} lists {len(missing)} questions that are not in the current "
                             f"question file. Run `prepare` with the same --split/--n-docs/--seed again.")
        print(f"Using the fixed sample {path.relative_to(ROOT.parent)} ({len(keys)} questions)")
        return [by_key[k] for k in keys]
    sample = stratified_sample(questions, n, seed, strategy)
    SAMPLES_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"split": qfile.get("split"), "n_docs": n_docs, "strategy": strategy,
                                "seed": seed, "label_shares_full_set": label_shares(questions),
                                "keys": [[q["doc_id"], q["hypothesis_key"]] for q in sample]}, indent=1))
    print(f"Created the fixed sample {path.relative_to(ROOT.parent)} ({len(sample)} questions) — "
          f"commit it, so later runs use the same questions")
    return sample


def print_class_table(rows: list[dict], weights: dict | None = None) -> None:
    rep = class_report(rows)
    lo, hi = rep["accuracy_ci"]
    print(f"\n  Per label (n={rep['n']}):  accuracy {pct(rep['accuracy'])} (95% {100*lo:.0f}–{100*hi:.0f}%)"
          f"  ·  macro-F1 {rep['macro_f1']:.3f}")
    if weights:
        from eval.metrics import weighted_accuracy
        print(f"  Accuracy re-weighted to the real label mix: {pct(weighted_accuracy(rows, weights))}")
    print(f"  {'label':<15}{'precision':>10}{'recall':>9}{'F1':>8}{'support':>9}")
    for lab in LABELS:
        m = rep["per_class"][lab]
        print(f"  {NAMES[lab]:<15}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}{m['support']:>9}")
    if weights:
        from eval.metrics import weighted_prf
        w = weighted_prf(rows, weights)
        print("  Re-weighted to the real label mix (what you'd see on real contracts):")
        for lab in LABELS:
            print(f"  {NAMES[lab]:<15}{w[lab]['precision']:>10.3f}{w[lab]['recall']:>9.3f}{w[lab]['f1']:>8.3f}")
    cs = citation_summary(rows)
    if cs and cs.get("answers_with_citations"):
        print(f"  Citations: supported {pct(cs['supported'])} · precision {pct(cs['precision'])} · "
              f"recall {pct(cs['recall'])} · right label but unsupported {pct(cs['right_label_unsupported'])} · "
              f"strict accuracy {pct(cs['strict_accuracy'])}")


def run(a) -> None:
    if not QUESTIONS_FILE.exists():
        raise SystemExit("Run `python -m eval.contractnli prepare` first.")
    qfile = json.loads(QUESTIONS_FILE.read_text())
    questions = qfile["questions"]
    embedder = Embedder(model=a.emb_model, backend=a.emb_backend)
    store = VectorStore(a.db, embedding_model=f"{a.emb_backend}:{a.emb_model}")
    n_docs = len({q["doc_id"] for q in questions})
    print(f"ContractNLI ({qfile.get('split')} split, chunks ≤ {qfile.get('max_tokens', 350)} tokens, "
          f"questions v{qfile.get('questions_version', 1)}): "
          f"{n_docs} NDAs · {len(questions)} questions "
          f"({sum(q['answerable'] for q in questions)} with evidence)\n")

    results = {"time": datetime.now().isoformat(timespec="seconds"), "retrieval": [], "rows": {}}
    reranker, retrievers = None, {}
    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        if name.endswith("+rerank") and reranker is None:
            from app.retrieval.reranker import Reranker
            print(f"Loading reranker {a.reranker_model} …")
            reranker = Reranker(a.reranker_model)
        r = Retriever(embedder, store, k=a.k, min_score=a.min_score, mode=name.split("+")[0],
                      reranker=reranker if name.endswith("+rerank") else None)
        retrievers[name] = r
        s, rows = eval_retrieval(name, r, questions)
        results["retrieval"].append(s)
        results["rows"][name] = rows

    print(f"RETRIEVAL  (questions with evidence; a hit = a retrieved chunk contains the evidence)")
    print(f"{'setup':<15}{'Hit@1':>8}{'Hit@3':>8}{'Hit@' + str(a.k):>8}{'MRR':>7}{'Evid.recall':>13}{'ms':>7}")
    for s in results["retrieval"]:
        print(f"{s['config']:<15}{pct(s['hit@1']):>8}{pct(s['hit@3']):>8}{pct(s['hit@k']):>8}"
              f"{s['mrr']:7.2f}{pct(s['evidence_recall']):>13}{s['ms']:7.0f}")

    # which question types are hard? (last config)
    last = results["retrieval"][-1]["config"]
    by_key = defaultdict(list)
    for r in results["rows"][last]:
        by_key[r["short"]].append(r)
    print(f"\nHit@3 per question type in '{last}' (worst first)")
    for short, rs in sorted(by_key.items(), key=lambda kv: sum(bool(r["rank"]) and r["rank"] <= 3
                                                               for r in kv[1]) / len(kv[1])):
        h3 = sum(bool(r["rank"]) and r["rank"] <= 3 for r in rs) / len(rs)
        print(f"  {pct(h3)}  ({len(rs):>3})  {short}")

    if a.llm:
        weights = None
        if a.sample:
            qs = fixed_sample(questions, qfile, a.sample, a.sample_seed, a.stratify)
            if a.stratify == "boost-false":
                weights = label_shares(questions)
        else:                                               # old behaviour (--max-questions)
            qs = questions
            if a.max_questions and len(qs) > a.max_questions:
                qs = random.Random(0).sample(qs, a.max_questions)
        qs = sorted(qs, key=lambda q: q["doc_id"])          # one document at a time
        if a.limit and a.limit < len(qs):
            # an evenly spread subset of the fixed sample (every n-th question): same questions
            # every time, all contracts covered; `compare` then pairs runs on the common questions
            step = len(qs) / a.limit
            qs = [qs[int(i * step)] for i in range(a.limit)]
            print(f"  --limit {a.limit}: using every {step:.1f}-th question of the sample")
        # only pass reasoning_effort when asked, so older llm.py versions keep working
        # only pass extra settings when asked, so older llm.py versions keep working
        kw = {}
        if a.reasoning_effort:
            kw["reasoning_effort"] = a.reasoning_effort
        if a.votes > 1:
            kw["temperature"] = a.vote_temperature      # votes only differ if answers can vary
        if a.llm_max_tokens:
            kw["max_tokens"] = a.llm_max_tokens         # room for thinking + answer
        base = LLM(provider=a.provider, model=a.model, **kw)
        llm = RetryLLM(base, pause=a.pause, max_wait=a.max_wait)
        judge = RetryLLM(LLM(provider=a.provider, model=a.judge_model or base.model),
                         pause=a.pause, max_wait=a.max_wait)
        setup = (f"{last}|k{a.k}|{qfile.get('max_tokens', 350)}|{llm.model}|judge:{judge.model}"
                 f"|re:{a.reasoning_effort or 'default'}|grade:{a.grading}"
                 f"|q{qfile.get('questions_version', 1)}|wd{a.whole_doc_tokens}|v{a.votes}"
                 f"|mt{a.llm_max_tokens or 'default'}"
                 + (f"|partly:{a.partly}" if a.partly != "judge" else "")
                 + (f"|fu:{a.followup}{a.followup_k if a.followup == 'more' else ''}" if a.followup != "none" else "")
                 + f"|{a.tag}")
        cache = ROOT / "results" / "contractnli_answers.jsonl"
        cache.parent.mkdir(exist_ok=True)
        print(f"\nEND-TO-END with '{last}', k={a.k}, answers by {llm.model}, judged by {judge.model}"
              f"  ({len(qs)} questions)")
        s, rows, stopped = eval_end_to_end(retrievers[last], llm, judge, qs, setup, cache, a.grading,
                                           a.whole_doc_tokens, a.votes, a.partly, a.followup, a.followup_k)
        if stopped:
            print(f"\n  ⏸  Stopped: {stopped}.\n     Progress is saved. Run the same command later "
                  f"(or tomorrow) to continue. Results below are for the {len(rows)} questions done so far.")
        if s:
            results["end_to_end"] = {"config": setup, **s, "rows": rows, "label_weights": weights,
                                     "sample": {"n": a.sample, "strategy": a.stratify, "seed": a.sample_seed}
                                     if a.sample else None}
            print(f"\n  Results for {s['n']}/{len(qs)} questions")
            print(f"  Label accuracy  {pct(s['label_accuracy'])}  (TRUE / FALSE / NOT_MENTIONED matches the lawyers)")
            print(f"  Answered        {pct(s['answered'])}  (TRUE/FALSE questions not called 'not found')")
            print(f"  Refused right   {pct(s['refused_right'])}  (NOT_MENTIONED questions answered 'not found')")
            print(f"  Cited right     {pct(s['cited_right'])}  (a citation contains the evidence)")
            print(f"  Time            {s['seconds']:.1f}s per question")
            empty = sum("returned an empty answer" in (r.get("answer") or "") for r in rows)
            if empty:
                print(f"  ⚠ Empty answers {empty}  (the model used all its tokens on thinking and wrote "
                      f"nothing; these count as 'not found'. Raise --llm-max-tokens)")
            print("  Mistakes (gold->pred):",
                  ", ".join(f"{k}: {v}" for k, v in s["confusion"].most_common()
                            if k.split("->")[0] != k.split("->")[1]) or "none")
            print_class_table(rows, weights)
            print_paper_comparison(rows, weights)
            print("\n  Full report (per-label intervals, citations, contradiction errors):"
                  "  python -m eval.contractnli analyze")

    out = ROOT / "results" / f"contractnli-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nSaved {out}")


# ------------------------------------------------------------------ mistakes
def mistakes(a) -> None:
    """Print every wrong answer of the latest end-to-end run, grouped by mistake type,
    so you can see WHY it was wrong (bad search, bad answer, or a wrong grade by the judge)."""
    files = sorted((ROOT / "results").glob("contractnli-*.json"))
    files = [f for f in files if "end_to_end" in json.loads(f.read_text())]
    if not files:
        raise SystemExit("No end-to-end results yet. Run `run --llm` first.")
    data = json.loads((Path(a.file) if a.file else files[-1]).read_text())
    all_rows = data["end_to_end"]["rows"]
    rows = [r for r in all_rows if r["pred"] != r["gold"]]
    print(f"{len(rows)} mistakes out of {len(all_rows)} in {a.file or files[-1].name}\n")

    # Where does each mistake come from?  Use the search results saved in the same run.
    setup = str(data["end_to_end"].get("config", "")).split("|")[0]
    ranks = {(r["doc_id"], r["key"]): r["rank"] for r in data.get("rows", {}).get(setup, [])}
    if ranks:
        search_miss = [r for r in rows if r["gold"] != "NOT_MENTIONED"
                       and ranks.get((r["doc_id"], r["hypothesis_key"])) is None]
        read_err = [r for r in rows if r["gold"] != "NOT_MENTIONED"
                    and ranks.get((r["doc_id"], r["hypothesis_key"])) is not None]
        nm_err = [r for r in rows if r["gold"] == "NOT_MENTIONED"]
        print("WHERE THE MISTAKES COME FROM")
        print(f"  {len(search_miss):>3}  search: the evidence was NOT among the retrieved chunks"
              f"   -> chunking / embeddings / query rewriting")
        print(f"  {len(read_err):>3}  answer: evidence WAS retrieved, but the answer is wrong"
              f"      -> prompt / model")
        print(f"  {len(nm_err):>3}  answer: contract is silent, but the model said yes/no"
              f"      -> prompt / model")
        print()
    for kind in sorted({f"{r['gold']}->{r['pred']}" for r in rows}):
        group = [r for r in rows if f"{r['gold']}->{r['pred']}" == kind]
        print(f"=== {kind}  ({len(group)})")
        for r in group[: a.limit]:
            print(f"  [{r['doc_id']} · {r['short']}]\n  Q: {r['question']}")
            print("  A: " + r["answer"].replace("\n", " ")[:400] + "\n")


# ------------------------------------------------------------------ regrade
def regrade(a) -> None:
    """Grade the SAVED answers of an earlier run again with the current judge prompt.
    Costs only the small judge calls (no search, no answer generation), so you get a fair
    'before' number for the new judge without spending your rate limit on a full rerun."""
    files = [f for f in sorted((ROOT / "results").glob("contractnli-*.json"))
             if "end_to_end" in json.loads(f.read_text())]
    if not files:
        raise SystemExit("No end-to-end results yet. Run `run --llm` first.")
    path = Path(a.file) if a.file else files[-1]
    data = json.loads(path.read_text())
    hyp = {(q["doc_id"], q["hypothesis_key"]): q["hypothesis"]
           for q in json.loads(QUESTIONS_FILE.read_text())["questions"]}
    judge = RetryLLM(LLM(provider=a.provider, model=a.judge_model), pause=a.pause, max_wait=a.max_wait)
    rows = data["end_to_end"]["rows"]
    print(f"Re-grading {len(rows)} saved answers from {path.name} with judge {judge.model}\n")
    changed = 0
    for r in rows:
        old = r["pred"]
        answer = (r.get("answer") or "").strip()
        new, how = grade(answer, hyp[(r["doc_id"], r["hypothesis_key"])], judge, a.grading, a.partly)
        r["graded_by"] = how
        if new != old:
            changed += 1
            print(f"  {r['short'][:38]:<38} gold={r['gold']:<13} {old} -> {new}")
        r["pred"] = new
    acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
    una = [r for r in rows if r["gold"] == "NOT_MENTIONED"]
    ans = [r for r in rows if r["gold"] != "NOT_MENTIONED"]
    conf = Counter(f"{r['gold']}->{r['pred']}" for r in rows if r["gold"] != r["pred"])
    by = Counter(r.get("graded_by") for r in rows)
    print(f"\n  {changed} grades changed  ·  graded by opening phrase: {by.get('opening', 0)}, "
          f"by LLM judge: {by.get('judge', 0)}")
    print(f"  Label accuracy  {pct(data['end_to_end']['label_accuracy'])}  ->  {pct(acc)}")
    print(f"  Answered        {pct(sum(r['pred'] != 'NOT_MENTIONED' for r in ans) / len(ans) if ans else None)}")
    print(f"  Refused right   {pct(sum(r['pred'] == 'NOT_MENTIONED' for r in una) / len(una) if una else None)}")
    print("  Mistakes (gold->pred):", ", ".join(f"{k}: {v}" for k, v in conf.most_common()) or "none")
    print_class_table(rows, data["end_to_end"].get("label_weights"))
    print_paper_comparison(rows, data["end_to_end"].get("label_weights"))
    data["end_to_end"].update(label_accuracy=acc, rows=rows, regraded_with=judge.model)
    out = path.with_name(path.stem + ("-regraded.json" if a.partly == "judge" else f"-partly-{a.partly}.json"))
    out.write_text(json.dumps(data, indent=1, default=str))
    print(f"\nSaved {out}")


# ------------------------------------------------------------------ analyze / compare
def _latest_e2e() -> Path:
    files = [f for f in sorted((ROOT / "results").glob("contractnli-*.json"))
             if "end_to_end" in json.loads(f.read_text())]
    if not files:
        raise SystemExit("No end-to-end results yet. Run `run --llm` first.")
    return files[-1]


def _load_rows(path: Path) -> tuple[dict, list[dict], dict]:
    """Rows of a saved run, with evidence/hypothesis added from the question file when the run
    is older and didn't store them. Also returns retrieval ranks for 'search vs reading' errors."""
    data = json.loads(path.read_text())
    rows = data["end_to_end"]["rows"]
    if QUESTIONS_FILE.exists():
        qs = {(q["doc_id"], q["hypothesis_key"]): q for q in json.loads(QUESTIONS_FILE.read_text())["questions"]}
        for r in rows:
            q = qs.get((r["doc_id"], r["hypothesis_key"]))
            if q and "evidence" not in r:
                r["evidence"] = q["evidence"]
    # Runs made before qa.py understood 【S1】-style citations saved those answers with no
    # citations. The answer text and the retrieved clauses were saved, so rebuild them here.
    from app.generation.qa import cited_labels
    for r in rows:
        if not r.get("citations") and r.get("retrieved") and r.get("answer"):
            idx = [int(lab[1:]) - 1 for lab in cited_labels(r["answer"])]
            rebuilt = [r["retrieved"][i] for i in idx if 0 <= i < len(r["retrieved"])]
            if rebuilt:
                r["citations"], r["citations_rebuilt"] = rebuilt, True
                r["cited_right"] = any(evidence_covered(c, e) for c in rebuilt for e in r.get("evidence", []))
    setup = str(data["end_to_end"].get("config", "")).split("|")[0]
    ranks = {(r["doc_id"], r["key"]): r["rank"] for r in data.get("rows", {}).get(setup, [])}
    return data, rows, ranks


def analyze(a) -> None:
    path = Path(a.file) if a.file else _latest_e2e()
    data, rows, ranks = _load_rows(path)
    e2e = data["end_to_end"]
    md = build_report(rows, title=f"ContractNLI evaluation — {path.name}", setup=str(e2e.get("config", "")),
                      ranks=ranks, weights=e2e.get("label_weights"), price_in=a.price_in, price_out=a.price_out)
    out = Path(a.out) if a.out else path.with_name(path.stem + "-report.md")
    out.write_text(md)
    print(md)
    print(f"\nSaved {out}")


def compare(a) -> None:
    """Controlled experiments: several runs on the SAME questions, side by side.
    The first run is the baseline; each other run gets the difference in accuracy and macro-F1
    with a 95% interval that resamples whole contracts (questions from one contract are related),
    plus McNemar's test on the individual questions."""
    from eval.metrics import (accuracy, cluster_bootstrap_ci, latency_cost, macro_f1,
                              paired_cluster_diff, weighted_accuracy)
    paths = list(a.runs or []) or [a.a, a.b]
    if len(paths) < 2 or not all(paths):
        raise SystemExit("Give at least two results files: --runs BASE.json NEW.json [MORE.json ...]")
    runs = []
    for p in paths:
        data, rows, _ = _load_rows(Path(p))
        tag = str(data["end_to_end"].get("config", "")).split("|")[-1] or Path(p).stem
        if Path(p).stem.endswith(("-regraded", "-not-mentioned", "-true")):
            tag += " (" + Path(p).stem.split("-", 2)[-1] + ")"
        runs.append({"path": p, "tag": tag, "rows": rows, "weights": data["end_to_end"].get("label_weights")})
    key = lambda r: (r["doc_id"], r["hypothesis_key"])  # noqa: E731
    common = set.intersection(*[{key(r) for r in run["rows"]} for run in runs])
    for run in runs:
        run["rows"] = [r for r in run["rows"] if key(r) in common]
    n_docs = len({k[0] for k in common})
    lines = [f"# Controlled comparison — {len(common)} common questions from {n_docs} contracts", "",
             "| Run | Accuracy (95% CI, by contract) | Acc. real mix | Macro-F1 | F1 Entail. | F1 Contr. | F1 NotMent. "
             "| Strict acc. | Cited evidence | s/question | Tokens in/out | $ per 1k q | Empty |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        rows = run["rows"]
        rep = class_report(rows)
        lo, hi = cluster_bootstrap_ci(rows, accuracy)
        cs = citation_summary(rows)
        lc = latency_cost(rows, a.price_in, a.price_out)
        wacc = weighted_accuracy(rows, run["weights"]) if run["weights"] else None
        f = {lab: rep["per_class"][lab]["f1"] for lab in LABELS}
        tok = f"{lc['tokens_in']:.0f}/{lc['tokens_out']:.0f}" if lc["tokens_in"] is not None else "n/a"
        cost = f"{lc['usd_per_1000']:.2f}" if "usd_per_1000" in lc else "n/a"
        lines.append(f"| {run['tag']} | {100*rep['accuracy']:.1f}% ({100*lo:.0f}–{100*hi:.0f}) | "
                     f"{pct(wacc).strip()} | {rep['macro_f1']:.3f} | {f['TRUE']:.3f} | {f['FALSE']:.3f} | "
                     f"{f['NOT_MENTIONED']:.3f} | {pct(cs.get('strict_accuracy')).strip() if cs else 'n/a'} | "
                     f"{pct(cs.get('supported')).strip() if cs else 'n/a'} | "
                     f"{lc['seconds_mean']:.1f} | {tok} | {cost} | {lc['empty']} |")
    base = runs[0]
    lines += ["", f"Differences against the baseline **{base['tag']}** (B − A):", "",
              "| Run | Δ accuracy (95% CI) | p (by contract) | McNemar p | only A right / only B right | Δ macro-F1 (95% CI) |",
              "|---|---|---|---|---|---|"]
    for run in runs[1:]:
        d = paired_cluster_diff(base["rows"], run["rows"], accuracy)
        dm = paired_cluster_diff(base["rows"], run["rows"], macro_f1)
        m = mcnemar(base["rows"], run["rows"])
        lines.append(f"| {run['tag']} | {100*d['diff']:+.1f} pts ({100*d['ci'][0]:+.1f} to {100*d['ci'][1]:+.1f}) | "
                     f"{d['p']:.3f} | {m['p_value']:.3f} | {m['a_only']} / {m['b_only']} | "
                     f"{dm['diff']:+.3f} ({dm['ci'][0]:+.3f} to {dm['ci'][1]:+.3f}) |")
    lines += ["", "A difference is convincing when its interval does not include 0. The contract-level interval "
              "is the stricter one: it allows for questions from the same contract being related.", ""]
    md = "\n".join(lines)
    print(md)
    if a.out:
        Path(a.out).write_text(md)
        print(f"Saved {a.out}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mistakes", help="show the wrong answers of the latest LLM run")
    m.add_argument("--file", help="a results file (default: the latest)")
    m.add_argument("--limit", type=int, default=6, help="examples per mistake type")
    g = sub.add_parser("regrade", help="grade saved answers again with the current judge prompt")
    g.add_argument("--file", help="a results file (default: the latest with LLM answers)")
    g.add_argument("--provider", default="groq")
    g.add_argument("--judge-model", help="default: the provider's default model")
    g.add_argument("--pause", type=float, default=2.0)
    g.add_argument("--grading", default="auto", choices=["auto", "judge"],
                   help="auto = read 'Yes./No./Partly./doesn't say' from the answer, judge only if needed")
    g.add_argument("--max-wait", type=float, default=120)
    g.add_argument("--partly", default="judge", choices=["judge", "true", "not-mentioned"],
                   help="how to grade answers that start with 'Partly.' (rule-based = no LLM calls)")
    z = sub.add_parser("analyze", help="Markdown report: per-label F1, citations, contradiction errors")
    z.add_argument("--file", help="a results file (default: the latest with LLM answers)")
    z.add_argument("--out", help="where to write the report (default: next to the results file)")
    z.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens (0 = local model)")
    z.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    c = sub.add_parser("compare", help="controlled comparison of runs on the same questions")
    c.add_argument("--runs", nargs="+", help="results files; the FIRST is the baseline")
    c.add_argument("--a", help="baseline results file (same as the first of --runs)")
    c.add_argument("--b", help="new results file")
    c.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens (0 = local model)")
    c.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    c.add_argument("--out", help="also save the table as Markdown")
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--emb-backend", default="local")
        p.add_argument("--emb-model", default="BAAI/bge-small-en-v1.5")
        p.add_argument("--db", default="./chroma_db")
    p = sub.choices["prepare"]
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--n-docs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=350, help="maximum chunk size (try 200 or 500)")
    p = sub.choices["run"]
    p.add_argument("--configs", default="dense,hybrid,hybrid+rerank")
    p.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--llm", action="store_true")
    p.add_argument("--max-questions", type=int, default=60,
                   help="old random sample; prefer --sample, which is fixed and stratified")
    p.add_argument("--sample", type=int, default=0,
                   help="fixed stratified sample of N questions, saved in eval/samples/ and re-used (e.g. 300)")
    p.add_argument("--stratify", default="boost-false", choices=["boost-false", "proportional"],
                   help="boost-false = all Contradiction questions + the rest proportional (default)")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="answer only N evenly spread questions of the sample (for rate-limited APIs)")
    p.add_argument("--partly", default="judge", choices=["judge", "true", "not-mentioned"],
                   help="how to grade answers that start with 'Partly.' (default: LLM judge)")
    p.add_argument("--followup", default="none", choices=["none", "more", "whole"],
                   help="second look for 'not mentioned' answers: 'more' = next clauses of the ranking, "
                        "'whole' = the rest of the contract")
    p.add_argument("--followup-k", type=int, default=15, help="with --followup more: look at clauses up to this rank")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between LLM calls")
    p.add_argument("--provider", default="groq")
    p.add_argument("--model")
    p.add_argument("--judge-model", help="model that grades the answers (default: same as --model). "
                   "A different model spreads the load over two rate limits, "
                   "e.g. llama-3.1-8b-instant on Groq")
    p.add_argument("--grading", default="auto", choices=["auto", "judge"],
                   help="auto = read 'Yes./No./Partly./doesn't say' from the answer, judge only if needed")
    p.add_argument("--whole-doc-tokens", type=int, default=0,
                   help="send the WHOLE contract if it is shorter than this many tokens (e.g. 8000); 0 = off")
    p.add_argument("--votes", type=int, default=1,
                   help="ask N times and take the majority answer (e.g. 3); slower")
    p.add_argument("--vote-temperature", type=float, default=0.7)
    p.add_argument("--llm-max-tokens", type=int,
                   help="max tokens the model may produce, thinking included (default 1500). "
                        "Use 6000+ with --reasoning-effort high, or answers come back empty")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                   help="how much gpt-oss 'thinks' before answering (default: llm.py's setting)")
    p.add_argument("--tag", default="v1", help="name for this experiment, e.g. 'prompt-v2'. "
                   "Change it after editing the prompt, otherwise saved answers are reused")
    p.add_argument("--max-wait", type=float, default=120,
                   help="longest rate-limit wait (seconds) before stopping; progress is saved")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "mistakes": mistakes, "regrade": regrade,
     "analyze": analyze, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()