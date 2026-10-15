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

    # 3. full pipeline: also calls the LLM (2 calls per question; start small!)
    python -m eval.contractnli run --configs hybrid --llm --max-questions 60 -k 3 \\
        --judge-model llama-3.1-8b-instant
    #    Free tier stopped you? Progress is saved: run the SAME command later to continue.

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


# ------------------------------------------------------------------ evidence matching
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def evidence_covered(chunk_text: str, evidence: str, n: int = 6, need: float = 0.5) -> bool:
    """Is this evidence sentence inside the chunk?  Our parser joins lines and fixes
    hyphens, so we compare normalised text, using word 6-grams so that evidence split
    across two chunks still counts if at least half of it is in this one."""
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


def grade(answer: str, hypothesis: str, judge, mode: str) -> tuple[str, str]:
    """Returns (label, how). mode 'auto' = opening phrase first, judge as fallback."""
    if mode == "auto":
        lab = label_from_opening(answer)
        if lab:
            return lab, "opening"
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


def eval_end_to_end(retriever, llm, judge, questions, setup: str, cache_path: Path,
                    grading: str = "auto", whole_doc_tokens: int = 0, votes: int = 1):
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
            try:
                tries = []
                for _ in range(max(1, votes)):
                    a = answer_question(q["question"], retriever, llm, q["doc_id"],
                                        whole_doc_tokens=whole_doc_tokens)
                    if not a.found:
                        lab, how_ = "NOT_MENTIONED", "not-found"
                    else:
                        lab, how_ = grade(a.text, q["hypothesis"], judge, grading)
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
                   "cited_right": any(evidence_covered(c.text, e) for c in a.citations for e in q["evidence"])}
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


def print_paper_comparison(rows: list[dict]) -> None:
    m = paper_metrics(rows)
    print(f"\n  Compared with the ContractNLI paper (Span NLI BERT-large, fine-tuned, full test set):")
    print(f"  {'':<22}{'yours':>10}{'paper':>10}")
    print(f"  {'Accuracy':<22}{m['accuracy']:>10.3f}{0.875:>10.3f}   "
          f"(95% range for yours: {m['ci_low']:.3f}–{m['ci_high']:.3f}, n={m['n']})")
    print(f"  {'F1 Contradiction':<22}{m['f1_contradiction']:>10.3f}{0.357:>10.3f}")
    print(f"  {'F1 Entailment':<22}{m['f1_entailment']:>10.3f}{0.834:>10.3f}")


def pct(x) -> str:
    return "  n/a " if x is None else f"{100 * x:5.1f}%"


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
        qs = questions
        if a.max_questions and len(qs) > a.max_questions:
            qs = random.Random(0).sample(qs, a.max_questions)
        qs = sorted(qs, key=lambda q: q["doc_id"])          # one document at a time
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
                 f"|mt{a.llm_max_tokens or 'default'}|{a.tag}")
        cache = ROOT / "results" / "contractnli_answers.jsonl"
        cache.parent.mkdir(exist_ok=True)
        print(f"\nEND-TO-END with '{last}', k={a.k}, answers by {llm.model}, judged by {judge.model}"
              f"  ({len(qs)} questions)")
        s, rows, stopped = eval_end_to_end(retrievers[last], llm, judge, qs, setup, cache, a.grading,
                                           a.whole_doc_tokens, a.votes)
        if stopped:
            print(f"\n  ⏸  Stopped: {stopped}.\n     Progress is saved. Run the same command later "
                  f"(or tomorrow) to continue. Results below are for the {len(rows)} questions done so far.")
        if s:
            results["end_to_end"] = {"config": setup, **s, "rows": rows}
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
            print_paper_comparison(rows)

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
        new, how = grade(answer, hyp[(r["doc_id"], r["hypothesis_key"])], judge, a.grading)
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
    print_paper_comparison(rows)
    data["end_to_end"].update(label_accuracy=acc, rows=rows, regraded_with=judge.model)
    out = path.with_name(path.stem + "-regraded.json")
    out.write_text(json.dumps(data, indent=1, default=str))
    print(f"\nSaved {out}")


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
    p.add_argument("--max-questions", type=int, default=60)
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
    {"prepare": prepare, "run": run, "mistakes": mistakes, "regrade": regrade}[a.cmd](a)


if __name__ == "__main__":
    main()