"""
run_eval.py  —  measure how well the app finds the right clause (and answers)

    # 1. retrieval only: fast, free, no LLM calls. Compares several setups side by side.
    python -m eval.run_eval --configs dense,hybrid,hybrid+rerank

    # 2. end-to-end: also calls the LLM for every question (costs API calls)
    python -m eval.run_eval --configs hybrid+rerank --llm
    python -m eval.run_eval --configs hybrid+rerank --llm --judge     # + LLM checks the content

WHAT IS MEASURED
Retrieval (for each setup):
  Hit@1 / Hit@3   right clause is the 1st result / among the first 3
  MRR             average of 1/rank of the right clause (1.0 = always first)
  Missed (FN)     answerable questions where NOTHING passed min_score -> app says "not found"
                  wrongly. This is the "not found" problem you saw.
  Let through     unanswerable questions where something passed min_score (the LLM still
                  gets a chance to refuse, so this is less serious)
  Best cut-off    the min_score that separates the two groups best, found by trying all values
  ms / query      retrieval time

End-to-end (--llm):
  Answered        answerable questions that got an answer (not "not found")
  Cited right     ...whose citations include the expected clause
  Correct (judge) ...whose content matches expected_answer, checked by the LLM itself
  Refused right   unanswerable questions answered with "not found"
  seconds         average LLM time and total time per question
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.generation.llm import LLM
from app.generation.qa import answer_question
from app.ingestion.embedder import Embedder
from app.ingestion.store import VectorStore
from app.retrieval.retriever import Retriever, Source

ROOT = Path(__file__).parent

try:                                    # reads GROQ_API_KEY etc. from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ------------------------------------------------------------------ helpers
def matches(src: Source, expected: list[str]) -> bool:
    num = (src.section_number or "").rstrip(".")
    return (num in expected) or ("intro" in expected and not num)


def first_relevant_rank(hits: list[Source], expected: list[str]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if matches(h, expected):
            return rank
    return None


def best_cutoff(rows: list[dict]) -> tuple[float, float, float]:
    """Try every possible min_score. Returns (balanced cut-off, its accuracy,
    'safe' cut-off = highest value that still lets every answerable question through)."""
    scores = sorted({r["top_score"] for r in rows} | {0.0})
    best_t, best_acc = 0.0, -1.0
    for t in scores:
        ok = sum((r["top_score"] >= t) == r["answerable"] for r in rows)
        acc = ok / len(rows)
        if acc > best_acc:
            best_t, best_acc = t, acc
    answerable_tops = [r["top_score"] for r in rows if r["answerable"]]
    safe = min(answerable_tops) if answerable_tops else 0.0
    return best_t, best_acc, safe


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


JUDGE_PROMPT = """You grade answers about a legal document.
Reference answer: {ref}
Candidate answer: {cand}
Does the candidate state the same key facts as the reference, without contradicting it?
Extra correct detail is fine. Reply with exactly one word: YES or NO."""


# ------------------------------------------------------------------ evaluation
def eval_retrieval(name: str, retriever: Retriever, questions: list[dict], doc_id: str | None):
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], doc_id)
        ms = 1000 * (time.perf_counter() - t0)
        rank = first_relevant_rank(hits, q["expected_sections"]) if q["answerable"] else None
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "answerable": q["answerable"], "rank": rank, "ms": ms,
            # the app keeps EVERY hit above min_score, so look at the best score, not hit #1
            "top_score": max((h.score for h in hits), default=0.0),
            "passes": any(h.score >= retriever.min_score for h in hits),
            "top_refs": [f"{h.short_ref} ({h.score:.2f})" for h in hits[:3]],
        })

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    t, acc, safe = best_cutoff(rows)
    summary = {
        "config": name, "min_score": retriever.min_score,
        "hit@1": sum(r["rank"] == 1 for r in ans) / len(ans),
        "hit@3": sum(r["rank"] is not None and r["rank"] <= 3 for r in ans) / len(ans),
        "mrr": statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in ans),
        "missed": sum(not r["passes"] for r in ans),
        "let_through": sum(r["passes"] for r in una),
        "best_cutoff": t, "best_cutoff_acc": acc, "safe_cutoff": safe,
        "ms": statistics.mean(r["ms"] for r in rows[1:] or rows),   # skip 1st (warm-up)
        "n_answerable": len(ans), "n_unanswerable": len(una),
    }
    return summary, rows


def eval_end_to_end(retriever: Retriever, llm: LLM, questions: list[dict], doc_id, judge: bool):
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        a = answer_question(q["question"], retriever, llm, doc_id)
        total = time.perf_counter() - t0
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "found": a.found, "answer": a.text, "llm_s": a.timings.get("llm", 0.0),
               "total_s": total, "cited": [c.short_ref for c in a.citations],
               "cited_right": any(matches(c, q["expected_sections"]) for c in a.citations)}
        if judge and q["answerable"] and a.found and q.get("expected_answer"):
            verdict = llm.complete("You are a strict grader.",
                                   JUDGE_PROMPT.format(ref=q["expected_answer"], cand=a.text))
            row["judge"] = verdict.strip().upper().startswith("YES")
        rows.append(row)
        print(f"  [{q['id']:>2}] {'✓' if (a.found == q['answerable']) else '✗'} "
              f"{total:4.1f}s  {q['question'][:60]}")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    summary = {
        "answered": sum(r["found"] for r in ans) / len(ans),
        "cited_right": sum(r["cited_right"] for r in ans) / len(ans),
        "refused_right": (sum(not r["found"] for r in una) / len(una)) if una else None,
        "llm_s": statistics.mean([r["llm_s"] for r in rows if r["llm_s"]] or [0]),
        "total_s": statistics.mean(r["total_s"] for r in rows),
    }
    if judge:
        judged = [r for r in ans if "judge" in r]
        summary["correct_judge"] = sum(r["judge"] for r in judged) / len(ans)
    return summary, rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(ROOT / "nda_questions.json"))
    ap.add_argument("--doc-id", help="document to test (default: the only one in the database)")
    ap.add_argument("--configs", default="dense,hybrid,hybrid+rerank",
                    help="comma list of: dense, hybrid, dense+rerank, hybrid+rerank")
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, help="override the default cut-off for every config")
    ap.add_argument("--llm", action="store_true", help="also run the full pipeline (calls the LLM)")
    ap.add_argument("--judge", action="store_true", help="let the LLM grade answer content")
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model")
    ap.add_argument("--emb-backend", default="local")
    ap.add_argument("--emb-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--db", default="./chroma_db")
    a = ap.parse_args()

    data = json.loads(Path(a.questions).read_text())
    questions = data["questions"]
    embedder = Embedder(model=a.emb_model, backend=a.emb_backend)
    store = VectorStore(a.db, embedding_model=f"{a.emb_backend}:{a.emb_model}")

    doc_id = a.doc_id
    if not doc_id:
        docs = sorted({m["doc_id"] for m in store.col.get(include=["metadatas"])["metadatas"]})
        if len(docs) != 1:
            raise SystemExit(f"Found {len(docs)} documents, pass one with --doc-id: {docs}")
        doc_id = docs[0]
    print(f"Document: {doc_id}   ·   {len(questions)} questions\n")

    reranker = None
    results = {"doc_id": doc_id, "time": datetime.now().isoformat(timespec="seconds"),
               "retrieval": [], "rows": {}}
    retrievers = {}
    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        mode = name.split("+")[0]
        use_rerank = name.endswith("+rerank")
        if use_rerank and reranker is None:
            from app.retrieval.reranker import Reranker
            print(f"Loading reranker {a.reranker_model} …")
            reranker = Reranker(a.reranker_model)
        r = Retriever(embedder, store, k=a.k, min_score=a.min_score, mode=mode,
                      reranker=reranker if use_rerank else None)
        retrievers[name] = r
        summary, rows = eval_retrieval(name, r, questions, doc_id)
        results["retrieval"].append(summary)
        results["rows"][name] = rows

    # ---- retrieval table
    print("RETRIEVAL")
    print(f"{'setup':<15}{'Hit@1':>7}{'Hit@3':>7}{'MRR':>6}{'Missed':>8}{'Let in':>8}"
          f"{'cut-off':>9}{'best':>7}{'acc':>7}{'safe':>7}{'ms':>7}")
    for s in results["retrieval"]:
        print(f"{s['config']:<15}{pct(s['hit@1']):>7}{pct(s['hit@3']):>7}{s['mrr']:6.2f}"
              f"{s['missed']:>5}/{s['n_answerable']:<2}{s['let_through']:>5}/{s['n_unanswerable']:<2}"
              f"{s['min_score']:9.2f}{s['best_cutoff']:7.2f}{pct(s['best_cutoff_acc']):>7}"
              f"{s['safe_cutoff']:7.2f}{s['ms']:7.0f}")
    print("  cut-off = min_score used now · best = cut-off that separates answerable/unanswerable best"
          " · safe = highest cut-off with no answerable question missed\n")

    # ---- failures of the last setup (usually the one you want to improve)
    last = results["retrieval"][-1]["config"]
    print(f"PROBLEMS in '{last}'")
    any_problem = False
    for r in results["rows"][last]:
        problem = None
        if r["answerable"] and r["rank"] != 1:
            problem = "right clause not first" if r["rank"] else "right clause not in top-k"
        if r["answerable"] and not r["passes"]:
            problem = (problem + " + " if problem else "") + "below cut-off → 'not found'"
        if not r["answerable"] and r["passes"]:
            problem = "unanswerable but passed cut-off"
        if problem:
            any_problem = True
            print(f"  [{r['id']:>2}] {r['question']}\n       {problem}   top: {' | '.join(r['top_refs'])}")
    if not any_problem:
        print("  none 🎉")

    # ---- end-to-end
    if a.llm:
        name = last
        llm = LLM(provider=a.provider, model=a.model)
        print(f"\nEND-TO-END with '{name}' and {llm.provider}:{llm.model}")
        summary, rows = eval_end_to_end(retrievers[name], llm, questions, doc_id, a.judge)
        results["end_to_end"] = {"config": name, "llm": f"{llm.provider}:{llm.model}",
                                 **summary, "rows": rows}
        print(f"\n  Answered       {pct(summary['answered'])}  (answerable questions not called 'not found')")
        print(f"  Cited right    {pct(summary['cited_right'])}")
        if "correct_judge" in summary:
            print(f"  Correct(judge) {pct(summary['correct_judge'])}")
        if summary["refused_right"] is not None:
            print(f"  Refused right  {pct(summary['refused_right'])}  (unanswerable → 'not found')")
        print(f"  Time           {summary['llm_s']:.1f}s LLM  ·  {summary['total_s']:.1f}s total per question")

    out = ROOT / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()