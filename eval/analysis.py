"""
analysis.py  —  turn a saved ContractNLI run into a readable report (Markdown)

    python -m eval.contractnli analyze                 # latest run with LLM answers
    python -m eval.contractnli analyze --file eval/results/contractnli-....json

The report has four parts:
  1. overall accuracy with a 95% interval, macro-F1, and (for stratified samples) accuracy
     re-weighted to the real label mix
  2. precision / recall / F1 for EACH label, with bootstrap intervals, and the confusion matrix
  3. citation quality: do the cited clauses contain the evidence the lawyers marked?
  4. a deep-dive into Contradiction (FALSE) errors: are they search misses or reading errors,
     and do exceptions, negation or conditions in the evidence make them more likely?
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from eval.metrics import (LABELS, NAMES, accuracy, bootstrap_ci, citation_failure_kind, citation_quality,
                          citation_summary, class_report, cluster_bootstrap_ci, doc_level, evidence_covered, latency_cost,
                          macro_f1, phenomena, prf, weighted_accuracy, weighted_prf)


def answer_opening(answer: str) -> str:
    t = re.sub(r"^[\s*_#>\-]+", "", answer or "").lower().replace("’", "'")
    if not t or "empty answer" in t[:60]:
        return "Empty"
    if re.match(r"yes\b", t):
        return "Yes"
    if re.match(r"no\b(?!t)", t):
        return "No"
    if t.startswith("partly"):
        return "Partly"
    if t.startswith(("the document doesn't say", "the document does not say", "i couldn't find")):
        return "Doesn't say"
    return "Other"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(lo_hi) -> str:
    lo, hi = lo_hi
    return f"{100 * lo:.0f}–{100 * hi:.0f}%"


def _short(text: str, n: int = 260) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def build_report(rows: list[dict], *, title: str, setup: str = "", ranks: dict | None = None,
                 weights: dict | None = None, k: int = 5, max_examples: int = 25,
                 price_in: float = 0.0, price_out: float = 0.0) -> str:
    """Full report: sections 1-4 (_core_report) + citation failures, the 'not mentioned'
    decision, a per-contract view, and latency/cost."""
    ranks = ranks or {}
    parts = [_core_report(rows, title=title, setup=setup, ranks=ranks, weights=weights, k=k,
                          max_examples=max_examples),
             _citation_failures(rows, ranks, max_examples),
             _not_mentioned(rows, ranks, k),
             _contracts(rows),
             _latency_cost(rows, price_in, price_out)]
    return "\n".join(parts)


def _core_report(rows: list[dict], *, title: str, setup: str = "", ranks: dict | None = None,
                 weights: dict | None = None, k: int = 5, max_examples: int = 25) -> str:
    """rows: end-to-end rows joined with 'evidence' (list of gold evidence sentences).
    ranks: {(doc_id, hypothesis_key): rank of the first retrieved chunk containing evidence, or None}."""
    ranks = ranks or {}
    rep = class_report(rows)
    out: list[str] = [f"# {title}", ""]
    if setup:
        out += [f"Setup: `{setup}`", ""]
    gold_mix = Counter(r["gold"] for r in rows)
    out += [f"Questions: **{rep['n']}** (Entailment {gold_mix['TRUE']} · Contradiction {gold_mix['FALSE']} · "
            f"NotMentioned {gold_mix['NOT_MENTIONED']})", ""]

    # ---- 1. overall
    out += ["## 1. Overall", "",
            "| Metric | Value | 95% interval |", "|---|---|---|",
            f"| Accuracy | {_pct(rep['accuracy'])} | {_ci(rep['accuracy_ci'])} |",
            f"| Macro-F1 (3 labels) | {rep['macro_f1']:.3f} | "
            f"{_ci(bootstrap_ci(rows, lambda rs: class_report(rs)['macro_f1']))} |",
            f"| Accuracy, interval resampling whole contracts | {_pct(rep['accuracy'])} | "
            f"{_ci(cluster_bootstrap_ci(rows, accuracy))} |",
            f"| Macro-F1, interval resampling whole contracts | {rep['macro_f1']:.3f} | "
            f"{_ci(cluster_bootstrap_ci(rows, macro_f1))} |"]
    if weights:
        out.append(f"| Accuracy re-weighted to the real label mix | {_pct(weighted_accuracy(rows, weights))} | |")
    out.append("")
    if gold_mix["FALSE"] < 30:
        out += [f"> ⚠ Only {gold_mix['FALSE']} Contradiction questions: the Contradiction F1 can swing by "
                f"±0.1 or more if one or two answers change. Use a bigger, stratified sample "
                f"(`run --sample 300`) before drawing conclusions about this class.", ""]

    # ---- 2. per class
    out += ["## 2. Per label", "",
            "| Label | Precision | Recall | F1 | F1 95% interval | Support |", "|---|---|---|---|---|---|"]
    for lab in LABELS:
        m = rep["per_class"][lab]
        ci = bootstrap_ci(rows, lambda rs, lab=lab: prf(rs, lab)["f1"])
        out.append(f"| {NAMES[lab]} | {m['precision']:.3f} | {m['recall']:.3f} | **{m['f1']:.3f}** | "
                   f"{ci[0]:.2f}–{ci[1]:.2f} | {m['support']} |")
    if weights:
        w = weighted_prf(rows, weights)
        out += ["", "Re-weighted to the real label mix (the sample has extra Contradictions, which makes "
                "Contradiction precision look higher than on real contracts; recall is unaffected):", "",
                "| Label | Precision | Recall | F1 |", "|---|---|---|---|"]
        for lab in LABELS:
            out.append(f"| {NAMES[lab]} | {w[lab]['precision']:.3f} | {w[lab]['recall']:.3f} | **{w[lab]['f1']:.3f}** |")
    out += ["", "Confusion matrix (rows = lawyers' label, columns = the app's answer):", "",
            "| gold \\ predicted | " + " | ".join(NAMES[p] for p in LABELS) + " |",
            "|---|" + "---|" * len(LABELS)]
    for g in LABELS:
        out.append(f"| **{NAMES[g]}** | " + " | ".join(str(rep["confusion"][g][p]) for p in LABELS) + " |")
    out.append("")

    # ---- answer format: which openings are reliable?
    fmt = defaultdict(list)
    for r in rows:
        fmt[answer_opening(r.get("answer", ""))].append(r)
    out += ["**How the answers start** (the prompt asks for Yes. / No. / Partly. / The document doesn't say this.)", "",
            "| Opening | Answers | Correct | Graded as |", "|---|---|---|---|"]
    for op in ("Yes", "No", "Partly", "Doesn't say", "Empty", "Other"):
        g = fmt.get(op, [])
        if g:
            as_ = ", ".join(f"{NAMES[k]} {v}" for k, v in Counter(r["pred"] for r in g).most_common())
            out.append(f"| {op} | {len(g)} | {sum(r['pred'] == r['gold'] for r in g)}/{len(g)} | {as_} |")
    out.append("")

    # ---- 3. citations
    out += ["## 3. Citation quality", ""]
    cs = citation_summary(rows)
    if cs:
        out += ["Does a citation actually contain the evidence sentence the lawyers marked? "
                "(Only TRUE/FALSE questions have marked evidence.)", "",
                "| Metric | Value | Meaning |", "|---|---|---|",
                f"| Supported answers | {_pct(cs['supported'])} | yes/no answers where ≥1 citation contains the evidence |",
                f"| Citation precision | {_pct(cs['precision'])} | share of cited clauses that contain evidence |",
                f"| Citation recall | {_pct(cs['recall'])} | share of evidence sentences that were cited |",
                f"| Right label, unsupported | {_pct(cs['right_label_unsupported'])} | correct yes/no, but no citation contains the evidence |",
                f"| Strict accuracy | {_pct(cs['strict_accuracy'])} | correct label AND (for yes/no) a supporting citation |",
                f"| Unsupported claims | {cs['unsupported_claims']} | contract is silent, but the app said yes/no and cited a clause |",
                ""]
    elif any("cited_right" in r for r in rows):
        ans = [r for r in rows if r["gold"] != "NOT_MENTIONED"]
        cr = sum(r.get("cited_right", False) for r in ans)
        out += [f"This run was saved before citations were stored, so only the old measure is available: "
                f"a citation contains the evidence for {cr}/{len(ans)} TRUE/FALSE questions ({_pct(cr / len(ans) if ans else None)}). "
                f"Re-run with the current code to get precision, recall and strict accuracy.", ""]

    # ---- 4. contradiction deep-dive
    out += ["## 4. Contradiction deep-dive", ""]
    false_gold = [r for r in rows if r["gold"] == "FALSE"]
    false_pred = [r for r in rows if r["pred"] == "FALSE" and r["gold"] != "FALSE"]
    if not false_gold:
        out += ["No Contradiction questions in this sample.", ""]
        return "\n".join(out)

    def retrieved(r):
        return ranks.get((r["doc_id"], r["hypothesis_key"])) is not None

    missed = [r for r in false_gold if r["pred"] != "FALSE"]
    out += [f"{len(false_gold)} Contradiction questions, {len(missed)} answered wrongly "
            f"(recall {_pct(1 - len(missed) / len(false_gold))}); "
            f"{len(false_pred)} other questions wrongly answered \"No\" (false alarms).", ""]
    if ranks:
        s_miss = [r for r in missed if not retrieved(r)]
        out += ["**Where the missed contradictions come from**", "",
                f"- {len(s_miss)} search: the evidence was not among the top-{k} clauses",
                f"- {len(missed) - len(s_miss)} reading: the evidence was retrieved, but the answer is wrong", ""]
    kinds = Counter(f"{r['gold']}→{r['pred']}" for r in missed)
    out += ["**What the app answered instead:** " +
            ", ".join(f"{NAMES[k.split('→')[1]]} ×{v}" for k, v in kinds.most_common()), ""]

    # phenomena: compare recall with / without each phenomenon, for FALSE and TRUE
    out += ["**Exceptions, negation and conditions in the evidence**", "",
            "Recall = share answered correctly. If a phenomenon makes the task harder, recall is "
            "lower *with* it than *without* it.", "",
            "| Phenomenon in evidence | Contradiction: with | without | Entailment: with | without |",
            "|---|---|---|---|---|"]
    true_gold = [r for r in rows if r["gold"] == "TRUE"]
    for ph in ("exception", "negation", "condition"):
        cells = []
        for group in (false_gold, true_gold):
            w = [r for r in group if ph in phenomena(" ".join(r.get("evidence", [])))]
            wo = [r for r in group if r not in w]
            for g in (w, wo):
                ok = sum(r["pred"] == r["gold"] for r in g)
                cells.append(f"{_pct(ok / len(g)) if g else 'n/a'} ({ok}/{len(g)})")
        out.append(f"| {ph} | " + " | ".join(cells) + " |")
    out.append("")

    # by question type
    by_type = defaultdict(list)
    for r in false_gold:
        by_type[r["short"]].append(r)
    out += ["**Contradiction recall per question type**", "", "| Question type | Correct | Recall |", "|---|---|---|"]
    for short, rs in sorted(by_type.items(), key=lambda kv: sum(r["pred"] == "FALSE" for r in kv[1]) / len(kv[1])):
        ok = sum(r["pred"] == "FALSE" for r in rs)
        out.append(f"| {short} | {ok}/{len(rs)} | {_pct(ok / len(rs))} |")
    out.append("")

    # examples
    out += [f"**Missed contradictions** ({min(len(missed), max_examples)} of {len(missed)} shown)", ""]
    for r in missed[:max_examples]:
        tags = ", ".join(phenomena(" ".join(r.get("evidence", [])))) or "none"
        where = "" if not ranks else (" · evidence retrieved" if retrieved(r) else " · evidence NOT retrieved")
        out += [f"- **{r['short']}** (`{r['doc_id']}`) — answered *{NAMES[r['pred']]}*{where} · phenomena: {tags}",
                f"  - Q: {r['question']}",
                f"  - Evidence: “{_short(' … '.join(r.get('evidence', [])))}”",
                f"  - App: {_short(r.get('answer', ''))}"]
    out.append("")
    if false_pred:
        out += ["**False alarms (the app said \"No\", lawyers said otherwise)**", ""]
        for r in false_pred[:max_examples]:
            out += [f"- **{r['short']}** (`{r['doc_id']}`) — gold *{NAMES[r['gold']]}*",
                    f"  - App: {_short(r.get('answer', ''))}"]
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ extra sections
FAILURE_TEXT = {
    "evidence_not_retrieved": "the evidence was not among the clauses the model saw",
    "cited_other_clause": "the evidence was retrieved, but a different clause was cited",
    "partial_evidence": "a cited clause holds part of the evidence (often split across chunks)",
    "no_citation": "the answer cites nothing",
}


def _citation_failures(rows: list[dict], ranks: dict, max_examples: int) -> str:
    """Correct yes/no answers whose citations don't contain the lawyers' evidence."""
    out = ["## 5. Citation failures: right label, unsupported citation", ""]
    have = [r for r in rows if "citations" in r and r["gold"] in ("TRUE", "FALSE") and r["pred"] == r["gold"]]
    bad = [r for r in have if not citation_quality(r["citations"], r.get("evidence", []))["supported"]]
    if not have:
        return "\n".join(out + ["This run didn't save citations. Re-run with the current code.", ""])
    out += [f"{len(bad)} of {len(have)} correct yes/no answers ({_pct(len(bad) / len(have))}) cite no clause "
            f"containing the evidence. Why:", "", "| Reason | Count | Meaning |", "|---|---|---|"]
    kinds = {}
    for r in bad:
        rank_known = (r["doc_id"], r["hypothesis_key"]) in ranks
        kinds.setdefault(citation_failure_kind(
            r, ranks.get((r["doc_id"], r["hypothesis_key"])) is not None if rank_known else None), []).append(r)
    for kind, text in FAILURE_TEXT.items():
        if kind in kinds:
            out.append(f"| {kind} | {len(kinds[kind])} | {text} |")
    out += ["", "\"cited_other_clause\" can still be a valid citation: the lawyers marked one evidence span, "
            "but another clause may say the same. Check those by hand:", ""]
    for kind in ("cited_other_clause", "partial_evidence", "evidence_not_retrieved", "no_citation"):
        for r in kinds.get(kind, [])[:min(10, max_examples)]:
            out += [f"- **{kind}** · {r['short']} (`{r['doc_id']}`) · {NAMES[r['gold']]}",
                    f"  - Evidence: “{_short(' … '.join(r.get('evidence', [])), 220)}”",
                    f"  - Cited: “{_short((r.get('citations') or [''])[0], 220)}”"]
    out.append("")
    return "\n".join(out)


def _not_mentioned(rows: list[dict], ranks: dict, k: int) -> str:
    """'Not in the top-k clauses' vs 'not in the contract': where do wrong 'not mentioned' answers come from?"""
    out = ["## 6. The \"not mentioned\" decision", ""]
    wrong_nm = [r for r in rows if r["pred"] == "NOT_MENTIONED" and r["gold"] != "NOT_MENTIONED"]
    false_claim = [r for r in rows if r["gold"] == "NOT_MENTIONED" and r["pred"] != "NOT_MENTIONED"]

    def retrieved(r):
        if r.get("retrieved") is not None and r.get("followup") is None:
            return any(e and evidence_covered(t, e)
                       for t in r["retrieved"] for e in r.get("evidence", []))
        return ranks.get((r["doc_id"], r["hypothesis_key"])) is not None

    if wrong_nm:
        inside = [r for r in wrong_nm if retrieved(r)]
        out += [f"{len(wrong_nm)} yes/no questions were answered \"the document doesn't say this\":", "",
                f"- **{len(wrong_nm) - len(inside)}** — the evidence was NOT in the top-{k} clauses. A follow-up search "
                f"(`--followup more` / `--followup whole`) can fix these.",
                f"- **{len(inside)}** — the evidence WAS in the top-{k} clauses, the model didn't recognise it. "
                f"More search can't fix these; they need a better prompt or model.", ""]
    out += [f"{len(false_claim)} questions where the contract is silent were answered yes/no "
            f"(a follow-up search can only make this number grow).", ""]
    fu = [r for r in rows if r.get("followup")]
    if fu:
        changed = [r for r in fu if r["pred"] != "NOT_MENTIONED"]
        good = sum(r["pred"] == r["gold"] for r in changed)
        out += [f"**Follow-up search used** on {len(fu)} \"not mentioned\" answers: {len(changed)} changed to yes/no "
                f"({good} correct, {len(changed) - good} wrong); {len(fu) - len(changed)} stayed \"not mentioned\" "
                f"after checking more of the contract.", ""]
        scope = Counter(r.get("not_found_scope") for r in rows if r["pred"] == "NOT_MENTIONED")
        out += ["How much of the contract was checked before answering \"not mentioned\": " +
                ", ".join(f"{k or 'unknown'} {v}" for k, v in scope.most_common()), ""]
    return "\n".join(out)


def _contracts(rows: list[dict]) -> str:
    docs = doc_level(rows)
    if not docs:
        return ""
    accs = sorted(d["accuracy"] for d in docs)
    mean = sum(accs) / len(accs)
    med = accs[len(accs) // 2]
    out = ["## 7. Per contract", "",
           f"{len(docs)} contracts, {len(rows) / len(docs):.1f} questions each on average. Accuracy per contract: "
           f"mean {_pct(mean)}, median {_pct(med)}, lowest {_pct(accs[0])}, highest {_pct(accs[-1])}; "
           f"{sum(a < 0.5 for a in accs)} contracts below 50%.", "",
           "Questions about the same contract are related, so section 1 also gives intervals that resample "
           "whole contracts; they are wider and more honest than the per-question ones.", "",
           "| Hardest contracts | Questions | Accuracy |", "|---|---|---|"]
    for d in docs[:8]:
        out.append(f"| `{d['doc_id']}` | {d['n']} | {_pct(d['accuracy'])} |")
    out.append("")
    return "\n".join(out)


def _latency_cost(rows: list[dict], price_in: float, price_out: float) -> str:
    lc = latency_cost(rows, price_in, price_out)
    out = ["## 8. Latency, tokens and failures", "",
           "| Metric | Value |", "|---|---|",
           f"| Seconds per question (mean / median / 90th pct) | {lc['seconds_mean']:.1f} / {lc['seconds_p50']:.1f} / "
           f"{lc['seconds_p90']:.1f} |" if lc["seconds_mean"] is not None else "| Seconds per question | n/a |",
           f"| Empty answers (model ran out of tokens while thinking) | {lc['empty']} |"]
    if lc["tokens_in"] is not None:
        out += [f"| Tokens per question: input / output (incl. thinking) | {lc['tokens_in']:.0f} / {lc['tokens_out']:.0f} |",
                f"| LLM calls per question (answer + judge) | {lc['calls']:.2f} |",
                f"| Questions where an answer call hit the max-token limit | {lc['hit_token_limit']} |"]
    if "usd_per_1000" in lc:
        out.append(f"| Cost per 1,000 questions (at the given prices) | ${lc['usd_per_1000']:.2f} |")
    out.append("")
    return "\n".join(out)