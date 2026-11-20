#!/usr/bin/env python3
"""Run natijalarini test to'plami bilan solishtirib, spetsifikatsiya 6-bo'limdagi metrikalarni hisoblaydi.

  python scripts/score_eval.py --csv eval/dev_set.csv --run runs/eval-2026-11-21 [--judge]

CSV ustunlari: id,batch,category,question,expected_answer,expected_doc_id,expected_article,answerable
Bir nechta kutilgan manba: expected_doc_id va expected_article ni ';' bilan ajrating (juft-juft).
--judge: javob to'g'riligini (0/1/2) va kontekst yetarliligini LLM-hakam bilan baholaydi (.env dagi LLM).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lexrag.textnorm import normalize_article_ref  # noqa: E402


def norm_doc(d: str | None) -> str:
    return re.sub(r"^lex:", "", (d or "").strip()).lstrip("-")


def expected_pairs(row: dict) -> list[tuple[str, str | None]]:
    docs = [x.strip() for x in re.split(r"[;|]", row.get("expected_doc_id") or "") if x.strip()]
    arts = [x.strip() for x in re.split(r"[;|]", row.get("expected_article") or "") if x.strip()]
    pairs = []
    for i, d in enumerate(docs):
        a = arts[i] if i < len(arts) else (arts[0] if len(arts) == 1 else None)
        pairs.append((norm_doc(d), normalize_article_ref(a) if a else None))
    if not docs and arts:
        pairs = [("", normalize_article_ref(a)) for a in arts]
    return pairs


def ctx_ids(c: dict) -> set[str]:
    ids = {norm_doc(c.get("doc_id"))}
    for a in c.get("alt_doc_ids", []) or []:
        ids.add(norm_doc(a))
    return ids


def match(c: dict, pair: tuple[str, str | None], level: str) -> bool:
    d, a = pair
    if d and d not in ctx_ids(c):
        return False
    if level == "doc":
        return True
    if a is None:
        return True
    return normalize_article_ref(c.get("article")) == a


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


JUDGE_SYSTEM = """You grade answers of a legal QA system for Uzbekistan law.
Compare the SYSTEM ANSWER with the EXPECTED ANSWER. Score:
2 = correct and complete (same legal meaning, numbers/terms match),
1 = partially correct (main point right but missing/extra important detail or minor number issue),
0 = wrong, refuses although expected answer exists, or contradicts.
If the expected answer says the law has no answer and the system also says no answer -> 2.
Also judge SUFFICIENCY: do the CONTEXTS contain enough information to produce the expected answer? (true/false)
Return ONLY JSON: {"score": 0|1|2, "sufficient": true|false, "reason": "short"}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", required=True, help="run papkasi (b01.json ...)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    results = {}
    for p in sorted(Path(args.run).glob("*.json")):
        j = json.loads(p.read_text("utf-8"))
        for r in j.get("results", []):
            results[r["id"]] = r

    K = args.k
    per = []
    hit_doc = hit_art = mrr = rec = prec = prec_doc = 0.0
    n_ans = 0
    n_noans = noans_ok = false_noans = 0
    errors = 0
    lat = []
    cit_total = cit_ok = cit_missing_ctx = stale = 0
    judge = None
    if args.judge:
        from lexrag.config import get_settings
        from lexrag.llm import LLM

        judge = LLM(get_settings().answer_endpoints)
    scores = []
    suff = []
    for qid, row in rows.items():
        r = results.get(qid)
        answerable = str(row.get("answerable", "true")).strip().lower() in ("true", "1", "yes", "ha")
        rec_row = {"id": qid, "category": row.get("category", ""), "answerable": answerable}
        if r is None:
            rec_row.update(status="missing")
            per.append(rec_row)
            errors += 1
            continue
        st = r.get("status")
        if st in ("error", "timeout"):
            errors += 1
        tm = (r.get("timing") or {}).get("total_ms")
        if tm is not None:
            lat.append(tm / 1000)
        ctx = (r.get("contexts") or [])[:K]
        ctx_chunk_ids = {c.get("chunk_id") for c in r.get("contexts") or []}
        for c in r.get("citations") or []:
            cit_total += 1
            if c.get("chunk_id") not in ctx_chunk_ids:
                cit_missing_ctx += 1
            if c.get("doc_status") and c["doc_status"] != "amalda":
                stale += 1
        rec_row.update(status=st, no_answer=r.get("no_answer"))
        if answerable:
            n_ans += 1
            pairs = expected_pairs(row)
            hd = any(match(c, p, "doc") for c in ctx for p in pairs) if pairs else False
            ha = any(match(c, p, "art") for c in ctx for p in pairs) if pairs else False
            rank = next((i for i, c in enumerate(ctx, 1) if any(match(c, p, "art") for p in pairs)), None)
            found = sum(1 for p in pairs if any(match(c, p, "art") for c in ctx))
            rel = sum(1 for c in ctx if any(match(c, p, "art") for p in pairs))
            rel_d = sum(1 for c in ctx if any(match(c, p, "doc") for p in pairs))
            hit_doc += hd
            hit_art += ha
            mrr += 1 / rank if rank else 0
            rec += found / len(pairs) if pairs else 0
            prec += rel / len(ctx) if ctx else 0
            prec_doc += rel_d / len(ctx) if ctx else 0
            if r.get("no_answer"):
                false_noans += 1
            for c in r.get("citations") or []:
                if any(match(c, p, "art") for p in pairs):
                    cit_ok += 1
            rec_row.update(hit_doc=hd, hit_article=ha, rank=rank, recall=found / len(pairs) if pairs else None)
            # xato sababi
            if r.get("no_answer") or st != "ok":
                cause = "retrieval" if not ha else "generation"
            else:
                cause = "" if ha else "retrieval"
                if ha and not any(any(match(c, p, "art") for p in pairs) for c in r.get("citations") or []):
                    cause = "source"
            rec_row["suspected_cause"] = cause
        else:
            n_noans += 1
            if r.get("no_answer"):
                noans_ok += 1
            rec_row["suspected_cause"] = "" if r.get("no_answer") else "hallucination?"
        if judge is not None:
            ctx_txt = "\n---\n".join((c.get("text") or "")[:1500] for c in ctx)
            exp = row.get("expected_answer") or ("(qonunda javob yo'q)" if not answerable else "")
            user = f"QUESTION: {row['question']}\nEXPECTED ANSWER: {exp}\nSYSTEM ANSWER: {r.get('answer')}\nSYSTEM no_answer={r.get('no_answer')}\nCONTEXTS:\n{ctx_txt}"
            try:
                jj, _ = judge.complete_json(JUDGE_SYSTEM, user, max_tokens=200)
                sc = int(jj.get("score", 0))
                scores.append(sc)
                suff.append(bool(jj.get("sufficient")))
                rec_row.update(judge_score=sc, sufficient=jj.get("sufficient"), judge_reason=jj.get("reason"))
            except Exception as e:
                rec_row.update(judge_error=str(e)[:200])
        per.append(rec_row)

    n = len(rows)
    summary = {
        "questions": n,
        "answerable": n_ans,
        f"hit@{K}_doc": round(hit_doc / n_ans, 4) if n_ans else None,
        f"hit@{K}_article": round(hit_art / n_ans, 4) if n_ans else None,
        "mrr": round(mrr / n_ans, 4) if n_ans else None,
        f"recall@{K}": round(rec / n_ans, 4) if n_ans else None,
        f"precision@{K}_article": round(prec / n_ans, 4) if n_ans else None,
        f"precision@{K}_doc": round(prec_doc / n_ans, 4) if n_ans else None,
        "no_answer_accuracy": f"{noans_ok}/{n_noans}",
        "false_no_answer": false_noans,
        "citation_accuracy": round(cit_ok / cit_total, 4) if cit_total else None,
        "citations_not_in_contexts": cit_missing_ctx,
        "stale_norm_citations": stale,
        "error_rate": round(errors / n, 4) if n else None,
        "latency_p50_s": round(pctl(lat, 0.5), 2),
        "latency_p95_s": round(pctl(lat, 0.95), 2),
    }
    if scores:
        summary["answer_correctness_avg"] = round(statistics.mean(scores), 3)
        summary["correct(2)/partial(1)/wrong(0)"] = f"{scores.count(2)}/{scores.count(1)}/{scores.count(0)}"
        summary["sufficiency"] = round(sum(suff) / len(suff), 4) if suff else None
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = Path(args.out or (Path(args.run) / "score"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    keys = sorted({k for p in per for k in p})
    with open(out / "per_question.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per)
    print(f"-> {out}/summary.json, per_question.csv")


if __name__ == "__main__":
    main()
