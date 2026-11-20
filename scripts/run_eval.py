#!/usr/bin/env python3
"""Tester ssenariysi (spetsifikatsiya 1-bo'lim): 100 savol -> 10 batch -> ketma-ket yuborish -> run_id/bNN.json.

  python scripts/run_eval.py --csv eval/dev_set.csv --base-url http://localhost:8000 --api-key KEY --run-id eval-2026-11-21

Qayta urinish: 429 -> Retry-After; 500/503/504 -> 3 martagacha (5s, 15s, 45s); 202 -> GET poll.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests


def load_items(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def post_with_retries(sess: requests.Session, url: str, body: dict, headers: dict, poll_base: str, timeout: float = 180) -> dict:
    backoff = [5, 15, 45]
    attempt = 0
    while True:
        try:
            r = sess.post(url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt >= len(backoff):
                raise
            print(f"   tarmoq xatosi: {e}; {backoff[attempt]}s dan keyin qayta", file=sys.stderr)
            time.sleep(backoff[attempt])
            attempt += 1
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 202:
            bid = body["batch_id"]
            for _ in range(600):
                time.sleep(2)
                g = sess.get(f"{poll_base}/v1/rag/batch/{bid}", headers=headers, timeout=60)
                if g.status_code == 200:
                    j = g.json()
                    if j.get("status") == "completed":
                        return j
                    if j.get("status") == "failed":
                        break
            raise RuntimeError(f"{bid}: async natija kelmadi")
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "2"))
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 503, 504) and attempt < len(backoff):
            time.sleep(backoff[attempt])
            attempt += 1
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--run-id", default=time.strftime("eval-%Y-%m-%d-%H%M"))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = load_items(Path(args.csv))
    out = Path(args.out) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    base = args.base_url.rstrip("/")
    sess = requests.Session()

    # batch ustuni bo'lsa — shundan, aks holda 10 tadan
    groups: dict[str, list[dict]] = {}
    if rows and rows[0].get("batch"):
        for r in rows:
            groups.setdefault(r["batch"], []).append(r)
    else:
        for i in range(0, len(rows), args.batch_size):
            groups[f"b{i // args.batch_size + 1:02d}"] = rows[i : i + args.batch_size]
    names = sorted(groups)
    t_all = time.time()
    for bi, name in enumerate(names, 1):
        items = [{"id": r["id"], "question": r["question"], **({"language": r["language"]} if r.get("language") else {})} for r in groups[name]]
        body = {
            "run_id": args.run_id,
            "batch_id": f"{args.run_id}-{name}",
            "batch_index": bi,
            "total_batches": len(names),
            "options": {"top_k": args.top_k, "return_contexts": True, "return_debug": args.debug, "temperature": 0, "seed": 42, "answer_language": "auto"},
            "items": items,
        }
        t0 = time.time()
        resp = post_with_retries(sess, f"{base}/v1/rag/batch", body, headers, base)
        (out / f"{name}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), "utf-8")
        s = resp.get("summary", {})
        print(f"{name}: {time.time() - t0:5.1f}s  ok={s.get('ok')} no_answer={s.get('no_answer')} error={s.get('error')}")
    print(f"Tayyor: {out} ({time.time() - t_all:.0f}s). Baholash: python scripts/score_eval.py --csv {args.csv} --run {out}")


if __name__ == "__main__":
    main()
