#!/usr/bin/env python3
"""Ishlayotgan serverni tez tekshirish (spetsifikatsiya checklisti bo'yicha).

  python scripts/smoke_test.py --base-url http://localhost:8000 --api-key KEY
"""
import argparse
import json
import sys
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()
    B = a.base_url.rstrip("/")
    H = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(("✅" if cond else "❌"), name, extra)

    h = requests.get(f"{B}/v1/health", timeout=30)
    check("GET /v1/health", h.status_code == 200, h.text[:200])
    m = requests.get(f"{B}/v1/meta", timeout=30)
    check("GET /v1/meta", m.status_code == 200)
    print(json.dumps(m.json().get("system"), ensure_ascii=False, indent=1))

    rid = time.strftime("smoke-%H%M%S")
    items = [
        {"id": "q001", "question": "Yillik asosiy mehnat taʼtilining eng kam muddati qancha?", "language": "uz-Latn"},
        {"id": "q002", "question": "Какой срок исковой давности установлен Гражданским кодексом?", "language": "ru"},
        {"id": "q003", "question": "Йиллик асосий меҳнат таътили неча кун?"},
        {"id": "q004", "question": "Toshkentda eng mazali osh qayerda?"},
    ]
    body = {"run_id": rid, "batch_id": f"{rid}-b01", "batch_index": 1, "total_batches": 1,
            "options": {"top_k": 5, "return_contexts": True, "return_debug": True, "temperature": 0, "seed": 42}, "items": items}
    t = time.time()
    r = requests.post(f"{B}/v1/rag/batch", json=body, headers=H, timeout=180)
    check("POST /v1/rag/batch", r.status_code == 200, f"{time.time() - t:.1f}s")
    j = r.json()
    res = j.get("results", [])
    check("results soni = items", len(res) == len(items))
    for x in res:
        ctx = {c["chunk_id"] for c in x.get("contexts", [])}
        cit_ok = all(c["chunk_id"] in ctx for c in x.get("citations", []))
        top = [f"{c['doc_id']}:{c['article']}" for c in x.get("contexts", [])[:3]]
        print(f"  {x['id']} [{x['status']}] {x['timing'].get('total_ms')}ms  top={top}\n     {str(x.get('answer'))[:300]}")
        check(f"  {x['id']} citation->contexts", cit_ok)
    check("no_answer savolida no_answer=true", res[-1].get("no_answer") is True)
    r2 = requests.post(f"{B}/v1/rag/batch", json=body, headers=H, timeout=180)
    check("idempotentlik (bir xil javob)", r2.json() == j)
    body11 = dict(body, batch_id=f"{rid}-b11", items=[dict(items[0], id=f"x{i}") for i in range(11)])
    check("11 ta savol -> 400", requests.post(f"{B}/v1/rag/batch", json=body11, headers=H, timeout=30).status_code == 400)
    body409 = dict(body, items=items[:1])
    check("boshqa mazmun, shu batch_id -> 409", requests.post(f"{B}/v1/rag/batch", json=body409, headers=H, timeout=30).status_code == 409)
    g = requests.get(f"{B}/v1/rag/batch/{rid}-b01", headers=H, timeout=30)
    check("GET /v1/rag/batch/{id}", g.status_code == 200 and g.json() == j)
    if a.api_key:
        check("noto'g'ri kalit -> 403", requests.post(f"{B}/v1/rag/batch", json=body, headers={"Authorization": "Bearer xato"}, timeout=30).status_code == 403)
    print("\nNATIJA:", "HAMMASI OK" if ok else "MUAMMO BOR")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
