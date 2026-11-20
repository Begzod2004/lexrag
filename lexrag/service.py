"""HTTP-freymvorkdan mustaqil API mantiqi: validatsiya, autentifikatsiya, rate limit, idempotentlik, sync/async."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .pipeline import Engine
from .storage import Store

log = logging.getLogger("lexrag.service")

LANG_OK = {"uz-Latn", "uz-Cyrl", "ru"}
ANSWER_LANG_OK = {"auto", "uz-Latn", "uz-Cyrl", "ru"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def err(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": {"code": code, "message": message, "retryable": retryable}}


class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = rate, burst
        self.state: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def take(self, key: str) -> float:
        """0 — ruxsat; >0 — necha soniyadan keyin qayta urinish mumkin."""
        if self.rate <= 0:
            return 0.0
        now = time.monotonic()
        with self.lock:
            tokens, ts = self.state.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - ts) * self.rate)
            if tokens >= 1:
                self.state[key] = (tokens - 1, now)
                return 0.0
            self.state[key] = (tokens, now)
            return (1 - tokens) / self.rate


def normalize_options(o: dict | None) -> tuple[dict | None, str | None]:
    o = dict(o or {})
    out = {
        "top_k": 5,
        "return_contexts": True,
        "return_debug": False,
        "temperature": 0.0,
        "seed": None,
        "as_of_date": None,
        "answer_language": "auto",
    }
    if "top_k" in o and o["top_k"] is not None:
        if not isinstance(o["top_k"], int) or isinstance(o["top_k"], bool) or not (1 <= o["top_k"] <= 50):
            return None, "options.top_k 1..50 oralig'idagi butun son bo'lishi kerak"
        out["top_k"] = o["top_k"]
    for b in ("return_contexts", "return_debug"):
        if b in o and o[b] is not None:
            if not isinstance(o[b], bool):
                return None, f"options.{b} bool bo'lishi kerak"
            out[b] = o[b]
    if o.get("temperature") is not None:
        if not isinstance(o["temperature"], (int, float)) or isinstance(o["temperature"], bool) or not (0 <= o["temperature"] <= 2):
            return None, "options.temperature 0..2"
        out["temperature"] = float(o["temperature"])
    if o.get("seed") is not None:
        if not isinstance(o["seed"], int) or isinstance(o["seed"], bool):
            return None, "options.seed butun son bo'lishi kerak"
        out["seed"] = o["seed"]
    if o.get("as_of_date") is not None:
        if not isinstance(o["as_of_date"], str) or not DATE_RE.match(o["as_of_date"]):
            return None, "options.as_of_date YYYY-MM-DD formatida bo'lishi kerak"
        out["as_of_date"] = o["as_of_date"]
    if o.get("answer_language") is not None:
        al = str(o["answer_language"])
        if al not in ANSWER_LANG_OK:
            return None, "options.answer_language: auto | uz-Latn | uz-Cyrl | ru"
        out["answer_language"] = al
    return out, None


def normalize_lang(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    low = s.lower().replace("_", "-")
    if low in ("uz-latn", "uz", "uzb", "uz-lat", "latn"):
        return "uz-Latn"
    if low in ("uz-cyrl", "uz-cyr", "cyrl"):
        return "uz-Cyrl"
    if low in ("ru", "rus", "ru-ru"):
        return "ru"
    return None  # noma'lum -> avtomatik aniqlash


def validate_batch(body) -> tuple[dict | None, dict | None]:
    if not isinstance(body, dict):
        return None, err("VALIDATION_ERROR", "So'rov JSON obyekt bo'lishi kerak")
    run_id = body.get("run_id")
    batch_id = body.get("batch_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return None, err("VALIDATION_ERROR", "run_id majburiy (string)")
    if not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id) > 200:
        return None, err("VALIDATION_ERROR", "batch_id majburiy (string, ≤200 belgi)")
    for f in ("batch_index", "total_batches"):
        v = body.get(f)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            return None, err("VALIDATION_ERROR", f"{f} musbat butun son bo'lishi kerak")
    opts, oerr = normalize_options(body.get("options"))
    if oerr:
        return None, err("VALIDATION_ERROR", oerr)
    items = body.get("items")
    if not isinstance(items, list) or not items:
        return None, err("VALIDATION_ERROR", "items 1–10 ta savoldan iborat ro'yxat bo'lishi kerak")
    if len(items) > 10:
        return None, err("TOO_MANY_ITEMS", f"Bitta batch'da ko'pi bilan 10 ta savol bo'lishi mumkin (yuborildi: {len(items)})")
    seen = set()
    clean = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            return None, err("VALIDATION_ERROR", f"items[{i}] obyekt bo'lishi kerak")
        qid = it.get("id")
        if not isinstance(qid, str) or not qid.strip():
            return None, err("VALIDATION_ERROR", f"items[{i}].id majburiy (string)")
        if qid in seen:
            return None, err("DUPLICATE_ID", f"Takroriy id: {qid}")
        seen.add(qid)
        q = it.get("question")
        if not isinstance(q, str) or not q.strip():
            return None, err("EMPTY_QUESTION", f"items[{i}].question bo'sh")
        if len(q) > 2000:
            return None, err("VALIDATION_ERROR", f"items[{i}].question 2000 belgidan uzun")
        clean.append({"id": qid, "question": q.strip(), "language": normalize_lang(it.get("language")), "language_raw": it.get("language")})
    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "batch_index": body.get("batch_index"),
        "total_batches": body.get("total_batches"),
        "options": opts,
        "items": clean,
    }, None


def request_hash(req: dict) -> str:
    canon = {"run_id": req["run_id"], "options": req["options"], "items": [{"id": i["id"], "question": i["question"], "language": i["language_raw"]} for i in req["items"]]}
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class Service:
    def __init__(self, engine: Engine, store: Store | None = None):
        self.engine = engine
        self.s = engine.s
        self.store = store or Store(self.s.db_path)
        self.store.fail_stale(0)  # qayta ishga tushganda yarim qolganlar
        self.store.purge(self.s.result_ttl_days)
        self.bucket = TokenBucket(self.s.rate_limit_per_sec, self.s.rate_limit_burst)
        self.events: dict[str, threading.Event] = {}
        self.ev_lock = threading.Lock()
        self.batch_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="batch")

    # ---------------- auth / limit ----------------
    def auth(self, header: str | None) -> tuple[int, dict] | None:
        if not self.s.api_keys:
            return None  # kalit sozlanmagan — ochiq rejim (faqat lokal test uchun)
        if not header or not header.lower().startswith("bearer "):
            return 401, err("UNAUTHORIZED", "Authorization: Bearer <API_KEY> header kerak")
        key = header.split(" ", 1)[1].strip()
        if key not in self.s.api_keys:
            return 403, err("FORBIDDEN", "API kalit noto'g'ri")
        return None

    def limit(self, header: str | None) -> tuple[int, dict, dict] | None:
        key = (header or "anon")[-24:]
        wait = self.bucket.take(key)
        if wait > 0:
            ra = max(1, math.ceil(wait))
            return 429, err("RATE_LIMITED", "So'rovlar juda tez yuborilyapti", True), {"Retry-After": str(ra)}
        return None

    # ---------------- batch ----------------
    def _build_response(self, req: dict, results: list[dict], received_at, t0: float) -> dict:
        completed = self.engine.now()
        summ = {"total": len(results), "ok": 0, "no_answer": 0, "error": 0}
        for r in results:
            st = r.get("status")
            if st == "ok":
                summ["ok"] += 1
            elif st == "no_answer":
                summ["no_answer"] += 1
            else:
                summ["error"] += 1
        return {
            "run_id": req["run_id"],
            "batch_id": req["batch_id"],
            "batch_index": req.get("batch_index"),
            "total_batches": req.get("total_batches"),
            "status": "completed",
            "system": self.engine.system_info(),
            "timing": {
                "received_at": received_at.isoformat(timespec="seconds"),
                "completed_at": completed.isoformat(timespec="seconds"),
                "total_ms": int((time.time() - t0) * 1000),
            },
            "summary": summ,
            "results": results,
        }

    def _job(self, req: dict, received_at, t0: float, previous: dict | None = None):
        bid = req["batch_id"]
        try:
            items = req["items"]
            prev_by_id = {}
            if previous:
                prev_by_id = {r["id"]: r for r in previous.get("results", []) if r.get("status") in ("ok", "no_answer")}
            todo = [it for it in items if it["id"] not in prev_by_id]
            fresh = self.engine.run_items(todo, req["options"], cache=self.store) if todo else []
            fresh_by_id = {r["id"]: r for r in fresh}
            results = [prev_by_id.get(it["id"]) or fresh_by_id[it["id"]] for it in items]
            resp = self._build_response(req, results, received_at, t0)
            self.store.set_status(bid, "completed", resp)
        except Exception as e:  # pragma: no cover
            log.exception("batch %s xato", bid)
            self.store.set_status(bid, "failed", {"batch_id": bid, "run_id": req["run_id"], "status": "failed", **err("INTERNAL", str(e)[:300], True)})
        finally:
            with self.ev_lock:
                ev = self.events.pop(bid, None)
            if ev:
                ev.set()

    def _start(self, req: dict, previous: dict | None = None) -> threading.Event:
        ev = threading.Event()
        with self.ev_lock:
            self.events[req["batch_id"]] = ev
        received_at = self.engine.now()
        self.batch_pool.submit(self._job, req, received_at, time.time(), previous)
        return ev

    def _wait(self, batch_id: str, ev: threading.Event | None, timeout: float) -> tuple[int, dict]:
        if ev is not None:
            ev.wait(timeout)
        row = self.store.get_batch(batch_id)
        if row and row["status"] == "completed":
            return 200, row["response"]
        if row and row["status"] == "failed":
            return 503, row["response"] or err("INTERNAL", "batch failed", True)
        return 202, {"batch_id": batch_id, "status": "processing"}

    def post_batch(self, body, async_mode: bool = False) -> tuple[int, dict]:
        req, e = validate_batch(body)
        if e:
            return 400, e
        h = request_hash(req)
        bid = req["batch_id"]
        wait_s = 0.0 if async_mode else self.s.sync_timeout_s
        row = self.store.get_batch(bid)
        if row is not None:
            if row["req_hash"] != h:
                return 409, err("BATCH_CONFLICT", f"batch_id '{bid}' boshqa mazmun bilan allaqachon yuborilgan; yangi batch_id ishlating")
            if row["status"] == "completed":
                resp = row["response"]
                retry_needed = any(r.get("status") in ("error", "timeout") and (r.get("error") or {}).get("retryable") for r in resp.get("results", []))
                if not retry_needed:
                    return 200, resp  # idempotent: saqlangan natija
                self.store.set_status(bid, "processing")
                ev = self._start(req, previous=resp)
                return self._wait(bid, ev, wait_s)
            if row["status"] == "processing":
                with self.ev_lock:
                    ev = self.events.get(bid)
                if ev is None:  # yetim yozuv
                    ev = self._start(req)
                return self._wait(bid, ev, wait_s)
            # failed -> qayta ishlash
            self.store.set_status(bid, "processing")
            ev = self._start(req)
            return self._wait(bid, ev, wait_s)
        if not self.store.create_batch(bid, req["run_id"], h, {k: v for k, v in req.items()}):
            return self.post_batch(body, async_mode)  # parallel poyga
        ev = self._start(req)
        return self._wait(bid, ev, wait_s)

    def get_batch(self, batch_id: str) -> tuple[int, dict]:
        row = self.store.get_batch(batch_id)
        if row is None:
            return 404, err("NOT_FOUND", f"batch_id '{batch_id}' topilmadi")
        if row["status"] == "completed":
            return 200, row["response"]
        if row["status"] == "failed":
            return 200, row["response"] or {"batch_id": batch_id, "status": "failed"}
        return 202, {"batch_id": batch_id, "run_id": row["run_id"], "status": "processing"}

    def query(self, body) -> tuple[int, dict]:
        if not isinstance(body, dict):
            return 400, err("VALIDATION_ERROR", "JSON obyekt kerak")
        q = body.get("question")
        if not isinstance(q, str) or not q.strip() or len(q) > 2000:
            return 400, err("EMPTY_QUESTION", "question 1–2000 belgi bo'lishi kerak")
        opts, oerr = normalize_options(body.get("options") or {k: body[k] for k in ("top_k", "return_debug", "answer_language") if k in body})
        if oerr:
            return 400, err("VALIDATION_ERROR", oerr)
        if "return_debug" not in (body.get("options") or {}) and "return_debug" not in body:
            opts["return_debug"] = True
        t0 = time.time()
        received = self.engine.now()
        res = self.engine.run_items([{"id": str(body.get("id") or "q"), "question": q.strip(), "language": normalize_lang(body.get("language"))}], opts, cache=self.store)[0]
        return 200, {
            "system": self.engine.system_info(),
            "timing": {"received_at": received.isoformat(timespec="seconds"), "completed_at": self.engine.now().isoformat(timespec="seconds"), "total_ms": int((time.time() - t0) * 1000)},
            "result": res,
        }

    def meta(self) -> tuple[int, dict]:
        idx = self.engine.index
        m = idx.meta if idx is not None else {}
        return 200, {
            "system": self.engine.system_info(),
            "corpus": {
                "documents": m.get("n_docs"),
                "articles": m.get("n_articles"),
                "chunks": m.get("n_chunks"),
                "doc_types": m.get("doc_types"),
                "languages": m.get("languages"),
                "built_at": m.get("built_at"),
                "chunking": "modda (article) darajasida; 6000 belgidan uzun moddalar qism (part) guruhlariga bo'linadi",
                "repealed_docs": "faqat amaldagi hujjatlar indekslanadi; 'kuchini yo'qotgan' hujjatlar (agar qo'shilsa) doc_status bilan belgilanadi va past baholanadi",
            },
            "limits": {"max_items": 10, "sync_timeout_s": self.s.sync_timeout_s, "rate_limit_per_sec": self.s.rate_limit_per_sec},
            "languages": ["uz-Latn", "uz-Cyrl", "ru"],
        }

    def health(self) -> tuple[int, dict]:
        ok = self.engine.index is not None
        return (200 if ok else 503), {
            "status": "ok" if ok else "degraded",
            "index_loaded": ok,
            "index_error": self.engine.index_error,
            "llm_configured": self.engine.llm_answer.available,
            "dense_enabled": bool(self.engine.embedder is not None and ok and self.engine.index.dense is not None),
            "time": self.engine.now().isoformat(timespec="seconds"),
        }
