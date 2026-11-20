"""SQLite: batch natijalari (idempotentlik, async) va javoblar keshi."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path | str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(p), check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    req_hash TEXT,
                    status TEXT,
                    request TEXT,
                    response TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS answers (
                    k TEXT PRIMARY KEY,
                    v TEXT,
                    created_at REAL
                );
                """
            )
            self.db.commit()

    # ---------------- batches ----------------
    def get_batch(self, batch_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT batch_id, run_id, req_hash, status, request, response, created_at, updated_at FROM batches WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "batch_id": row[0],
            "run_id": row[1],
            "req_hash": row[2],
            "status": row[3],
            "request": json.loads(row[4]) if row[4] else None,
            "response": json.loads(row[5]) if row[5] else None,
            "created_at": row[6],
            "updated_at": row[7],
        }

    def create_batch(self, batch_id: str, run_id: str, req_hash: str, request: dict) -> bool:
        """True — yangi yozuv yaratildi; False — allaqachon bor."""
        now = time.time()
        with self.lock:
            try:
                self.db.execute(
                    "INSERT INTO batches VALUES (?,?,?,?,?,?,?,?)",
                    (batch_id, run_id, req_hash, "processing", json.dumps(request, ensure_ascii=False), None, now, now),
                )
                self.db.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def set_status(self, batch_id: str, status: str, response: dict | None = None):
        with self.lock:
            self.db.execute(
                "UPDATE batches SET status=?, response=COALESCE(?, response), updated_at=? WHERE batch_id=?",
                (status, json.dumps(response, ensure_ascii=False) if response is not None else None, time.time(), batch_id),
            )
            self.db.commit()

    def fail_stale(self, older_than_s: float = 600):
        with self.lock:
            self.db.execute(
                "UPDATE batches SET status='failed' WHERE status='processing' AND updated_at < ?",
                (time.time() - older_than_s,),
            )
            self.db.commit()

    def purge(self, ttl_days: int):
        cutoff = time.time() - ttl_days * 86400
        with self.lock:
            self.db.execute("DELETE FROM batches WHERE created_at < ?", (cutoff,))
            self.db.commit()

    # ---------------- answers ----------------
    def get_answer(self, k: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT v FROM answers WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_answer(self, k: str, v: dict):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?)", (k, json.dumps(v, ensure_ascii=False), time.time()))
            self.db.commit()
