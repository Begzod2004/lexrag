"""lex.uz uchun ehtiyotkor HTTP mijoz: retry, tezlik cheklovi, diskda kesh."""
from __future__ import annotations

import gzip
import logging
import threading
import time
from pathlib import Path

import requests

from .parser import parse_listing

log = logging.getLogger("lexrag.lexuz")

BASE = "https://lex.uz"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36 LexRAG/1.0"


class LexClient:
    def __init__(self, cache_dir: Path, delay_s: float = 0.4, timeout: float = 60.0, use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.delay_s = delay_s
        self.timeout = timeout
        self._local = threading.local()
        self._lock = threading.Lock()
        self._last = 0.0

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": UA, "Accept-Language": "uz,ru;q=0.8,en;q=0.5"})
            self._local.s = s
        return s

    def _throttle(self):
        with self._lock:
            wait = self._last + self.delay_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _cache_path(self, key: str) -> Path:
        safe = key.strip("/").replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "-")
        return self.cache_dir / f"{safe}.html.gz"

    def get(self, path: str, use_cache: bool | None = None, retries: int = 4) -> str:
        cp = self._cache_path(path)
        use_cache = self.use_cache if use_cache is None else use_cache
        if use_cache and cp.exists():
            with gzip.open(cp, "rt", encoding="utf-8") as f:
                return f.read()
        url = path if path.startswith("http") else BASE + path
        last_err: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 200 and r.text:
                    r.encoding = "utf-8"
                    text = r.text
                    with gzip.open(cp, "wt", encoding="utf-8") as f:
                        f.write(text)
                    return text
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                last_err = RuntimeError(f"HTTP {r.status_code} {url}")
            except FileNotFoundError:
                raise
            except Exception as e:  # tarmoq xatosi
                last_err = e
            time.sleep(min(30, 2 ** attempt * 2))
        raise RuntimeError(f"lex.uz dan olinmadi: {url}: {last_err}")

    def listing(self, query_path: str, max_pages: int = 400) -> dict[str, str]:
        """Qidiruv natijalarini (barcha sahifalar) {doc_id: title} ko'rinishida qaytaradi.
        Sahifalash ASP.NET __doPostBack orqali ishlaydi."""
        url = BASE + query_path
        self._throttle()
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        r.encoding = "utf-8"
        page = parse_listing(r.text)
        found: dict[str, str] = dict(page["titles"])
        total = page["total"]
        log.info("listing %s: jami=%s, 1-sahifa=%d", query_path, total, len(page["ids"]))
        n = 1
        while page["next_target"] and n < max_pages:
            data = dict(page["hidden"])
            data["__EVENTTARGET"] = page["next_target"]
            data["__EVENTARGUMENT"] = ""
            self._throttle()
            r = self.session.post(url, data=data, timeout=self.timeout)
            r.raise_for_status()
            r.encoding = "utf-8"
            page = parse_listing(r.text)
            new = {k: v for k, v in page["titles"].items() if k not in found}
            if not new:
                break
            found.update(new)
            n += 1
            if n % 10 == 0:
                log.info("  ... %d sahifa, %d hujjat", n, len(found))
        return found
