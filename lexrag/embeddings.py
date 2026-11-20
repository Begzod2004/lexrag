"""Embedding provayderlari: openai (yoki OpenAI-mos), sentence-transformers (lokal), hash (offline test)."""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from .textnorm import tokenize

log = logging.getLogger("lexrag.emb")


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class Embedder:
    name = "base"
    dims = 0

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: str | None, base_url: str, dims: int | None = None, batch: int = 64, workers: int = 4, timeout: float = 60.0):
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY / OPENAI_API_KEY berilmagan")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.dims = dims or 0
        self.batch = batch
        self.workers = workers
        self.timeout = timeout
        self.name = f"openai:{model}" + (f"@{dims}" if dims else "")
        self._session = requests.Session()

    def _call(self, inputs: list[str]) -> np.ndarray:
        payload = {"model": self.model, "input": [t[:8000] if t else " " for t in inputs]}
        if self.dims and "text-embedding-3" in self.model:
            payload["dimensions"] = self.dims
        last = None
        for attempt in range(6):
            try:
                r = self._session.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    data = sorted(r.json()["data"], key=lambda d: d["index"])
                    return np.array([d["embedding"] for d in data], dtype=np.float32)
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404):
                    break
            except Exception as e:
                last = str(e)
            time.sleep(min(30, 1.5 * 2 ** attempt))
        raise RuntimeError(f"embedding xatosi: {last}")

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims or 1), dtype=np.float32)
        batches = [texts[i : i + self.batch] for i in range(0, len(texts), self.batch)]
        if len(batches) == 1:
            out = [self._call(batches[0])]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                out = list(ex.map(self._call, batches))
        m = np.vstack(out)
        self.dims = m.shape[1]
        return _normalize(m)


class STEmbedder(Embedder):
    def __init__(self, model: str, query_prefix: str = "", doc_prefix: str = "", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model, device=device)
        self.name = f"st:{model}"
        self.dims = self.model.get_sentence_embedding_dimension()
        self.qp, self.dp = query_prefix, doc_prefix

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        pre = self.qp if kind == "query" else self.dp
        m = self.model.encode([pre + t for t in texts], batch_size=16, normalize_embeddings=True, show_progress_bar=len(texts) > 500)
        return np.asarray(m, dtype=np.float32)


class HashEmbedder(Embedder):
    """Tarmoqsiz ishlaydigan oddiy 'embedding' (testlar uchun). Sifati past — ishlab chiqarishda ishlatmang."""

    def __init__(self, dims: int = 512):
        self.dims = dims
        self.name = f"hash@{dims}"

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        m = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, t in enumerate(texts):
            lang = "ru" if sum(1 for c in t if "Ѐ" <= c <= "ӿ") > len(t) * 0.3 and not any(c in t for c in "ўқғҳ") else "uz"
            for tok in tokenize(t, lang):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                m[i, h % self.dims] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(m)


def make_embedder(settings) -> Embedder | None:
    p = (settings.embedding_provider or "none").lower()
    if p == "none":
        return None
    if p == "openai":
        return OpenAIEmbedder(settings.embedding_model, settings.embedding_api_key, settings.embedding_base_url, settings.embedding_dims or None)
    if p == "st":
        import os

        return STEmbedder(settings.embedding_model, os.environ.get("EMBEDDING_QUERY_PREFIX", ""), os.environ.get("EMBEDDING_DOC_PREFIX", ""))
    if p == "hash":
        return HashEmbedder(settings.embedding_dims or 512)
    raise ValueError(f"Noma'lum EMBEDDING_PROVIDER: {p}")
