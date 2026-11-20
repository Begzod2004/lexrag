"""Indeks: BM25 (uz/ru) + dense embedding + hujjat/modda katalogi. Qurish va yuklash.

  python -m lexrag.index build            # data/docs -> data/index/current
  python -m lexrag.index stats
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy import sparse

from . import PIPELINE_VERSION
from .chunking import chunk_text, doc_chunks, index_text
from .config import Settings, get_settings
from .embeddings import Embedder, make_embedder
from .textnorm import fold, normalize_article_ref, tokenize

log = logging.getLogger("lexrag.index")

LANG_CODES = {"uz": 0, "ru": 1}


# ------------------------------------------------------------------ BM25
class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.W: sparse.csc_matrix | None = None
        self.rows: np.ndarray | None = None  # qator -> chunk indeksi

    def fit(self, token_lists: list[list[str]], rows: list[int]):
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for toks in token_lists:
            c = Counter(toks)
            for t, n in c.items():
                j = self.vocab.setdefault(t, len(self.vocab))
                indices.append(j)
                data.append(n)
            indptr.append(len(indices))
        N = len(token_lists)
        tf = sparse.csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)), shape=(N, max(1, len(self.vocab))))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) if N else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        # W = tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl))
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = tf.data * (self.k1 + 1) / denom
        self.W = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.rows = np.array(rows, dtype=np.int32)
        return self

    def search(self, tokens: list[str], n: int = 100) -> list[tuple[int, float]]:
        if self.W is None or not tokens:
            return []
        qc = Counter(t for t in tokens if t in self.vocab)
        if not qc:
            return []
        ids = np.array([self.vocab[t] for t in qc], dtype=np.int64)
        qw = np.array([self.idf[self.vocab[t]] * (1 + 0.5 * (c - 1)) for t, c in qc.items()], dtype=np.float32)
        scores = self.W[:, ids] @ qw
        scores = np.asarray(scores).ravel()
        if not scores.any():
            return []
        k = min(n, int((scores > 0).sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.rows[i]), float(scores[i])) for i in top]

    def save(self, path: Path):
        W = self.W.tocsc()
        np.savez_compressed(path, data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape), idf=self.idf, rows=self.rows)
        (path.with_suffix(".vocab.json")).write_text(json.dumps(self.vocab, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25":
        z = np.load(path)
        o = cls()
        o.W = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        o.idf = z["idf"]
        o.rows = z["rows"]
        o.vocab = json.loads(path.with_suffix(".vocab.json").read_text("utf-8"))
        return o


# ------------------------------------------------------------------ embedding kesh
class EmbCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS e (h TEXT PRIMARY KEY, v BLOB)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha1((model + "\x00" + text).encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(keys), 900):
            part = keys[i : i + 900]
            q = "SELECT h, v FROM e WHERE h IN (%s)" % ",".join("?" * len(part))
            for h, v in self.db.execute(q, part):
                out[h] = np.frombuffer(v, dtype=np.float16).astype(np.float32)
        return out

    def put_many(self, items: list[tuple[str, np.ndarray]]):
        self.db.executemany("INSERT OR REPLACE INTO e VALUES (?, ?)", [(h, v.astype(np.float16).tobytes()) for h, v in items])
        self.db.commit()


def embed_with_cache(embedder: Embedder, texts: list[str], cache: EmbCache, batch: int = 512) -> np.ndarray:
    keys = [EmbCache.key(embedder.name, t) for t in texts]
    have = cache.get_many(keys)
    missing = [i for i, k in enumerate(keys) if k not in have]
    log.info("embedding: %d matn, keshda %d, yangi %d", len(texts), len(texts) - len(missing), len(missing))
    for s in range(0, len(missing), batch):
        idx = missing[s : s + batch]
        vecs = embedder.embed([texts[i] for i in idx], kind="doc")
        cache.put_many([(keys[i], vecs[j]) for j, i in enumerate(idx)])
        for j, i in enumerate(idx):
            have[keys[i]] = vecs[j]
        log.info("  embedding %d/%d", min(s + batch, len(missing)), len(missing))
    dims = len(next(iter(have.values()))) if have else (embedder.dims or 1)
    m = np.zeros((len(texts), dims), dtype=np.float32)
    for i, k in enumerate(keys):
        m[i] = have[k]
    return m


# ------------------------------------------------------------------ Index
class Index:
    def __init__(self):
        self.meta: dict = {}
        self.docs: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.chunk_pos: dict[str, int] = {}
        self.article_chunks: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.bm25: dict[str, BM25] = {}
        self.dense: np.ndarray | None = None
        self.dense_rows: np.ndarray | None = None
        self.dense_langs: np.ndarray | None = None
        self.title_keys: list[tuple[str, set[str], str]] = []  # (doc_id, token set, folded title)

    # ---------- qurish ----------
    @classmethod
    def build(cls, docs: list[dict], settings: Settings, embedder: Embedder | None, out_dir: Path) -> "Index":
        t0 = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = cls()
        for d in docs:
            meta = {k: v for k, v in d.items() if k != "articles"}
            meta["n_articles"] = len(d["articles"])
            idx.docs[d["doc_id"]] = meta
            for c in doc_chunks(d):
                idx.chunks.append(c)
        idx._post_load()
        log.info("chunklar: %d (%d hujjat)", len(idx.chunks), len(idx.docs))

        uz_tokens, uz_rows, ru_tokens, ru_rows = [], [], [], []
        uz_texts, ru_texts = [], []
        for i, c in enumerate(idx.chunks):
            d = idx.docs[c["doc_id"]]
            t_uz = index_text(d, c, "uz-Latn")
            uz_tokens.append(tokenize(t_uz, "uz"))
            uz_rows.append(i)
            uz_texts.append(t_uz)
            if c["paragraphs"].get("ru"):
                t_ru = index_text(d, c, "ru")
                ru_tokens.append(tokenize(t_ru, "ru"))
                ru_rows.append(i)
                ru_texts.append(t_ru)
        idx.bm25["uz"] = BM25().fit(uz_tokens, uz_rows)
        idx.bm25["ru"] = BM25().fit(ru_tokens, ru_rows)
        idx.bm25["uz"].save(out_dir / "bm25_uz.npz")
        idx.bm25["ru"].save(out_dir / "bm25_ru.npz")
        log.info("BM25 tayyor (uz=%d, ru=%d qator)", len(uz_rows), len(ru_rows))

        emb_name = None
        if embedder is not None:
            cache = EmbCache(settings.data_dir / "emb_cache.sqlite3")
            texts = uz_texts + ru_texts
            mat = embed_with_cache(embedder, texts, cache)
            idx.dense = mat.astype(np.float16)
            idx.dense_rows = np.array(uz_rows + ru_rows, dtype=np.int32)
            idx.dense_langs = np.array([0] * len(uz_rows) + [1] * len(ru_rows), dtype=np.int8)
            np.save(out_dir / "dense.npy", idx.dense)
            np.save(out_dir / "dense_rows.npy", idx.dense_rows)
            np.save(out_dir / "dense_langs.npy", idx.dense_langs)
            emb_name = embedder.name

        with gzip.open(out_dir / "chunks.jsonl.gz", "wt", encoding="utf-8") as f:
            for c in idx.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (out_dir / "docs.json").write_text(json.dumps(idx.docs, ensure_ascii=False), "utf-8")
        snap = max((d.get("fetched_at") or "" for d in idx.docs.values()), default="") or date.today().isoformat()
        h = hashlib.sha1((",".join(sorted(idx.docs)) + str(emb_name)).encode()).hexdigest()[:6]
        idx.meta = {
            "index_version": f"lex-idx-{date.today().isoformat()}-{h}",
            "lex_snapshot_date": snap,
            "embedding_model": emb_name,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline_version": PIPELINE_VERSION,
            "n_docs": len(idx.docs),
            "n_chunks": len(idx.chunks),
            "n_articles": len(idx.article_chunks),
            "doc_types": dict(Counter(d.get("doc_type") for d in idx.docs.values())),
            "languages": {"uz-Latn": len(uz_rows), "ru": len(ru_rows), "uz-Cyrl": sum(1 for c in idx.chunks if c["paragraphs"].get("uz-Cyrl"))},
        }
        (out_dir / "meta.json").write_text(json.dumps(idx.meta, ensure_ascii=False, indent=1), "utf-8")
        log.info("indeks tayyor: %s (%.1fs)", out_dir, time.time() - t0)
        return idx

    # ---------- yuklash ----------
    @classmethod
    def load(cls, path: Path, dense_dtype: str = "float32") -> "Index":
        idx = cls()
        idx.meta = json.loads((path / "meta.json").read_text("utf-8"))
        idx.docs = json.loads((path / "docs.json").read_text("utf-8"))
        with gzip.open(path / "chunks.jsonl.gz", "rt", encoding="utf-8") as f:
            idx.chunks = [json.loads(l) for l in f]
        for lang in ("uz", "ru"):
            p = path / f"bm25_{lang}.npz"
            if p.exists():
                idx.bm25[lang] = BM25.load(p)
        if (path / "dense.npy").exists():
            idx.dense = np.load(path / "dense.npy", mmap_mode=None)
            if dense_dtype == "float32":
                idx.dense = idx.dense.astype(np.float32)  # tezroq (xotira 2x)
            idx.dense_rows = np.load(path / "dense_rows.npy")
            idx.dense_langs = np.load(path / "dense_langs.npy")
        idx._post_load()
        return idx

    def _post_load(self):
        self.chunk_pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.doc_index = {d: i for i, d in enumerate(self.docs)}
        self.chunk_doc = np.array([self.doc_index.get(c["doc_id"], -1) for c in self.chunks], dtype=np.int32)
        self.article_chunks = defaultdict(list)
        for i, c in enumerate(self.chunks):
            self.article_chunks[(c["doc_id"], c["art_key"])].append(i)
        self.title_keys = []
        for did, d in self.docs.items():
            for lang in ("uz-Latn", "ru", "uz-Cyrl"):
                t = d.get("titles", {}).get(lang)
                if t:
                    ft = fold(t if lang != "uz-Cyrl" else t)
                    self.title_keys.append((did, set(_title_tokens(ft)), ft))

    # ---------- qidiruv ----------
    def bm25_search(self, text: str, lang: str, n: int = 100) -> list[tuple[int, float]]:
        bm = self.bm25.get(lang)
        if bm is None:
            return []
        return bm.search(tokenize(text, lang), n)

    def bm25_scores(self, text: str, lang: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(ballar, qator->chunk) — keyin top_rows() bilan global yoki hujjat bo'yicha cheklangan top olinadi."""
        bm = self.bm25.get(lang)
        if bm is None or bm.W is None:
            return None
        qc = Counter(t for t in tokenize(text, lang) if t in bm.vocab)
        if not qc:
            return None
        ids = np.array([bm.vocab[t] for t in qc], dtype=np.int64)
        qw = np.array([bm.idf[bm.vocab[t]] * (1 + 0.5 * (c - 1)) for t, c in qc.items()], dtype=np.float32)
        return np.asarray(bm.W[:, ids] @ qw).ravel(), bm.rows

    def dense_scores(self, qmat: np.ndarray, block: int = 16384) -> np.ndarray | None:
        """qmat (m x d) -> ballar (N x m). Bitta matritsa ko'paytmasi barcha so'rovlar uchun."""
        if self.dense is None or qmat is None or len(qmat) == 0:
            return None
        q = np.asarray(qmat, dtype=np.float32).T
        if self.dense.dtype == np.float32:
            return self.dense @ q
        N = self.dense.shape[0]
        out = np.empty((N, q.shape[1]), dtype=np.float32)
        for s in range(0, N, block):
            out[s : s + block] = self.dense[s : s + block].astype(np.float32) @ q
        return out

    def top_rows(self, scores: np.ndarray, rows: np.ndarray, n: int, doc_mask_ids: set[str] | None = None, positive_only: bool = False) -> list[tuple[int, float]]:
        """Qator ballaridan chunk bo'yicha (max) top-n. doc_mask_ids berilsa — faqat shu hujjatlar.
        positive_only=True — BM25 uchun (0 ball = moslik yo'q)."""
        sc = scores
        if doc_mask_ids is not None:
            didx = [self.doc_index[d] for d in doc_mask_ids if d in self.doc_index]
            if not didx:
                return []
            mask = np.isin(self.chunk_doc[rows], np.array(didx, dtype=np.int32))
            sc = np.where(mask, scores, -np.inf)
        valid = int(np.isfinite(sc).sum()) if doc_mask_ids is not None else len(sc)
        k = min(n * 2, valid)
        if k <= 0:
            return []
        top = np.argpartition(-sc, k - 1)[:k]
        top = top[np.argsort(-sc[top])]
        best: dict[int, float] = {}
        for r in top:
            v = float(sc[r])
            if not np.isfinite(v) or (positive_only and v <= 0):
                continue
            ci = int(rows[r])
            if ci not in best:
                best[ci] = v
            if len(best) >= n:
                break
        return list(best.items())

    def dense_search(self, qvec: np.ndarray, n: int = 100) -> list[tuple[int, float]]:
        s = self.dense_scores(np.asarray(qvec)[None, :])
        if s is None:
            return []
        return self.top_rows(s[:, 0], self.dense_rows, n)

    def article_lookup(self, doc_id: str, article_ref: str) -> list[int]:
        key = normalize_article_ref(article_ref)
        if not key:
            return []
        return list(self.article_chunks.get((doc_id, key), []))

    def resolve_doc(self, name: str, top: int = 2, min_score: float = 0.5) -> list[str]:
        """'Mehnat kodeksi', 'Трудовой кодекс', 'Iste'molchilar huquqlarini himoya qilish' -> doc_id lar."""
        if not name:
            return []
        fn = fold(name)
        for alias, target in DOC_ALIASES.items():
            if re.search(rf"(^|\W){re.escape(alias)}($|\W)", fn):
                fn = target
                break
        q = set(_title_tokens(fn))
        if not q:
            return []
        scored: dict[str, float] = {}
        for did, toks, ft in self.title_keys:
            if not toks:
                continue
            inter = len(q & toks)
            if not inter:
                continue
            sc = inter / len(q) * 0.75 + inter / len(toks) * 0.25
            st = self.docs[did].get("doc_status")
            if st and st != "amalda":
                sc -= 0.15
            if sc > scored.get(did, 0):
                scored[did] = sc
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        return [d for d, sc in ranked[:top] if sc >= min_score]


_TITLE_STOP = {"ozbekiston", "respublikasining", "respublikasi", "togrisida", "togrisidagi", "qonuni", "haqida", "республики", "узбекистан", "узбекистана", "о", "об", "закон",
               "ўзбекистон", "республикасининг", "тўғрисида", "қонуни", "ning", "va", "и", "gi", "toʻgʻrisida"}


def _title_tokens(ft: str) -> list[str]:
    from .textnorm import cyr_to_lat, script_of

    if script_of(ft) == "cyr" and re.search(r"[ўқғҳ]", ft):
        ft = fold(cyr_to_lat(ft))
    lang = "ru" if script_of(ft) == "cyr" else "uz"
    return [t for t in tokenize(ft, lang) if t not in _TITLE_STOP and len(t) > 1]


DOC_ALIASES = {
    # qisqartmalar
    "mk": "mehnat kodeksi", "tk": "трудовой кодекс", "тк": "трудовой кодекс", "тк ру": "трудовой кодекс",
    "jk": "jinoyat kodeksi", "ук": "уголовный кодекс", "фк": "fuqarolik kodeksi", "гк": "гражданский кодекс", "fk": "fuqarolik kodeksi",
    "sk": "soliq kodeksi", "нк": "налоговый кодекс", "mjtk": "mamuriy javobgarlik togrisidagi kodeksi", "mjk": "mamuriy javobgarlik togrisidagi kodeksi",
    "коап": "кодекс об административной ответственности", "jpk": "jinoyat-protsessual kodeksi", "упк": "уголовно-процессуальный кодекс",
    "fpk": "fuqarolik protsessual kodeksi", "гпк": "гражданский процессуальный кодекс", "ok": "oila kodeksi", "ск": "семейный кодекс",
}


def load_index(settings: Settings | None = None) -> Index:
    s = settings or get_settings()
    return Index.load(s.index_dir, s.dense_dtype)


def main(argv=None):
    from .ingest import load_docs

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats"])
    ap.add_argument("--no-embed", action="store_true", help="dense embeddingsiz (faqat BM25)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    if args.cmd == "build":
        docs = list(load_docs(s.docs_dir))
        if not docs:
            raise SystemExit(f"{s.docs_dir} bo'sh. Avval: python -m lexrag.ingest all")
        emb = None if args.no_embed else make_embedder(s)
        Index.build(docs, s, emb, s.index_dir)
    else:
        idx = load_index(s)
        print(json.dumps(idx.meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
