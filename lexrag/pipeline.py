"""Savol -> javob: tahlil, qidiruv, rerank, generatsiya, javobni spetsifikatsiya formatiga keltirish."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from datetime import datetime
from zoneinfo import ZoneInfo

from . import PIPELINE_VERSION
from .chunking import _label_for_lang, chunk_lang_available
from .config import Settings, get_settings
from .embeddings import make_embedder
from .generate import NO_ANSWER_TEXT, build_sources, generate_answer
from .index import Index
from .llm import LLM, LLMError, Usage
from .retrieval import Analysis, CrossEncoderReranker, Retriever, analyze_query, detect_lang, llm_rerank
from .textnorm import fold

log = logging.getLogger("lexrag.pipeline")


class Engine:
    def __init__(self, settings: Settings | None = None, index: Index | None = None, embedder=None, llm_answer: LLM | None = None, llm_fast: LLM | None = None, load: bool = True):
        self.s = settings or get_settings()
        self.index = index
        self.index_error: str | None = None
        if self.index is None and load:
            try:
                self.index = Index.load(self.s.index_dir, self.s.dense_dtype)
            except Exception as e:
                self.index_error = f"{type(e).__name__}: {e}"
                log.error("indeks yuklanmadi (%s): %s", self.s.index_dir, e)
        self.embedder = embedder
        if self.embedder is None and self.index is not None and self.index.dense is not None:
            try:
                self.embedder = make_embedder(self.s)
            except Exception as e:
                log.error("embedder yaratilmadi: %s — faqat BM25 ishlaydi", e)
        if self.embedder is not None and self.index is not None and self.index.meta.get("embedding_model") not in (None, self.embedder.name):
            log.warning("Indeks embedding modeli (%s) sozlamadagidan (%s) farq qiladi!", self.index.meta.get("embedding_model"), self.embedder.name)
        self.llm_answer = llm_answer or LLM(self.s.answer_endpoints)
        self.llm_fast = llm_fast or LLM(self.s.fast_endpoints)
        self.retriever = Retriever(self.index, self.embedder, self.s) if self.index is not None else None
        self.cross = None
        if self.s.reranker == "cross_encoder":
            try:
                self.cross = CrossEncoderReranker(self.s.cross_encoder_model)
            except Exception as e:
                log.error("cross-encoder yuklanmadi: %s", e)
        # savollar va ichki sub-vazifalar uchun alohida pullar (ichma-ich kutishda deadlock bo'lmasligi uchun)
        self._pool = ThreadPoolExecutor(max_workers=max(4, self.s.max_parallel_questions * 3), thread_name_prefix="lexrag-q")
        self._sub_pool = ThreadPoolExecutor(max_workers=max(4, self.s.max_parallel_questions * 3), thread_name_prefix="lexrag-sub")

    # ------------------------------------------------------------ meta
    @property
    def reranker_label(self) -> str | None:
        if self.s.reranker == "llm" and self.llm_fast.available:
            return f"{self.llm_fast.label} (LLM listwise)"
        if self.s.reranker == "cross_encoder" and self.cross is not None:
            return self.cross.name
        return None

    def system_info(self) -> dict:
        m = self.index.meta if self.index is not None else {}
        return {
            "pipeline_version": PIPELINE_VERSION,
            "llm_model": self.llm_answer.label,
            "embedding_model": m.get("embedding_model"),
            "reranker": self.reranker_label,
            "query_rewriter": self.llm_fast.label if (self.s.query_rewrite and self.llm_fast.available) else None,
            "index_version": m.get("index_version"),
            "lex_snapshot_date": m.get("lex_snapshot_date"),
        }

    def now(self) -> datetime:
        return datetime.now(ZoneInfo(self.s.timezone))

    # ------------------------------------------------------------ kesh kaliti
    def cache_key(self, question: str, lang_hint: str | None, opts: dict) -> str:
        si = self.system_info()
        # qidiruv sozlamalari ham kalitda: RRF_K, RERANKER va h.k. o'zgarsa eski javob qaytmaydi
        knobs = [self.s.rrf_k, self.s.reranker, self.s.pool_size, self.s.gen_k, self.s.query_rewrite, self.s.adaptive_contexts,
                 self.s.min_contexts, self.s.min_context_score, self.s.doc_id_mode]
        payload = json.dumps(
            [fold(question), lang_hint, opts.get("top_k", 5), opts.get("answer_language", "auto"), opts.get("as_of_date"), opts.get("temperature", 0), opts.get("seed"), si, knobs],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ bitta savol
    def answer(self, qid: str, question: str, lang_hint: str | None, opts: dict) -> dict:
        t0 = time.time()
        usage = Usage()
        timing = {"retrieval_ms": 0, "rerank_ms": 0, "generation_ms": 0, "total_ms": 0}
        top_k = int(opts.get("top_k") or 5)
        seed = opts.get("seed")
        temperature = float(opts.get("temperature") or 0.0)
        lang = detect_lang(question, lang_hint)
        ans_lang = opts.get("answer_language") or "auto"
        if ans_lang not in ("uz-Latn", "uz-Cyrl", "ru"):
            ans_lang = lang
        debug: dict = {}
        res = {
            "id": qid,
            "status": "ok",
            "answer": None,
            "answer_language": ans_lang,
            "no_answer": False,
            "confidence": None,
            "citations": [],
            "contexts": [],
            "debug": debug,
            "timing": timing,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "error": None,
        }
        if self.index is None or self.retriever is None:
            res.update(status="error", error={"code": "INDEX_UNAVAILABLE", "message": self.index_error or "indeks yuklanmagan", "retryable": True})
            return res

        cands = []
        try:
            # 1) so'rov tahlili (LLM) va xom savol embeddingi parallel
            analysis = Analysis()
            fa = None
            if self.s.query_rewrite and self.llm_fast.available:
                fa = self._sub_pool.submit(analyze_query, self.llm_fast, question, usage, seed)
            raw_vec = None
            try:
                vecs = self.retriever.embed_queries([question])
                raw_vec = vecs[0] if vecs else None
            except Exception as e:
                log.warning("query embedding xato: %s", e)
                debug["embedding_error"] = str(e)
            if fa is not None:
                analysis = fa.result()
            # 2) gibrid qidiruv
            cands, rdbg = self.retriever.retrieve(question, lang, analysis, raw_vec)
            timing["retrieval_ms"] = int((time.time() - t0) * 1000)
            debug.update(
                {
                    "detected_language": lang,
                    "rewritten_query": analysis.uz or None,
                    "rewritten_query_ru": analysis.ru or None,
                    "keywords": analysis.keywords,
                    "hyde": analysis.hyde or None,
                    "doc_hints": analysis.docs,
                    "is_legal": analysis.legal,
                    "analysis_ms": analysis.ms,
                    **rdbg,
                }
            )
            if not cands:
                res.update(status="no_answer", no_answer=True, answer=NO_ANSWER_TEXT[ans_lang], confidence=0.9)
                return self._finish(res, t0, usage)
            # 3) rerank
            t1 = time.time()
            if self.s.reranker == "llm" and self.llm_fast.available:
                cands, rr = llm_rerank(self.llm_fast, self.index, question, lang, cands, usage, seed)
                debug.update(rr)
            elif self.s.reranker == "cross_encoder" and self.cross is not None:
                cands, rr = self.cross.rerank(self.index, question, lang, cands)
                debug.update(rr)
            timing["rerank_ms"] = int((time.time() - t1) * 1000)
            debug["top_candidates"] = [
                {"doc_id": c.doc_id, "article": self.index.chunks[c.chunk_idx]["article"], "fused": round(c.fused, 5), "rerank": c.rerank_score, "hits": c.hits}
                for c in cands[:10]
            ]
            # 4) generatsiya
            t2 = time.time()
            sources = build_sources(self.index, cands, ans_lang, self.s.gen_k)
            gen = generate_answer(self.llm_answer, self.index, question, ans_lang, sources, usage, seed, temperature)
            timing["generation_ms"] = int((time.time() - t2) * 1000)
            debug["quote_repairs"] = gen.repaired_quotes
            debug["relevant_sources"] = gen.relevant
        except LLMError as e:
            res.update(status="error", error={"code": e.code, "message": str(e)[:500], "retryable": e.retryable})
            res["contexts"] = self._contexts(cands, [], set(), top_k, ans_lang) if cands else []
            return self._finish(res, t0, usage)
        except Exception as e:  # kutilmagan xato — boshqa savollarga ta'sir qilmasin
            log.exception("savol %s: kutilmagan xato", qid)
            code = "RETRIEVAL_FAILED" if not cands else "LLM_ERROR"
            res.update(status="error", error={"code": code, "message": f"{type(e).__name__}: {e}"[:500], "retryable": True})
            res["contexts"] = self._contexts(cands, [], set(), top_k, ans_lang) if cands else []
            return self._finish(res, t0, usage)

        sid_map = {s.sid: s for s in sources}
        if gen.no_answer:
            res.update(status="no_answer", no_answer=True, answer=gen.answer, confidence=gen.confidence if gen.confidence is not None else 0.8)
            res["contexts"] = self._contexts(cands, [], set(), min(top_k, max(1, self.s.min_contexts)), ans_lang, sid_map)
            return self._finish(res, t0, usage)

        # iqtiboslar
        cited_idx: list[int] = []
        citations = []
        for c in gen.citations:
            src = sid_map[c["sid"]]
            ci = src.cand.chunk_idx
            if ci not in cited_idx:
                cited_idx.append(ci)
            citations.append(self._citation(c, src, ans_lang))
        relevant_idx = [sid_map[s].cand.chunk_idx for s in gen.relevant if s in sid_map]
        order = []
        for ci in relevant_idx + cited_idx:
            if ci not in order:
                order.append(ci)
        res["contexts"] = self._contexts(cands, order, set(cited_idx), top_k, ans_lang, sid_map)
        res["answer"] = gen.answer
        res["citations"] = citations
        conf = gen.confidence
        if conf is None:
            top_rr = max([c.rerank_score or 0 for c in cands[:3]] + [0])
            conf = round(0.5 + 0.4 * top_rr, 3)
        res["confidence"] = round(conf, 3)
        return self._finish(res, t0, usage)

    def _finish(self, res: dict, t0: float, usage: Usage) -> dict:
        res["timing"]["total_ms"] = int((time.time() - t0) * 1000)
        res["usage"] = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
        if usage.models:
            res["debug"]["llm_models"] = usage.models
        return res

    # ------------------------------------------------------------ formatlash
    def _doc_ref(self, doc_id: str, lang: str) -> tuple[str, str, list[str]]:
        d = self.index.docs[doc_id]
        ids = d.get("ids", {})
        alt = sorted({f"lex:{v}" for v in ids.values() if v and v.lstrip("-") != doc_id} | ({f"lex:-{doc_id}"} if "uz-Latn" in ids else set()))
        if self.s.doc_id_mode == "by_language" and lang == "ru" and ids.get("ru"):
            rid = ids["ru"]
            return f"lex:{rid}", f"https://lex.uz/ru/docs/{rid}", alt + [f"lex:{doc_id}"]
        return f"lex:{doc_id}", d.get("url") or f"https://lex.uz/docs/{doc_id}", alt

    def _citation(self, c: dict, src, ans_lang: str) -> dict:
        cand = src.cand
        d = self.index.docs[cand.doc_id]
        ch = self.index.chunks[cand.chunk_idx]
        doc_id, url, alt = self._doc_ref(cand.doc_id, ans_lang)
        title = d["titles"].get(src.lang) or d["titles"].get("uz-Latn") or ""
        return {
            "ref": c["ref"],
            "doc_id": doc_id,
            "doc_title": title,
            "doc_type": d.get("doc_type"),
            "url": url,
            "article": ch["article"],
            "part": f"{c['part']}-qism" if c.get("part") else None,
            "edition_date": ch.get("edition_date") or d.get("edition_date"),
            "article_edition_date": ch.get("article_edition_date"),
            "doc_status": d.get("doc_status", "amalda"),
            "chunk_id": ch["chunk_id"],
            "quote": c["quote"],
            "alt_doc_ids": alt,
        }

    def _context_entry(self, rank: int, cand, used: bool, ans_lang: str, sid_map: dict | None) -> dict:
        ch = self.index.chunks[cand.chunk_idx]
        d = self.index.docs[cand.doc_id]
        src = None
        if sid_map:
            for s in sid_map.values():
                if s.cand.chunk_idx == cand.chunk_idx:
                    src = s
                    break
        if src is not None:
            lang, paras = src.lang, src.paragraphs
        else:
            lang = chunk_lang_available(ch, ans_lang)
            paras = ch["paragraphs"].get(lang) or []
        atitle = ch["titles"].get(lang) or ch["titles"].get("uz-Latn") or ""
        head = f"{_label_for_lang(ch['article'], lang)}. {atitle}".strip()
        doc_id, url, _ = self._doc_ref(cand.doc_id, ans_lang)
        return {
            "rank": rank,
            "chunk_id": ch["chunk_id"],
            "doc_id": doc_id,
            "doc_title": d["titles"].get(lang) or d["titles"].get("uz-Latn"),
            "article": ch["article"],
            "text": head + "\n" + "\n".join(paras),
            "retrieval_score": round(float(cand.retrieval_score), 4),
            "rerank_score": round(float(cand.rerank_score if cand.rerank_score is not None else cand.retrieval_score), 4),
            "used_in_answer": used,
            "url": url,
            "doc_status": d.get("doc_status", "amalda"),
            "edition_date": ch.get("edition_date") or d.get("edition_date"),
            "language": lang,
        }

    def _contexts(self, cands, priority_idx: list[int], cited: set[int], top_k: int, ans_lang: str, sid_map: dict | None = None) -> list[dict]:
        by_idx = {c.chunk_idx: c for c in cands}
        chosen: list = []
        seen = set()
        for ci in priority_idx:
            if ci in by_idx and ci not in seen:
                chosen.append(by_idx[ci])
                seen.add(ci)
        for c in cands:
            if len(chosen) >= top_k:
                break
            if c.chunk_idx in seen:
                continue
            if self.s.adaptive_contexts and len(chosen) >= self.s.min_contexts:
                score = c.rerank_score if c.rerank_score is not None else c.retrieval_score
                if score < self.s.min_context_score:
                    continue
            chosen.append(c)
            seen.add(c.chunk_idx)
        # iqtibos keltirilganlar har doim kiradi (top_k dan oshsa ham)
        if len(chosen) > top_k:
            keep = [c for c in chosen if c.chunk_idx in cited]
            rest = [c for c in chosen if c.chunk_idx not in cited]
            chosen = (keep + rest)[: max(top_k, len(keep))]
            chosen.sort(key=lambda c: priority_idx.index(c.chunk_idx) if c.chunk_idx in priority_idx else 10**6)
        return [self._context_entry(i, c, c.chunk_idx in cited, ans_lang, sid_map) for i, c in enumerate(chosen, 1)]

    # ------------------------------------------------------------ batch
    def run_items(self, items: list[dict], opts: dict, cache=None) -> list[dict]:
        """items: [{'id','question','language'}] -> natijalar (tartib saqlanadi)."""
        futs = {}
        results: dict[str, dict] = {}
        keys = {}
        for it in items:
            key = self.cache_key(it["question"], it.get("language"), opts)
            keys[it["id"]] = key
            if cache is not None and self.s.answer_cache:
                hit = cache.get_answer(key)
                if hit is not None:
                    hit = dict(hit)
                    hit["id"] = it["id"]
                    hit.setdefault("debug", {})["cache"] = "hit"
                    results[it["id"]] = hit
                    continue
            futs[it["id"]] = self._pool.submit(self.answer, it["id"], it["question"], it.get("language"), opts)
        deadline = time.time() + self.s.question_timeout_s
        for qid, f in futs.items():
            try:
                r = f.result(timeout=max(0.1, deadline - time.time()))
            except FutTimeout:
                r = {
                    "id": qid, "status": "timeout", "answer": None, "no_answer": False, "citations": [], "contexts": [],
                    "timing": {"retrieval_ms": 0, "rerank_ms": 0, "generation_ms": 0, "total_ms": int(self.s.question_timeout_s * 1000)},
                    "error": {"code": "LLM_TIMEOUT", "message": f"Savol {self.s.question_timeout_s:.0f}s ichida tugamadi", "retryable": True},
                }
            except Exception as e:  # pragma: no cover
                r = {"id": qid, "status": "error", "answer": None, "no_answer": False, "citations": [], "contexts": [], "timing": {"total_ms": 0},
                     "error": {"code": "LLM_ERROR", "message": str(e)[:300], "retryable": True}}
            results[qid] = r
            if cache is not None and self.s.answer_cache and r.get("status") in ("ok", "no_answer"):
                cache.put_answer(keys[qid], r)
        out = []
        for it in items:
            r = dict(results[it["id"]])
            if not opts.get("return_contexts", True):
                r["contexts"] = []
            if not opts.get("return_debug", False):
                r.pop("debug", None)
            out.append(r)
        return out
