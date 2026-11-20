"""Gibrid qidiruv: so'rovni tahlil qilish (LLM), BM25 + dense, hujjat/modda ishoralari, RRF, rerank."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .chunking import chunk_header, chunk_lang_available, chunk_text
from .index import Index
from .llm import LLM, LLMError, Usage
from .textnorm import cyr_to_lat, detect_language, fold, normalize_article_ref, sentences, tokenize

log = logging.getLogger("lexrag.retrieval")

RRF_K = 60

ANALYZE_SYSTEM = """You turn a user's question about the legislation of the Republic of Uzbekistan into search queries for a lex.uz retrieval engine (articles of codes, laws, decrees in Uzbek and Russian).
Return ONLY a JSON object with keys:
"legal": true if the question concerns law, rights, duties, liability, taxes, state procedures etc. in Uzbekistan; false only if clearly unrelated to law.
"uz": the question rewritten in formal Uzbek (Latin script, oʻ gʻ) using the terminology of Uzbek codes and laws.
"ru": the same question in formal Russian legal language.
"keywords": 4-8 short legal terms (mix of Uzbek Latin and Russian) likely to appear in the relevant article.
"hyde": 1-2 sentences in Uzbek (Latin) written as the text of the article that would answer it (no article numbers).
"docs": up to 3 names of the most likely acts, e.g. "Mehnat kodeksi", "Fuqarolik kodeksi", "Soliq kodeksi", "Maʼmuriy javobgarlik toʻgʻrisidagi kodeks", "Jinoyat kodeksi", "Oila kodeksi", "Konstitutsiya", or a law title like "Isteʼmolchilarning huquqlarini himoya qilish toʻgʻrisida". [] if unsure.
"articles": up to 3 objects {"doc": "...", "article": "N"} ONLY if you are confident about the exact article number in the CURRENT edition (the Labour Code was replaced in 2023, the Constitution in 2023, the Tax Code in 2020). Otherwise [].
Never answer the question itself."""

RERANK_SYSTEM = """You are a legal retrieval reranker for Uzbekistan legislation (lex.uz).
Given a QUESTION and CANDIDATE fragments (articles), pick the candidates that are needed to answer the question correctly and completely, best first.
Prefer: the specific article that directly regulates the asked point (numbers, terms, conditions, sanctions); acts in force ("amalda") over repealed ones; the general code/law over secondary acts unless the question is about that act.
Include several candidates when the question needs several articles. Exclude irrelevant ones.
Return ONLY JSON: {"ranking": [{"id": "C3", "score": 95}, {"id": "C1", "score": 70}]} with at most 10 items; score 0-100 = usefulness."""

# Savoldagi aniq modda ishorasi: "Mehnat kodeksining 225-moddasi", "статья 15 ТК", "225-модда"
ART_REF_RE = re.compile(r"(\d+(?:-\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*[-–]?\s*(?:modda|модда)|(?:статья|статьи|статье|ст\.)\s*(\d+(?:-\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*)", re.I)
DOC_MENTION_RE = re.compile(
    r"([A-Za-zʻʼ'’Ѐ-ӿ\- ]{0,40}?(?:kodeks\w*|кодекс\w*|konstitutsiya\w*|конституци\w*))|(«[^»]{5,120}»|“[^”]{5,120}”|\"[^\"]{5,120}\")",
    re.I,
)


@dataclass
class Candidate:
    doc_id: str
    art_key: str
    chunk_idx: int
    fused: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float | None = None
    hits: dict = field(default_factory=dict)

    @property
    def akey(self) -> tuple[str, str]:
        return (self.doc_id, self.art_key)


@dataclass
class Analysis:
    legal: bool = True
    uz: str = ""
    ru: str = ""
    keywords: list[str] = field(default_factory=list)
    hyde: str = ""
    docs: list[str] = field(default_factory=list)
    articles: list[dict] = field(default_factory=list)
    ok: bool = False
    ms: int = 0


def analyze_query(llm: LLM, question: str, usage: Usage | None = None, seed: int | None = None) -> Analysis:
    t0 = time.time()
    try:
        j, _ = llm.complete_json(ANALYZE_SYSTEM, f"QUESTION:\n{question}", max_tokens=700, temperature=0.0, seed=seed, usage=usage)
    except (LLMError, ValueError) as e:
        log.warning("query analysis xato: %s", e)
        return Analysis(ms=int((time.time() - t0) * 1000))
    def _s(x):
        return x if isinstance(x, str) else ""
    def _l(x):
        return [str(i) for i in x if isinstance(i, (str, int))] if isinstance(x, list) else []
    arts = []
    for a in j.get("articles") or []:
        if isinstance(a, dict) and a.get("article"):
            arts.append({"doc": str(a.get("doc") or ""), "article": str(a["article"])})
    return Analysis(
        legal=bool(j.get("legal", True)),
        uz=_s(j.get("uz")),
        ru=_s(j.get("ru")),
        keywords=_l(j.get("keywords"))[:10],
        hyde=_s(j.get("hyde")),
        docs=_l(j.get("docs"))[:3],
        articles=arts[:3],
        ok=True,
        ms=int((time.time() - t0) * 1000),
    )


class Retriever:
    def __init__(self, index: Index, embedder, settings):
        self.index = index
        self.embedder = embedder
        self.s = settings

    # ------------------------------------------------------------ yordamchi
    def _to_articles(self, hits: list[tuple[int, float]]) -> list[tuple[tuple[str, str], int, float]]:
        seen = set()
        out = []
        for ci, sc in hits:
            c = self.index.chunks[ci]
            k = (c["doc_id"], c["art_key"])
            if k in seen:
                continue
            seen.add(k)
            out.append((k, ci, sc))
        return out

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        if self.embedder is None or self.index.dense is None or not texts:
            return []
        m = self.embedder.embed(texts, kind="query")
        return [m[i] for i in range(m.shape[0])]

    def doc_mentions(self, question: str) -> list[str]:
        found = []
        for m in DOC_MENTION_RE.finditer(question):
            s = (m.group(1) or m.group(2) or "").strip(" «»“”\"")
            if s:
                found.append(s)
        return found

    # ------------------------------------------------------------ asosiy
    def retrieve(self, question: str, lang: str, analysis: Analysis | None, raw_qvec: np.ndarray | None = None) -> tuple[list[Candidate], dict]:
        idx = self.index
        dbg: dict = {"lists": {}}
        a = analysis or Analysis()
        q_lat = cyr_to_lat(question) if lang == "uz-Cyrl" else question
        K = int(getattr(self.s, "rrf_k", RRF_K))

        lists: list[tuple[str, float, list[tuple[tuple[str, str], int, float]]]] = []

        # --- BM25 (ball vektorlari saqlanadi: keyin hujjat ichida cheklangan qidiruv uchun qayta ishlatiladi) ---
        bm_vectors: list[tuple[str, np.ndarray, np.ndarray]] = []  # (til, ballar, qatorlar)
        uz_variants = []
        if lang != "ru":
            uz_variants.append(("bm25_q", 1.0, q_lat))
        if a.uz:
            uz_variants.append(("bm25_uz", 0.9, a.uz))
        kw_uz = " ".join(k for k in a.keywords if not re.search(r"[\u0400-\u04ff]", k))
        kw_ru = " ".join(k for k in a.keywords if re.search(r"[\u0400-\u04ff]", k))
        if kw_uz:
            uz_variants.append(("bm25_kw", 0.6, kw_uz))
        if a.hyde:
            uz_variants.append(("bm25_hyde", 0.5, a.hyde))
        ru_variants = []
        if lang == "ru":
            ru_variants.append(("bm25_ru_q", 1.0, question))
        if a.ru:
            ru_variants.append(("bm25_ru", 0.7, a.ru))
        if kw_ru:
            ru_variants.append(("bm25_ru_kw", 0.4, kw_ru))
        for bl, variants in (("uz", uz_variants), ("ru", ru_variants)):
            for name, w, text in variants:
                r = idx.bm25_scores(text, bl)
                if r is None:
                    continue
                scores, rows = r
                if name in ("bm25_q", "bm25_uz", "bm25_ru_q", "bm25_ru"):
                    bm_vectors.append((name, scores, rows))
                lists.append((name, w, self._to_articles(idx.top_rows(scores, rows, 100, positive_only=True))))

        # --- dense (barcha so'rov vektorlari bitta matritsa ko'paytmasida) ---
        dense_texts = []
        if a.uz:
            dense_texts.append(("dense_uz", 1.0, a.uz))
        if a.ru:
            dense_texts.append(("dense_ru", 0.7, a.ru))
        if a.hyde:
            dense_texts.append(("dense_hyde", 0.8, a.hyde))
        if raw_qvec is None:
            dense_texts.insert(0, ("dense_q", 1.0, question))
        vecs: list[np.ndarray] = []
        if dense_texts and idx.dense is not None and self.embedder is not None:
            try:
                vecs = self.embed_queries([t for _, _, t in dense_texts])
            except Exception as e:  # embedding xizmati ishlamasa — BM25 bilan davom etamiz
                log.warning("dense embedding xato, faqat BM25: %s", e)
                dbg["dense_error"] = str(e)[:200]
                dense_texts = []
        dense_names = ([("dense_q", 1.0)] if raw_qvec is not None else []) + [(n, w) for n, w, _ in dense_texts[: len(vecs)]]
        qvecs = ([raw_qvec] if raw_qvec is not None else []) + vecs
        S = idx.dense_scores(np.vstack(qvecs)) if qvecs else None
        if S is not None:
            for j, (name, w) in enumerate(dense_names):
                lists.append((name, w, self._to_articles(idx.top_rows(S[:, j], idx.dense_rows, 100))))

        # --- birlashtirish (RRF) ---
        fused: dict[tuple[str, str], float] = {}
        best_chunk: dict[tuple[str, str], tuple[float, int]] = {}
        hits: dict[tuple[str, str], dict] = {}

        def add_list(name: str, w: float, items):
            dbg["lists"][name] = len(items)
            for rank, (k, ci, sc) in enumerate(items):
                contrib = w / (K + rank + 1)
                fused[k] = fused.get(k, 0.0) + contrib
                hits.setdefault(k, {})[name] = rank + 1
                if k not in best_chunk or contrib > best_chunk[k][0]:
                    best_chunk[k] = (contrib, ci)

        for name, w, items in lists:
            add_list(name, w, items)

        # --- hujjat ishoralari (LLM tahlili + savoldagi nomlar) ---
        hint_docs: list[str] = []
        for name in list(a.docs) + self.doc_mentions(question):
            for did in idx.resolve_doc(name):
                if did not in hint_docs:
                    hint_docs.append(did)
        dbg["hint_docs"] = hint_docs[:6]
        if hint_docs:
            hd = set(hint_docs[:4])
            # ishora qilingan hujjat(lar) ichida cheklangan qidiruv: global top-100 ga kirmagan moddalarni ham topadi
            in_doc: dict[tuple[str, str], float] = {}
            in_doc_chunk: dict[tuple[str, str], int] = {}

            def add_restricted(items):
                for rank, (k, ci, sc) in enumerate(self._to_articles(items)):
                    v = 1.0 / (K + rank + 1)
                    in_doc[k] = in_doc.get(k, 0.0) + v
                    in_doc_chunk.setdefault(k, ci)

            for _, scores, rows in bm_vectors:
                add_restricted(idx.top_rows(scores, rows, 40, doc_mask_ids=hd, positive_only=True))
            if S is not None:
                for j in range(min(S.shape[1], 3)):
                    add_restricted(idx.top_rows(S[:, j], idx.dense_rows, 40, doc_mask_ids=hd))
            ranked = sorted(in_doc, key=lambda k: -in_doc[k])[:60]
            add_list("doc_hint", 0.8, [(k, in_doc_chunk[k], in_doc[k]) for k in ranked])

        # --- aniq modda ishoralari ---
        direct: list[tuple[tuple[str, str], int, float]] = []
        explicit_refs = []
        for m in ART_REF_RE.finditer(question):
            explicit_refs.append(m.group(1) or m.group(2))
        if explicit_refs:
            docs_for_ref = hint_docs[:3]
            for ref in explicit_refs:
                for did in docs_for_ref:
                    for ci in idx.article_lookup(did, ref)[:1]:
                        c = idx.chunks[ci]
                        direct.append(((did, c["art_key"]), ci, 1.0))
        if direct:
            add_list("explicit_ref", 2.0, direct)
        guessed = []
        for art in a.articles:
            for did in idx.resolve_doc(art.get("doc", ""), top=2):
                for ci in idx.article_lookup(did, art["article"])[:1]:
                    c = idx.chunks[ci]
                    guessed.append(((did, c["art_key"]), ci, 1.0))
        if guessed:
            add_list("llm_article_guess", 1.0, guessed)
        dbg["direct"] = [f"{k[0]}:{k[1]}" for k, _, _ in direct + guessed]

        # --- holat bo'yicha jarima ---
        for k in list(fused):
            st = idx.docs.get(k[0], {}).get("doc_status")
            if st and st != "amalda":
                fused[k] *= 0.5

        ranked = sorted(fused, key=lambda k: -fused[k])[: self.s.pool_size]
        top = fused[ranked[0]] if ranked else 1.0
        cands = []
        for k in ranked:
            c = Candidate(
                doc_id=k[0],
                art_key=k[1],
                chunk_idx=best_chunk[k][1],
                fused=fused[k],
                retrieval_score=round(fused[k] / top, 4) if top else 0.0,
                hits=hits.get(k, {}),
            )
            cands.append(c)
        dbg["retrieved_total"] = len(fused)
        return cands, dbg


# ------------------------------------------------------------------ rerank
def _snippet(index: Index, cand: Candidate, lang: str, q_tokens: set[str], q_lang: str, max_first: int = 450) -> str:
    c = index.chunks[cand.chunk_idx]
    use = chunk_lang_available(c, lang)
    text = chunk_text(c, use)
    first = text[:max_first]
    best, best_sc = "", 0
    for snt in sentences(text[max_first:])[:60]:
        sc = len(q_tokens & set(tokenize(snt, q_lang)))
        if sc > best_sc:
            best, best_sc = snt, sc
    out = first + ("…" if len(text) > max_first else "")
    if best and best_sc >= 2:
        out += " … " + best[:260]
    return out.replace("\n", " ")


def llm_rerank(llm: LLM, index: Index, question: str, lang: str, cands: list[Candidate], usage: Usage | None = None, seed: int | None = None, n_in: int = 40) -> tuple[list[Candidate], dict]:
    t0 = time.time()
    pool = cands[:n_in]
    q_lang = "ru" if lang == "ru" else "uz"
    q_tokens = set(tokenize(question, q_lang))
    disp_lang = "ru" if lang == "ru" else "uz-Latn"
    lines = []
    for i, c in enumerate(pool, 1):
        d = index.docs[c.doc_id]
        ch = index.chunks[c.chunk_idx]
        head = chunk_header(d, ch, chunk_lang_available(ch, disp_lang))
        st = d.get("doc_status", "amalda")
        lines.append(f"[C{i}] {head} ({d.get('doc_type')}, {st})\n{_snippet(index, c, disp_lang, q_tokens, q_lang)}")
    user = f"QUESTION:\n{question}\n\nCANDIDATES:\n" + "\n\n".join(lines)
    try:
        j, _ = llm.complete_json(RERANK_SYSTEM, user, max_tokens=400, temperature=0.0, seed=seed, usage=usage)
    except (LLMError, ValueError) as e:
        log.warning("rerank xato: %s", e)
        return cands, {"rerank_error": str(e), "rerank_ms": int((time.time() - t0) * 1000)}
    order = []
    seen = set()
    for item in j.get("ranking") or []:
        if isinstance(item, dict):
            cid, sc = str(item.get("id", "")), item.get("score", 50)
        else:
            cid, sc = str(item), 50
        m = re.match(r"C?(\d+)", cid.strip())
        if not m:
            continue
        i = int(m.group(1)) - 1
        if 0 <= i < len(pool) and i not in seen:
            seen.add(i)
            try:
                scf = max(0.0, min(1.0, float(sc) / 100.0))
            except (TypeError, ValueError):
                scf = 0.5
            pool[i].rerank_score = round(scf, 3)
            order.append(pool[i])
    rest = [c for i, c in enumerate(cands) if not (i < len(pool) and i in seen)]
    for c in rest:
        c.rerank_score = 0.0 if c.rerank_score is None else c.rerank_score
    order.sort(key=lambda c: -(c.rerank_score or 0))
    return order + rest, {"rerank_ms": int((time.time() - t0) * 1000), "reranked": len(order)}


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model = CrossEncoder(model_name, max_length=1024)
        self.name = model_name

    def rerank(self, index: Index, question: str, lang: str, cands: list[Candidate], n_in: int = 40) -> tuple[list[Candidate], dict]:
        t0 = time.time()
        pool = cands[:n_in]
        disp = "ru" if lang == "ru" else "uz-Latn"
        pairs = []
        for c in pool:
            ch = index.chunks[c.chunk_idx]
            d = index.docs[c.doc_id]
            l2 = chunk_lang_available(ch, disp)
            pairs.append((question, chunk_header(d, ch, l2) + "\n" + chunk_text(ch, l2)[:2000]))
        scores = self.model.predict(pairs)
        scores = 1 / (1 + np.exp(-np.asarray(scores, dtype=np.float32)))
        for c, s in zip(pool, scores):
            c.rerank_score = round(float(s), 4)
        pool.sort(key=lambda c: -(c.rerank_score or 0))
        rest = cands[n_in:]
        for c in rest:
            c.rerank_score = 0.0
        return pool + rest, {"rerank_ms": int((time.time() - t0) * 1000)}


def detect_lang(question: str, hint: str | None) -> str:
    if hint in ("uz-Latn", "uz-Cyrl", "ru"):
        return hint
    if hint in ("uz", "uz-latn", "uz_Latn"):
        return "uz-Latn"
    if hint in ("uz-cyrl", "uz_Cyrl"):
        return "uz-Cyrl"
    return detect_language(question)


def fold_eq(a: str, b: str) -> bool:
    return fold(a) == fold(b)


__all__ = [
    "Analysis",
    "Candidate",
    "CrossEncoderReranker",
    "Retriever",
    "analyze_query",
    "detect_lang",
    "llm_rerank",
    "normalize_article_ref",
]
