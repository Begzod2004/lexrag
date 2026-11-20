"""Javob generatsiyasi: manbalar bilan prompt, JSON javob, iqtiboslarni (quote) manbaga solishtirib tekshirish."""
from __future__ import annotations

import difflib
import logging
import re
import time
from dataclasses import dataclass, field

from .chunking import _label_for_lang, chunk_lang_available
from .index import Index
from .llm import LLM, Usage
from .retrieval import Candidate
from .textnorm import fold, sentences, tokenize

log = logging.getLogger("lexrag.generate")

LANG_NAME = {
    "uz-Latn": "Uzbek in LATIN script (use oʻ, gʻ and ʼ correctly)",
    "uz-Cyrl": "Uzbek in CYRILLIC script (ўзбек кирилл ёзуви)",
    "ru": "Russian",
}

NO_ANSWER_TEXT = {
    "uz-Latn": "Taqdim etilgan qonunchilik manbalarida bu savolga javob topilmadi.",
    "uz-Cyrl": "Тақдим этилган қонунчилик манбаларида бу саволга жавоб топилмади.",
    "ru": "В предоставленных нормативных актах ответ на этот вопрос не найден.",
}

GEN_SYSTEM = """You are LexRAG, a precise legal assistant for the legislation of the Republic of Uzbekistan (official texts from lex.uz).
You get a QUESTION and numbered SOURCES [S1]..[Sn]. Each source is one article (modda) or fragment of an official act; its parts (qism) are marked like «1-qism:».

RULES
1. Use ONLY the SOURCES. Never add facts, numbers, dates or article numbers from memory.
2. Be exact. Copy numbers, terms, percentages, BHM (bazaviy hisoblash miqdori) multiples, day/month counts and conditions exactly as written; keep qualifiers such as "kamida", "koʻpi bilan", "...dan oshmasligi kerak", "ish kunlari"/"kalendar kunlari".
3. Answer first, directly (1-4 sentences). Then add the key conditions/exceptions if they matter. Name the act and article in the text (e.g. "Mehnat kodeksining 232-moddasiga koʻra ...").
4. Put citation markers like [1] or [1][2] after each factual sentence. Each marker refers to an item in "citations".
5. Each citation: "source" = S-number; "part" = part number (integer) where the supporting text is (null if unknown); "quote" = an EXACT verbatim excerpt (8-40 words) copied character-for-character from that source that supports the sentence.
6. If several articles or acts are needed, combine them and cite each one.
7. Rely on acts in force ("amalda") and their current text. Do not rely on a repealed act ("kuchini yoʻqotgan") except to say it is no longer in force.
8. Set "no_answer": true ONLY if the question is not about law, or none of the SOURCES contains the information needed. Then "answer" = one short sentence saying the provided legislation does not contain the answer, and "citations": []. If the sources answer the question even partly, answer it (no_answer=false) — do not refuse because the wording is colloquial.
9. "relevant_sources": the S-numbers of all sources that are actually relevant, most relevant first.
10. "confidence": probability (0..1) that your answer is fully correct.
11. Write "answer" in {LANG}.

Return ONLY this JSON object:
{"no_answer": false, "answer": "... [1]", "citations": [{"ref": 1, "source": "S2", "part": 1, "quote": "..."}], "relevant_sources": ["S2", "S4"], "confidence": 0.9}"""


@dataclass
class Source:
    sid: str
    cand: Candidate
    lang: str  # matn tili
    paragraphs: list[str]
    parts: list[int]
    header: str

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


@dataclass
class GenOutput:
    no_answer: bool
    answer: str
    citations: list[dict]
    relevant: list[str]
    confidence: float | None
    raw: dict = field(default_factory=dict)
    ms: int = 0
    repaired_quotes: int = 0


def build_sources(index: Index, cands: list[Candidate], lang: str, k: int, max_chars: int = 5000) -> list[Source]:
    out = []
    for i, c in enumerate(cands[:k], 1):
        ch = index.chunks[c.chunk_idx]
        d = index.docs[c.doc_id]
        use = chunk_lang_available(ch, lang)
        paras = ch["paragraphs"].get(use) or []
        parts = ch["parts"].get(use) or [1] * len(paras)
        # uzun matnni qisqartirish
        total, keep = 0, 0
        for p in paras:
            total += len(p) + 1
            if total > max_chars and keep > 0:
                break
            keep += 1
        title = d["titles"].get(use) or d["titles"].get("uz-Latn") or ""
        atitle = ch["titles"].get(use) or ch["titles"].get("uz-Latn") or ""
        label = _label_for_lang(ch["article"], use)
        header = f"{title} | {label}. {atitle}".strip()
        out.append(Source(f"S{i}", c, use, paras[:keep], parts[:keep], header))
    return out


def format_sources(index: Index, sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        d = index.docs[s.cand.doc_id]
        ch = index.chunks[s.cand.chunk_idx]
        meta = f"turi: {d.get('doc_type')}; holati: {d.get('doc_status', 'amalda')}; tahrir: {ch.get('edition_date') or d.get('edition_date') or '-'}"
        lines = [f"[{s.sid}] {s.header} ({meta})"]
        prev = None
        for p, part in zip(s.paragraphs, s.parts):
            if part != prev:
                lines.append(f"«{part}-qism:» {p}")
                prev = part
            else:
                lines.append(p)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_REF_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def _normalize_markers(answer: str) -> str:
    return _REF_GROUP.sub(lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,;]", m.group(1))), answer)


_DROP = set("'`‘’‛ʹʻʼʽ′´«»“”„\"‎‏­")
_SUPS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _cnorm(text: str) -> tuple[str, list[int]]:
    """Belgi darajasida normallashtirish + asl indekslar xaritasi (bo'shliqlar siqiladi)."""
    buf: list[str] = []
    mp: list[int] = []
    for i, ch in enumerate(text):
        if ch in _DROP:
            continue
        if ch.isspace() or ch == " ":
            if buf and buf[-1] == " ":
                continue
            buf.append(" ")
            mp.append(i)
            continue
        c = ch.lower().translate(_SUPS)
        if c == "ё":
            c = "е"
        if c in "—–":
            c = "-"
        buf.append(c)
        mp.append(i)
    return "".join(buf), mp


def _find_verbatim(quote: str, text: str) -> str | None:
    """Matnda quote'ni (apostrof/bo'shliq/registr farqlarini e'tiborsiz) topib, ASL matndagi bo'lakni qaytaradi."""
    if not quote or not text:
        return None
    if quote in text:
        return quote
    nq, _ = _cnorm(quote)
    nq = nq.strip(" .…")
    if len(nq) < 8:
        return None
    nt, mp = _cnorm(text)
    pos = nt.find(nq)
    if pos == -1:
        return None
    start = mp[pos]
    end = mp[pos + len(nq) - 1] + 1
    return text[start:end]


def _best_sentence(claim: str, text: str, lang: str) -> str:
    sents = sentences(text) or [text]
    tl = "ru" if lang == "ru" else "uz"
    ct = set(tokenize(claim, tl))
    best, best_sc = sents[0], -1.0
    for s in sents:
        st = set(tokenize(s, tl))
        if not st:
            continue
        sc = len(ct & st) / (len(st) ** 0.5)
        sc += 0.3 * difflib.SequenceMatcher(None, fold(claim)[:300], fold(s)[:300]).ratio()
        if sc > best_sc:
            best, best_sc = s, sc
    return best[:400]


def _part_of(quote: str, src: Source) -> int | None:
    fq = fold(quote)[:60]
    for p, part in zip(src.paragraphs, src.parts):
        if fq and fq in fold(p):
            return part
    return None


def generate_answer(llm: LLM, index: Index, question: str, lang: str, sources: list[Source], usage: Usage | None = None, seed: int | None = None, temperature: float = 0.0) -> GenOutput:
    t0 = time.time()
    system = GEN_SYSTEM.replace("{LANG}", LANG_NAME.get(lang, LANG_NAME["uz-Latn"]))
    user = f"QUESTION:\n{question}\n\nSOURCES:\n{format_sources(index, sources)}"
    j, _ = llm.complete_json(system, user, max_tokens=1400, temperature=temperature, seed=seed, usage=usage)
    return postprocess(j, sources, lang, int((time.time() - t0) * 1000))


def postprocess(j: dict, sources: list[Source], lang: str, ms: int = 0) -> GenOutput:
    by_sid = {s.sid: s for s in sources}
    no_answer = bool(j.get("no_answer", False))
    answer = str(j.get("answer") or "").strip()
    conf = j.get("confidence")
    try:
        conf = None if conf is None else max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    relevant = []
    for r in j.get("relevant_sources") or []:
        sid = f"S{re.sub(r'[^0-9]', '', str(r))}"
        if sid in by_sid and sid not in relevant:
            relevant.append(sid)

    if no_answer:
        return GenOutput(True, answer or NO_ANSWER_TEXT.get(lang, NO_ANSWER_TEXT["uz-Latn"]), [], relevant, conf, j, ms)

    answer = _normalize_markers(answer)
    raw_cits = j.get("citations") or []
    cits: dict[int, dict] = {}
    repaired = 0
    for rc in raw_cits:
        if not isinstance(rc, dict):
            continue
        try:
            ref = int(re.sub(r"[^0-9]", "", str(rc.get("ref", ""))) or 0)
        except ValueError:
            continue
        sid = f"S{re.sub(r'[^0-9]', '', str(rc.get('source', '')))}"
        src = by_sid.get(sid)
        if not ref or src is None:
            continue
        quote = str(rc.get("quote") or "").strip().strip("«»\"“”")
        quote = re.sub(r"^\s*«?\s*\d+\s*-\s*qism\s*:?\s*»?\s*", "", quote)  # prompt markerini olib tashlash
        quote = quote.strip(" .…").strip("«»\"“”")
        verb = _find_verbatim(quote, src.text)
        if verb is None:
            # LLM so'zma-so'z ko'chirmagan — manbadagi eng mos jumlani olamiz
            claim = quote or _sentence_with_ref(answer, ref)
            verb = _best_sentence(claim, src.text, lang)
            repaired += 1
        part = _part_of(verb, src)
        if part is None:
            try:
                part = int(rc.get("part")) if rc.get("part") not in (None, "") else None
            except (TypeError, ValueError):
                part = None
        if ref not in cits:
            cits[ref] = {"ref": ref, "sid": sid, "quote": verb, "part": part}

    # javobdagi markerlar bo'yicha tartiblash va qayta raqamlash
    used_refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    order = []
    for r in used_refs:
        if r in cits and r not in order:
            order.append(r)
    for r in sorted(cits):
        if r not in order:
            order.append(r)
    renum = {old: new for new, old in enumerate(order, 1)}

    def _sub(m):
        r = int(m.group(1))
        return f"[{renum[r]}]" if r in renum else ""

    answer = re.sub(r"\[(\d+)\]", _sub, answer)
    answer = re.sub(r"\s+([.,;:])", r"\1", answer).strip()
    citations = []
    for old in order:
        c = cits[old]
        citations.append({**c, "ref": renum[old]})

    if not citations and sources:
        # javob bor, lekin iqtibos yo'q — eng mos manbani biriktiramiz
        src = by_sid.get(relevant[0]) if relevant else sources[0]
        verb = _best_sentence(answer, src.text, lang)
        citations = [{"ref": 1, "sid": src.sid, "quote": verb, "part": _part_of(verb, src)}]
        if "[1]" not in answer:
            answer = answer.rstrip() + " [1]"
        repaired += 1
    for c in citations:
        if c["sid"] not in relevant:
            relevant.append(c["sid"])
    return GenOutput(False, answer, citations, relevant, conf, j, ms, repaired)


def _sentence_with_ref(answer: str, ref: int) -> str:
    for s in re.split(r"(?<=[.!?])\s+", answer):
        if f"[{ref}]" in s:
            return re.sub(r"\[\d+\]", "", s)
    return re.sub(r"\[\d+\]", "", answer)
