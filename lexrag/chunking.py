"""Hujjatlarni modda (va zarur bo'lsa qism guruhi) darajasidagi bo'laklarga ajratish."""
from __future__ import annotations

from .lexuz.parser import split_parts

LANGS = ("uz-Latn", "uz-Cyrl", "ru")
MAX_CHUNK_CHARS = 6000  # bundan uzun moddalar qismlarga bo'linadi
TARGET_CHUNK_CHARS = 3500

ART_WORD = {"uz-Latn": "modda", "uz-Cyrl": "модда", "ru": "Статья"}


def _label_for_lang(label: str, lang: str) -> str:
    """'57¹-modda' -> 'Статья 57¹' (ru) / '57¹-модда' (kirill)."""
    if label.endswith("-modda"):
        num = label[: -len("-modda")]
        if lang == "ru":
            return f"Статья {num}"
        if lang == "uz-Cyrl":
            return f"{num}-модда"
    if lang == "ru" and label.endswith("-band"):
        return f"Пункт {label[:-5]}"
    if lang == "uz-Cyrl" and label.endswith("-band"):
        return f"{label[:-5]}-банд"
    return label


def _chunk_prefix(doc: dict, key: str) -> str:
    if key == "preamble":
        return f"lex:{doc['doc_id']}#preamble"
    if key.startswith("ilova") or not doc.get("has_articles", True):
        return f"lex:{doc['doc_id']}#band-{key}"
    return f"lex:{doc['doc_id']}#art-{key}"


def _groups_by_parts(paras: list[str], parts: list[int]) -> list[tuple[int, int, int, int]]:
    """(part_from, part_to, para_from, para_to_exclusive) guruhlari, har biri ~TARGET_CHUNK_CHARS."""
    groups = []
    start = 0
    size = 0
    for i, p in enumerate(paras):
        new_part = i > 0 and parts[i] != parts[i - 1]
        if size >= TARGET_CHUNK_CHARS and i > start and (new_part or size >= MAX_CHUNK_CHARS):
            groups.append((parts[start], parts[i - 1], start, i))
            start, size = i, 0
        size += len(p) + 1
    groups.append((parts[start], parts[-1], start, len(paras)))
    return groups


def article_label(doc: dict, art: dict) -> str:
    return art["label"]


def doc_chunks(doc: dict) -> list[dict]:
    chunks: list[dict] = []
    for art in doc["articles"]:
        paras_by_lang = {l: art["paragraphs"].get(l) or [] for l in LANGS}
        base_lang = "uz-Latn" if paras_by_lang["uz-Latn"] else next((l for l in LANGS if paras_by_lang[l]), None)
        if base_lang is None:
            continue
        base = paras_by_lang[base_lang]
        parts_by_lang = {l: split_parts(p) for l, p in paras_by_lang.items() if p}
        total = sum(len(p) + 1 for p in base)
        prefix = _chunk_prefix(doc, art["key"])
        if total <= MAX_CHUNK_CHARS:
            groups = [(None, None, 0, len(base))]
        else:
            groups = _groups_by_parts(base, parts_by_lang[base_lang])
        for (pf, pt, a, b) in groups:
            texts: dict[str, list[str]] = {}
            parts: dict[str, list[int]] = {}
            for l, paras in paras_by_lang.items():
                if not paras:
                    continue
                if pf is None:
                    sel = list(range(len(paras)))
                else:
                    lp = parts_by_lang[l]
                    sel = [i for i, x in enumerate(lp) if pf <= x <= pt]
                    if l != base_lang and (not sel or parts_by_lang[base_lang][-1] != lp[-1]):
                        # qism soni mos kelmasa — proporsional kesim
                        n = len(paras)
                        lo = int(a * n / len(base))
                        hi = max(lo + 1, int(round(b * n / len(base))))
                        sel = list(range(lo, min(hi, n)))
                    elif l == base_lang:
                        sel = list(range(a, b))
                texts[l] = [paras[i] for i in sel]
                parts[l] = [parts_by_lang[l][i] for i in sel]
            if pf is None:
                cid = prefix
            else:
                cid = f"{prefix}#p{pf}" if pf == pt else f"{prefix}#p{pf}-{pt}"
                if any(c["chunk_id"] == cid for c in chunks[-6:]):
                    cid = f"{cid}.{a}"
            chunks.append(
                {
                    "chunk_id": cid,
                    "doc_id": doc["doc_id"],
                    "art_key": art["key"],
                    "article": art["label"],
                    "part_from": pf,
                    "part_to": pt,
                    "titles": art.get("titles", {}),
                    "headings": art.get("headings", {}),
                    "paragraphs": texts,
                    "parts": parts,
                    # hujjatning amaldagi (konsolidatsiya qilingan) tahriri sanasi
                    "edition_date": doc.get("edition_date") or art.get("edition_date"),
                    # moddaning o'ziga oxirgi o'zgartish kiritilgan sana (bo'lsa)
                    "article_edition_date": art.get("edition_date"),
                    "changed": bool(art.get("changed")),
                }
            )
    return chunks


def chunk_header(doc: dict, chunk: dict, lang: str) -> str:
    title = doc["titles"].get(lang) or doc["titles"].get("uz-Latn") or next(iter(doc["titles"].values()), "")
    heads = chunk["headings"].get(lang) or chunk["headings"].get("uz-Latn") or []
    atitle = chunk["titles"].get(lang) or chunk["titles"].get("uz-Latn") or ""
    lab = _label_for_lang(chunk["article"], lang)
    h = " / ".join(heads[-2:])
    return f"{title} | {h} | {lab}. {atitle}".replace("|  |", "|").strip(" |")


def chunk_text(chunk: dict, lang: str, fallback: bool = True) -> str:
    paras = chunk["paragraphs"].get(lang)
    if not paras and fallback:
        for l in ("uz-Latn", "uz-Cyrl", "ru"):
            if chunk["paragraphs"].get(l):
                paras = chunk["paragraphs"][l]
                break
    return "\n".join(paras or [])


def chunk_lang_available(chunk: dict, lang: str) -> str:
    """Berilgan tildagi matn bo'lmasa, mavjud eng yaqin tilni qaytaradi."""
    if chunk["paragraphs"].get(lang):
        return lang
    order = {"uz-Cyrl": ["uz-Latn", "ru"], "uz-Latn": ["uz-Cyrl", "ru"], "ru": ["uz-Latn", "uz-Cyrl"]}[lang]
    for l in order:
        if chunk["paragraphs"].get(l):
            return l
    return lang


def index_text(doc: dict, chunk: dict, lang: str, max_chars: int = 7000) -> str:
    return (chunk_header(doc, chunk, lang) + "\n" + chunk_text(chunk, lang, fallback=False))[:max_chars]
