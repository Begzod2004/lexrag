"""lex.uz dan korpusni yig'ish.

Misollar:
  python -m lexrag.ingest catalog --forms constitution,codes,laws
  python -m lexrag.ingest fetch --workers 3
  python -m lexrag.ingest fetch --ids 6257288,111453       # alohida hujjatlar
  python -m lexrag.ingest all --forms constitution,codes,laws,decrees --max-decrees 400
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from .config import get_settings
from .lexuz.client import LexClient
from .lexuz.parser import id_from_url, parse_card, parse_doc_page
from .textnorm import cyr_to_lat, fold, script_of

log = logging.getLogger("lexrag.ingest")

FORMS = {
    "constitution": ("4131", ""),
    "codes": ("3964", "&nature=1"),
    "laws": ("3968", "&nature=1"),
    "decrees": ("3973", "&nature=1"),  # Prezident farmonlari
    "resolutions": ("3972", "&nature=1"),  # Prezident / VM qarorlari
}

AMENDMENT_PAT = re.compile(
    r"(ozgartirish|ozgartish|qoshimcha|kiritish|oz kuchini yoqotgan deb|ratifikatsiya|tasdiqlash togrisida|tasdiqlash haqida|"
    r"узгартириш|ўзгартириш|кушимча|қўшимча|киритиш|уз кучини йукотган деб|ўз кучини йўқотган деб|ратификация|тасдиқлаш тўғрисида|тасдиклаш тугрисида|"
    r"qoshilish togrisida|қўшилиш тўғрисида)"
)
KARAKALPAK = re.compile(r"[әөүңұ]|республикасының|qaraqalpaqstan", re.I)


def is_amendment_title(title: str) -> bool:
    return bool(AMENDMENT_PAT.search(fold(title)))


def build_catalog(client: LexClient, forms: list[str], max_per_form: dict[str, int]) -> dict:
    catalog: dict[str, dict] = {}
    for form in forms:
        if form not in FORMS:
            raise SystemExit(f"Noma'lum forma: {form}. Mavjud: {', '.join(FORMS)}")
        fid, extra = FORMS[form]
        q = f"/uz/search/nat?lang=3&form_id={fid}&status=Y{extra}"
        found = client.listing(q)
        kept = 0
        for did, title in found.items():
            cid = str(abs(int(did)))
            if KARAKALPAK.search(title):
                continue
            if form != "constitution" and form != "codes" and is_amendment_title(title):
                continue
            if cid in catalog:
                continue
            if form in max_per_form and kept >= max_per_form[form]:
                break
            catalog[cid] = {"doc_id": cid, "title": title, "form": form}
            kept += 1
        log.info("%s: %d topildi, %d saqlandi", form, len(found), kept)
    return catalog


def _merge_pages(doc_id: str, pages: dict, card: dict | None) -> dict:
    """Til versiyalarini modda kaliti bo'yicha birlashtiradi."""
    base_lang = next((l for l in ("uz-Cyrl", "uz-Latn", "ru") if l in pages), None)
    if base_lang is None:
        raise RuntimeError("hech bir til versiyasi olinmadi")
    base = pages[base_lang]
    by_lang = {lang: {a.key: a for a in p.articles} for lang, p in pages.items()}
    articles = []
    base_articles = list(base.articles)
    # asosiy versiyada yo'q, lekin lotin versiyada bor moddalar (masalan, yangi qo'shilgan) — ularni ham olamiz
    if base_lang != "uz-Latn" and "uz-Latn" in pages:
        have = {a.key for a in base_articles}
        base_articles += [a for a in pages["uz-Latn"].articles if a.key not in have]
    for a in base_articles:
        item = {
            "key": a.key,
            "label": a.label,
            "titles": {},
            "headings": {},
            "paragraphs": {},
            "changed": a.changed,
            "edition_date": a.edition_date,
        }
        for lang in pages:
            la = by_lang[lang].get(a.key)
            if la is None:
                continue
            item["titles"][lang] = la.title
            item["headings"][lang] = la.headings
            item["paragraphs"][lang] = la.paragraphs
            if la.edition_date and (item["edition_date"] is None or la.edition_date > item["edition_date"]):
                item["edition_date"] = la.edition_date
            item["changed"] = item["changed"] or la.changed
        # lotin versiyasi yo'q bo'lsa — kirilldan transliteratsiya
        if "uz-Latn" not in item["paragraphs"] and "uz-Cyrl" in item["paragraphs"]:
            item["paragraphs"]["uz-Latn"] = [cyr_to_lat(p) for p in item["paragraphs"]["uz-Cyrl"]]
            item["titles"]["uz-Latn"] = cyr_to_lat(item["titles"].get("uz-Cyrl", ""))
            item["headings"]["uz-Latn"] = [cyr_to_lat(h) for h in item["headings"].get("uz-Cyrl", [])]
            item["latn_transliterated"] = True
        if any(item["paragraphs"].values()):
            articles.append(item)

    titles = {lang: p.title for lang, p in pages.items()}
    if "uz-Latn" not in titles and "uz-Cyrl" in titles:
        titles["uz-Latn"] = cyr_to_lat(titles["uz-Cyrl"])
    ids = {lang: p.page_id for lang, p in pages.items()}
    card = card or {}
    repealed = any(p.repealed for p in pages.values())
    status = card.get("doc_status") or ("kuchini yo'qotgan" if repealed else "amalda")
    if repealed:
        status = "kuchini yo'qotgan"
    title_l = fold(titles.get("uz-Latn", ""))
    doc_type = card.get("doc_type")
    if not doc_type:
        if "kodeks" in title_l:
            doc_type = "kodeks"
        elif "konstitutsiya" in title_l:
            doc_type = "konstitutsiya"
        else:
            doc_type = "qonun"
    edition = max([a["edition_date"] for a in articles if a.get("edition_date")] or [None], key=lambda x: x or "") or None
    eff = card.get("effective_date") or base.effective_date
    return {
        "doc_id": doc_id,
        "ids": ids,
        "titles": titles,
        "doc_type": doc_type,
        "doc_status": status,
        "adopted_date": card.get("adopted_date"),
        "effective_date": eff,
        "end_date": card.get("end_date") or base.repealed_date,
        "act_number": card.get("act_number"),
        "issuer": card.get("issuer"),
        "edition_date": edition or eff or card.get("adopted_date"),
        "url": f"https://lex.uz/docs/{doc_id}",
        "has_articles": base.has_articles,
        "fetched_at": date.today().isoformat(),
        "articles": articles,
    }


def fetch_document(client: LexClient, doc_id: str, languages: list[str]) -> dict:
    cid = str(abs(int(doc_id)))
    pages = {}
    links: dict[str, str] = {}
    errors = []
    try:
        cyr_html = client.get(f"/uz/docs/{cid}")
        p = parse_doc_page(cyr_html, "uz-Cyrl", cid)
        # ba'zi yangi hujjatlarda kirill sahifasi aslida lotincha bo'ladi
        sample = " ".join(" ".join(a.paragraphs[:2]) for a in p.articles[:5])
        if sample and script_of(sample) == "lat":
            p.lang = "uz-Latn"
            pages["uz-Latn"] = p
        else:
            pages["uz-Cyrl"] = p
        links = p.lang_links
    except Exception as e:
        errors.append(f"uz-Cyrl: {e}")
    if "uz-Latn" in languages and "uz-Latn" not in pages:
        try:
            lat_html = client.get(f"/uz/docs/-{cid}")
            p = parse_doc_page(lat_html, "uz-Latn", f"-{cid}")
            sample = " ".join(" ".join(a.paragraphs[:2]) for a in p.articles[:5])
            if p.articles and (not sample or script_of(sample) == "lat"):
                pages["uz-Latn"] = p
                links = links or p.lang_links
        except Exception as e:
            errors.append(f"uz-Latn: {e}")
    if "ru" in languages and links.get("ru"):
        ru_id = id_from_url(links["ru"])
        if ru_id and ru_id.lstrip("-") != cid:
            try:
                ru_html = client.get(f"/ru/docs/{ru_id}")
                p = parse_doc_page(ru_html, "ru", ru_id)
                if p.articles:
                    pages["ru"] = p
            except Exception as e:
                errors.append(f"ru: {e}")
    if not pages:
        raise RuntimeError("; ".join(errors) or "sahifalar yo'q")
    card = None
    try:
        card = parse_card(client.get(f"/uz/actinfo/card1/{cid}"))
    except Exception as e:
        errors.append(f"card: {e}")
    doc = _merge_pages(cid, pages, card)
    doc["fetch_errors"] = errors
    if card and card.get("raw"):
        doc["card"] = {k: v for k, v in card["raw"].items() if v}
    return doc


def save_doc(doc: dict, docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / f"{doc['doc_id']}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return path


def load_docs(docs_dir: Path):
    for p in sorted(Path(docs_dir).glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            yield json.load(f)


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="lex.uz korpusini yig'ish")
    ap.add_argument("cmd", choices=["catalog", "fetch", "all"])
    ap.add_argument("--forms", default="constitution,codes,laws", help="constitution,codes,laws,decrees,resolutions")
    ap.add_argument("--max-decrees", type=int, default=0, help="farmonlar sonini cheklash (0 = cheklovsiz)")
    ap.add_argument("--max-resolutions", type=int, default=0)
    ap.add_argument("--ids", default="", help="vergul bilan doc_id lar (katalogdan tashqari)")
    ap.add_argument("--ids-file", default="", help="har qatorda bitta doc_id")
    ap.add_argument("--workers", type=int, default=s.crawl_workers)
    ap.add_argument("--delay", type=float, default=s.crawl_delay_s)
    ap.add_argument("--languages", default=",".join(s.crawl_languages))
    ap.add_argument("--refresh", action="store_true", help="keshni e'tiborsiz qoldirib qayta yuklash")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = LexClient(s.raw_dir, delay_s=args.delay, use_cache=not args.refresh)
    cat_path = s.data_dir / "catalog.json"
    s.data_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd in ("catalog", "all"):
        limits = {}
        if args.max_decrees:
            limits["decrees"] = args.max_decrees
        if args.max_resolutions:
            limits["resolutions"] = args.max_resolutions
        catalog = build_catalog(client, [f.strip() for f in args.forms.split(",") if f.strip()], limits)
        old = json.loads(cat_path.read_text("utf-8")) if cat_path.exists() else {}
        old.update(catalog)
        cat_path.write_text(json.dumps(old, ensure_ascii=False, indent=1), "utf-8")
        log.info("katalog: %d hujjat -> %s", len(old), cat_path)
        if args.cmd == "catalog":
            return

    ids: list[str] = []
    if args.ids:
        ids += [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.ids_file:
        ids += [l.strip() for l in Path(args.ids_file).read_text("utf-8").splitlines() if l.strip() and not l.startswith("#")]
    if not ids:
        if not cat_path.exists():
            raise SystemExit("Avval: python -m lexrag.ingest catalog")
        ids = list(json.loads(cat_path.read_text("utf-8")).keys())
    ids = [str(abs(int(i))) for i in ids]
    if args.limit:
        ids = ids[: args.limit]
    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    todo = [i for i in ids if args.refresh or not (s.docs_dir / f"{i}.json.gz").exists()]
    log.info("yuklanadi: %d (jami %d, tayyor %d)", len(todo), len(ids), len(ids) - len(todo))
    ok = fail = 0

    def work(did: str):
        doc = fetch_document(client, did, langs)
        save_doc(doc, s.docs_dir)
        return did, len(doc["articles"]), doc["fetch_errors"]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(work, d): d for d in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            did = futs[fut]
            try:
                _, n_art, errs = fut.result()
                ok += 1
                if errs:
                    log.warning("%s: %d modda, ogohlantirish: %s", did, n_art, errs)
            except Exception as e:
                fail += 1
                log.error("%s: XATO %s", did, e)
            if i % 25 == 0:
                log.info("progress %d/%d (ok=%d, xato=%d)", i, len(todo), ok, fail)
    log.info("tayyor: ok=%d, xato=%d", ok, fail)
    if fail:
        sys.exit(0)


if __name__ == "__main__":
    main()
