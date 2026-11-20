"""lex.uz HTML sahifalarini tahlil qilish (hujjat matni, rekvizitlar kartochkasi, qidiruv ro'yxati).

Tuzilma (2026-11 holatiga ko'ra tekshirilgan):
  #divCont > div.<KLASS> > div[id]  — har bir element bitta paragraf
  KLASSlar: ACT_TITLE, TEXT_HEADER_DEFAULT (bo'lim/bob), CLAUSE_DEFAULT (modda sarlavhasi),
            ACT_TEXT / BY_DEFAULT (matn), TABLE_STD* (jadval), CHANGES_ORIGINS (tahrir izohi),
            COMMENT (LexUZ sharhi, 'Oldingi tahrirga qarang'), INDEXES_ON_REF (klassifikator).
  Lotin versiya: /uz/docs/-ID, kirill: /uz/docs/ID, rus: boshqa ID ('Рус' havolasidan olinadi).
  Kuchini yo'qotgan hujjatda .act_warning bloki bo'ladi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from lxml import html as LH

from ..textnorm import SUPERSCRIPT, clean_ws, key_to_label, sup_to_key

SKIP_CLASSES = {"INDEXES_ON_REF", "COMMENT", "PUBLICATION_ORIGIN", "ACT_FORM", "NEW_EDITION", "CHANGES_ORIGINS"}
TEXT_CLASSES_PREFIX = ("ACT_TEXT", "BY_DEFAULT", "TABLE_STD", "ACT_ESSENTIAL", "SIGNATURE", "APPLICATION", "ACT_TEXT_")

ART_RE = {
    "uz-Latn": re.compile(r"^\s*(\d+(?:-\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*[-–]\s*modda\b\.?\s*(.*)$", re.I | re.S),
    "uz-Cyrl": re.compile(r"^\s*(\d+(?:-\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*[-–]\s*модда\b\.?\s*(.*)$", re.I | re.S),
    "ru": re.compile(r"^\s*Статья\s+(\d+(?:-\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*\.?\s*(.*)$", re.I | re.S),
}
POINT_RE = re.compile(r"^\s*(\d{1,3})\.\s+\S")
ANNEX_RE = re.compile(r"(ilova|илова|приложение|nizom\b|низом\b|положение\b)", re.I)
DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")


@dataclass
class ParsedArticle:
    key: str
    label: str
    title: str
    paragraphs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    changed: bool = False
    elem_id: str | None = None

    @property
    def edition_date(self) -> str | None:
        return max_date(self.notes)


@dataclass
class ParsedPage:
    lang: str
    page_id: str
    title: str
    html_title: str
    effective_date: str | None
    repealed: bool
    repealed_date: str | None
    lang_links: dict[str, str]
    articles: list[ParsedArticle]
    has_articles: bool
    act_title: str | None = None


def parse_ddmmyyyy(s: str) -> str | None:
    m = DATE_RE.search(s or "")
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mth, d).isoformat()
    except ValueError:
        return None


def max_date(texts: list[str]) -> str | None:
    best = None
    for t in texts:
        for m in DATE_RE.finditer(t):
            try:
                d = date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
            except ValueError:
                continue
            if 1990 <= int(d[:4]) <= 2100 and (best is None or d > best):
                best = d
    return best


def _elem_text(el) -> str:
    """Element matni: <sup> -> ustki raqam, <br> -> yangi qator, jadval -> qatorlar."""
    el = LH.fromstring(LH.tostring(el, encoding="unicode"))  # nusxa (asl daraxtni buzmaslik)
    for junk in el.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " lx_elem2 ")]'):
        junk.getparent().remove(junk)
    for sup in el.xpath(".//sup"):
        t = (sup.text_content() or "").strip()
        repl = t.translate(SUPERSCRIPT) if t.isdigit() else t
        tail = sup.tail or ""
        parent = sup.getparent()
        prev = sup.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + repl + tail
        else:
            parent.text = (parent.text or "") + repl + tail
        parent.remove(sup)
    for br in el.xpath(".//br"):
        br.tail = "\n" + (br.tail or "")
    tables = el.xpath(".//table")
    if tables:
        rows = []
        for tr in el.xpath(".//tr"):
            cells = [clean_ws(td.text_content()) for td in tr.xpath("./td|./th")]
            cells = [c for c in cells if c and c not in {"—", "-", "–"}]
            if cells:
                rows.append(" — ".join(cells))
        return clean_ws("\n".join(rows))
    return clean_ws(el.text_content())


def _inner(el):
    found = el.xpath("./div[@id]")
    return found[-1] if found else el


def _first_class(el) -> str:
    return (el.get("class") or "").split(" ")[0]


def _lang_links(doc) -> dict[str, str]:
    out: dict[str, str] = {}
    for el in doc.xpath('//*[starts-with(@onclick, "openUrl")]'):
        m = re.search(r"openUrl\('([^']+)'\)", el.get("onclick", ""))
        if not m:
            continue
        label = clean_ws(el.text_content()).lower()
        url = m.group(1)
        if "/pdfs/" in url:
            continue
        if label in ("рус", "ру"):
            out["ru"] = url
        elif label in ("ўзб",):
            out["uz-Cyrl"] = url
        elif label in ("o’zb", "o'zb", "oʻzb", "o‘zb"):
            out["uz-Latn"] = url
    return out


def id_from_url(url: str) -> str | None:
    m = re.search(r"/docs/(-?\d+)", url or "")
    return m.group(1) if m else None


def split_parts(paragraphs: list[str]) -> list[int]:
    """Har bir paragraf uchun qism (qism/часть) raqami. Ro'yxat bandlari oldingi qismga tegishli."""
    parts: list[int] = []
    cur = 0
    prev = ""
    for p in paragraphs:
        s = p.strip()
        if not parts:
            cur = 1
        else:
            pe = prev.rstrip().rstrip("‎").rstrip()
            first = s[:1]
            starts_upper = bool(first) and (first.isupper() or first.isdigit() or first in "«\"“(")
            list_item = bool(re.match(r"^(\d{1,2}\)|[a-zа-яё]\)|[-–•])\s", s))
            if pe.endswith((":", ";", ",", "—", "–", "-")) or list_item:
                pass  # ro'yxat davom etyapti
            elif pe.endswith((".", "!", "?", ")", "»", "\"")) and starts_upper:
                cur += 1
            elif starts_upper and not pe.endswith((":", ";", ",")):
                cur += 1
        parts.append(cur)
        prev = s
    return parts


def parse_doc_page(html_text: str, lang: str, page_id: str) -> ParsedPage:
    doc = LH.fromstring(html_text)
    html_title = clean_ws(" ".join(doc.xpath("//title/text()")))
    title = ""
    lab = doc.xpath('//*[contains(@class,"docHeader__col__1")]//*[contains(@class,"docHeader__item-label")]')
    if lab:
        title = clean_ws(lab[0].text_content())
    if not title:
        title = re.sub(r"^\s*\d{2}\.\d{2}\.\d{4}\.?\s*", "", html_title)
    eff = None
    val = doc.xpath('//*[contains(@class,"docHeader__col__2")]//*[contains(@class,"docHeader__item-value")]')
    if val:
        eff = parse_ddmmyyyy(val[0].text_content())
    warn = doc.xpath('//*[contains(@class,"act_warning")]')
    repealed = False
    repealed_date = None
    if warn:
        wt = clean_ws(warn[0].text_content())
        if re.search(r"(kuchini yo|кучини йўқот|утратил|amalda emas|амалда эмас|не действ)", wt, re.I):
            repealed = True
            repealed_date = parse_ddmmyyyy(wt)

    cont = doc.xpath('//*[@id="divCont"]')
    articles: list[ParsedArticle] = []
    act_title = None
    if not cont:
        return ParsedPage(lang, page_id, title, html_title, eff, repealed, repealed_date, _lang_links(doc), [], False)

    # Sahifa tili e'lon qilinganidan farq qilishi mumkin (masalan, yangi hujjatning "kirill" sahifasi aslida lotincha),
    # shuning uchun modda sarlavhasi uchala yozuvda ham tan olinadi.
    _order = [lang] + [l for l in ART_RE if l != lang]

    class _AnyArt:
        @staticmethod
        def match(t: str):
            for l in _order:
                m = ART_RE[l].match(t)
                if m:
                    return m
            return None

    art_re = _AnyArt
    headings: list[str] = []
    cur: ParsedArticle | None = None
    preamble = ParsedArticle(key="preamble", label="preamble", title="")
    has_articles = False
    pending_changed = False
    seen_keys: dict[str, int] = {}

    def heading_level(t: str) -> int:
        tl = t.lower()
        if re.search(r"(qism\b|қисм\b|часть\b)", tl) and len(t) < 40:
            return 0
        if re.search(r"(bo[ʻ'‘’]?lim|бўлим|раздел)", tl):
            return 1
        if re.search(r"(\bbob\b|\bбоб\b|глава)", tl):
            return 2
        if re.search(r"(paragraf|параграф|§)", tl):
            return 3
        return 2

    def close():
        nonlocal cur
        if cur is not None:
            articles.append(cur)
        cur = None

    level_heads: dict[int, str] = {}
    for el in cont[0]:
        cls = _first_class(el)
        if not cls:
            continue
        if cls == "COMMENT":
            t = clean_ws(el.text_content())
            if re.search(r"(oldingi|олдинги|предыдущ)", t, re.I):
                if cur is not None:
                    cur.changed = True
                else:
                    pending_changed = True
            continue
        if cls == "CHANGES_ORIGINS":
            t = _elem_text(_inner(el))
            if t:
                (cur or preamble).notes.append(t)
                if cur is not None:
                    cur.changed = True
            continue
        if cls in SKIP_CLASSES:
            continue
        inner = _inner(el)
        text = _elem_text(inner)
        if not text:
            continue
        if cls == "ACT_TITLE":
            act_title = text
            continue
        if cls == "TEXT_HEADER_DEFAULT":
            close()
            lvl = heading_level(text)
            level_heads[lvl] = text.replace("\n", ". ")
            for k in list(level_heads):
                if k > lvl:
                    del level_heads[k]
            headings = [level_heads[k] for k in sorted(level_heads)]
            continue
        if cls == "CLAUSE_DEFAULT":
            m = art_re.match(text)
            if m:
                close()
                has_articles = True
                num = m.group(1)
                key = sup_to_key(num)
                if key in seen_keys:  # takroriy raqam (masalan, ilovadagi modda) — farqlash
                    seen_keys[key] += 1
                    key = f"{key}~{seen_keys[key]}"
                else:
                    seen_keys[key] = 1
                cur = ParsedArticle(
                    key=key,
                    label=key_to_label(key.split("~")[0]),
                    title=clean_ws(m.group(2)).rstrip("."),
                    headings=list(headings),
                    changed=pending_changed,
                    elem_id=inner.get("id"),
                )
                pending_changed = False
            else:
                close()
                level_heads[3] = text
                headings = [level_heads[k] for k in sorted(level_heads)]
            continue
        # matn
        if cls.startswith(TEXT_CLASSES_PREFIX) or cls.startswith("ACT_") or cls.startswith("TABLE"):
            if cur is not None:
                cur.paragraphs.append(text)
            else:
                preamble.paragraphs.append(text)
                if not preamble.headings and headings:
                    preamble.headings = list(headings)

    close()
    if not has_articles:
        articles = _points_from_blocks(cont[0], lang)
    else:
        if preamble.paragraphs and sum(len(p) for p in preamble.paragraphs) > 200:
            preamble.label = "muqaddima"
            articles.insert(0, preamble)
    return ParsedPage(lang, page_id, title, html_title, eff, repealed, repealed_date, _lang_links(doc), articles, has_articles, act_title)


def _points_from_blocks(container, lang: str) -> list[ParsedArticle]:
    """Moddasiz hujjatlar (farmon, qaror, nizom): yuqori darajadagi '1.', '2.' bandlar bo'yicha bo'lish."""
    out: list[ParsedArticle] = []
    annex = 0
    cur = ParsedArticle(key="preamble", label="muqaddima", title="")
    heads: list[str] = []
    last_point = 0

    def push():
        nonlocal cur
        if cur.paragraphs:
            out.append(cur)

    for el in container:
        cls = _first_class(el)
        if not cls:
            continue
        if cls == "CHANGES_ORIGINS":
            t = _elem_text(_inner(el))
            if t:
                cur.notes.append(t)
                cur.changed = True
            continue
        if cls in SKIP_CLASSES or cls == "ACT_TITLE":
            continue
        text = _elem_text(_inner(el))
        if not text:
            continue
        if cls == "TEXT_HEADER_DEFAULT" or cls == "CLAUSE_DEFAULT":
            if ANNEX_RE.search(text) and len(text) < 300:
                push()
                annex += 1
                last_point = 0
                heads = [text.replace("\n", ". ")]
                cur = ParsedArticle(key=f"ilova{annex}", label=f"{annex}-ilova", title=text.replace("\n", ". "), headings=list(heads))
            else:
                heads = (heads[:1] if annex else []) + [text.replace("\n", ". ")]
                cur.paragraphs.append(text)
            continue
        m = POINT_RE.match(text)
        if m and int(m.group(1)) in (last_point + 1, 1):
            n = int(m.group(1))
            if n == 1 and last_point > 1 and annex == 0:
                # yangi raqamlash ilovasiz boshlandi — ilova deb hisoblaymiz
                annex += 1
            push()
            last_point = n
            key = f"{n}" if annex == 0 else f"ilova{annex}-{n}"
            label = f"{n}-band" if annex == 0 else f"{annex}-ilova {n}-band"
            cur = ParsedArticle(key=key, label=label, title="", headings=list(heads))
            cur.paragraphs.append(text)
        else:
            cur.paragraphs.append(text)
    push()
    # juda uzun muqaddima/ilova bloklari keyinchalik chunker tomonidan bo'linadi
    return out


# ------------------------------ kartochka ------------------------------

DOC_TYPE_MAP = {
    "кодекс": "kodeks",
    "закон": "qonun",
    "конституция": "konstitutsiya",
    "указ": "farmon",
    "постановление": "qaror",
    "распоряжение": "farmoyish",
    "положение": "nizom",
    "приказ": "buyruq",
    "решение": "qaror",
}
STATUS_MAP = {"действующий": "amalda", "утративший силу": "kuchini yo'qotgan", "не вступивший в силу": "amalda emas"}


def parse_card(html_text: str) -> dict:
    doc = LH.fromstring(html_text)
    kv: dict[str, str] = {}
    for td in doc.xpath('//td[contains(concat(" ", normalize-space(@class), " "), " lbl ")]'):
        label = clean_ws(td.text_content())
        nxt = td.getnext()
        if nxt is not None:
            kv[label] = clean_ws(nxt.text_content())
    # Qabul qilgan organlar jadvali
    heads = [clean_ws(th.text_content()) for th in doc.xpath('//table[@id="bodyTab"]//thead//th')]
    rows = doc.xpath('//table[@id="bodyTab"]//tbody/tr')
    if rows and heads:
        cells = [clean_ws(td.text_content()) for td in rows[0].xpath("./td")]
        hdr = [h for h in heads if h != "Органы, принявшие акт"]
        for h, c in zip(hdr, cells):
            kv.setdefault(h, c)
    form = kv.get("Форма акта", "").lower()
    status_raw = kv.get("Статус акта", "").lower()
    out = {
        "doc_type": DOC_TYPE_MAP.get(form, form or None),
        "doc_status": STATUS_MAP.get(status_raw, "amalda" if not status_raw else status_raw),
        "adopted_date": parse_ddmmyyyy(kv.get("Дата принятия", "")),
        "act_number": kv.get("Номер акта") or None,
        "effective_date": parse_ddmmyyyy(kv.get("Дата вступления в силу", "")),
        "end_date": parse_ddmmyyyy(kv.get("Дата прекращения действия", "")),
        "issuer": kv.get("Наименование органа") or None,
        "kind": kv.get("Вид акта") or None,
        "raw": kv,
    }
    return out


# ------------------------------ qidiruv ro'yxati ------------------------------


def parse_listing(html_text: str) -> dict:
    doc = LH.fromstring(html_text)
    ids: list[str] = []
    titles: dict[str, str] = {}
    for a in doc.xpath('//a[contains(@href, "/docs/")]'):
        href = a.get("href", "")
        m = re.match(r"^/(?:uz|ru|en)/docs/(-?\d+)", href)
        if not m:
            continue
        t = clean_ws(a.text_content())
        if len(t) < 5:
            continue
        did = m.group(1)
        if did not in titles:
            ids.append(did)
            titles[did] = t
    hidden = {i.get("name"): i.get("value", "") for i in doc.xpath('//input[@type="hidden"]') if i.get("name")}
    nxt = None
    for a in doc.xpath('//a[starts-with(@href, "javascript:__doPostBack")]'):
        if clean_ws(a.text_content()).lower() in ("keyingisi", "keyingi", "следующая", "кейингиси"):
            m = re.search(r"__doPostBack\('([^']+)'", a.get("href", ""))
            if m:
                nxt = m.group(1)
    total = None
    m = re.search(r"(\d[\d\s]*)\s*hujjat topildi", doc.text_content())
    if m:
        total = int(re.sub(r"\s", "", m.group(1)))
    return {"ids": ids, "titles": titles, "hidden": hidden, "next_target": nxt, "total": total}
