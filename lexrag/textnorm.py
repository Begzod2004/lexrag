"""O'zbek (lotin/kirill) va rus matnlarini normallashtirish, transliteratsiya, tokenizatsiya."""
from __future__ import annotations

import re
import unicodedata

# lex.uz lotin matnida: oʻ gʻ uchun U+02BB, tutuq belgisi uchun U+02BC
OKINA = "ʻ"  # ʻ
TUTUQ = "ʼ"  # ʼ
APOSTROPHES = "'`‘’‛ʹʻʼʽ′´`"

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
FROM_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
SUP_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": TUTUQ, "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o" + OKINA, "қ": "q", "ғ": "g" + OKINA, "ҳ": "h",
}
_VOWELS_CYR = set("аеёиоуўэюяы")

UZ_CYR_SPECIFIC = set("ўқғҳЎҚҒҲ")
RU_SPECIFIC = set("ыщЫЩ")

UZ_CYR_WORDS = {
    "ва", "билан", "учун", "ҳам", "бу", "қандай", "неча", "қанча", "мумкин", "бўлади", "керак",
    "қайси", "нима", "ким", "агар", "бўлса", "ҳақида", "тўғрисида", "мазкур", "ушбу", "эмас",
}
RU_WORDS = {
    "и", "в", "на", "не", "что", "как", "какой", "какая", "какие", "каков", "ли", "для", "по", "от",
    "если", "срок", "сколько", "можно", "должен", "это", "при", "или", "статья", "закон", "кодекс",
}


def sup_to_key(s: str) -> str:
    """'57¹' -> '57-1', '244⁴' -> '244-4', '54-1' -> '54-1'."""
    s = s.strip()
    m = re.match(r"^(\d+)([" + SUP_CHARS + r"]+)$", s)
    if m:
        return f"{m.group(1)}-{m.group(2).translate(FROM_SUPERSCRIPT)}"
    return s


def key_to_label(key: str, unit: str = "modda") -> str:
    """'57-1' -> '57¹-modda'."""
    if "-" in key and key.replace("-", "").isdigit():
        base, sub = key.split("-", 1)
        return f"{base}{sub.translate(SUPERSCRIPT)}-{unit}"
    return f"{key}-{unit}"


def normalize_article_ref(ref: str | None) -> str | None:
    """Turli yozuvlarni bir xil kalitga keltiradi: '225-modda', 'Статья 225', '57¹', '57^1', 'art-57-1' -> '57-1'."""
    if not ref:
        return None
    s = str(ref).strip().lower()
    s = s.replace("art-", "").replace("статья", "").replace("ст.", "").replace("modda", "").replace("модда", "")
    s = s.replace("^", "-").strip(" .-—–")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(\d+)([" + SUP_CHARS + r"]+)", s)
    if m:
        return f"{m.group(1)}-{m.group(2).translate(FROM_SUPERSCRIPT)}"
    m = re.match(r"^(\d+)(?:-(\d+))?", s)
    if m:
        return m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
    return s or None


def cyr_to_lat(text: str) -> str:
    """O'zbek kirill -> lotin (1995 alifbosi). Rus matniga ishlatilmaydi."""
    out = []
    prev = ""
    for i, ch in enumerate(text):
        low = ch.lower()
        is_upper = ch != low
        if low == "е":
            # so'z boshida yoki unli/ъ/ь dan keyin -> ye
            rep = "ye" if (not prev or not prev.isalpha() or prev.lower() in _VOWELS_CYR or prev.lower() in "ъь") else "e"
        elif low == "ц":
            rep = "ts" if prev and prev.lower() in _VOWELS_CYR else "s"
        elif low in _CYR2LAT:
            rep = _CYR2LAT[low]
        else:
            out.append(ch)
            prev = ch
            continue
        if is_upper and rep:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            rep = rep.upper() if (nxt and nxt.isupper()) else rep[0].upper() + rep[1:]
        out.append(rep)
        prev = ch
    return "".join(out)


def script_of(text: str) -> str:
    lat = sum(1 for c in text if "a" <= c.lower() <= "z")
    cyr = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    return "cyr" if cyr > lat else "lat"


def detect_language(text: str) -> str:
    """'uz-Latn' | 'uz-Cyrl' | 'ru'."""
    if script_of(text) == "lat":
        return "uz-Latn"
    if any(c in UZ_CYR_SPECIFIC for c in text):
        return "uz-Cyrl"
    words = re.findall(r"[Ѐ-ӿ]+", text.lower())
    uz = sum(1 for w in words if w in UZ_CYR_WORDS)
    ru = sum(1 for w in words if w in RU_WORDS)
    if any(c in RU_SPECIFIC for c in text):
        ru += 2
    # o'zbekcha qo'shimchalar
    uz += sum(1 for w in words if re.search(r"(нинг|лар|лари|дан|даги|га|ни)$", w))
    ru += sum(1 for w in words if re.search(r"(ого|его|ому|ыми|ими|ость|ение|ется|ются|ый|ий|ая)$", w))
    return "uz-Cyrl" if uz > ru else "ru"


def to_latin_if_uzcyrl(text: str, lang: str | None = None) -> str:
    lang = lang or detect_language(text)
    return cyr_to_lat(text) if lang == "uz-Cyrl" else text


def unify_apostrophes(text: str) -> str:
    """Barcha apostrof turlarini lex.uz uslubiga: o'/g' -> ʻ, qolgani -> ʼ."""
    def rep(m: re.Match) -> str:
        before = m.string[m.start() - 1] if m.start() > 0 else ""
        return OKINA if before.lower() in ("o", "g") else TUTUQ

    return re.sub("[" + re.escape(APOSTROPHES) + "]", rep, text)


def fold(text: str) -> str:
    """Solishtirish uchun qattiq normallashtirish: kichik harf, apostroflarsiz, bo'shliqlar bitta."""
    t = unicodedata.normalize("NFC", text)
    t = re.sub("[" + re.escape(APOSTROPHES) + "]", "", t)
    t = t.replace("ё", "е").replace("Ё", "Е")
    # 57¹ -> 57-1 (aks holda '571' bo'lib, 571-modda bilan adashadi)
    t = re.sub(r"(\d)([" + SUP_CHARS + r"]+)", lambda m: m.group(1) + "-" + m.group(2).translate(FROM_SUPERSCRIPT), t)
    t = t.translate(FROM_SUPERSCRIPT)
    t = re.sub(r"[‎‏­ ]", " ", t)
    t = re.sub(r"[«»“”„\"]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def clean_ws(text: str) -> str:
    t = re.sub(r"[‎‏­]", "", text)
    t = t.replace(" ", " ")
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


# --------------------------- tokenizatsiya ---------------------------

UZ_STOP = {
    "va", "bilan", "uchun", "ham", "bu", "shu", "u", "bir", "yoki", "esa", "edi", "deb", "qanday", "qancha",
    "necha", "nima", "kim", "qaysi", "mumkin", "kerak", "boladi", "bolsa", "agar", "haqida", "hamda", "ushbu",
    "mazkur", "yoxud", "biroq", "lekin", "bolib", "bolgan", "etib", "qilib", "uning", "ular", "ularning",
    "oz", "har", "hech", "emas", "mi", "chi", "da", "ning", "dan", "ga", "ni", "nimalar", "qachon", "qayerda",
}
RU_STOP = {
    "и", "в", "во", "на", "не", "что", "как", "какой", "какая", "какие", "каков", "ли", "для", "по", "от", "до",
    "из", "к", "с", "со", "о", "об", "а", "но", "или", "это", "же", "бы", "при", "за", "у", "его", "ее", "их",
    "он", "она", "они", "то", "так", "также", "быть", "является", "сколько", "можно", "если", "ст",
}

_UZ_SUFFIXES = sorted(
    [
        "larining", "laridagi", "laridan", "lariga", "larini", "larida", "lardagi", "lardan", "larga", "larni",
        "larda", "lari", "lar", "ning", "dagi", "gacha", "dan", "ga", "ka", "qa", "da", "ni", "si", "ini",
        "ing", "imiz", "ingiz", "ish", "ishi", "ishga", "ishni", "i", "ligi", "lik", "chi", "lari", "mi",
    ],
    key=len,
    reverse=True,
)
_RU_SUFFIXES = sorted(
    [
        "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ых", "их", "ой", "ей", "ий", "ый", "ая",
        "яя", "ое", "ее", "ую", "юю", "ам", "ям", "ах", "ях", "ом", "ем", "ов", "ев", "ия", "ие", "ию", "ии",
        "ость", "ости", "ение", "ения", "ению", "ании", "ание", "ания", "ться", "тся", "ать", "ять", "ить",
        "а", "я", "о", "е", "ы", "и", "у", "ю", "ь",
    ],
    key=len,
    reverse=True,
)

_TOKEN_RE = re.compile(r"[0-9]+(?:[-_][0-9]+)?|[a-zа-яёўқғҳ]+", re.IGNORECASE)


def _stem_uz(w: str) -> str:
    for _ in range(3):
        for suf in _UZ_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        else:
            break
    return w[:8]


def _stem_ru(w: str) -> str:
    for suf in _RU_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:8]


def tokenize(text: str, lang: str) -> list[str]:
    """BM25 uchun tokenlar. lang: 'uz' (lotin; kirill bo'lsa avval transliteratsiya qilinadi) yoki 'ru'."""
    if lang == "uz" and script_of(text) == "cyr":
        text = cyr_to_lat(text)
    t = fold(text)
    # 57¹ -> 57-1 (fold allaqachon ¹ -> 1 qilgan; '57¹-modda' -> '571-modda' bo'lib qolmasligi uchun
    # indekslashda sup_to_key ishlatiladi, bu yerda raqamlar alohida token)
    toks = []
    for m in _TOKEN_RE.finditer(t):
        w = m.group(0)
        if w[0].isdigit():
            toks.append(w.replace("_", "-"))
            continue
        if lang == "ru":
            if w in RU_STOP or len(w) < 2:
                continue
            toks.append(_stem_ru(w))
        else:
            if w in UZ_STOP or len(w) < 2:
                continue
            toks.append(_stem_uz(w))
    return toks


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]
