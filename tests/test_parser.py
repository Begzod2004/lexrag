from conftest import fx, make_docs

from lexrag.chunking import doc_chunks
from lexrag.ingest import is_amendment_title
from lexrag.lexuz.parser import parse_card, parse_doc_page, parse_listing, split_parts


def test_labour_latin_page():
    p = parse_doc_page(fx("labour_latn.html"), "uz-Latn", "-6257288")
    assert p.title == "Oʻzbekiston Respublikasining Mehnat kodeksi"
    assert p.effective_date == "2023-04-30"
    assert not p.repealed
    assert p.lang_links["ru"] == "/uz/docs/6257291"
    keys = [a.key for a in p.articles]
    assert keys == ["1", "2", "3", "4", "5"]
    a4 = p.articles[3]
    assert a4.label == "4-modda"
    assert a4.changed and a4.edition_date == "2026-11-13"
    # 4-moddaning 2-qismi (tahrir izohi shuni aytadi)
    parts = split_parts(a4.paragraphs)
    assert parts == [1, 2, 3, 4, 5]
    assert a4.paragraphs[1].startswith("Mehnat va mashgʻulotlar sohasida kamsitish taqiqlanadi")
    # ro'yxat bandlari bitta qismda
    a5 = p.articles[4]
    assert split_parts(a5.paragraphs) == [1, 2, 2, 2, 2, 2, 2, 2]
    assert a5.headings[-1] == "1-bob. Asosiy qoidalar"


def test_russian_page_article_keys():
    p = parse_doc_page(fx("labour_ru.html"), "ru", "6257291")
    assert [a.key for a in p.articles] == ["1", "2", "3"]
    assert p.articles[0].label == "1-modda"
    assert p.lang_links["uz-Cyrl"] == "/ru/docs/6257288"


def test_criminal_sup_tables_and_empty_clause():
    p = parse_doc_page(fx("criminal_latn.html"), "uz-Latn", "-111453")
    keys = [a.key for a in p.articles]
    assert "57-1" in keys and "244-4" in keys and "571" not in keys
    a = next(a for a in p.articles if a.key == "57-1")
    assert a.label == "57¹-modda"
    assert a.edition_date == "2019-05-03"
    t = next(a for a in p.articles if a.key == "58")
    assert "Ancha miqdor — bazaviy hisoblash miqdorining yuz baravaridan" in t.paragraphs[0]
    assert any("analoglari" in x for x in t.paragraphs)  # BY_DEFAULT matni
    a244 = next(a for a in p.articles if a.key == "244-4")
    assert split_parts(a244.paragraphs) == [1, 1, 2]
    assert "BIRINCHI BOʻLIM. UMUMIY QOIDALAR" in p.articles[0].headings


def test_card_and_listing():
    c = parse_card(fx("card1.html"))
    assert c["doc_type"] == "kodeks" and c["doc_status"] == "amalda"
    assert c["adopted_date"] == "2022-10-28" and c["effective_date"] == "2023-04-30"
    assert c["act_number"] == "ЎРҚ-798"
    l = parse_listing(fx("search_codes.html"))
    assert l["ids"] == ["7655343", "6257288", "4492619", "111453"]
    assert l["next_target"] == "ucFoundActsControl$LinkButton1"
    assert l["total"] == 21
    assert "__VIEWSTATE" in l["hidden"]


def test_merge_and_chunks():
    labour, criminal, consumer = make_docs()
    assert labour["doc_type"] == "kodeks"
    assert labour["ids"] == {"uz-Latn": "-6257288", "ru": "6257291"}
    a1 = labour["articles"][0]
    assert set(a1["paragraphs"]) == {"uz-Latn", "ru"}
    ch = doc_chunks(labour)
    assert ch[0]["chunk_id"] == "lex:6257288#art-1"
    assert ch[0]["edition_date"] == "2026-11-13"
    crc = {c["chunk_id"] for c in doc_chunks(criminal)}
    assert "lex:111453#art-57-1" in crc


def test_long_article_split():
    doc = {
        "doc_id": "1", "titles": {"uz-Latn": "T"}, "has_articles": True, "edition_date": None,
        "articles": [{"key": "9", "label": "9-modda", "titles": {}, "headings": {},
                      "paragraphs": {"uz-Latn": [f"Qism {i}. " + "soʻz " * 300 + "." for i in range(8)]}}],
    }
    ch = doc_chunks(doc)
    assert len(ch) > 1
    assert all(c["chunk_id"].startswith("lex:1#art-9#p") for c in ch)
    assert len({c["chunk_id"] for c in ch}) == len(ch)


def test_amendment_filter():
    assert is_amendment_title("Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида")
    assert is_amendment_title("Oʻzbekiston Respublikasining Mehnat kodeksiga oʻzgartirish kiritish haqida")
    assert not is_amendment_title("Isteʼmolchilarning huquqlarini himoya qilish toʻgʻrisida")
    assert not is_amendment_title("Ўзбекистон Республикасининг Меҳнат кодекси")
