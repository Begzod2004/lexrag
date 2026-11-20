from conftest import fx

from lexrag.ingest import fetch_document


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, path, use_cache=None, retries=4):
        self.calls.append(path)
        if path not in self.pages:
            raise FileNotFoundError(path)
        return self.pages[path]


def test_fetch_document_all_languages():
    cyr = fx("labour_latn.html").replace("modda.", "модда.").replace("Oʻzbekiston Respublikasining Mehnat kodeksi", "Ўзбекистон Республикасининг Меҳнат кодекси")
    # kirill sahifani taqlid qilish: matnni kirillga o'tkazmasdan, faqat sarlavhalar (skript aniqlash lotin deb topadi)
    client = FakeClient({
        "/uz/docs/6257288": cyr,
        "/uz/docs/-6257288": fx("labour_latn.html"),
        "/ru/docs/6257291": fx("labour_ru.html"),
        "/uz/actinfo/card1/6257288": fx("card1.html"),
    })
    doc = fetch_document(client, "6257288", ["uz-Cyrl", "uz-Latn", "ru"])
    assert doc["doc_id"] == "6257288"
    assert doc["doc_type"] == "kodeks" and doc["doc_status"] == "amalda"
    assert doc["act_number"] == "ЎРҚ-798"
    assert "ru" in doc["ids"] and doc["ids"]["ru"] == "6257291"
    assert [a["key"] for a in doc["articles"]] == ["1", "2", "3", "4", "5"]
    assert "ru" in doc["articles"][0]["paragraphs"]
    assert doc["edition_date"] == "2026-11-13"
    assert "/ru/docs/6257291" in client.calls


def test_fetch_document_missing_versions():
    client = FakeClient({"/uz/docs/111453": fx("criminal_latn.html").replace("modda.", "модда.")})
    doc = fetch_document(client, "-111453", ["uz-Cyrl", "uz-Latn", "ru"])
    assert doc["doc_id"] == "111453"
    assert doc["articles"] and doc["fetch_errors"]  # lotin/kartochka topilmadi, lekin hujjat saqlandi
    assert all("uz-Latn" in a["paragraphs"] for a in doc["articles"])
