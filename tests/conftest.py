import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lexrag.config import Settings  # noqa: E402
from lexrag.embeddings import HashEmbedder  # noqa: E402
from lexrag.index import Index  # noqa: E402
from lexrag.ingest import _merge_pages  # noqa: E402
from lexrag.lexuz.parser import parse_card, parse_doc_page  # noqa: E402
from lexrag.llm import LLMResult  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def fx(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def make_docs() -> list[dict]:
    lab_lat = parse_doc_page(fx("labour_latn.html"), "uz-Latn", "-6257288")
    lab_ru = parse_doc_page(fx("labour_ru.html"), "ru", "6257291")
    card = parse_card(fx("card1.html"))
    labour = _merge_pages("6257288", {"uz-Latn": lab_lat, "ru": lab_ru}, card)
    cr = parse_doc_page(fx("criminal_latn.html"), "uz-Latn", "-111453")
    criminal = _merge_pages("111453", {"uz-Latn": cr}, {"doc_type": "kodeks", "doc_status": "amalda"})
    # sintetik qonun: iste'molchilar huquqlari (retrieval testlari uchun)
    consumer = {
        "doc_id": "9990001",
        "ids": {"uz-Latn": "-9990001"},
        "titles": {"uz-Latn": "Isteʼmolchilarning huquqlarini himoya qilish toʻgʻrisida", "ru": "О защите прав потребителей"},
        "doc_type": "qonun",
        "doc_status": "amalda",
        "edition_date": "2024-05-01",
        "url": "https://lex.uz/docs/9990001",
        "has_articles": True,
        "fetched_at": "2026-11-20",
        "articles": [
            {
                "key": "15",
                "label": "15-modda",
                "titles": {"uz-Latn": "Sifatsiz tovarni almashtirish", "ru": "Замена товара ненадлежащего качества"},
                "headings": {"uz-Latn": ["3-bob. Tovar sotish"]},
                "paragraphs": {
                    "uz-Latn": [
                        "Isteʼmolchi sifatsiz tovarni sotib olgan kundan eʼtiboran oʻn kun ichida almashtirishni talab qilishga haqli.",
                        "Kafolat muddati belgilangan tovarlar uchun talab kafolat muddati davomida qoʻyilishi mumkin.",
                    ],
                    "ru": [
                        "Потребитель вправе потребовать замены товара ненадлежащего качества в течение десяти дней со дня покупки.",
                        "Для товаров с гарантийным сроком требование может быть предъявлено в течение гарантийного срока.",
                    ],
                },
                "edition_date": "2024-05-01",
                "changed": False,
            },
            {
                "key": "16",
                "label": "16-modda",
                "titles": {"uz-Latn": "Pulni qaytarish"},
                "headings": {"uz-Latn": ["3-bob. Tovar sotish"]},
                "paragraphs": {"uz-Latn": ["Sotuvchi tovar qaytarilgan kundan eʼtiboran uch kun ichida pulni qaytarishi shart."]},
                "edition_date": None,
                "changed": False,
            },
        ],
    }
    return [labour, criminal, consumer]


class FakeLLM:
    """Promptga qarab deterministik JSON qaytaradi."""

    def __init__(self, label="fake-model", gen_override=None, fail=False):
        self._label = label
        self.calls = []
        self.gen_override = gen_override
        self.fail = fail

    @property
    def available(self):
        return True

    @property
    def label(self):
        return self._label

    def complete_json(self, system, user, max_tokens=1200, temperature=0.0, seed=None, usage=None):
        from lexrag.llm import LLMError

        self.calls.append(system[:40])
        if self.fail:
            raise LLMError("LLM_TIMEOUT", "fake timeout", True)
        res = LLMResult("{}", self._label, 100, 20, 5)
        if usage is not None:
            usage.add(res)
        q = user.split("\n", 2)[1] if "\n" in user else user
        if "search queries" in system:
            if "ob-havo" in q.lower():
                return {"legal": False, "uz": q, "ru": "", "keywords": [], "hyde": "", "docs": [], "articles": []}, res
            if "kamsit" in q.lower():
                return {"legal": True, "uz": "mehnat sohasida kamsitishni taqiqlash", "ru": "запрещение дискриминации в сфере труда",
                        "keywords": ["kamsitish", "дискриминация"], "hyde": "Mehnat sohasida kamsitish taqiqlanadi.",
                        "docs": ["Mehnat kodeksi"], "articles": [{"doc": "Mehnat kodeksi", "article": "4"}]}, res
            return {"legal": True, "uz": q, "ru": "", "keywords": [], "hyde": "", "docs": [], "articles": []}, res
        if "reranker" in system:
            ids = re.findall(r"\[C(\d+)\]", user)
            return {"ranking": [{"id": f"C{i}", "score": 90 - n * 10} for n, i in enumerate(ids[:5])]}, res
        # generatsiya
        if self.gen_override is not None:
            return self.gen_override(user), res
        if "ob-havo" in q.lower():
            return {"no_answer": True, "answer": "Javob topilmadi.", "citations": [], "relevant_sources": [], "confidence": 0.9}, res
        m = re.search(r"\[S(\d+)\][^\n]*\n(?:«\d+-qism:» )?([^\n]+)", user.split("SOURCES:", 1)[1])
        sid, text = m.group(1), m.group(2)
        quote = " ".join(text.split()[:10])
        return {"no_answer": False, "answer": f"Qonunga koʻra: {quote} [1].", "citations": [{"ref": 1, "source": f"S{sid}", "part": 1, "quote": quote}],
                "relevant_sources": [f"S{sid}"], "confidence": 0.8}, res


@pytest.fixture(scope="session")
def docs():
    return make_docs()


@pytest.fixture()
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.api_keys = ["test-key"]
    s.embedding_provider = "hash"
    s.embedding_dims = 256
    s.reranker = "llm"
    s.query_rewrite = True
    s.rate_limit_per_sec = 1000
    s.rate_limit_burst = 1000
    s.answer_cache = True
    s.sync_timeout_s = 20
    return s


@pytest.fixture()
def index(docs, settings):
    emb = HashEmbedder(256)
    return Index.build(docs, settings, emb, settings.index_dir)


@pytest.fixture()
def engine(index, settings):
    from lexrag.pipeline import Engine

    return Engine(settings, index=index, embedder=HashEmbedder(256), llm_answer=FakeLLM("fake-answer"), llm_fast=FakeLLM("fake-fast"))


@pytest.fixture()
def client(engine):
    from starlette.testclient import TestClient

    from lexrag.api import create_app
    from lexrag.service import Service

    svc = Service(engine)
    return TestClient(create_app(svc))


AUTH = {"Authorization": "Bearer test-key"}


def batch(n=2, bid="run1-b01", questions=None, **opts):
    qs = questions or ["Mehnat sohasida kamsitish taqiqlanadimi?", "Ertaga ob-havo qanday bo'ladi?", "Sifatsiz tovarni necha kun ichida almashtirish mumkin?"]
    items = [{"id": f"q{i+1:03d}", "question": qs[i % len(qs)]} for i in range(n)]
    return {"run_id": "run1", "batch_id": bid, "batch_index": 1, "total_batches": 10, "options": {"top_k": 5, "return_contexts": True, "return_debug": True, "temperature": 0, "seed": 42, **opts}, "items": items}


def dumps(o):
    return json.dumps(o, ensure_ascii=False)
