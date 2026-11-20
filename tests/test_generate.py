from lexrag.generate import Source, _find_verbatim, postprocess
from lexrag.retrieval import Candidate


def _src(sid, paras, parts=None):
    return Source(sid, Candidate("1", "5", 0), "uz-Latn", paras, parts or list(range(1, len(paras) + 1)), "H")


def test_find_verbatim_tolerant():
    text = "Isteʼmolchi sifatsiz tovarni  sotib olgan kundan eʼtiboran oʻn kun ichida almashtirishni talab qilishga haqli."
    assert _find_verbatim("sotib olgan kundan e'tiboran o'n kun ichida", text) == "sotib olgan kundan eʼtiboran oʻn kun ichida"
    assert _find_verbatim("tovarni sotib olgan", text) == "tovarni  sotib olgan"
    assert _find_verbatim("butunlay boshqa jumla bu yerda", text) is None


def test_postprocess_repairs_and_renumbers():
    s1 = _src("S1", ["Birinchi qism matni: xodim yigirma bir kalendar kun taʼtil oladi.", "Ikkinchi qism matni boshqa narsa haqida."])
    s2 = _src("S2", ["Uchinchi manba: ish beruvchi uch kun ichida toʻlaydi."])
    j = {
        "no_answer": False,
        "answer": "Taʼtil 21 kun [3]. Toʻlov uch kunda [5, 3].",
        "citations": [
            {"ref": 3, "source": "S1", "part": 2, "quote": "xodim yigirma bir kalendar kun ta'til oladi"},
            {"ref": 5, "source": "S2", "part": 1, "quote": "ish beruvchi uch kun ichida toʻlaydi (parafraz emas)"},
            {"ref": 7, "source": "S9", "quote": "yo'q manba"},
        ],
        "relevant_sources": ["S2", "S1", "S7"],
        "confidence": 1.4,
    }
    out = postprocess(j, [s1, s2], "uz-Latn")
    assert out.answer == "Taʼtil 21 kun [1]. Toʻlov uch kunda [2][1]."
    assert [c["ref"] for c in out.citations] == [1, 2]
    assert out.citations[0]["quote"] == "xodim yigirma bir kalendar kun taʼtil oladi"
    assert out.citations[0]["part"] == 1  # quote qaysi qismda ekanidan aniqlanadi
    assert out.citations[1]["quote"] in s2.text  # tuzatilgan (repair)
    assert out.repaired_quotes == 1
    assert out.relevant == ["S2", "S1"]
    assert out.confidence == 1.0


def test_quote_with_prompt_marker():
    s1 = _src("S1", ["Birinchi qism.", "Xodim oʻn toʻrt kalendar kun oldin ogohlantirgan holda bekor qilishga haqli."])
    j = {"answer": "14 kun [1].", "citations": [{"ref": 1, "source": "S1", "quote": "«2-qism:» Xodim o'n to'rt kalendar kun oldin ogohlantirgan holda"}]}
    out = postprocess(j, [s1], "uz-Latn")
    assert out.citations[0]["quote"] == "Xodim oʻn toʻrt kalendar kun oldin ogohlantirgan holda"
    assert out.citations[0]["part"] == 2 and out.repaired_quotes == 0


def test_postprocess_no_answer():
    out = postprocess({"no_answer": True, "answer": "", "citations": [{"ref": 1}]}, [], "ru")
    assert out.no_answer and out.citations == [] and "не найден" in out.answer


def test_answer_without_citations_gets_one():
    s1 = _src("S1", ["Pul uch kun ichida qaytariladi."])
    out = postprocess({"no_answer": False, "answer": "Pul uch kunda qaytariladi.", "citations": []}, [s1], "uz-Latn")
    assert out.citations and out.answer.endswith("[1]")
