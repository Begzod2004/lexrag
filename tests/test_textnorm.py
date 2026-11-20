from lexrag.textnorm import (
    cyr_to_lat,
    detect_language,
    fold,
    key_to_label,
    normalize_article_ref,
    sup_to_key,
    tokenize,
    unify_apostrophes,
)


def test_cyr_to_lat_basic():
    assert cyr_to_lat("Меҳнат кодекси") == "Mehnat kodeksi"
    assert cyr_to_lat("ўғил") == "oʻgʻil"
    assert cyr_to_lat("Ўзбекистон") == "Oʻzbekiston"
    assert cyr_to_lat("ер") == "yer"
    assert cyr_to_lat("шахс") == "shaxs"
    assert cyr_to_lat("таътил") == "taʼtil"


def test_detect_language():
    assert detect_language("Yillik asosiy mehnat taʼtilining eng kam muddati qancha?") == "uz-Latn"
    assert detect_language("Йиллик асосий меҳнат таътилининг энг кам муддати қанча?") == "uz-Cyrl"
    assert detect_language("Какой срок исковой давности установлен Гражданским кодексом?") == "ru"
    assert detect_language("Мехнат шартномаси неча кун ичида тузилади?") in ("uz-Cyrl",)


def test_article_refs():
    assert normalize_article_ref("225-modda") == "225"
    assert normalize_article_ref("Статья 15") == "15"
    assert normalize_article_ref("57¹-modda") == "57-1"
    assert normalize_article_ref("57^1") == "57-1"
    assert normalize_article_ref("art-244-4") == "244-4"
    assert sup_to_key("244⁴") == "244-4"
    assert key_to_label("57-1") == "57¹-modda"
    assert key_to_label("12") == "12-modda"


def test_fold_and_apostrophes():
    assert fold("Oʻzbekiston  Respublikasi") == fold("O'zbekiston Respublikasi") == "ozbekiston respublikasi"
    assert fold("57¹-modda") == "57-1-modda"
    assert unify_apostrophes("o'g'il ta'til") == "oʻgʻil taʼtil"


def test_tokenize_stemming_consistent():
    a = tokenize("taʼtilga chiqish huquqi", "uz")
    b = tokenize("ta'tilning", "uz")
    assert a[0] == b[0] == "tatil"
    assert tokenize("Меҳнат таътили", "uz") == tokenize("Mehnat taʼtili", "uz")
    r1 = tokenize("трудового отпуска", "ru")
    r2 = tokenize("трудовой отпуск", "ru")
    assert r1 == r2
