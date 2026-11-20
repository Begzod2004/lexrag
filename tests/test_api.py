"""Spetsifikatsiyadagi 7-bo'lim checklisti bo'yicha testlar."""
import time

from conftest import AUTH, FakeLLM, batch


def test_health_meta(client):
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    m = client.get("/v1/meta").json()
    for k in ("pipeline_version", "llm_model", "embedding_model", "reranker", "index_version", "lex_snapshot_date"):
        assert k in m["system"]
    assert m["system"]["embedding_model"] == "hash@256"


def test_auth(client):
    assert client.post("/v1/rag/batch", json=batch()).status_code == 401
    assert client.post("/v1/rag/batch", json=batch(), headers={"Authorization": "Bearer wrong"}).status_code == 403


def test_batch_basic_format(client):
    r = client.post("/v1/rag/batch", json=batch(3), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    j = r.json()
    assert j["status"] == "completed" and j["run_id"] == "run1" and j["batch_id"] == "run1-b01"
    for k in ("pipeline_version", "llm_model", "embedding_model", "reranker", "index_version", "lex_snapshot_date"):
        assert k in j["system"]
    assert set(j["timing"]) >= {"received_at", "completed_at", "total_ms"}
    assert j["timing"]["received_at"].endswith("+05:00")
    assert j["summary"]["total"] == 3
    ids = [x["id"] for x in j["results"]]
    assert ids == ["q001", "q002", "q003"]
    by = {x["id"]: x for x in j["results"]}
    ok = by["q001"]
    assert ok["status"] == "ok" and ok["no_answer"] is False and ok["answer"]
    assert ok["citations"], "javobda citations bo'lishi kerak"
    ctx_ids = {c["chunk_id"] for c in ok["contexts"]}
    for c in ok["citations"]:
        for f in ("ref", "doc_id", "url", "article", "part", "edition_date", "doc_status", "quote", "chunk_id"):
            assert f in c
        assert c["doc_id"].startswith("lex:") and c["url"].startswith("https://lex.uz/docs/")
        assert c["chunk_id"] in ctx_ids, "citation chunk_id contexts ichida bo'lishi shart"
        assert f"[{c['ref']}]" in ok["answer"]
        # quote manba matnidan aynan olingan
        ctx = next(x for x in ok["contexts"] if x["chunk_id"] == c["chunk_id"])
        assert c["quote"] in ctx["text"]
    for x in ok["contexts"]:
        for f in ("rank", "chunk_id", "doc_id", "article", "text", "retrieval_score", "rerank_score", "used_in_answer"):
            assert f in x
    assert [x["rank"] for x in ok["contexts"]] == list(range(1, len(ok["contexts"]) + 1))
    assert len(ok["contexts"]) <= 5
    assert set(ok["timing"]) >= {"retrieval_ms", "rerank_ms", "generation_ms", "total_ms"}
    # javobsiz savol
    na = by["q002"]
    assert na["status"] == "no_answer" and na["no_answer"] is True and na["citations"] == []
    assert j["summary"]["ok"] + j["summary"]["no_answer"] + j["summary"]["error"] == 3


def test_validation(client):
    b = batch(10, bid="v-b1")
    assert client.post("/v1/rag/batch", json=b, headers=AUTH).status_code == 200
    b = batch(11, bid="v-b2")
    r = client.post("/v1/rag/batch", json=b, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "TOO_MANY_ITEMS"
    b = batch(2, bid="v-b3")
    b["items"][1]["id"] = b["items"][0]["id"]
    assert client.post("/v1/rag/batch", json=b, headers=AUTH).status_code == 400
    b = batch(2, bid="v-b4")
    b["items"][0]["question"] = "   "
    assert client.post("/v1/rag/batch", json=b, headers=AUTH).status_code == 400
    b = batch(0, bid="v-b5")
    b["items"] = []
    assert client.post("/v1/rag/batch", json=b, headers=AUTH).status_code == 400
    r = client.post("/v1/rag/batch", content=b"{bad json", headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_idempotency_and_conflict(client, engine):
    b = batch(2, bid="idem-b1")
    r1 = client.post("/v1/rag/batch", json=b, headers=AUTH).json()
    calls_before = len(engine.llm_answer.calls)
    r2 = client.post("/v1/rag/batch", json=b, headers=AUTH).json()
    assert r1 == r2, "bir xil batch_id -> saqlangan natija"
    assert len(engine.llm_answer.calls) == calls_before, "qayta hisoblanmasligi kerak"
    b2 = batch(2, bid="idem-b1", questions=["Boshqa savol?"])
    assert client.post("/v1/rag/batch", json=b2, headers=AUTH).status_code == 409
    g = client.get("/v1/rag/batch/idem-b1", headers=AUTH)
    assert g.status_code == 200 and g.json() == r1
    assert client.get("/v1/rag/batch/yoq-batch", headers=AUTH).status_code == 404


def test_determinism_same_sources(client):
    q = ["Sifatsiz tovarni necha kun ichida almashtirish mumkin?"]
    a = client.post("/v1/rag/batch", json=batch(1, bid="det-1", questions=q), headers=AUTH).json()["results"][0]
    b = client.post("/v1/rag/batch", json=batch(1, bid="det-2", questions=q), headers=AUTH).json()["results"][0]
    assert [c["chunk_id"] for c in a["contexts"]] == [c["chunk_id"] for c in b["contexts"]]
    assert [c["chunk_id"] for c in a["citations"]] == [c["chunk_id"] for c in b["citations"]]


def test_languages_and_apostrophes(client):
    qs = ["Sifatsiz tovarni almashtirish muddati qancha?", "Сифатсиз товарни алмаштириш муддати қанча?", "Какой срок замены товара ненадлежащего качества?"]
    j = client.post("/v1/rag/batch", json=batch(3, bid="lang-1", questions=qs), headers=AUTH).json()
    langs = [r["answer_language"] for r in j["results"]]
    assert langs == ["uz-Latn", "uz-Cyrl", "ru"]
    top = [r["contexts"][0]["chunk_id"] for r in j["results"]]
    assert all(t == "lex:9990001#art-15" for t in top), top
    ru = j["results"][2]
    assert "Потребитель" in ru["contexts"][0]["text"]  # rus savolga rus matni
    lat = j["results"][0]["contexts"][0]["text"]
    assert "oʻn kun" in lat  # ʻ belgisi buzilmagan
    raw = client.post("/v1/rag/batch", json=batch(1, bid="lang-2", questions=["Isteʼmolchi tovarni oʻn kun ichida almashtira oladimi?"]), headers=AUTH)
    assert "oʻn" in raw.content.decode("utf-8")  # \u escape emas, UTF-8


def test_one_failure_does_not_break_batch(client, engine):
    def gen(user):
        if "kamsit" in user.split("\n")[1]:  # faqat savol qatori
            raise RuntimeError("generatsiya yiqildi")
        return FakeLLM().complete_json("x", user)[0]

    engine.llm_answer.gen_override = gen
    j = client.post("/v1/rag/batch", json=batch(3, bid="fail-1"), headers=AUTH)
    assert j.status_code == 200
    res = {r["id"]: r for r in j.json()["results"]}
    assert res["q001"]["status"] == "error" and res["q001"]["error"]["code"]
    assert res["q001"]["error"]["retryable"] is True
    assert res["q003"]["status"] == "ok"
    assert j.json()["summary"]["error"] == 1
    # qayta yuborilganda faqat xato savol qayta hisoblanadi
    engine.llm_answer.gen_override = None
    j2 = client.post("/v1/rag/batch", json=batch(3, bid="fail-1"), headers=AUTH).json()
    res2 = {r["id"]: r for r in j2["results"]}
    assert res2["q001"]["status"] == "ok"
    assert res2["q003"] == res["q003"]


def test_llm_down_returns_error_objects(client, engine):
    engine.llm_answer.fail = True
    j = client.post("/v1/rag/batch", json=batch(1, bid="down-1", questions=["Pulni qaytarish muddati?"]), headers=AUTH).json()
    r = j["results"][0]
    assert r["status"] == "error" and r["error"]["code"] == "LLM_TIMEOUT" and r["error"]["retryable"]
    assert r["answer"] is None and r["citations"] == []
    engine.llm_answer.fail = False


def test_async_mode(client):
    r = client.post("/v1/rag/batch?mode=async", json=batch(2, bid="async-1"), headers=AUTH)
    assert r.status_code in (200, 202)
    for _ in range(100):
        g = client.get("/v1/rag/batch/async-1", headers=AUTH)
        if g.status_code == 200:
            break
        assert g.json()["status"] == "processing"
        time.sleep(0.05)
    assert g.status_code == 200 and g.json()["status"] == "completed"


def test_rate_limit(client, engine):
    svc = client.app.state.get_service()
    svc.bucket.rate, svc.bucket.burst = 1.0, 1
    svc.bucket.state.clear()
    client.post("/v1/rag/batch", json=batch(1, bid="rl-1"), headers=AUTH)
    r = client.post("/v1/rag/batch", json=batch(1, bid="rl-2"), headers=AUTH)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_query_endpoint(client):
    r = client.post("/v1/rag/query", json={"question": "Pulni qaytarish muddati qancha?"}, headers=AUTH)
    assert r.status_code == 200
    j = r.json()
    assert j["result"]["status"] in ("ok", "no_answer") and "debug" in j["result"]


def test_return_flags(client):
    j = client.post("/v1/rag/batch", json=batch(1, bid="flags-1", return_contexts=False, return_debug=False), headers=AUTH).json()
    r = j["results"][0]
    assert r["contexts"] == [] and "debug" not in r


def test_rrf_k_changes_cache_key(engine):
    k1 = engine.cache_key("Savol?", None, {"top_k": 5})
    engine.s.rrf_k = 10
    k2 = engine.cache_key("Savol?", None, {"top_k": 5})
    engine.s.rrf_k = 60
    assert k1 != k2
