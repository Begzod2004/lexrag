# LexRAG: suhbatdan to'liq topshirish (handoff) hujjati

> Claude Code uchun. Bu hujjat claude.ai'dagi suhbatda (2026-11-21, 01:08–02:18, Toshkent vaqti) qilingan barcha ishlarni, topilmalarni va qarorlarni jamlaydi. Foydalanuvchi bilan **o'zbek tilida** gaplashing. U Data Science kursida o'qiyapti, shuning uchun har bir yangi narsani tushuntirib bering.
> Repo: https://github.com/Begzod2004/lexrag (hozircha **public**, bitta asosiy commit). Qisqa kontekst `CLAUDE.md` da, formulalar `docs/QANDAY_ISHLAYDI.md` da.

---

## 1. Vazifa va kontekst

- Foydalanuvchi (GitHub: **Begzod2004**) 3 oylik intensiv "Data Scientist / ML-AI Engineer" kursida o'qiydi. Kursning 8-haftasi RAG, 10-haftasi baholash.
- Topshiriq: **lex.uz uchun RAG API** yozish. Asos: «RAG API — test spetsifikatsiyasi (lex.uz)» hujjati (muallif @jurabek, 2026-11-03). Foydalanuvchining talabi: "eng zo'ri bizniki bo'lishi kerak". U Claude ishlatilishini so'radi, shuning uchun asosiy LLM Claude.
- Jamoalar reytingi (foydalanuvchi yuborgan jadval):

| Tizim (jamoa) | To'g'ri | Qisman | Xato | Ortiqcha «topilmadi» | Recall@5 | Precision@5 | Hit@5 | MRR@5 | Sufficiency | Latency p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| LexRAG-Haiku (Alibek, Muhammad, Abdulaziz) | **92** | 6 | 2 | ≈2 | 54,2% | 58,1% | **88,5%** | **0,79** | 38,5% | 5,8 s |
| Dalil (5-guruh, Dilshod) | ≈66 | ≈12 | ≈3 | 15 (+2 xato) | 54,6% | 57,0% | 76,6% | 0,68 | 44,7% | 4,3 s |
| Qonunchi (Xafizullo) | ≈59 | — | — | ≈13 | 54,4% | 53,8% | 78,8% | 0,73 | 48,1% | 5,5 s |
| lex.gamerly.uz (Batirniyaz) | ≈57 | ≈21 | ≈2 | 15 | — | — | — | — | — | 6,7 s |
| Render (8-komanda, Sardor) | ≈47 | — | — | ≈39 | 56,8% | 55,4% | 82,7% | 0,70 | 40,4% | 3,6 s |
| Vercel (Azizbek) | ≈30 | ≈12 | ≈11 | ≈45–51 | 56,7% | 24,6% | 61,5% | 0,43 | 44,2% | 10,3 s |

  Xulosa: boshqa jamoalar asosan **korpus torligi** sababli yutqazgan ("ortiqcha topilmadi" ko'p). Recall@5 hammada ≈54% atrofida, Sufficiency esa past.

## 2. Spetsifikatsiya qisqacha (to'liq matni foydalanuvchida, PDF)

**Endpointlar:** `POST /v1/rag/batch` (majburiy, 1–10 savol), `GET /v1/rag/batch/{batch_id}` (majburiy), `POST /v1/rag/query` (ixtiyoriy, debug), `GET /v1/meta` (majburiy), `GET /v1/health` (majburiy).

**Texnik talablar:** `Authorization: Bearer <API_KEY>`; UTF-8; uz-Latn, uz-Cyrl va ru tillari, `oʻ gʻ ʼ` belgilari buzilmasligi kerak; 10 tadan ko'p savolga `400`; bitta batch ≤120 s (sync), oshsa async rejim `202` + GET; rate limit ≥1 batch/s, oshsa `429` + `Retry-After`; `temperature=0` + `seed`, bir xil savol bir xil manbalarni qaytarishi kerak; bir xil `batch_id` qayta kelsa saqlangan natija qaytadi; har javobda `system` obyekti bo'ladi.

**Request:** `run_id`, `batch_id`, `batch_index`, `total_batches`, `options{top_k=5, return_contexts, return_debug, temperature, seed, as_of_date, answer_language=auto|uz-Latn|uz-Cyrl|ru}`, `items[{id, question (1–2000 belgi), language?}]`.

**Response:** `run_id, batch_id, status, system{pipeline_version, llm_model, embedding_model, reranker, index_version, lex_snapshot_date}, timing{received_at, completed_at, total_ms}, summary{total, ok, no_answer, error}, results[]`. Har bir natija: `id, status(ok|no_answer|error|timeout), answer ([1] havolalar bilan), answer_language, no_answer, confidence, citations[{ref, doc_id "lex:…", doc_title, doc_type, url, article "N-modda", part "N-qism", edition_date, doc_status "amalda", chunk_id "lex:ID#art-N#p1", quote}], contexts[{rank, chunk_id, doc_id, article, text, retrieval_score, rerank_score, used_in_answer}], debug, timing{retrieval_ms, rerank_ms, generation_ms, total_ms}, usage, error`.

**Qoidalar:** results soni items soniga teng; har `citations[].chunk_id` `contexts` ichida bo'lishi shart; `doc_id` lex.uz raqamiga barqaror bog'langan bo'ladi. Bitta savoldagi xato batch'ni yiqitmaydi: batch baribir `200` qaytaradi, xato `error{code, message, retryable}` da bo'ladi. HTTP kodlar: `200/202/400/401/403/409/429/500/503/504` (tester 5xx da 3 marta qayta urinadi: 5, 15, 45 s). Xato kodlari: `RETRIEVAL_FAILED, INDEX_UNAVAILABLE, LLM_TIMEOUT, LLM_ERROR, CONTEXT_TOO_LONG, INVALID_LANGUAGE`.

**Test to'plami (100 savol, 10 batch, har birida 10 ta):** faktik bitta modda 35, raqamlar (muddat, summa, foiz, BHM) 15, bir nechta modda/hujjat 15, o'zgargan yoki kuchini yo'qotgan normalar 10, rus/kirill 10, javobsiz 10, noaniq/og'zaki 5.
CSV formati: `id,batch,category,question,expected_answer,expected_doc_id,expected_article,answerable`.

**Maqsad metrikalar:** Hit@5 (hujjat) ≥90%, Hit@5 (modda) ≥80%, MRR ≥0.70, context precision ≥60%, answer correctness o'rtacha ≥1.6/2, faithfulness ≥95%, citation accuracy ≥90%, to'qilgan manba 0, eskirgan norma 0, no-answer ≥9/10, latency p50 ≤5 s va p95 ≤12 s, xato ulushi ≤2%.

## 3. lex.uz bo'yicha topilmalar (built-in brauzerda tekshirilgan, 2026-11-21)

- **URL:** `/uz/docs/ID` o'zbekcha kirill (kanonik ID), `/uz/docs/-ID` lotin (manfiy ID), rus versiyasining ID'si boshqa: masalan Mehnat kodeksi `6257288` → rus `6257291`. Rus ID sahifadagi `openUrl('/uz/docs/6257291')` ("Рус" tugmasi) dan olinadi. Rus sahifadan o'zbekcha ID'ga qaytish havolasi: "Ўзб" → `/ru/docs/6257288`.
- Sahifalar **server tomonida render qilinadi** (JS kerak emas), `requests` + `lxml` yetarli. Kodekslarning HTML hajmi ~3 MB.
- **DOM:** `#divCont > div.<KLASS> > div[id]`. Klasslar: `ACT_TITLE`, `TEXT_HEADER_DEFAULT` (bo'lim/bob, ichida `<br>` bo'lishi mumkin), `CLAUSE_DEFAULT` (`N-modda. Nomi`, ba'zan bo'sh), `ACT_TEXT`, `BY_DEFAULT` (bo'sh yoki matnli), `TABLE_STD`/`TABLE_STD2`, `CHANGES_ORIGINS` (`(4-moddaning ikkinchi qismi ... 13.11.2026-y. ...)`), `COMMENT` ("Oldingi tahrirga qarang", "Qarang: ...", "LexUZ sharhi"), `INDEXES_ON_REF` (klassifikator, yashirin), `PUBLICATION_ORIGIN`, `ACT_FORM`, `NEW_EDITION`. Har bir elementda `.lx_elem2` menyusi bor ("Hujjatga taklif yuborish" va h.k.), uni olib tashlash kerak.
- `57<sup>1</sup>-modda`: `textContent` uni `571-modda` deb o'qiydi, bu xato. Kod `<sup>` ni `¹` ga o'giradi, kalit `57-1` bo'ladi. Konstitutsiya moddalarida nom yo'q (`106-modda.`). Rus sahifada format `Статья 1. ...`.
- Lotin matnda `ʻ` (U+02BB, oʻ/gʻ uchun) va `ʼ` (U+02BC, tutuq belgisi) ishlatiladi.
- **Kuchini yo'qotgan hujjat:** sahifada `.act_warning` bloki bo'ladi ("Ҳужжат кучини йўқотган 30.04.2023"). Misol: eski Mehnat kodeksi `142859`.
- **Rekvizitlar kartochkasi:** `/uz/actinfo/card1/{ID}`. Jadvalda `TD.lbl` dan keyin `TD.codL/codR` keladi: "Форма акта" (Кодекс/Закон/Указ/Постановление), "Статус акта" (Действующий / Утративший силу), "Дата вступления в силу", "Дата прекращения действия"; `table#bodyTab` ichida "Дата принятия" va "Номер акта" (masalan ЎРҚ-798).
- **Qidiruv/katalog:** `/uz/search/nat?lang=3&form_id=…&status=Y&nature=1`.
  - `form_id`: 4131 Konstitutsiya, 3964 Kodeks, 3968 Qonun, 3973 Farmon, 3972 Qaror, 588 Farmoyish.
  - `lang`: 3 = ЎЗБ (kirill), 4 = O'ZB (lotin), 1 = РУС, 2 = ENG.
  - `status`: Y amaldagi, R kuchini yo'qotgan, N amalda emas. `nature`: 1 normativ, 2 operativ, 3 o'zgartirish va qo'shimchalar.
  - Bir sahifada 20 natija. Sahifalash ASP.NET `__doPostBack('ucFoundActsControl$LinkButton1','')` orqali ("Keyingisi"): yashirin `__VIEWSTATE` maydonlari bilan POST qilinadi. Hujjatlar soni "N hujjat topildi" matnida.
  - Amaldagi hujjatlar soni: **21 kodeks**, **1255 qonun** (`nature=1`, ichida o'zgartirish kirituvchi qonunlar ham bor, ular sarlavha bo'yicha filtrlanadi), **1121 farmon**, 2 konstitutsiya (Qoraqalpog'iston konstitutsiyasini o'tkazib yuboramiz: unda ә ө ү ң harflari bor).
  - lex.uz'ning o'z matn qidiruvi sanaga ko'ra saralaydi, ahamiyatlilik bo'yicha emas. Shuning uchun uni "jonli fallback" sifatida ishlatish foydasiz.
- **Amaldagi kodekslar (kanonik ID):** Suv 7655343, Mehnat 6257288, Shaharsozlik 5307951, Soliq 4674902, Saylov 4386848 (4492619 qoraqalpoqcha), Maʼmuriy sud ishlarini yuritish 3527353, Iqtisodiy protsessual 3523891, Fuqarolik protsessual 3517337, Bojxona 2876354, Byudjet 2304138, Uy-joy 106136, Oila 104720, Yer 152653, Jinoyat-ijroiya 163629, Fuqarolik II qism 180552, Fuqarolik I qism 111189, Jinoyat-protsessual 111460, Jinoyat 111453, Maʼmuriy javobgarlik 97664, Havo 55594. Konstitutsiya (2023): 6445145.
- **Tekshirilgan faktlar** (`eval/dev_set.csv` shular asosida tuzilgan):
  - Mehnat kodeksi: 217-m. har yilgi asosiy eng kam taʼtil 21 kalendar kun; 218-m. 18 yoshgacha bo'lganlarga 30 kun; 130-m. sinov muddati ≤3 oy (rahbarlar ≤6 oy); 132-m. sinovda kamida 3 kun oldin ogohlantiriladi; 160-m. xodim o'z tashabbusi bilan ketsa 14 kalendar kun oldin ogohlantiradi; 182-m. ish haftasi ≤40 soat; 118-m. ishga qabul 16 yoshdan; 404-m. homiladorlik va tug'ish taʼtili 70+56 kun (qiyin tug'ishda 70); 4-m. 2-qismi 2026-11-13 dagi O'RQ-1189 bilan tahrir qilingan.
  - Fuqarolik kodeksi I qism: 150-m. umumiy daʼvo muddati 3 yil; 22-m. to'liq muomala layoqati 18 yoshda.
  - Oila kodeksi: 15-m. nikoh yoshi 18. Jinoyat kodeksi: 17-m. javobgarlik 16 yoshdan (ayrim jinoyatlar uchun 14).
  - Soliq kodeksi: 258-m. QQS 12% (36-bob holatlarida 0%); 381-m. rezident jismoniy shaxslar uchun daromad solig'i 12%.
  - Konstitutsiya: 106-m. Prezident kamida 35 yoshda bo'lishi kerak, 7 yilga saylanadi.

## 4. Arxitektura va qarorlarning sababi

```
savol ─┬─> [Claude Haiku] tahlil: uz/ru qayta yozish, kalit so'zlar, HyDE, ehtimoliy hujjat/modda
       └─> savol embeddingi (OpenAI text-embedding-3-large @1024)
→ BM25(uz, translit+stemming) + BM25(ru) + dense (barcha so'rovlar bitta matmul'da)
→ RRF (k=60, og'irliklar bilan) + hujjat ichida cheklangan qidiruv + aniq/taxminiy modda havolalari
→ kuchini yo'qotgan ×0.5 → top-40 → [Haiku] listwise rerank → top-8
→ [Claude Sonnet] JSON javob: answer[n], citations{source, part, quote}, relevant_sources, confidence
→ quote verifikatsiyasi (mos kelmasa manbadagi eng yaqin jumla olinadi), [n] qayta raqamlanadi
→ contexts: relevant/cited birinchi, keyin rerank ≥0.35, kamida 3 ta
```

- **Chunk = modda.** Spetsifikatsiya `article` va `part` aniqligini talab qiladi, Sufficiency ham butun moddada yuqori bo'ladi. 6000 belgidan uzun moddalar qism guruhlariga bo'linadi. Qism quyidagicha aniqlanadi: oldingi paragraf "." bilan tugasa va keyingisi katta harf bilan boshlansa, yangi qism. ":" ";" "," "—" bilan tugasa, ro'yxat davom etyapti.
- **3 til.** Lotin matn indekslanadi (BM25 + dense), rus matn alohida indekslanadi. Kirill savol transliteratsiya qilinadi. Kontekst javob tilida qaytariladi: rus savolga rus matni, kirillga kirill.
- `doc_id` = o'zbekcha kanonik ID, boshqa ID'lar `alt_doc_ids` da. Noaniqlik bor: tester rus savollar uchun rus ID kutishi mumkin. Shu holat uchun `DOC_ID_MODE=by_language` sozlamasi bor.
- `edition_date` = hujjatning amaldagi tahriri sanasi (barcha `CHANGES_ORIGINS` ichidagi eng so'nggi sana). Moddaning o'ziga kiritilgan oxirgi o'zgartish `article_edition_date` da.
- **Takrorlanuvchanlik:** retrieval deterministik. Javob keshining kaliti: savol, opsiyalar, `system_info` va qidiruv sozlamalari (`knobs`).
- **Idempotentlik:** kanonik so'rovdan SHA-256 xesh olinadi. Xesh boshqacha bo'lsa `409`. Saqlangan natijada `retryable` xato bo'lsa, qayta yuborilganda faqat o'sha savollar qayta hisoblanadi.
- **LLM fallback:** Claude → OpenAI → Qwen (`.env` dagi kalitlarga qarab).
- **Latency taxmini:** p50 ≈ 5–7 s (Haiku tahlili ~1.2 s + Haiku rerank ~1.2 s + Sonnet ~3 s). Tezlatish uchun `RERANKER=none` yoki javobni Haiku yozadi.

## 5. Fayllar va buyruqlar

`lexrag/` paketi: `config.py` (barcha `.env` sozlamalari), `textnorm.py` (translit, fold, stemming, til aniqlash), `lexuz/{parser,client}.py`, `ingest.py`, `chunking.py`, `embeddings.py` (openai | st | hash | none), `index.py` (BM25, dense, EmbCache, `resolve_doc`, `DOC_ALIASES`), `llm.py` (Anthropic, OpenAI va Qwen REST'i to'g'ridan-to'g'ri `requests` orqali; Qwen uchun `enable_thinking=false`), `retrieval.py`, `generate.py`, `pipeline.py` (Engine), `storage.py` (SQLite), `service.py`, `api.py` (Starlette).
`scripts/`: `run_eval.py` (tester ssenariysi), `score_eval.py` (metrikalar, `--judge`), `smoke_test.py`. `eval/dev_set.csv`. `tests/` (34 test, haqiqiy lex.uz HTML bo'laklari `tests/fixtures/` da).

```bash
pip install -r requirements.txt && cp .env.example .env   # kalitlarni yozing
python -m lexrag.ingest all --forms constitution,codes,laws          # ~40–90 daqiqa
python -m lexrag.ingest all --forms decrees --max-decrees 300        # BHM va eng kam ish haqi kabi raqamlar uchun
python -m lexrag.index build                                         # embedding ≈ $6–8, data/emb_cache.sqlite3 ga keshlanadi
uvicorn lexrag.api:app --host 0.0.0.0 --port 8000
python scripts/smoke_test.py --base-url http://localhost:8000 --api-key KEY
python scripts/run_eval.py --csv eval/dev_set.csv --api-key KEY --run-id dev-1
python scripts/score_eval.py --csv eval/dev_set.csv --run runs/dev-1 --judge
cloudflared tunnel --url http://localhost:8000                       # testerga ochiq URL
```

Muhim `.env` sozlamalari: `LEXRAG_API_KEYS`, `ANTHROPIC_API_KEY`, `CLAUDE_ANSWER_MODEL=claude-sonnet-5-5`, `CLAUDE_FAST_MODEL=claude-haiku-5-5`, `OPENAI_API_KEY`, `OPENAI_ANSWER_MODEL=gpt-5.6-sol` (model nomini tekshiring), `QWEN_BASE_URL=http://10.20.9.24:8000/v1`, `QWEN_MODEL=Qwen/Qwen3-8B-AWQ`, `EMBEDDING_PROVIDER/MODEL/DIMS`, `RERANKER=llm|cross_encoder|none`, `QUERY_REWRITE`, `RRF_K=60`, `LEXRAG_POOL_SIZE=40`, `LEXRAG_GEN_K=8`, `ADAPTIVE_CONTEXTS`, `MIN_CONTEXTS=3`, `MIN_CONTEXT_SCORE=0.35`, `DOC_ID_MODE`, `DENSE_DTYPE`.

## 6. Nima tekshirilgan va nima tekshirilmagan

- ✅ 34 test o'tadi: parser (haqiqiy lex.uz bo'laklarida), ingest (soxta mijoz bilan), spetsifikatsiya checklisti (API), generatsiya va quote tuzatish, baholash skripti. Retrieval tezligi sintetik 80k qatorda o'lchangan: dense ~65 ms, BM25 <1 ms.
- ❌ **Haqiqiy lex.uz crawl va haqiqiy LLM/embedding chaqiruvlari hali ishga tushirilmagan.** Claude sandbox'ida tarmoq yopiq edi. Birinchi real ishga tushirishda shularga e'tibor bering:
  1. `ingest` loglari: har bir hujjatdagi moddalar soni mantiqiymi (Mehnat kodeksida ~594 ta, Jinoyatda ~408 ta)? `fetch_errors` bormi?
  2. LLM model nomlari (Claude va OpenAI) 404 bermayaptimi?
  3. Promptlar haqiqiy javoblarda JSON'ni to'g'ri qaytaryaptimi? `debug.quote_repairs` soni ko'p emasmi?
  4. dev_set bo'yicha Hit@5 va MRR qancha chiqdi? `per_question.csv` da har xatoning sababi yozilgan.

## 7. Ochiq masalalar va g'oyalar

- `as_of_date` hisobga olinmaydi: indeksda faqat amaldagi tahrir bor.
- "O'zgargan/kuchini yo'qotgan" kategoriyasi uchun: hozir faqat amaldagi hujjatlar yig'iladi. Kerak bo'lsa `status=R` dagi asosiy eski kodekslarni `kuchini yo'qotgan` belgisi bilan qo'shish mumkin (ball ×0.5, LLM ularga tayanmaydi).
- BHM va eng kam ish haqi miqdorlari farmonlarda yozilgan, ularni korpusga albatta qo'shing (`--forms decrees`).
- Sozlash uchun nomzodlar: `RRF_K`, `LEXRAG_GEN_K`, `MIN_CONTEXT_SCORE`, `LEXRAG_POOL_SIZE`. RRF'ning sof taʼsirini ko'rish uchun avval `RERANKER=none` bilan solishtiring.
- Xavfsizlik: suhbatda OpenAI kaliti, Qwen kaliti va sales bazasining paroli ochiq yozilgan edi. Ularni **almashtirish tavsiya qilindi**, ular hech qaysi faylga yozilmagan. Claude GitHub App o'rnatildi; "All repositories" tanlangan bo'lishi mumkin, uni faqat `lexrag` bilan cheklash tavsiya qilindi. Repo public; musobaqa tugaguncha private qilish tavsiya qilindi.
- RAG'ga aloqasi yo'q materiallar (foydalanuvchi suhbatga yuborgan): `raw_data.csv` (kredit skoring, 37 ustun: Client_ID ... Decision, Is_Emergency), `Topshiriqlar.docx` (orders_raw.csv bo'yicha 30 ta pandas data-cleaning topshirig'i), roadmap docx, "Linear Regression in 3 Minutes" videosi, sales PostgreSQL bazasi (10.20.9.24:5440, `sales` DB, `sales_reader` foydalanuvchi). Ular alohida qilinishi mumkin.

## 8. O'rgatish konteksti

Foydalanuvchi "qanday qilding, qanday formulalar" deb so'radi. `docs/QANDAY_ISHLAYDI.md` da quyidagilar tushuntirilgan: BM25 (IDF, k1 va b, raqamli misol), kosinus o'xshashlik va L2-normallashtirish, `S = D·Qᵀ`, xotira hisobi, HyDE, RRF, argpartition, sigmoid, quote verifikatsiyasi, token bucket, Hit@k, MRR, Recall, Precision va persentil. Shuningdek RRF'dagi k haqida savoliga javob berilgan: 1-o'rinning 10-o'ringa nisbati `(k+10)/(k+1)`; ikkita ro'yxatdagi r-o'rin bitta ro'yxatdagi 1-o'rinni `r < k+2` bo'lsa yengadi; 60 RRF maqolasidan (Cormack va boshq., 2009) qolgan empirik standart. Har gal yangi narsani shu uslubda, misol bilan tushuntiring.

## 9. Claude Code uchun tavsiya etilgan birinchi so'rov

> "docs/HANDOFF.md va CLAUDE.md ni o'qi. Keyin .env ni birga to'ldiramiz, korpusni yig'amiz (avval `--limit 5` bilan sinab ko'ramiz), indeks quramiz, smoke test va dev_set bo'yicha baholashni o'tkazamiz. Har qadamni o'zbekcha tushuntirib bor."
