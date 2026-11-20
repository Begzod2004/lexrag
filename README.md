# LexRAG: lex.uz bo'yicha RAG API

O'zbekiston qonunchiligi (lex.uz) bo'yicha savol-javob tizimi. **«RAG API — test spetsifikatsiyasi (lex.uz)»** hujjatiga to'liq mos keladi: `POST /v1/rag/batch`, `GET /v1/rag/batch/{id}`, `POST /v1/rag/query`, `GET /v1/meta`, `GET /v1/health`.

## Nega kuchli (jadvaldagi jamoalardan farqi)

| Muammo (boshqalarda) | LexRAG yechimi |
|---|---|
| «Ortiqcha topilmadi» ko'p (13–51 ta), korpus tor | Barcha **amaldagi kodekslar (21) + Konstitutsiya + amaldagi qonunlar (~1255 tadan o'zgartirish aktlari chiqarib tashlangan)**, ixtiyoriy farmonlar. 3 tilda: lotin, kirill, rus |
| Recall@5 ≈ 54% | **Gibrid qidiruv**: BM25 (o'zbekcha stemming, kirill→lotin translit) + dense embedding + LLM so'rov tahlili (uz/ru qayta yozish, HyDE, kalit so'zlar) + **hujjat ichida cheklangan qidiruv** («Mehnat kodeksi» deb aniqlansa, faqat shu kodeks ichidan ham qidiriladi) + aniq modda havolalari |
| Noto'g'ri tartib (MRR) | **LLM-reranker** (Claude Haiku) 40 nomzoddan eng keraklilarini tanlaydi; generatsiya `relevant_sources` qaytaradi va kontekstlar shu tartibda beriladi |
| Sufficiency past (38–48%) | Chunk = **butun modda** (uzun moddalar qism guruhlariga bo'linadi), sarlavha + bob + modda nomi bilan |
| Gallyutsinatsiya / to'qilgan manba | Javob faqat manbadan; har `quote` manba matniga **belgi-belgi solishtiriladi**, mos kelmasa manbadagi eng yaqin jumla bilan almashtiriladi; `citation.chunk_id` doim `contexts` ichida |
| Rus/kirill savollar | Rus savolga rus matni, kirill savolga kirill matni; `article` har doim `N-modda` formatida |
| Generatsiya modeli | **Claude Sonnet** (javob) + **Claude Haiku** (tahlil, rerank). Claude ishlamasa OpenAI, keyin kursdagi Qwen'ga avtomatik o'tadi |

## Arxitektura

```
savol ─┬─> [Haiku] so'rov tahlili: uz/ru qayta yozish, HyDE, kalit so'zlar, ehtimoliy hujjat/modda
       └─> savol embeddingi
            │
            ▼
  BM25(uz) + BM25(ru) + dense(uz+ru)  ──>  RRF birlashtirish  <── hujjat ishoralari, aniq modda havolalari
            │                                (kuchini yo'qotgan hujjat ×0.5)
            ▼
  top-40 modda ──> [Haiku] listwise rerank ──> top-8 ──> [Sonnet] javob + iqtiboslar + relevant_sources
            │
            ▼
  quote verifikatsiya, [n] qayta raqamlash, contexts (iqtibos keltirilganlar birinchi), JSON (spec 4-bo'lim)
```

## O'rnatish

Python 3.10+ kerak.

**Windows (PowerShell):**
```powershell
cd lexrag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env        # kalitlarni yozing
```

**Linux/macOS:**
```bash
cd lexrag && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
```

`.env` da kamida quyidagilar:
- `ANTHROPIC_API_KEY`: Claude (javob va rerank uchun asosiy model).
- `OPENAI_API_KEY`: embedding (`text-embedding-3-large`) uchun.
- `LEXRAG_API_KEYS`: tester uchun alohida kalit (o'zingiz o'ylab toping).

## 1) Korpusni yig'ish (lex.uz)

```bash
# Konstitutsiya + 21 kodeks + amaldagi qonunlar (o'zgartirish aktlarisiz)
python -m lexrag.ingest all --forms constitution,codes,laws

# Tavsiya: BHM, eng kam ish haqi kabi raqamlar uchun so'nggi Prezident farmonlarini ham qo'shing
python -m lexrag.ingest all --forms decrees --max-decrees 300

# Alohida hujjatlar (masalan, test to'plamida uchragan qaror):
python -m lexrag.ingest fetch --ids 1234567,7654321
```
- Har bir hujjat uchun 4 ta so'rov (kirill, lotin, rus versiyalari + rekvizitlar kartochkasi). Taxminan **40–90 daqiqa** ketadi. Xom HTML `data/raw/` ga keshlanadi: qayta ishga tushirsangiz faqat yetishmaganlari yuklanadi.
- Natija: `data/docs/<doc_id>.json.gz` (moddalar, qismlar, tahrir sanalari, holati).

## 2) Indeksni qurish

```bash
python -m lexrag.index build      # BM25 + embedding
python -m lexrag.index stats
```
Embedding narxi taxminan 50M token × $0.13/1M ≈ **$6–8** (bir marta). Embeddinglar `data/emb_cache.sqlite3` ga keshlanadi, shuning uchun hujjat qo'shib qayta qursangiz faqat yangilari hisoblanadi. API kalit bo'lmasa: `python -m lexrag.index build --no-embed` (faqat BM25, sifati pastroq).

## 3) Serverni ishga tushirish

```bash
uvicorn lexrag.api:app --host 0.0.0.0 --port 8000
```
Tekshirish:
```bash
python scripts/smoke_test.py --base-url http://localhost:8000 --api-key <LEXRAG_API_KEYS dagi kalit>
```

## 4) Testerga ochiq URL berish

Tester serverga internetdan kira olishi kerak. Variantlar:
- **Cloudflare tunnel** (eng oson, kompyuteringizdan): `cloudflared tunnel --url http://localhost:8000` buyrug'i `https://xxxx.trycloudflare.com` beradi.
- **VPS** (2 vCPU, 4 GB RAM yetadi):
  `docker build -t lexrag . && docker run -d -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data lexrag`
- Render/Vercel'ning bepul tariflari indeks uchun xotira yetmasligi mumkin (~0.5–1 GB RAM kerak).

Testerga beriladi: **base URL** va **API kalit** (`LEXRAG_API_KEYS`).

## 5) O'zingiz baholash (spec 6-bo'lim metrikalari)

```bash
python scripts/run_eval.py --csv eval/dev_set.csv --base-url http://localhost:8000 --api-key KEY --run-id dev-1
python scripts/score_eval.py --csv eval/dev_set.csv --run runs/dev-1 --judge
```
`eval/dev_set.csv` da 20 ta savol bor. Ularning moddalari lex.uz matni bilan qo'lda tekshirilgan (Mehnat, Fuqarolik, Oila, Jinoyat, Soliq kodekslari, Konstitutsiya; rus, kirill, javobsiz va og'zaki savollar). O'z savollaringizni shu formatda qo'shing. Natijada Hit@5 (hujjat/modda), MRR, Recall@5, Precision@5, no-answer aniqligi, ortiqcha «topilmadi», citation aniqligi, latency p50/p95, `--judge` bilan esa javob to'g'riligi (0/1/2) va sufficiency hisoblanadi. `per_question.csv` har xatoga sabab qo'yadi (retrieval / generation / source).

## Sozlash profillari

| Profil | `.env` |
|---|---|
| **Eng kuchli (standart)** | `CLAUDE_ANSWER_MODEL=claude-sonnet-5-5`, `RERANKER=llm`, `QUERY_REWRITE=true` |
| Eng zo'r sifat, sekinroq | `CLAUDE_ANSWER_MODEL=claude-opus-5-5`, `LEXRAG_GEN_K=10` |
| Tez (p50 ≈ 3–4 s) | `CLAUDE_ANSWER_MODEL=claude-haiku-5-5`, `RERANKER=none`, `LEXRAG_GEN_K=10` |
| Faqat kursdagi Qwen (internet LLMsiz) | `LLM_PROVIDER=qwen`, `QWEN_BASE_URL=http://10.20.9.24:8000/v1` (faqat o'sha tarmoqda ishlaydi) |
| GPU bor, lokal embedding | `EMBEDDING_PROVIDER=st`, `EMBEDDING_MODEL=BAAI/bge-m3`, `pip install sentence-transformers` |

`DOC_ID_MODE=canonical` bo'lsa, `doc_id` doim o'zbekcha hujjat raqami bo'ladi (`lex:6257288`). Rus versiya ID'si (`lex:6257291`) va lotin (`lex:-6257288`) `alt_doc_ids` da beriladi. Agar tester rus savollar uchun rus ID kutsa, `DOC_ID_MODE=by_language` qiling.

## Spetsifikatsiya checklisti (7-bo'lim)

- [x] `POST /v1/rag/batch`: 1–10 savol qabul qiladi, 11 tasiga `400 TOO_MANY_ITEMS`; takroriy `id` yoki bo'sh savolga `400`
- [x] Javobda har `id` bir marta, tartib saqlanadi
- [x] `citations`: `doc_id`, `url`, `article`, `part`, `edition_date`, `doc_status`, `quote` (+ `doc_title`, `doc_type`, `alt_doc_ids`)
- [x] `contexts`: matn, `retrieval_score`, `rerank_score`, `used_in_answer`
- [x] Har `citations[].chunk_id` `contexts` ichida bor (testlar bilan tekshirilgan)
- [x] Javobsiz savolda `no_answer: true`, `citations: []`
- [x] Bitta savoldagi xato boshqalarga ta'sir qilmaydi (`status: error` + `error{code,message,retryable}`). Shu `batch_id` qayta yuborilsa, faqat xato savollar qayta hisoblanadi
- [x] `system`: model, embedding, reranker, indeks versiyasi, lex.uz snapshot sanasi
- [x] `temperature=0` + `seed`: retrieval deterministik, javoblar keshlanadi, demak bir xil savol bir xil manbalarni qaytaradi
- [x] Bir xil `batch_id` qayta yuborilsa saqlangan natija qaytadi (qayta hisoblanmaydi); boshqa mazmun bilan yuborilsa `409`
- [x] Lotin, kirill, rus; `oʻ`/`gʻ`/`ʼ` buzilmaydi (UTF-8, `\u` escape yo'q)
- [x] Sync ≤110 s, oshsa `202` + `GET` poll; `?mode=async` bilan darhol `202`
- [x] `GET /v1/meta`, `GET /v1/health`
- [x] `Authorization: Bearer`, `401`/`403`; rate limit `429` + `Retry-After`

## Spetsifikatsiyadagi ochiq savollarga javoblar

- **Ma'lumot sanasi:** `lex_snapshot_date` = korpus yuklangan sana (`/v1/meta`). Yangilash uchun `ingest fetch --refresh` va `index build` buyruqlari, tavsiya: haftada bir marta.
- **Chunking:** modda bo'yicha (`lex:ID#art-N`). 6000 belgidan uzun moddalar qism guruhlariga bo'linadi (`#art-N#p3-5`). Moddasiz hujjatlar (farmon, qaror) bandlar bo'yicha (`#band-N`).
- **Kuchini yo'qotgan hujjatlar:** faqat amaldagilar yig'iladi. Agar qo'shilsa, ular `doc_status: "kuchini yo'qotgan"` deb belgilanadi, ball ×0.5 qilinadi va LLM ularga tayanmaydi. `edition_date` hujjatning amaldagi tahriri sanasi. Moddaning o'ziga kiritilgan oxirgi o'zgartish sanasi `article_edition_date` da.

## O'rganish uchun

Tizim qanday qurilgani, BM25, kosinus o'xshashlik, RRF, MRR va boshqa formulalar: [docs/QANDAY_ISHLAYDI.md](docs/QANDAY_ISHLAYDI.md).

## Testlar

```bash
pytest -q        # 34 test: parser (lex.uz'ning haqiqiy HTML bo'laklari), API checklist, generatsiya, baholash
```

## Muammolar

- `index_loaded: false`: avval `ingest` va `index build` buyruqlarini bajaring, `LEXRAG_DATA_DIR` ni tekshiring.
- `LLM_ERROR ... 401`: API kalit noto'g'ri. `404 model`: `.env` dagi model nomini tekshiring.
- lex.uz sekin yoki bloklayapti: `CRAWL_DELAY_S=1` va `CRAWL_WORKERS=2` qilib qayta ishga tushiring, kesh saqlanib qoladi.
