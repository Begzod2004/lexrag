# LexRAG: Claude Code uchun loyiha konteksti

lex.uz (O'zbekiston qonunchiligi) bo'yicha RAG API. Maqsad: kursdagi «RAG API — test spetsifikatsiyasi (lex.uz)» bo'yicha eng yaxshi natija (jadvaldagi lider LexRAG-Haiku: 92 to'g'ri, Hit@5 88.5%, MRR 0.79).
Foydalanuvchi bilan **o'zbek tilida** gaplashing.

## Buyruqlar
- Testlar: `pytest -q` (tarmoqsiz ishlaydi, soxta LLM va hash-embedding bilan)
- Korpus: `python -m lexrag.ingest all --forms constitution,codes,laws` (+ `--forms decrees --max-decrees 300`)
- Indeks: `python -m lexrag.index build` (`--no-embed` = faqat BM25)
- Server: `uvicorn lexrag.api:app --port 8000`
- Tekshirish: `python scripts/smoke_test.py --base-url http://localhost:8000 --api-key KEY`
- Baholash: `scripts/run_eval.py` → `scripts/score_eval.py --judge` (dev to'plam: `eval/dev_set.csv`, 20 ta savol, moddalari lex.uz bilan tekshirilgan)

## Arxitektura (fayllar)
- `lexrag/lexuz/` — lex.uz HTML parser va crawler. `#divCont` ichidagi klasslar: `CLAUSE_DEFAULT` (modda), `ACT_TEXT`/`BY_DEFAULT` (matn), `CHANGES_ORIGINS` (tahrir sanasi), `TABLE_STD`. Lotin = `/uz/docs/-ID`, rus ID boshqa ("Рус" havolasidan). `57<sup>1</sup>` → kalit `57-1`, yorliq `57¹-modda`.
- `lexrag/ingest.py` — katalog (qidiruv sahifasi, `__doPostBack` sahifalash) + hujjatni 3 tilda yig'ish → `data/docs/*.json.gz`
- `lexrag/chunking.py` — modda darajasida chunk, >6000 belgi bo'lsa qism guruhlari
- `lexrag/index.py` — BM25 (scipy sparse), dense (numpy), embedding kesh (SQLite)
- `lexrag/retrieval.py` — LLM so'rov tahlili (HyDE), BM25+dense, RRF (`RRF_K`), hujjat ichida cheklangan qidiruv, LLM rerank
- `lexrag/generate.py` — javob prompti, quote verifikatsiya (belgi darajasida normallashtirib qidirish)
- `lexrag/pipeline.py` — savol oqimi, contexts tartibi, javob keshi
- `lexrag/service.py` + `api.py` — Starlette API, validatsiya, idempotentlik (SHA-256), async 202, token-bucket rate limit
- `docs/HANDOFF.md` — claude.ai suhbatining to'liq xulosasi (spetsifikatsiya, lex.uz topilmalari, qarorlar, tekshirilmaganlar)
- `docs/QANDAY_ISHLAYDI.md` — formulalar va tushuntirishlar (foydalanuvchi o'rganyapti — yangi narsani tushuntirib bering)

## Qoidalar
- Javob formati spetsifikatsiyaning 4-bo'limiga qat'iy mos bo'lishi shart; har `citations[].chunk_id` `contexts` ichida bo'lsin (testlar bor).
- `article` doim `N-modda` formatida (rus savolda ham). `doc_id` = `lex:<o'zbekcha ID>`, boshqa ID'lar `alt_doc_ids` da.
- Kalitlar faqat `.env` da (gitignore). Ularni kodga, commit'ga yoki loglarga yozmang.
- Yangi qidiruv sozlamasi qo'shilsa — `Engine.cache_key` dagi `knobs` ro'yxatiga ham qo'shing.

## Holat va keyingi qadamlar
- [x] Kod, 34 test, README, dev to'plam
- [ ] Haqiqiy korpusni yig'ish va indeks qurish (foydalanuvchi kompyuterida/serverida; kalitlar kerak: ANTHROPIC_API_KEY, OPENAI_API_KEY)
- [ ] dev_set bo'yicha birinchi run → `per_question.csv` dagi xatolar sababiga qarab tuzatish (retrieval / generation / source)
- [ ] `RRF_K`, `LEXRAG_GEN_K`, `MIN_CONTEXT_SCORE` ni dev to'plamda sozlash
- [ ] Testerga ochiq URL (cloudflared yoki VPS) va API kalit berish
