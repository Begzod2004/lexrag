# LexRAG qanday ishlaydi: qadam-baqadam va matematikasi bilan

Bu qo'llanma tizim qanday qurilganini, har bir qarorning sababini va ishlatilgan formulalarni tushuntiradi. Roadmap'dagi mavzular bilan bog'lanishi oxirida berilgan.

---

## 0. Umumiy g'oya: RAG nima?

**RAG = Retrieval-Augmented Generation**, ya'ni "qidirib topib, keyin javob yozish".

LLM (Claude, GPT, Qwen) O'zbekiston qonunlarini aniq yodlab olmagan. Undan bevosita so'rasangiz, modda raqamini yoki muddatni o'ylab topishi (gallyutsinatsiya) mumkin. Shuning uchun ish ikki bosqichga bo'linadi:

1. **Retrieval:** savolga tegishli moddalarni bazadan topamiz.
2. **Generation:** LLM'ga *faqat shu moddalarni* berib, "faqat shulardan foydalanib javob ber" deymiz.

Spetsifikatsiya ham shu ikki qismni alohida baholaydi: qidiruv uchun Hit@5, MRR va Recall, generatsiya uchun to'g'rilik va faithfulness.

```
lex.uz ──> [1] yig'ish ──> [2] bo'lish (chunking) ──> [3] indekslash (BM25 + embedding)
                                                              │
savol ──> [4] so'rovni tahlil qilish ──> [5] qidiruv + RRF ──> [6] rerank ──> [7] LLM javob ──> [8] tekshirish ──> JSON
```

---

## 1. Ma'lumot yig'ish: lex.uz tuzilmasini qanday aniqladim

lex.uz'da rasmiy API yo'q, shuning uchun HTML'ni o'qiymiz (scraping). Avval brauzerda sahifani ochib, JavaScript bilan DOM tuzilmasini o'rgandim (DevTools'dagi kabi):

```js
const dc = document.getElementById('divCont');
for (const c of dc.children) counts[c.className]++;
// => CLAUSE_DEFAULT: 594, ACT_TEXT: 3028, CHANGES_ORIGINS: 96, COMMENT: 466 ...
```

Topilgan qoidalar:

| HTML klassi | Ma'nosi |
|---|---|
| `CLAUSE_DEFAULT` | modda sarlavhasi: `217-modda. Har yilgi asosiy eng kam ...` |
| `ACT_TEXT`, `BY_DEFAULT` | modda matni (har biri bitta paragraf) |
| `TEXT_HEADER_DEFAULT` | bo'lim / bob sarlavhasi |
| `CHANGES_ORIGINS` | tahrir izohi: `(4-moddaning ikkinchi qismi ... 13.11.2026-y. ...)` |
| `TABLE_STD` | jadval (masalan, "ancha miqdor — BHMning 100–300 baravari") |
| `.act_warning` | "Hujjat kuchini yo'qotgan" ogohlantirishi |

URL qoidalari:
- `/uz/docs/6257288`: o'zbekcha **kirill** (asosiy ID),
- `/uz/docs/-6257288`: **lotin** (manfiy ID!),
- rus versiyasining ID'si boshqa (`6257291`); u sahifadagi "Рус" tugmasidan olinadi.

Hujjatlar ro'yxati qidiruv sahifasidan olinadi: `form_id=3964` (kodeks), `3968` (qonun), `status=Y` (amaldagi), `nature=1` (normativ). Sahifalash ASP.NET `__doPostBack` orqali ishlaydi, shuning uchun yashirin `__VIEWSTATE` maydonlarini qaytarib POST qilamiz.

`57<sup>1</sup>-modda` kabi moddalarni ehtiyot qilish kerak: oddiy `textContent` ularni `571-modda` deb o'qiydi (571-modda bilan adashadi!). Shuning uchun `<sup>` ni `¹` belgisiga, kalitni esa `57-1` ga o'giramiz.

---

## 2. Bo'lish (chunking): nega butun modda?

Bitta **chunk** = qidiruv birligi. Variantlar:
- belgilangan uzunlik (masalan, 500 token): moddani o'rtasidan kesib yuboradi, `article` maydoni noaniq bo'ladi;
- **modda bo'yicha:** huquqiy matnning tabiiy birligi. Spetsifikatsiya ham `article` va `part` darajasida aniqlik so'raydi.

Qoida: modda ≤ 6000 belgi bo'lsa, bitta chunk (`lex:6257288#art-217`). Uzunroq bo'lsa, qismlar guruhiga bo'linadi (`#art-4#p3-5`).

**Qismni (part) aniqlash algoritmi.** Paragraflar ketma-ketligida yangi qism quyidagicha boshlanadi:

```
agar oldingi paragraf ":" ";" "," "—" bilan tugasa  →  ro'yxat davom etyapti (shu qism)
agar oldingi "." bilan tugasa VA joriy katta harf bilan boshlansa  →  yangi qism
```

Tekshiruv: 4-moddaning tahrir izohida "4-moddaning **ikkinchi** qismi" deyilgan, algoritm ham aynan shu paragrafni 2-qism deb topdi.

---

## 3. Matnni normallashtirish

**Transliteratsiya (kirill → lotin).** Kirill savol lotin indeksda ham topilishi uchun: `ў→oʻ, қ→q, ғ→gʻ, ҳ→h, ш→sh, ч→ch, ю→yu, я→ya`. Qiyin holat `е`: so'z boshida yoki unlidan keyin `ye`, aks holda `e` (`ер→yer`, `меҳнат→mehnat`).

**Apostroflar.** Odamlar `o'`, `oʻ`, `o‘`, `o`` deb turlicha yozadi. Qidiruv uchun hammasini olib tashlaymiz (`to'g'ri → togri`), shunda yozilish farqi xalaqit bermaydi.

**Stemming (so'z o'zagini ajratish).** O'zbek tili agglyutinativ, ya'ni bitta so'zga ko'p qo'shimcha qo'shiladi: `taʼtil-lar-i-ning`. Qo'shimchalarni uzunidan qisqasiga qarab kesamiz (o'zak kamida 4 harf qolishi sharti bilan):

```
taʼtilga → tatilga → tatil        taʼtilning → tatil        taʼtillari → tatil
```

Natijada "taʼtilga", "taʼtilning" va "taʼtili" bitta so'z sifatida hisoblanadi.

---

## 4. BM25: kalit so'z bo'yicha qidiruv (asosiy formula)

BM25 "TF-IDF"ning yaxshilangan varianti. Har bir hujjat $d$ ning so'rov $q$ ga moslik balli:

$$
\text{score}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d)+k_1\left(1-b+b\,\dfrac{|d|}{\text{avgdl}}\right)}
$$

$$
\text{IDF}(t)=\ln\!\left(1+\frac{N-n(t)+0.5}{n(t)+0.5}\right)
$$

Belgilar:
- $f(t,d)$: $t$ so'zi $d$ hujjatda necha marta uchragani (term frequency),
- $|d|$: hujjat uzunligi (so'zlar soni), $\text{avgdl}$: o'rtacha uzunlik,
- $N$: jami hujjatlar, $n(t)$: $t$ so'zi bor hujjatlar soni,
- $k_1=1.2$, $b=0.75$: standart parametrlar.

**Intuitsiya:**
1. **IDF:** kam uchraydigan so'z qimmatliroq. "va" hamma joyda bor, $n\approx N$, demak IDF ≈ 0. "taʼtil" kam uchraydi, demak IDF katta.
2. **To'yinish ($k_1$):** so'z 1 marta va 2 marta uchrashi o'rtasidagi farq katta, 20 va 21 marta o'rtasida esa kichik. Kasr $f\to\infty$ da $k_1+1$ ga intiladi, cheksiz o'smaydi.
3. **Uzunlik jarimasi ($b$):** uzun moddada so'z tasodifan ko'proq uchraydi, shuning uchun uzun hujjat biroz "jarimalanadi".

**Misol (raqamlar bilan).** $N=40\,000$, "tatil" $n=300$ moddada bor:

$$\text{IDF}=\ln\left(1+\frac{40000-300+0.5}{300+0.5}\right)=\ln(133.1)\approx 4.89$$

217-moddada $f=2$, $|d|=15$, $\text{avgdl}=120$:

$$\frac{2\cdot 2.2}{2+1.2\,(1-0.75+0.75\cdot 15/120)}=\frac{4.4}{2+1.2\cdot 0.344}=\frac{4.4}{2.41}\approx 1.83$$

Shu so'z uchun hissa $4.89\times 1.83\approx 8.9$ ball.

**Tezlik uchun hiyla (matritsa).** Kasrli qismni oldindan har bir (hujjat, so'z) juftligi uchun hisoblab, **siyrak matritsa** $W$ ga (N × V, ko'p kataklari 0) saqlaymiz. So'rov vaqtida ballar bitta matritsa-vektor ko'paytmasi bo'ladi:

$$\mathbf{s}=W_{[:,\,q]}\cdot \text{IDF}_{[q]}$$

`scipy.sparse.csc_matrix` ustunlarni tez kesadi. 40 000 hujjatda bitta qidiruv taxminan **0.7 ms** oldi.

---

## 5. Embedding va kosinus o'xshashlik: ma'no bo'yicha qidiruv

BM25 faqat bir xil so'zlarni topadi. "Ishdan bo'shamoqchiman" va "mehnat shartnomasini bekor qilish" bir xil ma'noda, lekin so'zlari boshqa. Buni **embedding** hal qiladi: model har bir matnni vektorga aylantiradi ($\mathbf{v}\in\mathbb{R}^{1024}$), ma'nosi yaqin matnlarning vektorlari ham yaqin bo'ladi.

**Kosinus o'xshashlik:**

$$\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\,\|\mathbf b\|}=\frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\sqrt{\sum_i b_i^2}}\in[-1,1]$$

**Hiyla: L2-normallashtirish.** Har bir vektorni oldindan uzunligiga bo'lamiz: $\hat{\mathbf a}=\mathbf a/\|\mathbf a\|$. Shunda $\|\hat{\mathbf a}\|=1$ va kosinus oddiy skalyar ko'paytmaga aylanadi: $\cos=\hat{\mathbf a}\cdot\hat{\mathbf b}$.

**Barcha so'rovlarni birdaniga hisoblash.** $D$: indeksdagi barcha vektorlar ($N\times d$), $Q$: so'rov vektorlari ($m\times d$; savol, uz/ru qayta yozilgan variantlar va HyDE):

$$S = D\,Q^{\top}\quad (N\times m)$$

Hisob hajmi $N\cdot d\cdot m$: $80\,000\times1024\times 6\approx 5\cdot10^8$ ko'paytma, bu taxminan 60 ms.

**Xotira hisobi:** $N\cdot d\cdot 4$ bayt (float32) $=80\,000\cdot1024\cdot4\approx 328$ MB. float16 yarmini oladi, lekin sekinroq. `EMBEDDING_DIMS=1024`: `text-embedding-3-large` vektori aslida 3072 o'lchamli, uni 1024 ga qisqartirish mumkin (Matryoshka xossasi), xotira 3 baravar kamayadi.

**HyDE (Hypothetical Document Embedding).** Savol ("necha kun?") va modda matni ("...yigirma bir kalendar kunni tashkil etadi") uslubi farq qiladi. Shuning uchun LLM'dan "javob beradigan modda qanday yozilgan bo'lardi?" deb 1–2 gap yozdirib, *shuni* ham embedding qilamiz. U haqiqiy moddaga yaqinroq tushadi.

---

## 6. RRF: bir nechta ro'yxatni birlashtirish

Bizda 8–12 ta ro'yxat bor: BM25 (savol, qayta yozilgan, kalit so'zlar, rus), dense (savol, uz, ru, HyDE), hujjat ishorasi va modda ishorasi. Ularning ballari turli shkalada: BM25 = 8.9, kosinus = 0.62. To'g'ridan-to'g'ri qo'shib bo'lmaydi. Shuning uchun **ball emas, o'rin (rank)** ishlatiladi:

$$\text{RRF}(a)=\sum_{l\in\text{ro'yxatlar}}\frac{w_l}{k+\text{rank}_l(a)},\qquad k=60$$

**Misol.** 217-modda BM25'da 1-o'rinda, dense'da 3-o'rinda ($w=1$):

$$\frac{1}{61}+\frac{1}{63}=0.0164+0.0159=0.0323$$

218-modda faqat BM25'da 2-o'rinda: $1/62=0.0161$. Demak ikki ro'yxatda ham yuqorida turgan modda yutadi. $k=60$ birinchi o'rinlar orasidagi farqni yumshatadi: bitta ro'yxatdagi 1-o'rin hamma narsani hal qilib qo'ymaydi.

Qo'shimcha qoidalar:
- kuchini yo'qotgan hujjat balli $\times 0.5$ qilinadi;
- `retrieval_score` $=\text{RRF}(a)/\max\text{RRF}$, ya'ni 0..1 oralig'iga normallashtiriladi.

**Hujjat ichida cheklangan qidiruv.** LLM "bu Mehnat kodeksi" desa, BM25/dense ballaridan faqat shu kodeks qatorlarini qoldiramiz (boshqalarini $-\infty$ qilamiz) va yana top-40 olamiz. Shunda global top-100 ga kirmagan, lekin to'g'ri modda ham topiladi.

**Top-k ni tez tanlash.** Hammasini saralash $O(N\log N)$. `np.argpartition` esa eng katta $k$ tasini $O(N)$ da ajratadi, keyin faqat shu $k$ tasi saralanadi: $O(N + k\log k)$.

---

## 7. Rerank

RRF'dan keyin 40 nomzod qoladi. **Claude Haiku** savol va 40 ta qisqa parchani o'qib, eng keraklilarini 0–100 ball bilan tartiblaydi (*listwise reranking*). LLM "aynan qaysi modda shu masalani tartibga soladi"ni yaxshi tushunadi.

Muqobil variant (GPU bo'lsa) cross-encoder `bge-reranker-v2-m3`. U xom ball $x$ qaytaradi, biz uni sigmoid bilan 0..1 ga o'tkazamiz:

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

---

## 8. Generatsiya va tekshirish

Claude Sonnet'ga top-8 modda `[S1]..[S8]` deb raqamlab beriladi. Promptdagi asosiy qoidalar:
- faqat manbalardan foydalan, raqamlarni aynan ko'chir;
- har gapdan keyin `[1]` qo'y, har iqtibosga manbadan **so'zma-so'z** `quote` ber;
- manbalarda javob yo'q bo'lsa `no_answer: true` qaytar, lekin qisman javob bo'lsa ham javob ber (ortiqcha "topilmadi" bo'lmasin);
- `relevant_sources`: foydali manbalar, eng muhimi birinchi. Kontekstlar shu tartibda qaytariladi, bu MRR va Hit@5'ni oshiradi.

**Quote'ni tekshirish algoritmi.** LLM ba'zan apostrofni o'zgartiradi (`e'tiboran` va `eʼtiboran`). Shuning uchun:
1. ikkala matnni belgi darajasida normallashtiramiz (kichik harf, apostroflarsiz, bo'shliqlar siqilgan) va **har bir belgining asl indeksini** eslab qolamiz;
2. normallashgan matndan qidiramiz. Topilsa, asl matndagi bo'lakni qaytaramiz. Natijada quote manbaga 100% mos bo'ladi;
3. topilmasa, manbadagi eng mos jumlani olamiz. Jumla balli:

$$\text{ball}(S)=\frac{|C\cap S|}{\sqrt{|S|}}+0.3\cdot\text{ratio}(C,S)$$

bu yerda $C$ va $S$ da'vo va jumlaning so'zlar to'plami, $\text{ratio}$ esa `difflib.SequenceMatcher` o'xshashligi. $\sqrt{|S|}$ ga bo'lishdan maqsad: uzun jumla faqat uzunligi hisobiga yutib ketmasin.

---

## 9. API matematikasi

**Idempotentlik.** So'rovning kanonik JSON'idan (kalitlar saralangan holda) SHA-256 xesh olinadi. Bir xil `batch_id` va bir xil xesh bo'lsa, saqlangan javob qaytadi. Xesh boshqa bo'lsa, `409`.

**Rate limit: Token bucket algoritmi.** Har bir kalitning "chelagi" bor: sig'imi $B$ (burst), soniyasiga $r$ ta token bilan to'ladi:

$$\text{tokens}\leftarrow\min\bigl(B,\ \text{tokens}+\Delta t\cdot r\bigr)$$

So'rov kelganda tokens ≥ 1 bo'lsa, o'tkazib yuboriladi va bitta token olinadi. Aks holda javob `429` bo'ladi va kutish vaqti yuboriladi:

$$\text{Retry-After}=\left\lceil\frac{1-\text{tokens}}{r}\right\rceil$$

---

## 10. Baholash metrikalari (spetsifikatsiya 6-bo'lim)

$N$: javobi bor savollar soni, $\text{rank}_i$: to'g'ri modda kontekstlarda nechanchi o'rinda.

| Metrika | Formula | Ma'nosi |
|---|---|---|
| Hit@5 | $\frac{1}{N}\sum_i \mathbb 1[\text{rank}_i\le5]$ | to'g'ri modda top-5 da bormi |
| MRR | $\frac{1}{N}\sum_i \frac{1}{\text{rank}_i}$ | qanchalik yuqorida (1-o'rin = 1, 2-o'rin = 0.5) |
| Recall@5 | $\frac{\lvert \text{topilgan kutilganlar}\rvert}{\lvert\text{kutilganlar}\rvert}$ | bir nechta modda kerak bo'lsa, nechtasi topildi |
| Precision@5 | $\frac{\lvert\text{tegishli kontekstlar}\rvert}{\lvert\text{qaytarilgan kontekstlar}\rvert}$ | qaytarilganlarning nechtasi foydali |

**MRR misoli** (testdagi): 1-savolda to'g'ri modda 2-o'rinda, 2-savolda 1-o'rinda:

$$\text{MRR}=\frac{1}{2}\left(\frac12+\frac11\right)=0.75$$

**Precision va recall muvozanati.** Kam kontekst qaytarsak precision oshadi, lekin recall tushishi mumkin. Shuning uchun *moslashuvchan* qoida ishlatilgan: iqtibos keltirilganlar doim kiradi, kamida 3 ta kontekst beriladi, qolganlari faqat rerank balli ≥ 0.35 bo'lsa qo'shiladi.

**Latency persentili (p50, p95)** chiziqli interpolyatsiya bilan hisoblanadi. Saralangan $x_0\le\dots\le x_{n-1}$ uchun:

$$k=(n-1)p,\quad f=\lfloor k\rfloor,\quad c=f+1,\quad P_p=x_f+(x_c-x_f)(k-f)$$

p50 mediana, p95 esa "100 savoldan 95 tasi shundan tez" degani.

---

## 11. Vaqt byudjeti (nega ~5–7 soniya)

| Bosqich | Vaqt |
|---|---|
| Haiku so'rov tahlili ‖ savol embeddingi (parallel) | ~1.0–1.5 s |
| BM25 + dense + RRF | ~0.1 s |
| Haiku rerank (40 nomzod) | ~1.0–1.5 s |
| Sonnet javob | ~2.5–4 s |

10 ta savol **parallel** ishlanadi, shuning uchun batch ham taxminan 10 soniyada tugaydi. Tezroq kerak bo'lsa: `RERANKER=none` (−1.2 s) yoki javob uchun Haiku.

---

## 12. Roadmap bilan bog'lanish

| Roadmap haftasi | Shu loyihada |
|---|---|
| 1-hafta (Python, testlar) | `pytest`, 33 test, toza modullar |
| 2-hafta (data, SQL) | scraping, JSON/SQLite saqlash |
| 3-hafta (matematika) | kosinus, normallashtirish, matritsa ko'paytmasi, logarifm (IDF), persentil |
| 6-hafta (NLP, embedding) | tokenizatsiya, stemming, embedding, MTEB modellari |
| 7-hafta (LLM) | Claude/OpenAI/Qwen API, temperature=0, JSON chiqish |
| 8-hafta (RAG) | chunking, hybrid search, RRF, reranker, HyDE |
| 9-hafta (serving) | REST API, Docker, rate limit, idempotentlik |
| 10-hafta (baholash) | Hit@k, MRR, Recall, faithfulness, LLM-hakam |

**Mustaqil mashq qilish uchun:**
1. `k1` va `b` ni o'zgartirib (`lexrag/index.py`, `BM25`), `score_eval.py` bilan Hit@5 qanday o'zgarishini ko'ring.
2. RRF'da `k=60` o'rniga 10 va 200 qo'yib ko'ring (`retrieval.py`, `RRF_K`).
3. `RERANKER=none` va `llm` ni solishtiring: MRR va latency qanday o'zgaradi?
4. BM25 formulasini qog'ozda 3 ta qisqa hujjat uchun qo'lda hisoblang, keyin `BM25().fit(...)` natijasi bilan solishtiring.
