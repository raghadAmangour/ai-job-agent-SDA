# AI Job Agent — تطبيق Streamlit

تطبيق ويب يحوّل مسار Phase 2 → 3 → 4 → 5 → 6 → 7 → 8 (من دفاتر Colab الأصلية)
إلى تجربة تلقائية: أي زائر يرفع سيرته الذاتية، يملأ تفضيلاته، ويحصل على
أفضل الوظائف المطابقة مع شرح وشات تفاعلي — بدون أي رفع ملفات يدوي بين
المراحل.

## البنية

```
app.py                      # واجهة Streamlit الرئيسية
src/
  config.py                 # كل القيم القابلة للتعديل (أسماء الموديلات، top_k، ...)
  shared.py                 # تنظيف النصوص + تطبيع المهارات (مطابق لـ Phase 1/2)
  resume_reader.py          # قراءة PDF/DOCX/TXT (Phase 2)
  phase2_profile.py         # استخراج بروفايل المرشح من السيرة (LLM)
  phase3_retrieval.py       # الاسترجاع بالـ embeddings (تشابه جيب)
  phase4_filtering.py       # الفلاتر الصارمة (دولة/مدينة/راتب/...)
  phase5_matching.py        # درجة التطابق (equal-weight + CRITIC)
  phase6_rerank.py          # إعادة الترتيب بالـ Cross-Encoder
  phase7_explain.py         # شرح النتائج بالـ LLM (مع تحقق صارم من عدم الاختلاق)
  phase8_agent.py           # وكيل المحادثة (function-calling على نتائج جاهزة)
data/
  jobs_prepared.parquet     # ناتج Phase 1 — قاعدة الوظائف (9,619 وظيفة)
  job_embeddings.npy        # ناتج Phase 1 — متجهات الوظائف
  job_ids.npy               # ناتج Phase 1 — معرّفات الوظائف
  skill_vocabulary.parquet  # مُعاد بناؤه محليًا من jobs_prepared (IDF لكل مهارة)
requirements.txt
.streamlit/secrets.toml.example
```

**Phase 1 لا تُشغَّل داخل التطبيق** — بياناتها جاهزة ومُرفَقة في `data/`.
لتحديث قاعدة الوظائف مستقبلًا، شغّل دفتر Phase 1 كما هو خارج التطبيق ثم
استبدل الثلاثة ملفات في `data/`.

## التشغيل محليًا

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# عدّل .streamlit/secrets.toml وضع مفتاح OpenAI الحقيقي
streamlit run app.py
```

## النشر على Streamlit Community Cloud

1. ادفع هذا المجلد كاملًا (بما فيه `data/`) إلى مستودع GitHub.
   الملفات الثلاثة في `data/` مجموعها ~80MB — أقل من حد GitHub (100MB لكل
   ملف)، فلا حاجة لـ Git LFS.
2. اذهب إلى https://share.streamlit.io → **New app**.
3. اختر المستودع، الفرع، وحدّد `app.py` كملف رئيسي.
4. من **Settings → Secrets** أضف:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. اضغط **Deploy**. أول تشغيل سيأخذ وقتًا أطول لتحميل نموذج
   الـ Cross-Encoder (~1-2 دقيقة) — بعدها يبقى محمّلًا في الذاكرة
   (`st.cache_resource`) لكل الجلسات التالية.

## نقاط مهمة قبل النشر العام

- **اسم الموديل**: الدفاتر الأصلية تستخدم `"gpt-5.6-luna"` في
  `src/config.py` (`EXTRACTION_MODEL`, `EXPLANATION_MODEL`, `AGENT_MODEL`).
  إذا لم يكن هذا الموديل متاحًا في حسابك، غيّره إلى موديل فعلي متاح لديك
  (مثل `gpt-4o` أو `gpt-4o-mini`).
- **التكلفة**: كل تشغيل تحليل كامل = استدعاء استخراج بروفايل + embedding +
  حتى 10-20 استدعاء LLM لشرح الوظائف (Phase 7) + استدعاءات المحادثة
  (Phase 8). المفتاح في `secrets.toml` يُحاسَب على حساب مالك التطبيق —
  التطبيق يحدّ من عدد رسائل المحادثة لكل جلسة (`MAX_CHAT_TURNS_PER_SESSION`
  في `app.py`) كحماية أساسية، لكن يُنصح بمراقبة الاستخدام من لوحة OpenAI.
- **الذاكرة**: خيار "استخدام Cross-Encoder" في الشريط الجانبي مفعّل
  افتراضيًا. لو واجهت بطئًا أو نفاد ذاكرة على الطبقة المجانية من Streamlit
  Cloud، بإمكان المستخدم إيقافه من الشريط الجانبي دون فقدان بقية المراحل.
- **حماية إضافية (اختياري)**: للحد من إساءة الاستخدام على نشر عام، فكّر
  بإضافة كلمة مرور بسيطة (`st.text_input` + مقارنة مع `st.secrets`) قبل
  عرض التطبيق.
