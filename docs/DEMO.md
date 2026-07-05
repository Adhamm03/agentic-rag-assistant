# Demo GIF — shot list & questions

Copy-paste these questions while recording (`docs/demo.gif`). They all hit the
committed corpus (`data/pdfs/`) and are drawn from the ground-truth set, so the
answers and citations are known-good. Do a dry run first; on the final take,
trim the CPU wait/spinner frames so the GIF stays ~15–20s.

Have the index ingested and both servers up (`uvicorn` on :8000, `npm run dev`)
before you start.

## The 3-shot sequence (do these in order)

**1. Cited English answer** — the money shot; make sure `[source:page]` shows,
then expand the **Sources** panel.
```
What is the maximum annual coverage ceiling per member?
```
_Expected: EGP 500,000 per member per policy year, cited to 01_policy_terms_en.pdf._

**2. Anti-hallucination / abstain** — your strongest differentiator. Ask
something not in the docs; show it refuse instead of guessing.
```
Does the policy cover LASIK eye surgery?
```
_Expected: "Not found in the provided documents" (LASIK isn't addressed)._

**3. Arabic + RTL** — flip the EN/AR toggle first, then ask in Arabic.
```
ما هو الحد الأقصى السنوي للتغطية لكل عضو؟
```
_Expected: خمسمائة ألف جنيه مصري, cited to 02_coverage_summary_ar.pdf._

## Optional 4th shot (if time allows — real wow factor)

**Cross-lingual retrieval** — an Arabic question answered from the *English*
policy doc (proves language-agnostic retrieval, not just a translated UI).
```
ما رقم الخط الساخن للطوارئ؟
```
_Expected: 19240, retrieved from 01_policy_terms_en.pdf._

## Priority if you must cut

Keep **1 → 2 → 3**. Shot 2 (refusal) and shot 3 (Arabic) are what clients don't
see in other RAG demos — never drop those.

## More known-good questions (swap in if you prefer)

| Question | Expected |
|---|---|
| What co-payment applies to outpatient consultations? | 10%, at point of service |
| How long is the waiting period for maternity benefits? | 10 months continuous cover |
| What is the annual limit for dental treatment and its co-pay? | EGP 3,000/yr, 20% co-pay |
| Do I need a referral to see a specialist? | No referral; 10% co-pay applies |
| ما نسبة التحمل على علاج الأسنان؟ | عشرون بالمائة (20%) |
| Is a co-payment charged for in-network inpatient treatment? | No (answered from the Arabic doc) |
