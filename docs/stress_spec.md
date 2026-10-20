# Stress-set specification (Phase 3, generated 2026-10-18, run `make_stress-20261018-025910`)

All stress rows are **synthetic** (`is_synthetic=True`), written by Claude (Opus 5.5). They are used **only for evaluation**: never for training, model selection or early stopping. Generator: `src/make_stress.py --seed 2026`; templates: `src/stress_templates.py` (invented content, typed slots, so no slot-fill nonsense). Native-script multilingual sets were **cut** (owner, 2026-10-18): the app translates non-English SMS to English with ML Kit before the model runs.

## Base messages
25 English templates per class (spam = scam, caution, promo, ham), each rendered once with seeded, type-matched fills: bank/shop/telecom names, realistic Indian rupee amounts, 4–6 digit **random** OTPs, look-alike phishing domains (for example `secure-hdfcbank-verify.com`, `bit.ly/...`), and random Indian mobile numbers.

## stress_spoof (750 rows)
Each base text is crossed with **6 sender types, independent of the label**:
`registered_ST` (XX-BRAND-S/T), `registered_P` (XX-BRAND-P), `registered_G` (government header -G), `personal_mobile` (+91…), `unknown_alnum` (6 random letters), `spoof_lookalike` (XX-HDFCBN-S, AMAZ0N, lCICIB, …).
Scam texts additionally get one **camouflage** variant (prefix "Security Alert: " or suffix "Do not share your OTP with anyone." / "Beware of fraud calls."). These phrases trigger the app's safety-advisory veto (R1).
True label = intent of the text, except that **caution texts from a personal mobile number are labelled spam** (taxonomy: OTP-themed texts from personal numbers).
Purpose: measure the rule layer's dependence on sender headers (R0 trusted substring, R4 suffix rules) and the effect of camouflage on R1.

## stress_obfuscation (500 rows; 487 unique)
Each base text × {none, typos (swap/drop/duplicate inside about 35% of words), spacing ("K.Y.C", "v-e-r-i-f-y" in about 30% of words), homoglyph (Cyrillic а е о с р х і у for 40% of eligible letters), leetspeak (a→4 e→3 i→1 o→0 s→5 t→7 in about 50% of words)}. The label is unchanged. 13 rows are identical to their `none` version because the random transform skipped every word; they are kept and the unique count is reported.

## stress_hinglish (60 rows)
15 romanized Hindi/Marathi-English messages per class. **Needs native-speaker verification** before any claim is made. Romanized text may not be translated by ML Kit, so the model sees it directly.

## digit_probe (72 rows)
Caution OTP templates and ham amount templates rendered 3× with random digits and 1× with patterned 112233-style digits. This checks whether a model relies on the surrogate-digit artifact (00_audit §3) when choosing D0/D1/D2 in Phase 5.

## Uniqueness and leakage
Normalized-text overlap with train/val/test = 0 for all four sets.

## Known train/deploy mismatch to resolve in Phase 5 (placeholders)
Training text has scrubbed placeholders (`<URL>`, `<PHONE>`, `<NAME>`, `[OTP]`, …), while deployed and stress inputs contain real links and numbers. Options: **U0** apply the same masking at inference (app change: run the scrubber's URL/phone/email regex before tokenising); **U1** re-insert realistic random URLs/numbers in place of placeholders during training (no app change). Choose on validation and report the result on the stress sets.
