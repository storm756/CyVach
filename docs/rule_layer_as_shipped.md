# CyVach rule layer, exactly as shipped (SMSReaderApp)

Source files were read on 2026-10-17 from `C:\Users\Acer\Documents\New folder\SMSReaderApp\`:
`src/ml/scamDetector.ts` (371 lines, modified 2026-02-10), `src/components/SMSReader.tsx` (544 lines, 2026-03-21), and `src/utils/SenderParser.ts` (140 lines, 2026-02-10).

## Processing order for one message

```
classifyText(body)                                   # scamDetector.ts, sender is NOT used here
  H1  missed-call override  -> ham 1.00
  H2  utility-transaction override -> ham 0.98
  L   ML Kit language ID; if not 'en'/'und' -> on-device translation to English
  M   WordPiece (custom JS port) -> ONNX INT8 -> softmax -> argmax over [ham, spam, promo, caution]
getFinalClassification(address, body, aiResult)       # SMSReader.tsx, applied in order, first match returns
  R0  trusted-sender substring whitelist
  R1  safety-advisory veto
  R2  mobile-number confidence gate
  R3  mobile-number + suspicious keyword -> spam
  R4  TRAI suffix rules (-P, -G, -S/-T)
  else aiResult
parseSender(address)                                  # SenderParser.ts, display only (name + ✓ tick); no effect on the label
```

## Pre-model heuristics (`classifyText`, text only, all lower-cased)

- **H1 Missed call.** If the text contains `"missed call from"` or `"last missed call"` **and** any of `jio`, `airtel`, `vi`, `bsnl`, `vodafone` as a *substring*, return `ham` with confidence 1.0. Note: the substring `vi` matches very common words ("via", "service", "visit"), so the operator check is almost always true.
- **H2 Utility transaction.** If the text contains `processed`, `bill generated` or `payment received`, **and** any of `airtel`, `jio`, `act fibernet`, `tatasky`, return `ham` with confidence 0.98.
- On an inference exception, the result is `ham` with confidence 0.5 (fail-open).

## Model input (`buildInputs`)

- The tokenizer is a hand-written JS WordPiece, not the HF tokenizer. It lower-cases, collapses whitespace, splits on every character outside `[a-z0-9]` (so accented and Indic characters become single-character tokens), then does greedy longest-match WordPiece. On failure it emits `[UNK]` but keeps the sub-pieces already matched (HF maps the whole word to `[UNK]`). It applies no accent stripping. JS iterates UTF-16 code units, so emoji become 2 × `[UNK]`.
- Input is always padded to 128 (`[CLS] … [SEP]`, 126 content tokens max), with batch 1.
- The output tensor must have length 4. Otherwise the result is `ham` with confidence 0.5.

## Post-model rules (`getFinalClassification`)

`text = body.toLowerCase()`, `sender = address.toUpperCase()`.

- **R0 Trusted sender (strongest).** `TRUSTED_SENDERS = [HDFC, ICICI, SBI, AXIS, KOTAK, YESBNK, BOB, IDFC, PNB, LENSKT, AMAZON, FLIPKART, MYNTRA, ZOMATO, SWIGGY, JIO, MYJIO, AIRTEL, AIRINF, VI, VODAFONE, IDEA, BSNL, NSE, BSE, CDSL, NSDL, ITDEPT, GOV, IRCTC, CRIS, MSEDCL, MAHADISCOM, ADHAR, UIDAI, VAHAN, EPFO]`. The match is a **substring** match on the upper-cased address.
  - If AI = `promo`, keep promo. Otherwise return **`ham` 0.99**. This includes AI = `caution` and AI = `spam`.
  - Consequences: (a) an OTP from a bank header is shown as **ham**, never caution; (b) a spoofed header containing any of these substrings vetoes a scam verdict, **even when the body has a phishing link**; (c) short entries `VI`, `BOB`, `BSE`, `GOV`, `NSE`, `CRIS`, `IDEA` match many unrelated headers (for example any header containing "VI").
- **R1 Safety advisory.** If the text contains `beware`, `security alert`, `do not share` or `report to`, **and** AI = `spam`, return `ham` 0.95. A scam that adds "Do not share your OTP" is therefore vetoed to ham.
- **R2 Mobile-number gate.** If AI = `spam` with confidence < 0.985 and the address matches `^(\+91|91|0)?[6-9]\d{9}$` (after removing spaces and hyphens), return `caution` with the AI confidence.
- **R3 Mobile + keyword.** If the address is a mobile number and the text contains any of `otp, parcel, verify, pin, blocked, account, kyc, pan, update, refund` as a **substring**, return **`spam` 1.0**. Substring matching means `pin` matches "shopping" and "happiness", `pan` matches "company", and `account` matches any bank alert forwarded by a friend.
- **R4 TRAI suffixes**, checked on the original-case address:
  - ends with `-P` or ` P` → `promo` 1.0 (overrides the AI, including spam)
  - ends with `-G` or ` G` → `ham` 1.0 (overrides the AI, including spam)
  - ends with `-S` or `-T`, **and** AI = `spam` with confidence < 0.999 → `ham` 0.95
- Otherwise the AI result is returned unchanged.

## UI

- Only a final label of `spam` triggers a push notification.
- `parseSender` maps headers to brand names and shows a ✓ for `KNOWN_SENDERS` hits and for numeric short codes of 6 digits or fewer. This is display only, but a **spoofable "verified" tick** is shown for any sender whose 6-char key matches. Phone numbers longer than 9 characters are treated as unverified.

## Differences from the brief and the TODO

1. **There is no URL/link rule anywhere in the shipped code.** The TODO's "link-based trust revocation" (Algorithm 1) and the Phase 7 "+URL/link rules" ablation step have no counterpart in `SMSReader.tsx`, `scamDetector.ts` or `SenderParser.ts`. Trusted headers are trusted unconditionally (R0), even when the message contains a link.
2. **There is no "OTP override" rule** of the kind the TODO describes. The only OTP-related logic is R3 (`otp` substring + mobile-number sender → spam). For trusted bank headers, R0 *removes* the caution label from OTPs.
3. A `-T` header is not trusted unconditionally. It only downgrades spam with confidence < 0.999 (R4). R0 runs earlier and is the stronger rule.
4. The ONNX model in the app was exported with **opset 12** and with only the batch axis dynamic (`v7_3.py`), not opset 17 with a dynamic sequence axis. The app always pads to 128, so this works.
5. The pipeline includes **ML Kit language ID plus translation** before the model (`TranslationService.ts`). Desktop evaluation cannot reproduce that step exactly, so multilingual results on desktop are model-only.
6. The `scamDetector.ts` comment still says "267MB file", but the shipped asset is 67,678,083 bytes.
