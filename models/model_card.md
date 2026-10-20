# Model card: CyVach re-tuned DistilBERT (Phase 5, 2026-10-18)

**Base:** `distilbert-base-uncased`, fine-tuned from scratch (not from v7.3). **Labels (fixed order):** 0 ham, 1 spam (= scam/fraud), 2 promo, 3 caution.
**Training data:** 5,882 real, scrubbed, non-native-script SMS (ham 4,786 / promo 679 / spam 331 / caution 86). Sources: owner + friend exports, inbox file, UCI SMS Spam Collection, Mishra & Soni 2022. **No synthetic rows** (they reduced val F1 in the sweep). Labels: Claude silver + original (see `docs/AI_USE_LOG.md`).
**Config** (selected on val by the pre-registered rule in `src/sweep.py`; `reports/05_sweep.md`): class-weighted CE (S0), lr 5e-5, batch 32, up to 5 epochs (best epoch on val), max_len 128, weight decay 0.01, warm-up 6%, no label smoothing, fp16 mixed precision on a T4 (Colab), code `25a4bb1`.
**Input preprocessing during training:** U1 (placeholders such as `<URL>`/`[OTP]` filled with realistic random values) and D1 (digits re-randomised per batch). **Neither needs any app change:** at inference the app passes raw text, as it does now.

## Five fresh seeds (1–5), validation set (1,001 real non-native-script messages; 15 caution), run `calibrate-20261018-191340`
| | mean ± std (n = 5) |
|---|---|
| Macro-F1 | 0.891 ± 0.008 |
| F1 caution / spam | 0.815 ± 0.037 / 0.839 ± 0.043 |
| MCC | 0.905 ± 0.014 |
| Legit flagged as scam | 0.9% ± 0.8% |
| Temperature T | 1.35 ± 0.15 |
| ECE (15 bins) before → after T | 0.022 ± 0.005 → 0.014 ± 0.004 (in-sample on val) |

Seed run ids: `train-20261018-132741` (1), `-132939` (2), `-133131` (3), `-133323` (4), `-133515` (5). Per-epoch losses: `logs/<run>_epochs.csv`.
**Deploy candidate (pre-registered: the median val-F1 seed, to avoid picking the luckiest):** seed 3, `train-20261018-133131`, T = 1.397.
Size of each FP32 checkpoint (`model.safetensors`): see Phase 6 (reported in bytes, MB and MiB).

## Limitations
- Caution has only 86 real training and 15 val examples, so its F1 is noisy (std 0.037).
- Labels are mostly AI silver labels, with an owner check on the test set only.
- The test set has **not** been evaluated yet; that happens once, after Gate 3.
- English and Hinglish only. Native-script SMS is expected to be translated to English on the device (ML Kit) before the model.
