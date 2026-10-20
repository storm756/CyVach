# Data statement (draft paper section)

*All numbers below are generated; see `results/paper_numbers.md` for each value's run id.*

**Sources.** We combine (i) the UCI SMS Spam Collection [Almeida & Gómez Hidalgo 2011; CC BY 4.0], (ii) the SMS Phishing dataset of Mishra & Soni [2022; CC BY 4.0], and (iii) real SMS exported with the CyVach app from the phones of the author and about 6–8 contributors, mostly students, who gave **verbal consent** (no ethics-committee approval was sought). An earlier, AI-pseudonymised and "class-balanced" file was audited. Its ham class turned out to be cloned (466 rows → 52 near-duplicate clusters), 440 of its rows look machine-generated, and its surrogate digits correlate with the label (Cramér's V = 0.58). Its non-synthetic rows are used, and its suspected-generated rows were excluded from training after they lowered validation F1. Email corpora (CEAS-08) and files without a verifiable licence were excluded.

**Privacy.** Phone exports were pseudonymised **locally** with a deterministic rule-based scrubber (phone, account, card, UPI, email, URL, Aadhaar/PAN-like patterns and several name contexts), followed by a residual audit. No cloud model saw unscrubbed text. Real messages are not released or quoted in the paper; only scrubbed, aggregate statistics are reported.

**Labels.** Four classes: *ham* (benign, including transactional notices), *caution* (legitimate but security-sensitive, e.g. OTPs and login alerts), *scam* (called `spam` in the app; fraud, phishing, malware) and *promo* (non-deceptive marketing). When a message fits several classes, the most severe wins. Labels for items where the source label and the old model disagreed (plus a stratified sample) were proposed by an LLM (Claude Opus 5.5) from written guidelines; the other training items keep their source label. Test labels were proposed blind by the same LLM; the author checked 180 of them (3.9% overridden). **There is a single human annotator, and most labels are model-proposed.**

**Splits.** Text was normalised, exact duplicates were merged, and near-duplicates were clustered (MinHash, Jaccard 0.8); splits are cluster-disjoint (overlap 0 at cluster, text and id level). The test set (533 real messages; main subset 519 after removing 14 native-script messages, which the app translates to English before classification) was frozen and hashed before training and evaluated once. Its class mix (ham 197, promo 147, caution 88, scam 87) was deliberately enriched for rare classes; validation keeps the natural distribution.

| Split | ham | scam | promo | caution | total |
|---|---|---|---|---|---|
| train (used) | 4,786 | 331 | 679 | 86 | 5,882 |
| val | 802 | 62 | 122 | 15 | 1,001 |
| test (main) | 197 | 87 | 147 | 88 | 519 |

All rows used for training and evaluation are real; no synthetic message was used in the final model's training. Effective unique items (near-duplicate clusters) per class are given in Table `tab_data.tex`.

**Stress sets** (spoofed senders, obfuscation, Hinglish, random-digit probe) are synthetic, written by the LLM and a seeded generator. They are used for evaluation only and are not representative of real traffic; the Hinglish set needs native-speaker verification.

**Known limitations.** Single annotator with LLM-proposed labels; small test set (CIs about ±0.03 macro-F1); validation has only 15 caution items; UCI messages are 2000s-era UK texts; the contributors are a small, non-random group; consent was verbal.
