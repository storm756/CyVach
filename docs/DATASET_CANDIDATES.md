# Dataset candidates (Phase 0, Step D)

Searched 2026-10-17 (web search plus the dataset pages listed). **Nothing below has been downloaded**, except the UCI SMS Spam Collection, which is an approved source in CLAUDE.md §3.2. Licences are quoted from the dataset page as retrieved on that date. Re-check each one before release.

Mapping notation: our classes are `ham / spam(=scam) / promo / caution`.

## Ranked shortlist (recommendation for Gate 1)

| Rank | Dataset | URL | Licence (verbatim from page) | Size | Lang | Labels | Real/synthetic | Access | Mapping to our 4 classes | Red flags | Recommend |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **UCI SMS Spam Collection** (Almeida & Gómez Hidalgo 2011) | https://archive.ics.uci.edu/dataset/228/sms+spam+collection, DOI 10.24432/C5CC84 | "Creative Commons Attribution 4.0 International (CC BY 4.0)" | 5,574 | en (UK/SG) | ham/spam | real | direct (already approved, downloaded) | ham→ham; spam→**needs relabel** (mix of premium-rate scams, prize scams and plain marketing → spam or promo) | Old (2011), non-Indian, many exact duplicates (measured in `results/compare_uci.json`) | **Use** (approved) |
| 2 | **SMS Phishing Dataset for ML and Pattern Recognition** (Mishra & Soni, Mendeley Data v1, 2022) | https://data.mendeley.com/datasets/f45bkkt8pr/1, DOI 10.17632/f45bkkt8pr.1 | "CC BY 4.0" | 5,971 (4,844 ham, 489 spam, 638 smishing) | en (page does not state) | ham/spam/smishing | page says collected from the Internet via screenshots + OCR; reads as real | direct | ham→ham; smishing→spam; spam→**relabel** (spam or promo) | Likely heavy overlap with UCI ham (to be measured on a ≤50-row sample after approval); OCR noise | **Use, if approved** (adds smishing examples, which we are short of) |
| 3 | **Smishing Dataset I – Smishtank.com** (Timko & Rahman, CODASPY 2024) | https://smishtank.com/dataset, paper arXiv:2402.18430 | **No dataset licence stated.** Only the arXiv paper is "CC BY-NC-SA 4.0" | 1,062 smishing | mostly en (not stated) | smishing only, with brand and URL metadata | real (user submissions) | direct page (not verified that it needs no login) | all→spam | Licence unclear → **ask authors**; US-centric | **Only if you approve emailing the authors / accept the risk** |
| 4 | **Hindi Spam SMS Dataset** (IEEE DataPort) | https://ieee-dataport.org/documents/spam-sms-hindi-language, DOI 10.21227/5y8x-n678 | **No licence shown on page** | 3,894 | hi | spam/ham | real (students' inboxes) | **login + IEEE DataPort subscription** | ham→ham; spam→relabel | Licence missing; subscription required; one page comment says files were not uploaded | Manual download by you, and only if a licence can be found; otherwise **do not use** |
| 5 | **A Balanced Dataset for Spam and Smishing Detection using LLMs** (Mendeley vmg875v4xs) | https://data.mendeley.com/datasets/vmg875v4xs | "CC BY 4.0" | 10,191 (3,397 per class) | en | ham/spam/smishing | **LLM-generated** from Mishra & Soni; model not named | direct | as #2 | Synthetic, unnamed generator, built from #2 (leakage risk if both are used) | **Do not use** for training or testing; at most as flagged synthetic augmentation |
| 6 | alusci/sms-otp-spam-dataset (Hugging Face) | https://huggingface.co/datasets/alusci/sms-otp-spam-dataset | "MIT" (tag) | 10,000 | en | valid / not valid | **synthetic**, generation not documented | direct | OTP texts → caution; labels do not map | Highly templated; labels appear inconsistent with content | **Do not use** |
| 7 | dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset (HF) | https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset | "gpl" (tag) | 5.57k × 21 languages (incl. hi, mr) | multi | ham/spam | apparently machine-translated UCI (not documented) | direct | as UCI | GPL (copyleft) for data; translation quality unknown; duplicates UCI | **Do not use** |
| 8 | Ngadou/Spam_SMS (HF) | https://huggingface.co/datasets/Ngadou/Spam_SMS | "cc" (unspecified) | 3,000 rows shown | en | 0/1 | claims UCI re-upload, but preview rows look like **emails** | direct | — | Unverified Kaggle re-upload; content mismatch | **Do not use** |

## Not available (described in papers, no usable release found)

| Source | Status |
|---|---|
| Agarwal, Suarez-Tangil & Vasek, "An Overview of 7726 User Reports: Uncovering SMS Scams and Scammer Strategies", arXiv:2508.05276 (NDSS 2026) | **Not available.** The arXiv page has no data-availability statement or dataset link. Its CC BY 4.0 covers the paper only. Data came from a mobile network operator. |
| SpamDam (Li et al., arXiv:2404.09481), 76,577 SMS spam texts from Twitter/Weibo | **Not available / licence unclear.** The project page (chasesecurity.github.io/SpamDam) states no dataset licence or download. Its CC BY-SA 4.0 covers the website. Also social-media-derived (scraping concerns). |
| Dravidian spam SMS (IEEE DataPort; en/ta/te/kn/ml) | Not our target languages; IEEE DataPort subscription. Not pursued. |
| Kor-Smishing, Bangalabarta (Bangla), Azerbaijani SMS | Not our languages. Not pursued. |
| Indian transactional/OTP/promotional SMS corpus (TRAI categories) | **None found** on Zenodo, Hugging Face, Mendeley or UCI. |
| Hinglish / Marathi / Gujarati SMS spam corpus | **None found** with an open licence. Hinglish corpora found (L3Cube-HingCorpus, PHINC, HinGE) are not SMS and not spam-labelled. |

## Consequence for the plan

- No openly licensed source covers **caution** (OTP / login alerts) or **Indian promo** messages. Those classes will rely on the owner's files plus flagged synthetic data. That is a limitation to state in the paper.
- Multilingual evaluation would rest on synthetic stress sets only (cut line 1 applies if no native-speaker check happens).

Sources: [arXiv 2508.05276](https://arxiv.org/abs/2508.05276v2), [Mendeley f45bkkt8pr](https://data.mendeley.com/datasets/f45bkkt8pr/1), [Mendeley vmg875v4xs](https://data.mendeley.com/datasets/vmg875v4xs), [Smishtank paper](https://arxiv.org/html/2402.18430v2), [SpamDam](https://chasesecurity.github.io/SpamDam), [IEEE DataPort Hindi](https://ieee-dataport.org/documents/spam-sms-hindi-language), [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), [HF alusci](https://huggingface.co/datasets/alusci/sms-otp-spam-dataset), [HF dbarbedillo](https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset), [HF Ngadou](https://huggingface.co/datasets/Ngadou/Spam_SMS), HF dataset search API.
