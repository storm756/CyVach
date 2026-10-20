# CyVach: auditing and rebuilding a deployed on-device SMS fraud classifier

Research code and evaluation outputs for the paper *CyVach: Auditing and Rebuilding a Deployed On-Device SMS Fraud
Classifier* (I. S. Pavase, ICTCS 2026). This repository contains the **research pipeline**, not the CyVach Android app.

## What is here
| Path | Content |
|---|---|
| `src/` | Every script used in the paper (audit, local scrubber, corpus build, splits, training, ONNX export, evaluation, statistics, figures) |
| `tests/` | Unit tests for the Python port of the app's rule layer (`python tests/test_rules.py`) |
| `docs/` | Label guidelines, the exact LLM labeling prompt (`AI_USE_LOG.md`), stress-set spec, rule-layer description, data sources, data statement, dataset search log |
| `data/stress/` | The **synthetic** stress sets (spoofed senders, obfuscation, Hinglish, digit probe, benign camouflage controls) |
| `results/` | All metrics as JSON with run ids, per-item predictions keyed by opaque item ids, `paper_numbers.md` (every number in the paper with its run id), LaTeX tables |
| `figures/` | Paper figures (PDF) |
| `colab/` | Notebooks used for GPU training |

## What is not here, and why
- **Real SMS messages** (raw or scrubbed) from volunteers' phones: they are personal data shared under verbal consent and are never published. Scripts that read them will not run without the private files.
- **Model weights**: not released.
- **The app source**: the CyVach app is covered by Indian patent application 202621074740.
- Public datasets are available from their sources: UCI SMS Spam Collection (doi:10.24432/C5CC84) and Mishra & Soni (2022, doi:10.17632/f45bkkt8pr.1).

## Reproducing
See `README_repro.md` and `make_all.ps1`. With the private data absent, the analyses that work from saved outputs
(`src/stats.py`, `src/abstain.py`, `src/make_figures.py`, `src/make_tables.py`, `src/make_paper_numbers.py`) and the
rule-layer tests still run.

## AI use
LLM assistance (Claude Opus 5.5 via Claude Code) was used for labeling, stress-set writing, code and drafting; Google
Antigravity integrated the model into the app for the on-device benchmark. Details: `docs/AI_USE_LOG.md`.
