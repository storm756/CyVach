# Reproducing the CyVach re-tune results

## Environment
- Windows 11, Python 3.13 venv in `.venv`, packages pinned in `requirements.txt` (CPU torch 2.14.1). Install with
  `uv pip install -r requirements.txt` (torch from the PyTorch CPU index).
- Training was done on Google Colab (Tesla T4, torch 2.11.0+cu130, transformers 5.20.1, fp16 autocast) with
  `colab/phase5_sweep.ipynb` and `colab/phase5_seeds.ipynb`. Every run JSON records its own versions under `env`.
- Exported ONNX, parity, latency and all evaluation ran locally on CPU (onnxruntime 1.31.0).

## Inputs that scripts cannot regenerate
| Input | Why |
|---|---|
| `data_raw/` | Owner and public files (hashes in `docs/DATA_SOURCES.md`); kept read-only, never committed |
| `data/name_seeds.txt` | Local private name list for the scrubber; never committed or printed |
| `annotation/claude_gold_pass1.csv`, `annotation/claude_silver_pass1.csv` | LLM labels (protocol in `docs/AI_USE_LOG.md`) |
| `annotation/check_results.csv` | Owner's accept/override check of 180 test labels |
| `annotation/gold_candidates.csv`, `data/test.csv` | Frozen test set (SHA-256 `7fa0c6c4bdaf604a22c7185e39d3c605a266fa7ef0b5072a5c85527ce04e201b`; `make_splits.py` and `final_test.py` assert it) |
| `model_v7_3/`, app assets (`vocab.txt`, `sms_spam_tiny.onnx`) | Legacy model being evaluated |
| `models/train-20261018-13*` | The 5 final seeds trained on Colab (about 257 MB each; not committed) |

## One command
```powershell
.\make_all.ps1             # pipeline + every evaluation output from saved models/predictions
.\make_all.ps1 -Train      # also re-train (sweep + 5 seeds); a GPU is strongly advised
.\make_all.ps1 -RerunTest  # also repeat the frozen-test evaluation (deterministic reproduction; never for tuning)
```
Outputs: `figures/*.pdf`, `results/tables/*.tex`, `results/paper_numbers.md`, and `results/*.json` (each has a `run_id`, config and code version; `logs/runs.csv` lists all runs).

## Pipeline (script → purpose)
| Phase | Scripts |
|---|---|
| 0 Audit | `audit_inventory.py`, `audit_stress400.py`, `compare_uci.py`, `check_onnx.py`, `scrub.py --audit` |
| 1 Corpus | `build_known_names.py`, `build_pool.py` (local scrub, normalise, dedupe, MinHash clusters), `predict_v73.py`, `label_batches.py` (make/ingest LLM batches), `agreement.py` |
| 2 Test set | `sample_gold.py`, `check_app.py` (owner check, Flask on 127.0.0.1:5057), `assemble_labels.py`, `make_splits.py` |
| 3 Stress | `stress_templates.py`, `make_stress.py` (spec: `docs/stress_spec.md`) |
| 4 Baselines | `baselines.py` (TF-IDF LR/SVM), `eval_v73.py` |
| 5 Train | `train.py`, `sweep.py`, `placeholders.py`, `calibrate.py` |
| 6 Export | `export_onnx.py`, `bench_onnx.py`, `app_tokenizer.py` |
| 7 Ablation | `rules.py` (+ `tests/test_rules.py`, 28 tests), `final_test.py`, `ablation.py`, `stress_eval.py` |
| 8 Statistics | `stats.py`, `abstain.py`, `error_analysis.py` |
| 9 Assets | `make_figures.py`, `make_tables.py`, `make_paper_numbers.py` |

## Notes
- Seeds: split 42, sweep 13/21, final 1–5, bootstrap 0, placeholder fill 1234, stress 2026.
- GPU vs CPU: validation scores logged during Colab training use fp16; CPU FP32 re-evaluation of seed 3 gives the same val macro-F1 (0.888, `export_onnx-20261018-194435`).
- Desktop-CPU latency is not phone latency. The on-device protocol is in the docstring of `src/bench_onnx.py`.
