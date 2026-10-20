# Regenerate every table and figure from raw data (see README_repro.md).
# Usage:  .\make_all.ps1                 # data pipeline + all evaluation outputs from saved models/predictions
#         .\make_all.ps1 -Train          # also re-run the sweep and the 5 final seeds (GPU strongly advised; see colab/)
#         .\make_all.ps1 -RerunTest      # also re-run the frozen-test evaluation (deterministic reproduction, NOT tuning)
# Inputs that no script can regenerate: data_raw/ (owner + public files), data/name_seeds.txt (local, private),
# annotation/claude_*_pass1.csv (LLM labels), annotation/check_results.csv (owner check), model_v7_3/,
# models/train-* (trained on Colab) and the app assets (vocab.txt, deployed ONNX).
param([switch]$Train, [switch]$RerunTest)
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Set-Location $PSScriptRoot

function Run([string]$script, [string[]]$a = @()) {
    Write-Host ">> $script $($a -join ' ')"
    # Windows PowerShell 5.1 turns native stderr (e.g. library warnings) into errors under "Stop"; judge by exit code only
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $py "src\$script" @a 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "$script failed (exit $code)" }
}

# Phase 0: audit (counts only)
Run audit_inventory.py
Run audit_stress400.py
Run compare_uci.py
Run check_onnx.py
Run scrub.py @("--audit", "sms_export_2026-01-20.csv", "Friend_sms_export_2026-10-17.csv", "2sms_export_2026-10-18.csv", "3sms_export_2026-10-18.csv", "--out", "pii_audit_all.json")

# Phase 1-2: pool, labels, splits (the test file is never rewritten; make_splits asserts its SHA-256)
Run build_known_names.py
Run build_pool.py
Run predict_v73.py
Run assemble_labels.py
Run make_splits.py
Run agreement.py

# Phase 3-4: stress sets, baselines on val, v7.3 on val
Run make_stress.py
Run baselines.py
Run eval_v73.py @("--split", "val")
$ErrorActionPreference = "Continue"; & $py (Join-Path "tests" "test_rules.py") 2>&1 | ForEach-Object { "$_" }; $code = $LASTEXITCODE; $ErrorActionPreference = "Stop"
if ($code -ne 0) { throw "rule-layer tests failed" }

# Phase 5: training (optional; the reported models were trained on a Colab T4 with these exact commands)
if ($Train) {
    Run sweep.py @("--stage", "A")
    Run sweep.py @("--stage", "B")
    foreach ($s in 1..5) {
        Run train.py @("--seed", "$s", "--balance", "S0", "--digits", "D1", "--placeholders", "U1", "--synthetic-ratio", "0",
                       "--lr", "5e-5", "--epochs", "5", "--max-len", "128", "--label-smoothing", "0", "--batch", "32")
    }
    Write-Host "NOTE: new run ids differ; update SEEDS in src/final_test.py before using new models."
}
Run calibrate.py @("--runs", "train-20261018-132741", "train-20261018-132939", "train-20261018-133131", "train-20261018-133323", "train-20261018-133515")

# Phase 6: export, quantize, parity on val, desktop latency
Run export_onnx.py
Run bench_onnx.py

# Phase 7-8: the single test evaluation (only with -RerunTest), then everything derived from saved predictions
if ($RerunTest) {
    Run eval_v73.py @("--split", "test")
    Run final_test.py
    Run baselines.py @("--final-test")
}
Run ablation.py
Run stress_eval.py
Run stats.py
Run abstain.py
Run error_analysis.py

# Phase 10 (post-hoc; saved test predictions, val and stress sets only)
Run posthoc.py
Run tfidf_extra.py
Run ablation_val.py

# Phase 9: paper assets
Run make_figures.py
Run make_tables.py
Run make_paper_numbers.py
Run make_paper_pack.py
Write-Host "Done. Figures: figures\*.pdf  Tables: results\tables\*.tex  Numbers: results\paper_numbers.md"
