"""Build release/ - the public GitHub repository content (https://github.com/storm756/CyVach).

Included: research code (src/, tests/, make_all.ps1, requirements, Colab notebooks), docs (label guidelines with the
exact LLM labeling prompt in AI_USE_LOG, stress-set spec, rule-layer description, dataset search log, data statement,
data sources, provenance), the SYNTHETIC stress sets, result files (metrics, run logs, per-item predictions keyed by
opaque ids), paper tables/figures and the model card.
Excluded: every real message (data/ except data/stress/, annotation/, data_raw/), name lists, model weights, the app.
Checks (build fails on any hit): (1) no real message - first 40 normalized characters of every real pool message,
searched in every text file; (2) no private name seed from data/name_seeds.txt.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, normalize, save_result  # noqa: E402

OUT = ROOT / "release"
DIRS = {"src": "src", "tests": "tests", "colab": "colab", "figures": "figures", "results/tables": "results/tables"}
FILES = ["make_all.ps1", "requirements.txt", "requirements.in", "README_repro.md", "models/model_card.md",
         "docs/label_guidelines.md", "docs/stress_spec.md", "docs/rule_layer_as_shipped.md", "docs/DATASET_CANDIDATES.md",
         "docs/AI_USE_LOG.md", "docs/DATA_SOURCES.md", "docs/DATA_STATEMENT.md", "docs/PROVENANCE.md",
         "results/paper_numbers.md"]
SKIP_SUFFIX = {".zip", ".onnx", ".pyc", ".tsv"}
# Reviewed generic matches (no personal content): Android's stock quick-reply text, and a well-known scam template
# that also appears in the legacy template-generated data. Any other match fails the build.
ALLOW = {"sorry, i can't talk right now.", "your electricity will be disconnected to"}

README = """# CyVach: auditing and rebuilding a deployed on-device SMS fraud classifier

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
"""


def main():
    run = new_run("make_release", {})
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for src, dst in DIRS.items():
        for f in (ROOT / src).rglob("*"):
            if f.is_file() and f.suffix not in SKIP_SUFFIX and "__pycache__" not in f.parts:
                t = OUT / dst / f.relative_to(ROOT / src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
    for f in FILES:
        t = OUT / f
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, t)
    for f in (ROOT / "results").glob("*.json"):
        shutil.copy2(f, OUT / "results" / f.name)
    for f in (ROOT / "results").glob("*.csv"):
        shutil.copy2(f, OUT / "results" / f.name)
    (OUT / "data" / "stress").mkdir(parents=True)
    for f in (ROOT / "data" / "stress").glob("*.csv"):
        shutil.copy2(f, OUT / "data" / "stress" / f.name)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitignore").write_text("data_raw/\ndata/*\n!data/stress/\nannotation/\nmodels/*\n!models/model_card.md\n*.onnx\n*.safetensors\n.venv/\n__pycache__/\ntools/\n", encoding="utf-8")
    # leak checks
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)
    real = pool[pool["is_synthetic"].astype(str) != "True"]["text"]
    keys = {normalize(t)[:40] for t in real if len(normalize(t)) >= 25}
    seeds_f = ROOT / "data" / "name_seeds.txt"
    seeds = [s.strip().lower() for s in seeds_f.read_text(encoding="utf-8").splitlines() if s.strip()] if seeds_f.exists() else []
    hits, seed_hits = [], []
    for f in OUT.rglob("*"):
        if f.is_file() and f.suffix in (".md", ".tex", ".txt", ".py", ".json", ".csv", ".ps1", ".ipynb", ".in"):
            body = f.read_text(encoding="utf-8", errors="ignore")
            nb = normalize(body)
            hits += [str(f.relative_to(OUT)) for k in keys if k in nb and k not in ALLOW]
            low = body.lower()
            seed_hits += [str(f.relative_to(OUT)) for s_ in seeds if s_ in low]  # informational: the author's name is public
    files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    res = {"n_files": len(files), "real_messages_checked": len(keys), "leak_hits": len(hits), "leak_files": sorted(set(hits)),
           "name_seed_files": sorted(set(seed_hits))}
    save_result(run, res, "release.json")
    print(json.dumps({k: v for k, v in res.items()}, indent=1)); print(run["run_id"])
    if hits:
        raise SystemExit("real message text found in release/")


if __name__ == "__main__":
    main()
