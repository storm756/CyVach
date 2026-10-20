"""Second-annotator agreement on the 180 owner-checked test items (run after src/annotate_blind.py is complete).

Compares the blind second annotator (annotation/second_annotator.csv; last answer per item, skips dropped) with:
  final test labels (data/test.csv), Claude's blind proposals (annotation/claude_gold_pass1.csv),
  and the author's checked labels (annotation/check_results.csv).
Reports raw agreement and Cohen's kappa, plus the confusion matrix against the final labels.
Output: results/kappa.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402


def main():
    run = new_run("kappa", {})
    sec = pd.read_csv(ROOT / "annotation" / "second_annotator.csv").drop_duplicates("item_id", keep="last")
    sec = sec[sec["label"] != "skip"]
    test = pd.read_csv(ROOT / "data" / "test.csv", keep_default_na=False)[["item_id", "final_label"]]
    claude = pd.read_csv(ROOT / "annotation" / "claude_gold_pass1.csv")[["item_id", "claude_label"]]
    owner = pd.read_csv(ROOT / "annotation" / "check_results.csv").drop_duplicates("item_id", keep="last")[["item_id", "final_label"]].rename(columns={"final_label": "owner_label"})
    d = sec.merge(test, on="item_id").merge(claude, on="item_id").merge(owner, on="item_id")
    res = {"n_annotated": int(len(sec)), "n_compared": int(len(d))}
    for ref in ("final_label", "claude_label", "owner_label"):
        res[f"vs_{ref}"] = {"agreement": float((d["label"] == d[ref]).mean()), "k": int((d["label"] == d[ref]).sum()),
                            "cohen_kappa": float(cohen_kappa_score(d["label"], d[ref], labels=LABELS))}
    res["confusion_second_rows_final_cols"] = confusion_matrix(d["label"], d["final_label"], labels=LABELS).tolist()
    res["second_annotator_borderline"] = int(d["borderline"].sum())
    save_result(run, res, "kappa.json")
    print(json.dumps(res, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
