"""Silver-label self-agreement: Claude pass 1 vs an independent pass 2 on a random 10%."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402


def main():
    run = new_run("silver_self_agreement", {"sample": "annotation/batches/silver_p2.tsv (10%, seed 99)"})
    b = pd.read_csv(ROOT / "annotation/batches/silver_p2.tsv", sep="\t", keep_default_na=False)
    p2 = pd.read_csv(ROOT / "annotation/batches/silver_p2.labels.tsv", sep="\t", keep_default_na=False, quoting=3)
    assert set(b["id"]) == set(p2["id"]) and not p2["id"].duplicated().any(), "pass-2 ids do not match the sample"
    p1 = pd.read_csv(ROOT / "annotation/claude_silver_pass1.csv")
    m = p2.rename(columns={"id": "item_id", "label": "p2"}).merge(p1[["item_id", "claude_label"]], on="item_id")
    kappa = cohen_kappa_score(m["claude_label"], m["p2"], labels=LABELS)
    out = {"n": int(len(m)), "agreement": float((m["claude_label"] == m["p2"]).mean()), "cohen_kappa": float(kappa),
           "confusion_pass1_rows_pass2_cols": confusion_matrix(m["claude_label"], m["p2"], labels=LABELS).tolist(),
           "labels": LABELS,
           "disagreements": m.loc[m["claude_label"] != m["p2"], ["item_id", "claude_label", "p2"]].to_dict("records")}
    save_result(run, out, "silver_self_agreement.json")
    print({k: v for k, v in out.items() if k != "disagreements"}); print("disagreements:", out["disagreements"]); print(run["run_id"])


if __name__ == "__main__":
    main()
