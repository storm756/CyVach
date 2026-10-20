"""Phase 8: error analysis of the deploy model on the main test subset, from saved predictions.

The categories of the 50 most confident errors were assigned once by Claude Opus 5.5 (in-session, 2026-10-18, see
docs/AI_USE_LOG.md) and are stored by item id in results/error_analysis_top50.csv (no text). This script re-derives the
top-50 list from the saved predictions, checks it matches the stored ids, and recomputes the counts and error rates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402

COL = "seed3_int8_quint8|filled"


def main():
    run = new_run("error_analysis", {"model": COL, "top": 50, "categorised_by": "Claude Opus 5.5 (in-session), item ids only stored"})
    p = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    t = pd.read_csv(ROOT / "data" / "test.csv", keep_default_na=False)[["item_id", "source"]]
    p = p.merge(t, on="item_id")
    m = p["native_script"].astype(str) != "True"
    p["pred"], p["conf"] = p[f"{COL}|raw"], p[[f"{COL}|p_{l}" for l in LABELS]].max(axis=1)
    err = m & (p["pred"] != p["true"])
    top = p[err].sort_values("conf", ascending=False, kind="stable").head(50)
    cats = pd.read_csv(ROOT / "results" / "error_analysis_top50.csv")
    assert list(top["item_id"]) == list(cats["item_id"]), "top-50 errors differ from the categorised list"
    by = pd.DataFrame({"n": p[m].groupby("source").size(), "errors": err[m].groupby(p[m]["source"]).sum()})
    by["error_rate"] = (by["errors"] / by["n"]).round(4)
    res = {"n_errors_main": int(err.sum()), "top50_category_counts": cats["category"].value_counts().to_dict(),
           "error_rate_by_source": by.reset_index().to_dict("records"),
           "top50_true_pred": cats.groupby(["final_label", "pred"]).size().rename("n").reset_index().to_dict("records")}
    save_result(run, res, "error_analysis.json")
    print(res["n_errors_main"], res["top50_category_counts"]); print(by); print(run["run_id"])


if __name__ == "__main__":
    main()
