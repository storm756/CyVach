"""Phase 5: temperature scaling on VAL logits for each final-seed model; ECE before/after; seed summary.

Fits a single temperature T > 0 per model by minimising val NLL (bounded scalar search over log T).
ECE: 15 equal-width confidence bins, top-label. Note: T is fitted and ECE measured on the same val set
(in-sample); out-of-sample ECE is measured once on the test set after Gate 3.
Writes T into models/<run>/temperature.json and results/calibrate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from metrics import evaluate  # noqa: E402


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(logits, y, t):
    return float(-log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="train run ids of the final seeds")
    args = ap.parse_args()
    run = new_run("calibrate", vars(args))
    rows = []
    for rid in args.runs:
        d = ROOT / "models" / rid
        lg, y = np.load(d / "val_logits.npy"), np.load(d / "val_labels.npy")
        res = minimize_scalar(lambda lt: nll(lg, y, np.exp(lt)), bounds=(-3, 3), method="bounded")
        t = float(np.exp(res.x))
        tr = json.loads((ROOT / "results" / f"{rid}.json").read_text(encoding="utf-8"))
        ev = evaluate([LABELS[i] for i in y], [LABELS[i] for i in lg.argmax(1)])
        r = {"run_id": rid, "seed": tr["config"]["seed"], "best_epoch": tr["result"]["best_val"]["epoch"],
             "val_macro_f1": ev["macro_f1"], "val_mcc": ev["mcc"], "val_acc": ev["accuracy"],
             **{f"val_f1_{l}": ev["per_class"][l]["f1"] for l in LABELS},
             "val_fpr_legit_as_scam": ev["fpr_legit_as_scam"], "probe_acc": tr["result"]["best_val"].get("probe_acc"),
             "temperature": t, "nll_before": nll(lg, y, 1.0), "nll_after": nll(lg, y, t),
             "ece_before": ece(softmax(lg, 1), y), "ece_after": ece(softmax(lg / t, 1), y)}
        (d / "temperature.json").write_text(json.dumps({"temperature": t, "fitted_on": "val", "calibrate_run": run["run_id"]}))
        rows.append(r)
    keys = [k for k in rows[0] if k not in ("run_id", "seed")]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1))}
               for k in keys if rows[0][k] is not None}
    f1s = [r["val_macro_f1"] for r in rows]
    median_run = rows[int(np.argsort(f1s)[len(f1s) // 2])]["run_id"]
    out = {"seeds": rows, "summary": summary, "deploy_candidate_median_val_f1": median_run}
    save_result(run, out, "calibrate.json")
    for r in rows:
        print(r["seed"], r["run_id"], f"F1 {r['val_macro_f1']:.4f} caution {r['val_f1_caution']:.3f} T {r['temperature']:.3f} "
              f"ECE {r['ece_before']:.4f}->{r['ece_after']:.4f}")
    for k in ("val_macro_f1", "val_f1_caution", "val_f1_spam", "val_mcc", "val_fpr_legit_as_scam", "ece_before", "ece_after", "temperature"):
        print(f"{k}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    print("median-F1 seed (deploy candidate):", median_run); print(run["run_id"])


if __name__ == "__main__":
    main()
