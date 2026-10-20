"""Phase 8: caution-as-abstention (reject option) - tuned on VAL, applied once to saved TEST probabilities.

Confidence = max temperature-scaled probability (T fitted on val, seed 3), or negative entropy.
1. Risk-coverage curves (accept the most confident items first) and AURC, on val and test.
2. Caution-as-abstention rule: if the model predicts ham or promo with calibrated confidence < tau, output caution
   (an uncertain message is shown as "be careful" instead of "safe"; spam and caution predictions are kept).
   tau is chosen on VAL from {none, 0.40, 0.45, ..., 0.95} by the highest val macro-F1 (pre-registered); ties -> no rule.
   Also reported: scam "warned" rate (true scam predicted spam OR caution) and the share of legit messages newly
   shown as caution.
Val probabilities: seed-3 FP32 val logits saved at training time (Colab, fp16 autocast). Test: deploy INT8 QUInt8, main subset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from final_test import SEEDS  # noqa: E402
from metrics import evaluate  # noqa: E402

TAUS = [None] + [round(x, 2) for x in np.arange(0.40, 0.951, 0.05)]


def risk_coverage(P, y, score):
    order = np.argsort(-score)
    err = (P.argmax(1) != y)[order].astype(float)
    n = np.arange(1, len(y) + 1)
    risk = np.cumsum(err) / n
    cov = n / len(y)
    pts = {f"{c:.1f}": float(risk[max(0, int(np.ceil(c * len(y))) - 1)]) for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    return {"aurc": float(risk.mean()), "risk_at_coverage": pts, "curve": {"coverage": cov[::5].round(4).tolist(), "risk": risk[::5].round(4).tolist()}}


def apply_rule(P, tau):
    pred = P.argmax(1).copy()
    if tau is not None:
        low = (P.max(1) < tau) & np.isin(pred, [LABELS.index("ham"), LABELS.index("promo")])
        pred[low] = LABELS.index("caution")
    return pred


def summary(P, y, tau):
    pred = apply_rule(P, tau)
    e = evaluate([LABELS[i] for i in y], [LABELS[i] for i in pred])
    scam, cau = LABELS.index("spam"), LABELS.index("caution")
    legit = np.isin(y, [LABELS.index("ham"), LABELS.index("promo")])
    base = P.argmax(1)
    return {"macro_f1": e["macro_f1"], "f1_caution": e["per_class"]["caution"]["f1"], "scam_recall": e["per_class"]["spam"]["recall"],
            "scam_warned_rate": float(np.isin(pred[y == scam], [scam, cau]).mean()),
            "legit_newly_caution": float(((pred == cau) & (base != cau) & legit).sum() / legit.sum()),
            "n_changed": int((pred != base).sum())}


def main():
    run = new_run("abstain", {"taus": TAUS})
    rid = SEEDS[3]
    T = json.loads((ROOT / "models" / rid / "temperature.json").read_text())["temperature"]
    vl, vy = np.load(ROOT / "models" / rid / "val_logits.npy"), np.load(ROOT / "models" / rid / "val_labels.npy")
    Pv = softmax(vl / T, 1)
    pf = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    pf = pf[pf["native_script"].astype(str) != "True"]
    Pt = pf[[f"seed3_int8_quint8|filled|p_{l}" for l in LABELS]].values.clip(1e-12, 1)
    Pt = softmax(np.log(Pt) / T, 1)
    ty = np.array([LABELS.index(t) for t in pf["true"]])
    ent = lambda P: (P * np.log(P.clip(1e-12))).sum(1)  # negative entropy: higher = more confident
    res = {"temperature": T, "risk_coverage": {
        "val": {"maxprob": risk_coverage(Pv, vy, Pv.max(1)), "neg_entropy": risk_coverage(Pv, vy, ent(Pv))},
        "test": {"maxprob": risk_coverage(Pt, ty, Pt.max(1)), "neg_entropy": risk_coverage(Pt, ty, ent(Pt))}}}
    val_grid = {str(t): summary(Pv, vy, t) for t in TAUS}
    best = max(TAUS, key=lambda t: (round(val_grid[str(t)]["macro_f1"], 6), t is None))
    res["val_grid"] = val_grid
    res["tau_selected_on_val"] = best
    res["test_no_rule"] = summary(Pt, ty, None)
    res["test_with_selected_rule"] = summary(Pt, ty, best)
    res["test_grid_for_reference_only"] = {str(t): summary(Pt, ty, t) for t in TAUS}
    save_result(run, res, "abstain.json")
    print(pd.DataFrame(val_grid).T.round(4).to_string())
    print("tau selected on val:", best)
    print("test no rule:", {k: round(v, 4) for k, v in res["test_no_rule"].items()})
    print("test with rule:", {k: round(v, 4) for k, v in res["test_with_selected_rule"].items()})
    for s in ("val", "test"):
        for k, v in res["risk_coverage"][s].items():
            print(s, k, "AURC", round(v["aurc"], 4), v["risk_at_coverage"])
    print(run["run_id"])


if __name__ == "__main__":
    main()
