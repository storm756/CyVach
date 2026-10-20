"""Phase 8: statistics from SAVED test predictions (no model is run on test here).

- Point metrics + confusion matrices (counts, row-normalised) on the main subset (test minus native script).
- Cluster-level bootstrap (1000 resamples of near-duplicate clusters, seed 0): 95% percentile CIs for macro-F1, MCC,
  accuracy, per-class F1, legit-flagged-as-scam; paired CIs for differences vs the new deploy model.
- Exact McNemar tests (two-sided binomial on discordant pairs): new deploy INT8 vs v7.3 INT8, vs TF-IDF LR/SVM.
- Test ECE (15 bins) of the 5 seeds and the deploy INT8, before/after the temperature fitted on VAL.
- Class-balanced view: 1000 stratified samples of real test items, n = smallest class per class (no synthetic items).
Outputs: results/stats.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import ece  # noqa: E402
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from final_test import SEEDS  # noqa: E402
from metrics import evaluate  # noqa: E402

B, SEED = 1000, 0
DEPLOY = "new_int8_quint8"


def load():
    p = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    b = pd.read_csv(ROOT / "results" / "preds_baselines_test.csv", keep_default_na=False)
    assert (p["item_id"].values == b["item_id"].values).all()
    p["native_script"] = p["native_script"].astype(str) == "True"
    m = ~p["native_script"].values
    models = {DEPLOY: p["seed3_int8_quint8|filled|raw"], "new_int8_quint8+rules": p["seed3_int8_quint8|filled|pipeline"],
              "new_int8_qint8": p["seed3_int8_qint8|filled|raw"],
              **{f"new_seed{s}_fp32": p[f"seed{s}_fp32|filled|raw"] for s in SEEDS},
              "v73_int8": p["v73_int8|filled|raw"], "v73_int8+rules": p["v73_int8|filled|pipeline"],
              "v73_fp32": p["v73_fp32|filled|raw"], "tfidf_lr": b["lr"], "tfidf_svm": b["svm"]}
    return p[m].reset_index(drop=True), {k: v.values[m] for k, v in models.items()}


def scalars(y, pred):
    e = evaluate(y, pred)
    out = {"macro_f1": e["macro_f1"], "mcc": e["mcc"], "accuracy": e["accuracy"], "legit_as_scam": e["fpr_legit_as_scam"]}
    out.update({f"f1_{c}": e["per_class"][c]["f1"] for c in LABELS})
    out["scam_recall"] = e["per_class"]["spam"]["recall"]
    return out


def main():
    run = new_run("stats", {"B": B, "seed": SEED})
    df, models = load()
    y = df["true"].values
    res = {"n_main": int(len(y)), "n_clusters": int(df["cluster_id"].nunique()), "point": {}, "ci95": {}, "diff_vs_deploy_ci95": {},
           "mcnemar": {}, "confusion": {}}
    for k, pr in models.items():
        res["point"][k] = scalars(y, pr)
        cm = pd.crosstab(pd.Categorical(y, LABELS), pd.Categorical(pr, LABELS), dropna=False).values
        res["confusion"][k] = {"counts": cm.tolist(), "row_normalised": (cm / cm.sum(1, keepdims=True)).round(4).tolist()}
    # cluster bootstrap
    rng = np.random.default_rng(SEED)
    cl = df["cluster_id"].values
    uniq = np.unique(cl)
    idx_by = {c: np.where(cl == c)[0] for c in uniq}
    boots = {k: [] for k in models}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by[c] for c in pick])
        for k, pr in models.items():
            boots[k].append(scalars(y[ix], pr[ix]))
    for k in models:
        bdf = pd.DataFrame(boots[k])
        res["ci95"][k] = {c: [float(bdf[c].quantile(.025)), float(bdf[c].quantile(.975))] for c in bdf}
        if k != DEPLOY:
            d = pd.DataFrame(boots[DEPLOY]) - bdf
            res["diff_vs_deploy_ci95"][k] = {c: {"diff": res["point"][DEPLOY][c] - res["point"][k][c],
                                                 "ci95": [float(d[c].quantile(.025)), float(d[c].quantile(.975))]}
                                             for c in ("macro_f1", "mcc", "accuracy", "legit_as_scam", "f1_spam", "f1_caution")}
    # McNemar (exact)
    ok_new = models[DEPLOY] == y
    for k in ("v73_int8", "v73_int8+rules", "tfidf_lr", "tfidf_svm", "new_int8_qint8", "new_int8_quint8+rules"):
        ok = models[k] == y
        b_, c_ = int((ok_new & ~ok).sum()), int((~ok_new & ok).sum())
        res["mcnemar"][f"{DEPLOY}_vs_{k}"] = {"new_right_other_wrong": b_, "new_wrong_other_right": c_,
                                              "p_exact": float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0}
    # seeds
    f1s = [res["point"][f"new_seed{s}_fp32"]["macro_f1"] for s in SEEDS]
    res["seeds_macro_f1"] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s, ddof=1)), "values": f1s}
    # ECE with val temperature
    pf = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    pf = pf[pf["native_script"].astype(str) != "True"].reset_index(drop=True)
    yi = np.array([LABELS.index(t) for t in pf["true"]])
    res["ece_test"] = {}
    for name, col, rid in [*[(f"seed{s}_fp32", f"seed{s}_fp32|filled", SEEDS[s]) for s in SEEDS], ("deploy_int8_quint8", "seed3_int8_quint8|filled", SEEDS[3])]:
        P = pf[[f"{col}|p_{l}" for l in LABELS]].values.clip(1e-12, 1)
        T = json.loads((ROOT / "models" / rid / "temperature.json").read_text())["temperature"]
        res["ece_test"][name] = {"T_from_val": T, "before": ece(P, yi), "after": ece(softmax(np.log(P) / T, 1), yi)}
    e = pd.DataFrame(res["ece_test"]).T
    res["ece_test_seed_summary"] = {c: {"mean": float(e.loc[[f"seed{s}_fp32" for s in SEEDS], c].mean()),
                                        "std": float(e.loc[[f"seed{s}_fp32" for s in SEEDS], c].std())} for c in ("before", "after")}
    # class-balanced view (stratified sampling of real items)
    n_min = int(pd.Series(y).value_counts().min())
    by_cls = {c: np.where(y == c)[0] for c in LABELS}
    bal = {k: [] for k in models}
    for _ in range(B):
        ix = np.concatenate([rng.choice(by_cls[c], n_min, replace=False) for c in LABELS])
        for k, pr in models.items():
            e2 = evaluate(y[ix], pr[ix]); bal[k].append((e2["accuracy"], e2["macro_f1"]))
    res["balanced_view"] = {"n_per_class": n_min, **{k: {"accuracy_mean": float(np.mean([a for a, _ in v])),
                                                          "macro_f1_mean": float(np.mean([f for _, f in v])),
                                                          "macro_f1_std": float(np.std([f for _, f in v]))} for k, v in bal.items()}}
    save_result(run, res, "stats.json")
    t = pd.DataFrame({k: {"F1": res["point"][k]["macro_f1"], "CI": "[%.3f, %.3f]" % tuple(res["ci95"][k]["macro_f1"]),
                          "MCC": res["point"][k]["mcc"], "legit->scam": res["point"][k]["legit_as_scam"],
                          "bal F1": res["balanced_view"][k]["macro_f1_mean"]} for k in models}).T
    print(t.to_string())
    print(json.dumps({"diff": {k: {c: (round(v[c]["diff"], 4), [round(x, 4) for x in v[c]["ci95"]]) for c in ("macro_f1", "legit_as_scam")}
                               for k, v in res["diff_vs_deploy_ci95"].items()}, "mcnemar": res["mcnemar"],
                      "ece": res["ece_test"], "ece_seeds": res["ece_test_seed_summary"], "seeds": res["seeds_macro_f1"],
                      "clusters": res["n_clusters"]}, indent=1))
    print(run["run_id"])


if __name__ == "__main__":
    main()
