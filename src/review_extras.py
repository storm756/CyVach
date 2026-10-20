"""POST-HOC analyses requested by the v3 reviewers (saved test predictions, val and synthetic stress data only).

1. Benign camouflage controls: the 50 legitimate (ham + caution) base texts of the spoof set, from their registered
   -S/-T sender, each with the 3 camouflage phrases used on scams ("Security Alert: " prefix, "Do not share your OTP
   with anyone." / "Beware of fraud calls." suffix) -> 150 synthetic legit messages (written to
   data/stress/benign_camouflage.csv). Compared with the 150 camouflaged scams of the spoof set: scam recall,
   legit accuracy, balanced accuracy; model only and with the app rules; and how many scam verdicts R1 removes.
2. Prevalence-adjusted scam precision: PPV = R*pi / (R*pi + F*(1-pi)) with pi = scam share of the natural-distribution
   validation set, R = test scam recall, F = test legit-flagged-as-scam rate.
3. Equivalence (TOST) of deploy INT8 vs TF-IDF: paired item bootstrap (1000, seed 0) of the macro-F1 difference;
   90% CI inside +-0.05 => equivalent at margin 0.05 (margin chosen post-hoc, stated as such).
4. Caution provenance: sources of caution items in train / val / test (does test caution share its phone with train?).
Output: results/review_extras.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS  # noqa: E402
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from final_test import ONNX, onnx_logits  # noqa: E402
from metrics import evaluate  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402
from subsets import load_split  # noqa: E402

PHRASES = [("prefix", "Security Alert: "), ("suffix", " Do not share your OTP with anyone."), ("suffix", " Beware of fraud calls.")]


def camouflage_controls(app):
    sp = pd.read_csv(ROOT / "data" / "stress" / "stress_spoof.csv", keep_default_na=False)
    base = sp[(sp["sender_kind"] == "registered_ST") & sp["final_label"].isin(["ham", "caution"]) & (sp["camouflage"] == 0)]
    rows = []
    for r in base.itertuples():
        for i, (where, ph) in enumerate(PHRASES):
            t = ph + r.text if where == "prefix" else r.text.rstrip() + ph
            rows.append({"item_id": f"bc_{r.base_id}_{i}", "base_id": r.base_id, "sender": r.sender, "text": t,
                         "final_label": r.final_label, "phrase": i, "is_synthetic": True})
    ben = pd.DataFrame(rows)
    ben.to_csv(ROOT / "data" / "stress" / "benign_camouflage.csv", index=False)
    scam = sp[(sp["camouflage"] == 1)][["item_id", "sender", "text", "final_label"]]
    allm = pd.concat([scam.assign(group="camouflaged_scam"), ben[["item_id", "sender", "text", "final_label"]].assign(group="camouflaged_legit")], ignore_index=True)
    texts = allm["text"].tolist()
    probs = {"new_int8": softmax(onnx_logits(ROOT / ONNX["int8_quint8"], texts, app), 1),
             "v73_int8": softmax(onnx_logits(APP_ASSETS / "sms_spam_tiny.onnx", texts, app, batch=1), 1)}
    out = {"n_camouflaged_scams": int((allm.group == "camouflaged_scam").sum()), "n_camouflaged_legit": int((allm.group == "camouflaged_legit").sum()),
           "n_legit_base_texts": int(base["base_id"].nunique()), "models": {}}
    for k in ("lr", "svm"):
        pl = joblib.load(ROOT / "models" / "tfidf" / f"tfidf_{k}.joblib")
        probs[f"tfidf_{k}"] = pl.predict(texts)  # labels (SVM has no probabilities)
    for name, P in probs.items():
        if isinstance(P, np.ndarray) and P.ndim == 2:
            raw = np.array([LABELS[i] for i in P.argmax(1)]); conf = P.max(1)
        else:
            raw = np.asarray(P); conf = np.full(len(raw), 0.99)  # TF-IDF: no calibrated confidence; rules use 0.99
        pipe = np.array([pipeline(s, t, lambda _t, p=p, c=c: (p, float(c)), ALL_LAYERS).label
                         for s, t, p, c in zip(allm["sender"], texts, raw, conf)])
        nosafety = np.array([pipeline(s, t, lambda _t, p=p, c=c: (p, float(c)), ALL_LAYERS - {"safety"}).label
                             for s, t, p, c in zip(allm["sender"], texts, raw, conf)])
        sc = (allm.group == "camouflaged_scam").values
        r = {}
        for view, pr in (("model_only", raw), ("with_rules", pipe), ("rules_without_R1", nosafety)):
            rec = float((pr[sc] == "spam").mean())
            legit_ok = float((pr[~sc] == allm["final_label"].values[~sc]).mean())
            legit_not_scam = float((pr[~sc] != "spam").mean())
            r[view] = {"scam_recall": rec, "scam_k": int((pr[sc] == "spam").sum()), "legit_exact_acc": legit_ok,
                       "legit_not_flagged_scam": legit_not_scam, "legit_flagged_scam_k": int((pr[~sc] == "spam").sum()),
                       "balanced_acc_scam_vs_notscam": (rec + legit_not_scam) / 2}
        r["R1_removed_scam_verdicts_on_camouflaged_scams"] = int(((nosafety == "spam") & (pipe != "spam") & sc).sum())
        out["models"][name] = r
    return out


def ppv_and_tost():
    st = json.loads((ROOT / "results" / "stats.json").read_text(encoding="utf-8"))["result"]
    va = load_split(ROOT / "data" / "val.csv", drop_native=True)
    pi = float((va["final_label"] == "spam").mean())
    ppv = {}
    for k in ("new_int8_quint8", "new_int8_quint8+rules", "tfidf_lr", "tfidf_svm", "v73_int8", "v73_int8+rules"):
        R, F = st["point"][k]["scam_recall"], st["point"][k]["legit_as_scam"]
        ppv[k] = {"recall": R, "legit_as_scam": F, "ppv_at_val_prevalence": R * pi / (R * pi + F * (1 - pi))}
    # paired bootstrap of macro-F1 difference
    p = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    b = pd.read_csv(ROOT / "results" / "preds_baselines_test.csv", keep_default_na=False)
    m = (p["native_script"].astype(str) != "True").values
    y = p["true"].values[m]
    new = p["seed3_int8_quint8|filled|raw"].values[m]
    rng = np.random.default_rng(0)
    tost = {}
    for k in ("lr", "svm"):
        o = b[k].values[m]
        d = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            d.append(evaluate(y[i], new[i])["macro_f1"] - evaluate(y[i], o[i])["macro_f1"])
        lo, hi = np.quantile(d, [0.05, 0.95])
        tost[f"tfidf_{k}"] = {"diff_point": evaluate(y, new)["macro_f1"] - evaluate(y, o)["macro_f1"], "ci90": [float(lo), float(hi)],
                              "margin": 0.05, "equivalent_at_margin": bool(lo > -0.05 and hi < 0.05),
                              "smallest_margin_shown": float(max(abs(lo), abs(hi)))}
    return {"val_scam_prevalence": pi, "ppv": ppv, "tost": tost}


def caution_provenance():
    out = {}
    for s in ("train", "val", "test"):
        d = load_split(ROOT / "data" / f"{s}.csv", drop_native=True)
        if s == "train":
            d = d[d["is_synthetic"].astype(str) != "True"]
        c = d[d["final_label"] == "caution"]
        out[s] = {"n": int(len(c)), "by_source": c["source"].value_counts().to_dict(),
                  "by_label_source": c["label_source"].value_counts().to_dict()}
    return out


def main():
    run = new_run("review_extras", {"post_hoc": True})
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    res = {"post_hoc": True, "camouflage_controls": camouflage_controls(app), **ppv_and_tost(), "caution_provenance": caution_provenance()}
    save_result(run, res, "review_extras.json")
    print(json.dumps(res, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
