"""POST-HOC analyses requested by the review of paper v5/humanised v2 (saved test predictions, stress data only;
no model is trained or selected here, and no test prediction is recomputed except the synthetic stress sets).

1. Model agreement with the blind second annotator on the 180 owner-checked test items (Cohen's kappa), next to each
   model's agreement with the final labels on the same items.
2. Per-rule table: test items each rule layer changes (changed / fixed / broke, macro-F1), applied alone to the raw
   new-INT8 predictions and on top of R0 (the "trusted" layer).
3. Single-device caution: metrics with the author's-phone (source export2026) caution items removed from the main
   test subset: 4-class macro-F1, 3-class (ham/scam/promo) macro-F1, and caution P/R/F1 on the remaining caution items.
4. PPV range: scam precision at 1%, 6.2% (validation) and 20% prevalence, from test scam recall and legit-as-scam rate.
5. Table 3 uncertainty: camouflaged-scam catches (of 150) with a 95% CI from a cluster bootstrap over the 25 base scam
   texts (2000 resamples, seed 0), model only, with rules, and rules without R1; benign controls clustered by base text.
6. ECE check: test ECE of the deployed INT8 before/after temperature with 10 and 15 bins.
Output: results/review2_extras.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from calibrate import ece  # noqa: E402
from check_onnx import APP_ASSETS  # noqa: E402
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from eval_v73 import address_for_rules  # noqa: E402
from final_test import ONNX, SEEDS, onnx_logits  # noqa: E402
from metrics import evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from review_extras import PHRASES  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402
from subsets import load_split  # noqa: E402

NEW, V73 = "seed3_int8_quint8|filled", "v73_int8|filled"
LAYERS = {"H1/H2": "pre", "R0": "trusted", "R1": "safety", "R2/R3": "mobile", "R4": "suffix"}


def load():
    df = load_split(ROOT / "data" / "test.csv", drop_native=False).reset_index(drop=True)
    pr = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    bl = pd.read_csv(ROOT / "results" / "preds_baselines_test.csv", keep_default_na=False)
    assert (pr["item_id"].values == df["item_id"].values).all() and (bl["item_id"].values == df["item_id"].values).all()
    return df, pr, bl


def second_annotator(df, pr, bl):
    sec = pd.read_csv(ROOT / "annotation" / "second_annotator.csv").drop_duplicates("item_id", keep="last")
    sec = sec[sec["label"] != "skip"].set_index("item_id")["label"]
    preds = {"new_int8": pr[f"{NEW}|raw"], "tfidf_lr": bl["lr"], "v73_int8": pr[f"{V73}|raw"], "final_labels": df["final_label"]}
    idx = df["item_id"].isin(sec.index).values
    s2 = sec.loc[df["item_id"][idx]].values
    out = {"n": int(idx.sum()), "n_native_script": int(df["native_script"].values[idx].sum())}
    for k, p in preds.items():
        p = np.asarray(p)[idx]
        out[k] = {"vs_second": {"agree_k": int((p == s2).sum()), "kappa": float(cohen_kappa_score(s2, p, labels=LABELS))}}
        if k != "final_labels":
            y = df["final_label"].values[idx]
            out[k]["vs_final_same_items"] = {"agree_k": int((p == y).sum()), "kappa": float(cohen_kappa_score(y, p, labels=LABELS))}
    return out


def per_rule(df, pr):
    frng = random.Random(1234)
    texts = [fill(t, frng) for t in df["text"]]
    m = ~df["native_script"].values
    y = df["final_label"].values
    P = pr[[f"{NEW}|p_{l}" for l in LABELS]].values
    raw = np.array([LABELS[i] for i in P.argmax(1)]); conf = P.max(1)

    def run(layers):
        return np.array([pipeline(address_for_rules(r), t, lambda _t, p=p, c=c: (p, float(c)), frozenset(layers)).label
                         for (_, r), t, p, c in zip(df.iterrows(), texts, raw, conf)])

    r0 = run({"trusted"})
    out = {}
    for name, layer in LAYERS.items():
        for base_name, base, layers in (("alone", raw, {layer}), ("on_top_of_R0", r0, {layer, "trusted"})):
            if name == "R0" and base_name == "on_top_of_R0":
                continue
            pred = run(layers)
            ch = (pred != base) & m
            out[f"{name}|{base_name}"] = {"changed": int(ch.sum()), "fixed": int((ch & (pred == y)).sum()),
                                          "broke": int((ch & (base == y)).sum()),
                                          "macro_f1": evaluate(y[m], pred[m])["macro_f1"],
                                          "macro_f1_base": evaluate(y[m], base[m])["macro_f1"]}
    full = run(ALL_LAYERS)
    ch = (full != raw) & m
    out["all|alone"] = {"changed": int(ch.sum()), "fixed": int((ch & (full == y)).sum()), "broke": int((ch & (raw == y)).sum()),
                        "macro_f1": evaluate(y[m], full[m])["macro_f1"], "macro_f1_base": evaluate(y[m], raw[m])["macro_f1"]}
    return out


def single_device(df, pr, bl):
    m = ~df["native_script"].values
    own = (df["source"].values == "export2026") & (df["final_label"].values == "caution")
    keep = m & ~own
    y = df["final_label"].values
    preds = {"new_int8": pr[f"{NEW}|raw"].values, "tfidf_lr": bl["lr"].values, "v73_int8": pr[f"{V73}|raw"].values,
             "new_int8+rules": pr[f"{NEW}|pipeline"].values}
    out = {"n_main": int(m.sum()), "n_removed_author_caution": int((own & m).sum()), "n_kept": int(keep.sum()),
           "caution_kept": int((keep & (y == "caution")).sum()), "models": {}}
    for k, p in preds.items():
        full, red = evaluate(y[m], p[m]), evaluate(y[keep], p[keep])
        three = f1_score(y[keep], p[keep], labels=["ham", "spam", "promo"], average="macro", zero_division=0)
        ck = keep & (y == "caution")
        out["models"][k] = {"macro_f1_full": full["macro_f1"], "caution_f1_full": full["per_class"]["caution"]["f1"],
                            "macro_f1_without_author_caution": red["macro_f1"], "macro_f1_3class_ham_scam_promo": float(three),
                            "caution_on_kept": red["per_class"]["caution"],
                            "caution_recall_k": int((p[ck] == "caution").sum()),
                            "author_caution_recall_k": int((p[own & m] == "caution").sum())}
    return out


def ppv_range():
    st = json.loads((ROOT / "results" / "stats.json").read_text(encoding="utf-8"))["result"]["point"]
    out = {}
    for k in ("new_int8_quint8", "new_int8_quint8+rules", "tfidf_lr", "v73_int8", "v73_int8+rules"):
        R, F = st[k]["scam_recall"], st[k]["legit_as_scam"]
        out[k] = {"recall": R, "legit_as_scam": F,
                  **{f"ppv@{pi}": R * pi / (R * pi + F * (1 - pi)) for pi in (0.01, 0.062, 0.20)}}
    va = load_split(ROOT / "data" / "val.csv", drop_native=True)
    out["val_scam_prevalence"] = float((va["final_label"] == "spam").mean())
    return out


def camouflage_ci(app):
    sp = pd.read_csv(ROOT / "data" / "stress" / "stress_spoof.csv", keep_default_na=False)
    scam = sp[sp["camouflage"] == 1][["base_id", "sender", "text"]].assign(group="scam")
    ben = pd.read_csv(ROOT / "data" / "stress" / "benign_camouflage.csv", keep_default_na=False)[["base_id", "sender", "text"]].assign(group="legit")
    allm = pd.concat([scam, ben], ignore_index=True)
    texts = allm["text"].tolist()
    probs = {"new_int8": softmax(onnx_logits(ROOT / ONNX["int8_quint8"], texts, app), 1),
             "v73_int8": softmax(onnx_logits(APP_ASSETS / "sms_spam_tiny.onnx", texts, app, batch=1), 1)}
    probs["tfidf_lr"] = joblib.load(ROOT / "models" / "tfidf" / "tfidf_lr.joblib").predict(texts)
    rng = np.random.default_rng(0)
    out = {"n_scam_base_texts": int(scam["base_id"].nunique()), "n_legit_base_texts": int(ben["base_id"].nunique()), "models": {}}
    for name, P in probs.items():
        if isinstance(P, np.ndarray) and P.ndim == 2:
            raw = np.array([LABELS[i] for i in P.argmax(1)]); conf = P.max(1)
        else:
            raw = np.asarray(P); conf = np.full(len(raw), 0.99)
        views = {"model_only": raw}
        for v, layers in (("with_rules", ALL_LAYERS), ("rules_without_R1", ALL_LAYERS - {"safety"})):
            views[v] = np.array([pipeline(s, t, lambda _t, p=p, c=c: (p, float(c)), layers).label
                                 for s, t, p, c in zip(allm["sender"], texts, raw, conf)])
        r = {}
        for v, pred in views.items():
            for grp in ("scam", "legit"):
                g = (allm["group"] == grp).values
                hit = pred[g] == "spam"
                bases = allm["base_id"].values[g]
                ub = np.unique(bases)
                per = {b: (hit[bases == b].sum(), (bases == b).sum()) for b in ub}
                boots = []
                for _ in range(2000):
                    s = rng.choice(ub, len(ub))
                    k, n = sum(per[b][0] for b in s), sum(per[b][1] for b in s)
                    boots.append(150 * k / n)
                lo, hi = np.quantile(boots, [0.025, 0.975])
                r[f"{v}|{grp}_flagged_scam"] = {"k": int(hit.sum()), "n": int(g.sum()), "ci95_of150": [float(lo), float(hi)]}
        out["models"][name] = r
    return out


def ece_check(df, pr):
    m = ~df["native_script"].values
    P = pr[[f"{NEW}|p_{l}" for l in LABELS]].values[m].clip(1e-12, 1)
    y = np.array([LABELS.index(t) for t in df["final_label"].values[m]])
    T = json.loads((ROOT / "models" / SEEDS[3] / "temperature.json").read_text())["temperature"]
    Pc = softmax(np.log(P) / T, 1)
    return {"T": T, **{f"bins{b}": {"before": ece(P, y, b), "after": ece(Pc, y, b)} for b in (10, 15)}}


def main():
    run = new_run("review2_extras", {"post_hoc": True})
    df, pr, bl = load()
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    res = {"post_hoc": True, "second_annotator_models": second_annotator(df, pr, bl), "per_rule": per_rule(df, pr),
           "single_device_caution": single_device(df, pr, bl), "ppv_range": ppv_range(),
           "camouflage_ci": camouflage_ci(app), "ece_check": ece_check(df, pr)}
    save_result(run, res, "review2_extras.json")
    print(json.dumps(res, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
