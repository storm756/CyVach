"""Phase 4 baselines: TF-IDF (word 1-2 + char_wb 2-5) with Logistic Regression and LinearSVC.

Selection on VAL only. `--final-test` runs the selected configs ONCE on the frozen test set
(do not use until Gate 3 / final evaluation). Train: native-script rows dropped, synthetic capped per class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, save_result  # noqa: E402
from metrics import cap_synthetic, evaluate  # noqa: E402
from subsets import load_split  # noqa: E402


def featurize(train_text, *others):
    w = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr = hstack([w.fit_transform(train_text), c.fit_transform(train_text)]).tocsr()
    return Xtr, [hstack([w.transform(t), c.transform(t)]).tocsr() for t in others]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    run = new_run("baselines", vars(args))
    tr = cap_synthetic(load_split(ROOT / "data/train.csv", drop_native=True), seed=args.seed)
    va = load_split(ROOT / "data/val.csv", drop_native=True)
    Xtr, (Xva,) = featurize(tr["text"], va["text"])
    grid = {"lr": [0.5, 2, 8], "svm": [0.1, 0.5, 2]}
    res = {"train_rows": int(len(tr)), "train_labels": tr["final_label"].value_counts().to_dict(), "val_rows": int(len(va)), "models": {}}
    best = {}
    for kind, Cs in grid.items():
        res["models"][kind] = {}
        for C in Cs:
            m = (LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=args.seed) if kind == "lr"
                 else LinearSVC(C=C, class_weight="balanced", random_state=args.seed))
            m.fit(Xtr, tr["final_label"])
            ev = evaluate(va["final_label"], m.predict(Xva))
            res["models"][kind][str(C)] = ev
            if kind not in best or ev["macro_f1"] > best[kind][1]:
                best[kind] = (C, ev["macro_f1"])
    res["selected_on_val"] = {k: {"C": v[0], "val_macro_f1": v[1]} for k, v in best.items()}
    if args.final_test:
        te_all = load_split(ROOT / "data/test.csv", drop_native=False)
        te = te_all[~te_all["native_script"]]
        Xtr, (Xte, Xnat) = featurize(tr["text"], te["text"], te_all[te_all["native_script"]]["text"])
        res["test"] = {}
        import pandas as pd
        out = pd.DataFrame({"item_id": te_all["item_id"].values, "native_script": te_all["native_script"].values, "true": te_all["final_label"].values})
        for kind, (C, _) in best.items():
            m = (LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=args.seed) if kind == "lr"
                 else LinearSVC(C=C, class_weight="balanced", random_state=args.seed))
            m.fit(Xtr, tr["final_label"])
            out[kind] = m.predict(featurize(tr["text"], te_all["text"])[1][0])
            res["test"][kind] = {"main": evaluate(te["final_label"], m.predict(Xte)),
                                 "native_script": evaluate(te_all[te_all["native_script"]]["final_label"], m.predict(Xnat))}
    if args.final_test:
        out.to_csv(ROOT / "results" / "preds_baselines_test.csv", index=False)
    p = save_result(run, res, "baselines_final.json" if args.final_test else "baselines_val.json")
    print(json.dumps({"selected": res["selected_on_val"], "train_rows": res["train_rows"], "train_labels": res["train_labels"],
                      "val_per_class_f1": {k: {c: round(v["per_class"][c]["f1"], 3) for c in v["per_class"]}
                                           for k, v in ((k, res["models"][k][str(best[k][0])]) for k in best)}}, indent=1))
    print(run["run_id"])


if __name__ == "__main__":
    main()
