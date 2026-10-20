"""Phase 10 (POST-HOC): TF-IDF baselines on the stress sets, serialized size and desktop-CPU latency.

The LR (C=8) and LinearSVC (C=0.5) baselines are rebuilt exactly as in src/baselines.py (same features, train rows,
seed and C chosen on val) as sklearn Pipelines (FeatureUnion of word 1-2 and char_wb 2-5 TF-IDF), and must reproduce
the saved test predictions (results/preds_baselines_test.csv) item for item; the script aborts otherwise.
Stress sets: evaluated on raw text (TF-IDF sees no sender, so no rules).
Size: joblib file of the full pipeline (vectorizers + model), bytes/MB/MiB.
Latency: bench_onnx protocol - one message (the same MSG), 1 thread (threadpoolctl), 50 warm-ups + 500 timed
pipeline.predict calls incl. featurization; median/p95 ms. Desktop CPU, not phone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_onnx import MSG  # noqa: E402
from check_onnx import size_units  # noqa: E402
from common import ROOT, new_run, save_result  # noqa: E402
from metrics import cap_synthetic, evaluate  # noqa: E402
from subsets import load_split  # noqa: E402

SETS = {"stress_spoof": "camouflage", "stress_obfuscation": "transform", "stress_hinglish": None, "digit_probe": "digit_style"}


def pipe(kind, seed=13):
    feats = FeatureUnion([("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                          ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True))])
    clf = (LogisticRegression(C=8, max_iter=3000, class_weight="balanced", random_state=seed) if kind == "lr"
           else LinearSVC(C=0.5, class_weight="balanced", random_state=seed))
    return Pipeline([("f", feats), ("m", clf)])


def main():
    run = new_run("tfidf_extra", {"post_hoc": True, "C": {"lr": 8, "svm": 0.5}})
    tr = cap_synthetic(load_split(ROOT / "data/train.csv", drop_native=True), seed=13)
    te = load_split(ROOT / "data/test.csv", drop_native=False)
    saved = pd.read_csv(ROOT / "results" / "preds_baselines_test.csv", keep_default_na=False)
    out = ROOT / "models" / "tfidf"; out.mkdir(parents=True, exist_ok=True)
    res = {"post_hoc": True, "models": {}}
    for kind in ("lr", "svm"):
        pl = pipe(kind).fit(tr["text"], tr["final_label"])
        pred = pl.predict(te["text"])
        assert (pred == saved[kind].values).all(), f"{kind}: rebuilt pipeline does not reproduce saved test predictions"
        path = out / f"tfidf_{kind}.joblib"
        joblib.dump(pl, path, compress=3)
        r = {"reproduces_saved_test_predictions": True, "size": size_units(path.stat().st_size),
             "n_features": int(sum(len(v.vocabulary_) for _, v in pl.named_steps["f"].transformer_list))}
        with threadpool_limits(1):
            for _ in range(50):
                pl.predict([MSG])
            t = []
            for _ in range(500):
                t0 = time.perf_counter_ns(); pl.predict([MSG]); t.append((time.perf_counter_ns() - t0) / 1e6)
        r["latency_1thread"] = {"median_ms": round(float(np.median(t)), 3), "p95_ms": round(float(np.percentile(t, 95)), 3), "runs": 500, "warmup": 50}
        r["stress"] = {}
        for s, by in SETS.items():
            d = pd.read_csv(ROOT / "data" / "stress" / f"{s}.csv", keep_default_na=False)
            pr = pl.predict(d["text"])
            ok = pr == d["final_label"].values
            e = evaluate(d["final_label"], pr)
            r["stress"][s] = {"n": int(len(d)), "accuracy": float(ok.mean()), "k": int(ok.sum()), "macro_f1": e["macro_f1"]}
            if by:
                r["stress"][s]["accuracy_by_" + by] = {str(k): {"acc": float(v.mean()), "k": int(v.sum()), "n": int(len(v))}
                                                       for k, v in pd.Series(ok).groupby(d[by].astype(str).values)}
            if s == "stress_spoof":
                sc = (d["final_label"] == "spam").values
                for cam in ("0", "1"):
                    mm = sc & (d["camouflage"].astype(str) == cam).values
                    r["stress"][s][f"scam_recall_camouflage_{cam}"] = {"acc": float(ok[mm].mean()), "k": int(ok[mm].sum()), "n": int(mm.sum())}
        res["models"][kind] = r
        print(kind, r["size"], r["latency_1thread"], {s: round(v["accuracy"], 3) for s, v in r["stress"].items()})
        print("  spoof by camouflage", r["stress"]["stress_spoof"]["accuracy_by_camouflage"], "scam cam1", r["stress"]["stress_spoof"]["scam_recall_camouflage_1"])
        print("  obfuscation", {k: round(v["acc"], 3) for k, v in r["stress"]["stress_obfuscation"]["accuracy_by_transform"].items()})
    save_result(run, res, "tfidf_extra.json"); print(run["run_id"])


if __name__ == "__main__":
    main()
