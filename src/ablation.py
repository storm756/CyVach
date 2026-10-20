"""Phase 7: rule-layer ablation, computed from the SAVED final-test probabilities (no new model run on test).

Layers (src/rules.py): pre (H1/H2), trusted (R0 whitelist), safety (R1), mobile (R2/R3), suffix (R4 TRAI suffix).
"Header rules" = trusted + suffix. The shipped app has no URL/link rule, so "+URL rules" is reported as not applicable.
Configs: raw model; raw + each layer alone; header rules (trusted+suffix); full; full minus each layer.
For each: main-subset macro-F1, scam recall, caution F1, legit-flagged-as-scam rate, and how many items the rules
changed versus the raw model (fixed = wrong->right, broke = right->wrong).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from eval_v73 import address_for_rules  # noqa: E402
from metrics import evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402
from subsets import load_split  # noqa: E402

LAYERS = ["pre", "trusted", "safety", "mobile", "suffix"]


def configs():
    c = {"raw_model": frozenset()}
    c.update({f"+{l}": frozenset({l}) for l in LAYERS})
    c["+header(trusted+suffix)"] = frozenset({"trusted", "suffix"})
    c["full"] = ALL_LAYERS
    c.update({f"full-{l}": ALL_LAYERS - {l} for l in LAYERS})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-on-val", action="store_true")
    ap.add_argument("--models", nargs="+", default=["seed3_int8_quint8|filled", "v73_int8|filled", "seed3_fp32|filled"])
    args = ap.parse_args()
    tag = "_DRYRUN_val" if args.dry_run_on_val else ""
    run = new_run("ablation" + tag, vars(args))
    df = load_split(ROOT / "data" / ("val.csv" if args.dry_run_on_val else "test.csv"), drop_native=False).reset_index(drop=True)
    if args.dry_run_on_val:
        df = df.head(60)
    pr = pd.read_csv(ROOT / "results" / f"preds_final_test{tag}.csv", keep_default_na=False)
    assert (pr["item_id"].values == df["item_id"].values).all()
    frng = random.Random(1234)
    texts = {"filled": [fill(t, frng) for t in df["text"]], "raw": df["text"].tolist()}
    m = ~df["native_script"].values
    y = df["final_label"].values
    res = {"n_main": int(m.sum()), "url_rule": "not applicable: the shipped app has no URL/link rule", "models": {}}
    for name in args.models:
        view = name.split("|")[1]
        P = pr[[f"{name}|p_{l}" for l in LABELS]].values
        raw = np.array([LABELS[i] for i in P.argmax(1)]); conf = P.max(1)
        out = {}
        for cname, layers in configs().items():
            pred = np.array([pipeline(address_for_rules(r), t, lambda _t, p=p, c=c: (p, float(c)), layers).label
                             for (_, r), t, p, c in zip(df.iterrows(), texts[view], raw, conf)])
            ev = evaluate(y[m], pred[m])
            ch = (pred != raw) & m
            out[cname] = {"macro_f1": ev["macro_f1"], "scam_recall": ev["per_class"]["spam"]["recall"],
                          "caution_f1": ev["per_class"]["caution"]["f1"], "legit_as_scam": ev["fpr_legit_as_scam"],
                          "changed": int(ch.sum()), "fixed": int((ch & (pred == y)).sum()), "broke": int((ch & (raw == y)).sum())}
        res["models"][name] = out
        print("\n" + name)
        print(pd.DataFrame(out).T.round(4).to_string())
    save_result(run, res, f"ablation{tag}.json"); print(run["run_id"])


if __name__ == "__main__":
    main()
