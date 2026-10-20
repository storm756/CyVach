"""Phase 10 (POST-HOC): rule-layer ablation on VAL, plus the "R0 never overrides caution" variant.

VAL: the deploy model (seed-3 INT8 QUInt8, app tokenizer) is run on val with the same "filled" view as selection
(placeholders filled, random.Random(1234)); v7.3 INT8 val probabilities come from results/preds_v73_val.csv (raw view,
eval_v73 run). Configs: every config of src/ablation.py plus "full_R0_keeps_caution" (R0 replaced by trusted_nc).
TEST: only the new variant is applied ONCE to the SAVED test probabilities (no model run), labelled
"post-hoc, motivated by the test ablation".
Outputs: results/ablation_val.json
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import configs  # noqa: E402
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS  # noqa: E402
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from eval_v73 import address_for_rules  # noqa: E402
from final_test import ONNX, onnx_logits  # noqa: E402
from metrics import evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402
from subsets import load_split  # noqa: E402

VARIANT = (ALL_LAYERS - {"trusted"}) | {"trusted_nc"}


def run_configs(df, texts, P, cfgs, mask):
    y = df["final_label"].values
    raw = np.array([LABELS[i] for i in P.argmax(1)]); conf = P.max(1)
    out = {}
    for cname, layers in cfgs.items():
        pred = np.array([pipeline(address_for_rules(r), t, lambda _t, p=p, c=c: (p, float(c)), layers).label
                         for (_, r), t, p, c in zip(df.iterrows(), texts, raw, conf)])
        e = evaluate(y[mask], pred[mask]); ch = (pred != raw) & mask
        legit = np.isin(y, ["ham", "promo", "caution"]) & mask
        out[cname] = {"macro_f1": e["macro_f1"], "f1_caution": e["per_class"]["caution"]["f1"], "scam_recall": e["per_class"]["spam"]["recall"],
                      "legit_as_scam": e["fpr_legit_as_scam"], "legit_as_scam_k_n": f"{int((legit & (pred == 'spam')).sum())}/{int(legit.sum())}",
                      "changed": int(ch.sum()), "fixed": int((ch & (pred == y)).sum()), "broke": int((ch & (raw == y)).sum())}
    return out


def main():
    run = new_run("ablation_val", {"post_hoc": True, "variant": sorted(VARIANT)})
    cfgs = {**configs(), "full_R0_keeps_caution": VARIANT}
    va = load_split(ROOT / "data/val.csv", drop_native=False).reset_index(drop=True)
    vm = ~va["native_script"].values
    frng = random.Random(1234)
    vt = [fill(t, frng) for t in va["text"]]
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    Pn = softmax(onnx_logits(ROOT / ONNX["int8_quint8"], vt, app), 1)
    v7 = pd.read_csv(ROOT / "results" / "preds_v73_val.csv", keep_default_na=False)
    assert (v7["item_id"].values == va["item_id"].values).all()
    P7 = v7[[f"onnx_int8_app_tokenizer_p_{l}" for l in LABELS]].values
    res = {"post_hoc": True, "val": {"n_main": int(vm.sum()),
                                     "new_int8_deploy_filled": run_configs(va, vt, Pn, cfgs, vm),
                                     "v73_int8_raw_view": run_configs(va, va["text"].tolist(), P7, cfgs, vm)}}
    # TEST: the variant only, once, from saved probabilities
    te = load_split(ROOT / "data/test.csv", drop_native=False).reset_index(drop=True)
    tm = ~te["native_script"].values
    frng = random.Random(1234)
    tt = [fill(t, frng) for t in te["text"]]
    pf = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    assert (pf["item_id"].values == te["item_id"].values).all()
    sel = {"raw_model": frozenset(), "full": ALL_LAYERS, "full_R0_keeps_caution": VARIANT}
    res["test_variant_once"] = {"label": "post-hoc, motivated by the test ablation"}
    for name in ("seed3_int8_quint8|filled", "v73_int8|filled"):
        P = pf[[f"{name}|p_{l}" for l in LABELS]].values
        res["test_variant_once"][name] = run_configs(te, tt, P, sel, tm)
    save_result(run, res, "ablation_val.json")
    for k in ("new_int8_deploy_filled", "v73_int8_raw_view"):
        print("\nVAL", k); print(pd.DataFrame(res["val"][k]).T.drop(columns=["legit_as_scam_k_n"]).round(4).to_string())
    for k in ("seed3_int8_quint8|filled", "v73_int8|filled"):
        print("\nTEST (post-hoc variant)", k); print(pd.DataFrame(res["test_variant_once"][k]).T.round(4).to_string())
    print(run["run_id"])


if __name__ == "__main__":
    main()
