"""THE single final run on the frozen test set (Gate 3 approved by the owner, 2026-10-18).

Pre-registered before running (nothing here is tuned on test):
- Asserts data/test.csv SHA-256 == the frozen hash; aborts otherwise.
- Main metric subset: test minus native_script rows (src/subsets.py). Native-script rows reported separately.
- Text views: "filled" (MAIN) = placeholders filled by src/placeholders.fill with random.Random(1234) in file order,
  exactly the protocol used for val during selection; "raw" (SECONDARY) = scrubbed text with placeholders, which is
  what v7.3's earlier single test run saw (eval_v73-20261018-023957). v7.3 is additionally scored on "filled" so
  both models are compared on identical input; this is an extra measurement condition, not a selection.
- Models: 5 final seeds (PyTorch FP32, HF tokenizer); deploy candidate seed 3 as ONNX INT8 QUInt8 and QInt8 (app
  tokenizer, padded to 128); v7.3 FP32 and deployed INT8 ONNX (batch 1) on "filled".
- Rules (src/rules.py, full pipeline) use the UNCALIBRATED softmax confidence, as the app does. Mobile-number senders
  (scrubbed) get the stand-in address 9876543210, as in eval_v73.
Outputs: results/preds_final_test.csv (ids, labels, probabilities; no text), results/final_test.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS  # noqa: E402
from common import LABELS, ROOT, new_run, save_result, sha256_file  # noqa: E402
from eval_v73 import address_for_rules  # noqa: E402
from metrics import evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402
from subsets import load_split  # noqa: E402

FROZEN_SHA = "7fa0c6c4bdaf604a22c7185e39d3c605a266fa7ef0b5072a5c85527ce04e201b"
SEEDS = {1: "train-20261018-132741", 2: "train-20261018-132939", 3: "train-20261018-133131",
         4: "train-20261018-133323", 5: "train-20261018-133515"}
ONNX = {"int8_quint8": "models/onnx/cyvach_int8_quint8.onnx", "int8_qint8": "models/onnx/cyvach_int8_qint8.onnx"}


def torch_logits(path, texts):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, max_length=128, padding=True, return_tensors="pt")
            out.append(m(**enc).logits.numpy())
    return np.concatenate(out)


def onnx_logits(path, texts, app, batch=32):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    enc = [app.encode(t) for t in texts]
    out = []
    for i in range(0, len(enc), batch):
        out.append(s.run(["logits"], {"input_ids": np.array([e[0] for e in enc[i:i + batch]], dtype=np.int64),
                                      "attention_mask": np.array([e[1] for e in enc[i:i + batch]], dtype=np.int64)})[0])
    return np.concatenate(out)


def main():
    dry = "--dry-run-on-val" in sys.argv  # identical code path on VAL, to catch crashes before the single test run
    path = ROOT / "data" / ("val.csv" if dry else "test.csv")
    sha = sha256_file(path)
    if not dry:
        assert sha == FROZEN_SHA, f"test.csv hash mismatch: {sha}"
    run = new_run("final_test_DRYRUN_val" if dry else "final_test", {"dry_run_on_val": dry, "frozen_sha256": sha, "seeds": SEEDS, "onnx": ONNX})
    torch.set_num_threads(4)
    df = load_split(path, drop_native=False).reset_index(drop=True)
    if dry:
        df = df.head(60)  # crash check only
    frng = random.Random(1234)
    views = {"filled": [fill(t, frng) for t in df["text"]], "raw": df["text"].tolist()}
    app = AppTokenizer(APP_ASSETS / "vocab.txt")

    logits = {}
    for v, texts in views.items():
        for s, rid in SEEDS.items():
            logits[f"seed{s}_fp32|{v}"] = torch_logits(ROOT / "models" / rid, texts)
            print("done", s, v, flush=True)
        for k, p in ONNX.items():
            logits[f"seed3_{k}|{v}"] = onnx_logits(ROOT / p, texts, app)
    logits["v73_fp32|filled"] = torch_logits(ROOT / "model_v7_3", views["filled"])
    logits["v73_int8|filled"] = onnx_logits(APP_ASSETS / "sms_spam_tiny.onnx", views["filled"], app, batch=1)

    main_m = ~df["native_script"].values
    y = df["final_label"].values
    preds = pd.DataFrame({"item_id": df["item_id"], "cluster_id": df["cluster_id"], "native_script": df["native_script"],
                          "true": y, "label_source": df["label_source"], "sender_kind": df["sender_kind"]})
    res = {"test_sha256": sha, "n": int(len(df)), "n_main": int(main_m.sum()), "n_native": int((~main_m).sum()),
           "main_label_counts": pd.Series(y[main_m]).value_counts().to_dict(), "models": {}}
    for name, lg in logits.items():
        pr = softmax(lg, 1)
        raw = np.array([LABELS[i] for i in pr.argmax(1)])
        v = name.split("|")[1]
        full = [pipeline(address_for_rules(r), t, lambda _t, p=p, c=c: (p, float(c)), ALL_LAYERS)
                for (_, r), t, p, c in zip(df.iterrows(), views[v], raw, pr.max(1))]
        pipe = np.array([f.label for f in full])
        for k, l in enumerate(LABELS):
            preds[f"{name}|p_{l}"] = pr[:, k].round(6)
        preds[f"{name}|raw"], preds[f"{name}|pipeline"] = raw, pipe
        res["models"][name] = {"main_raw": evaluate(y[main_m], raw[main_m]), "main_pipeline": evaluate(y[main_m], pipe[main_m]),
                               "native_raw": evaluate(y[~main_m], raw[~main_m]) if (~main_m).any() else None,
                               "rules_fired_main": pd.Series([f.rule for f in full])[main_m].value_counts().to_dict()}
    # seed summary (main, filled + raw)
    for v in views:
        for kind in ("main_raw", "main_pipeline"):
            f1 = [res["models"][f"seed{s}_fp32|{v}"][kind]["macro_f1"] for s in SEEDS]
            res[f"seeds_{v}_{kind}_macro_f1"] = {"mean": float(np.mean(f1)), "std": float(np.std(f1, ddof=1)), "values": f1}
    # FP32 vs INT8 parity on test (seed 3, filled, all rows)
    ref = logits["seed3_fp32|filled"]
    res["parity_seed3_filled"] = {}
    for k in ONNX:
        q = logits[f"seed3_{k}|filled"]
        lp, lq = np.log(softmax(ref, 1)), np.log(softmax(q, 1))
        res["parity_seed3_filled"][k] = {"argmax_agree": float((ref.argmax(1) == q.argmax(1)).mean()),
                                         "mean_kl_fp32_to_int8": float((softmax(ref, 1) * (lp - lq)).sum(1).mean()),
                                         "macro_f1_diff_main": res["models"][f"seed3_{k}|filled"]["main_raw"]["macro_f1"]
                                         - res["models"]["seed3_fp32|filled"]["main_raw"]["macro_f1"]}
    tag = "_DRYRUN_val" if dry else ""
    preds.to_csv(ROOT / "results" / f"preds_final_test{tag}.csv", index=False)
    save_result(run, res, f"final_test{tag}.json")
    for name, d in res["models"].items():
        a, b = d["main_raw"], d["main_pipeline"]
        print(f"{name:28s} raw F1 {a['macro_f1']:.4f} MCC {a['mcc']:.3f} caution {a['per_class']['caution']['f1']:.3f} "
              f"legit->scam {a['fpr_legit_as_scam']:.3f} | pipeline F1 {b['macro_f1']:.4f} legit->scam {b['fpr_legit_as_scam']:.3f}")
    print(json.dumps({k: v for k, v in res.items() if k.startswith("seeds_") or k == "parity_seed3_filled"}, indent=1))
    print(run["run_id"])


if __name__ == "__main__":
    main()
