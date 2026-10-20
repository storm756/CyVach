"""Evaluate the legacy v7.3 model (FP32 checkpoint and deployed INT8 ONNX) on val and, ONCE, on the frozen test set.

- FP32: HF tokenizer from model_v7_3/, dynamic padding.
- INT8 ONNX: the app's own tokenizer port (src/app_tokenizer.py), padded to 128, batch 1 (the graph only supports batch 1).
- Each is scored as raw model and as the full shipped pipeline (src/rules.py, all layers). Rules need the sender:
  senders were scrubbed, so a row whose ORIGINAL sender was a mobile number (sender_kind == mobile_number)
  gets the stand-in address "9876543210" (only the mobile-number test in R2/R3 depends on it).
- Main test metric = test minus native_script rows (src/subsets.py); native_script reported separately.
Per-item predictions are saved (results/preds_v73_<split>.csv: ids + labels + probabilities, no text).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS  # noqa: E402
from common import LABELS, ROOT, new_run, save_result, sha256_file  # noqa: E402
from metrics import evaluate  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402
from subsets import load_split  # noqa: E402


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def address_for_rules(row) -> str:
    return "9876543210" if row["sender_kind"] == "mobile_number" else row["sender"]


def score(df, probs):
    pred = [LABELS[i] for i in probs.argmax(1)]
    conf = probs.max(1)
    full = [pipeline(address_for_rules(r), r["text"], lambda t, p=p, c=c: (p, float(c)), ALL_LAYERS)
            for (_, r), p, c in zip(df.iterrows(), pred, conf)]
    return pred, [f.label for f in full], [f.rule for f in full]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    args = ap.parse_args()
    run = new_run("eval_v73", vars(args))
    path = ROOT / "data" / f"{args.split}.csv"
    df = load_split(path, drop_native=False).reset_index(drop=True)
    torch.set_num_threads(4)
    ck = ROOT / "model_v7_3"
    tok = AutoTokenizer.from_pretrained(str(ck))
    model = AutoModelForSequenceClassification.from_pretrained(str(ck)).eval()
    texts = df["text"].tolist()
    lg = np.zeros((len(texts), 4), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, max_length=128, padding=True, return_tensors="pt")
            lg[i:i + 32] = model(**enc).logits.numpy()
    p_fp32 = softmax(lg)
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    sess = ort.InferenceSession(str(APP_ASSETS / "sms_spam_tiny.onnx"), providers=["CPUExecutionProvider"])
    lo = np.zeros_like(lg)
    for i, t in enumerate(texts):
        ids, mask = app.encode(t)
        lo[i] = sess.run(["logits"], {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.array([mask], dtype=np.int64)})[0][0]
    p_int8 = softmax(lo)

    out = pd.DataFrame({"item_id": df["item_id"], "native_script": df["native_script"], "true": df["final_label"]})
    res = {"split": args.split, "split_sha256": sha256_file(path), "n": int(len(df)),
           "n_native_script": int(df["native_script"].sum()), "models": {}}
    for name, probs in (("fp32", p_fp32), ("onnx_int8_app_tokenizer", p_int8)):
        raw, full, rule = score(df, probs)
        out[f"{name}_raw"], out[f"{name}_pipeline"], out[f"{name}_rule"] = raw, full, rule
        for k, l in enumerate(LABELS):
            out[f"{name}_p_{l}"] = probs[:, k].round(5)
        m = ~df["native_script"]
        res["models"][name] = {
            "main_raw": evaluate(df.loc[m, "final_label"], np.array(raw)[m]),
            "main_pipeline": evaluate(df.loc[m, "final_label"], np.array(full)[m]),
            "native_script_raw": evaluate(df.loc[~m, "final_label"], np.array(raw)[~m]) if (~m).any() else None,
            "rule_fired_counts_main": pd.Series(np.array(rule)[m]).value_counts().to_dict(),
        }
    agree = (out["fp32_raw"] == out["onnx_int8_app_tokenizer_raw"]).mean()
    res["fp32_vs_int8_raw_agreement"] = float(agree)
    out.to_csv(ROOT / "results" / f"preds_v73_{args.split}.csv", index=False)
    save_result(run, res, f"eval_v73_{args.split}.json")
    short = {n: {k: (round(v["macro_f1"], 4), round(v["accuracy"], 4)) for k, v in d.items() if isinstance(v, dict) and "macro_f1" in v}
             for n, d in res["models"].items()}
    print(json.dumps({"n": res["n"], "native": res["n_native_script"], "agree_fp32_int8": round(agree, 4),
                      "macroF1_acc": short}, indent=1))
    for n, d in res["models"].items():
        print(n, "main_raw per-class F1", {c: round(v["f1"], 3) for c, v in d["main_raw"]["per_class"].items()},
              "| pipeline", {c: round(v["f1"], 3) for c, v in d["main_pipeline"]["per_class"].items()})
    print(run["run_id"])


if __name__ == "__main__":
    main()
