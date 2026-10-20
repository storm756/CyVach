"""Run the v7.3 FP32 checkpoint on data/pool.csv (proxy labels for sampling and silver selection).

Output goes to data/v73_preds.csv (kept separate from the pool so that the
test-set labelling tool never reads it). Prints counts only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    run = new_run("predict_v73", vars(args))
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)
    ck = ROOT / "model_v7_3"
    tok = AutoTokenizer.from_pretrained(str(ck))
    model = AutoModelForSequenceClassification.from_pretrained(str(ck)).eval()
    texts = pool["text"].tolist()
    order = np.argsort([len(t) for t in texts])  # length-sorted for dynamic padding
    logits = np.zeros((len(texts), 4), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            idx = order[i:i + args.batch]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            logits[idx] = model(**enc).logits.numpy()
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    out = pd.DataFrame({"item_id": pool["item_id"], "v73_pred": [LABELS[k] for k in p.argmax(1)],
                        "v73_conf": p.max(1).round(4)})
    for k, l in enumerate(LABELS):
        out[f"v73_logit_{l}"] = logits[:, k].round(4)
    out.to_csv(ROOT / "data" / "v73_preds.csv", index=False)
    pool = pool.merge(out[["item_id", "v73_pred"]], on="item_id")
    summary = {
        "n": len(out),
        "pred_counts": out["v73_pred"].value_counts().to_dict(),
        "crosstab_label_vs_pred": {f"{a}->{b}": int(n) for (a, b), n in pool.groupby(["label", "v73_pred"]).size().items()},
        "agree_with_original_label": int((pool["label"] == pool["v73_pred"]).sum()),
        "agree_by_source": {s: f"{int((g.label == g.v73_pred).sum())}/{len(g)}" for s, g in pool.groupby("source")},
    }
    save_result(run, summary, "predict_v73.json")
    import json
    print(json.dumps(summary, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
