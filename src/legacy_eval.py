"""POST-HOC same-recipe data ablation, evaluation: each legacy-trained model is scored ONCE on the frozen test set.

Models: train runs on data/legacy_L0.csv (legacy file as-is) and data/legacy_L1.csv (legacy messages after cleaning),
seeds 1-3, trained with the final recipe (see results/<run>.json configs). Evaluation protocol identical to the final
5 seeds: PyTorch FP32, HF tokenizer, "filled" view (placeholders filled with random.Random(1234)), main subset.
Also reports the val macro-F1 logged at training time. Output: results/legacy_eval.json (post-hoc).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result, sha256_file  # noqa: E402
from final_test import FROZEN_SHA, torch_logits  # noqa: E402
from metrics import evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from subsets import load_split  # noqa: E402

RUNS = {"L0": ["train-20261019-191916", "train-20261019-194218", "train-20261019-200340"],
        "L1": ["train-20261019-202419", "train-20261019-203249", "train-20261019-204113"]}


def main():
    path = ROOT / "data" / "test.csv"
    assert sha256_file(path) == FROZEN_SHA
    run = new_run("legacy_eval", {"post_hoc": True, "runs": RUNS})
    df = load_split(path, drop_native=False).reset_index(drop=True)
    frng = random.Random(1234)
    texts = [fill(t, frng) for t in df["text"]]
    m = ~df["native_script"].values
    y = df["final_label"].values
    res = {"post_hoc": True, "n_main": int(m.sum()), "conditions": {}}
    for cond, rids in RUNS.items():
        rows = []
        for rid in rids:
            tr = json.loads((ROOT / "results" / f"{rid}.json").read_text(encoding="utf-8"))
            pred = np.array([LABELS[i] for i in torch_logits(ROOT / "models" / rid, texts).argmax(1)])
            e = evaluate(y[m], pred[m])
            legit = np.isin(y[m], ["ham", "promo", "caution"])
            rows.append({"run_id": rid, "seed": tr["config"]["seed"], "train_rows": tr["result"]["train_rows"],
                         "val_macro_f1": tr["result"]["best_val"]["macro_f1"], "test_macro_f1": e["macro_f1"], "test_mcc": e["mcc"],
                         **{f"test_f1_{l}": e["per_class"][l]["f1"] for l in LABELS},
                         "legit_as_scam": f"{int((legit & (pred[m] == 'spam')).sum())}/{int(legit.sum())}"})
            print(cond, rows[-1], flush=True)
        f1 = [r["test_macro_f1"] for r in rows]
        res["conditions"][cond] = {"runs": rows, "test_macro_f1_mean": float(np.mean(f1)), "test_macro_f1_std": float(np.std(f1, ddof=1)),
                                   "val_macro_f1_mean": float(np.mean([r["val_macro_f1"] for r in rows]))}
    save_result(run, res, "legacy_eval.json")
    print({c: (round(v["test_macro_f1_mean"], 3), round(v["test_macro_f1_std"], 3)) for c, v in res["conditions"].items()}); print(run["run_id"])


if __name__ == "__main__":
    main()
