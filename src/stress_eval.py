"""Phase 3/7: evaluate deployed-form models on the synthetic stress sets (evaluation only, never used for selection).

Models: new seed-3 INT8 QUInt8 / QInt8 (app tokenizer), new seed-3 FP32 (PyTorch), v7.3 deployed INT8 (batch 1).
Stress texts contain realistic URLs/numbers (no placeholders), so no filling is applied.
- stress_spoof: raw model and full pipeline (senders are part of the design), broken down by sender_kind and camouflage.
- stress_obfuscation: by transform. stress_hinglish: overall. digit_probe: by digit_style.
Outputs: results/stress_eval.json, results/preds_stress.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS  # noqa: E402
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from final_test import ONNX, SEEDS, onnx_logits, torch_logits  # noqa: E402
from metrics import evaluate  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402

SETS = {"stress_spoof": "sender_kind", "stress_obfuscation": "transform", "stress_hinglish": None, "digit_probe": "digit_style"}


def main():
    run = new_run("stress_eval", {})
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    res, allp = {}, []
    for sname, by in SETS.items():
        d = pd.read_csv(ROOT / "data" / "stress" / f"{sname}.csv", keep_default_na=False)
        texts = d["text"].tolist()
        lg = {"new_int8_quint8": onnx_logits(ROOT / ONNX["int8_quint8"], texts, app),
              "new_int8_qint8": onnx_logits(ROOT / ONNX["int8_qint8"], texts, app),
              "new_fp32": torch_logits(ROOT / "models" / SEEDS[3], texts),
              "v73_int8": onnx_logits(APP_ASSETS / "sms_spam_tiny.onnx", texts, app, batch=1)}
        res[sname] = {"n": int(len(d)), "n_unique_text": int(d["text"].nunique()), "label_counts": d["final_label"].value_counts().to_dict(), "models": {}}
        for mname, L in lg.items():
            P = softmax(L, 1); raw = np.array([LABELS[i] for i in P.argmax(1)])
            r = {"raw": evaluate(d["final_label"], raw)}
            r["raw"] = {k: r["raw"][k] for k in ("accuracy", "macro_f1", "per_class", "fpr_legit_as_scam")}
            if by:
                r["raw_accuracy_by_" + by] = pd.Series(raw == d["final_label"].values).groupby(d[by].values).mean().round(4).to_dict()
            if sname == "stress_spoof":
                pipe = np.array([pipeline(s, t, lambda _t, p=p, c=c: (p, float(c)), ALL_LAYERS).label
                                 for s, t, p, c in zip(d["sender"], texts, raw, P.max(1))])
                e = evaluate(d["final_label"], pipe)
                r["pipeline"] = {k: e[k] for k in ("accuracy", "macro_f1", "per_class", "fpr_legit_as_scam")}
                r["pipeline_accuracy_by_sender_kind"] = pd.Series(pipe == d["final_label"].values).groupby(d["sender_kind"].values).mean().round(4).to_dict()
                r["pipeline_accuracy_by_camouflage"] = pd.Series(pipe == d["final_label"].values).groupby(d["camouflage"].values).mean().round(4).to_dict()
                r["raw_accuracy_by_camouflage"] = pd.Series(raw == d["final_label"].values).groupby(d["camouflage"].values).mean().round(4).to_dict()
                allp.append(pd.DataFrame({"set": sname, "item_id": d["item_id"], "model": mname, "true": d["final_label"], "raw": raw, "pipeline": pipe}))
            else:
                allp.append(pd.DataFrame({"set": sname, "item_id": d["item_id"], "model": mname, "true": d["final_label"], "raw": raw}))
            res[sname]["models"][mname] = r
            line = f"{sname:20s} {mname:16s} raw acc {r['raw']['accuracy']:.3f} F1 {r['raw']['macro_f1']:.3f}"
            if "pipeline" in r:
                line += f" | pipeline acc {r['pipeline']['accuracy']:.3f} F1 {r['pipeline']['macro_f1']:.3f}"
            print(line, flush=True)
    pd.concat(allp).to_csv(ROOT / "results" / "preds_stress.csv", index=False)
    save_result(run, res, "stress_eval.json"); print(run["run_id"])


if __name__ == "__main__":
    main()
