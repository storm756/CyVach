"""Inspect the deployed ONNX model and the app's tokenizer assets.

- graph signature, opset, op/quantization counts, size (bytes, MB, MiB)
- tokenizer parity: HF tokenizer (app's tokenizer.json) vs exact port of the app's JS tokenizer
- prediction agreement between the two tokenizations on non-private text
- if model_v7_3/ exists: FP32 vs ONNX agreement (otherwise reported as "not run")
Only non-private files are used as text sources. Prints counts only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from common import FILE_SCHEMA, LABELS, RAW, ROOT, load_csv, new_run, save_result, sha256_file  # noqa: E402

APP_ASSETS = Path(r"C:\Users\Acer\Documents\New folder\SMSReaderApp\android\app\src\main\assets")
TEXT_FILES = ["ultimate_stress_test_400.csv", "Indian_sms_scam.csv", "spam.csv", "financial_fraud_training_10000.csv"]


def size_units(n: int) -> dict:
    return {"bytes": n, "MB": round(n / 1e6, 3), "MiB": round(n / 2**20, 3)}


def run_onnx(sess, ids, mask, batch=1):  # deployed graph only supports batch=1 (fixed Reshape)
    out = []
    for i in range(0, len(ids), batch):
        out.append(sess.run(["logits"], {"input_ids": np.array(ids[i:i + batch], dtype=np.int64),
                                         "attention_mask": np.array(mask[i:i + batch], dtype=np.int64)})[0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=400, help="max texts per file (seeded sample)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    run = new_run("check_onnx", vars(args))
    onnx_path = APP_ASSETS / "sms_spam_tiny.onnx"
    res: dict = {"onnx_path": str(onnx_path), "onnx_sha256": sha256_file(onnx_path),
                 "onnx_size": size_units(onnx_path.stat().st_size)}
    m = onnx.load(str(onnx_path))
    res["opset"] = [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import]
    res["producer"] = f"{m.producer_name} {m.producer_version}"
    def sig(vs):
        out = []
        for v in vs:
            tt = v.type.tensor_type
            dims = [d.dim_param or d.dim_value for d in tt.shape.dim] if tt.HasField("shape") else "no shape info"
            out.append({"name": v.name, "elem_type": onnx.TensorProto.DataType.Name(tt.elem_type), "shape": dims})
        return out
    res["inputs"], res["outputs"] = sig(m.graph.input), sig(m.graph.output)
    ops = Counter(n.op_type for n in m.graph.node)
    res["op_counts"] = dict(ops.most_common())
    res["value_info_entries"] = len(m.graph.value_info)
    init_types = Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer)
    res["initializer_dtypes"] = dict(init_types)

    # tokenizer assets
    cfg = json.loads((APP_ASSETS / "tokenizer_config.json").read_text(encoding="utf-8"))
    res["tokenizer_config"] = {k: cfg.get(k) for k in ("do_lower_case", "model_max_length", "tokenizer_class", "strip_accents", "tokenize_chinese_chars")}
    vocab_lines = (APP_ASSETS / "vocab.txt").read_text(encoding="utf-8").splitlines()
    res["vocab_lines"] = len(vocab_lines)
    from transformers import BertTokenizerFast
    hf = BertTokenizerFast(tokenizer_file=str(APP_ASSETS / "tokenizer.json"))
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    res["hf_vocab_size"] = hf.vocab_size

    rng = np.random.default_rng(args.seed)
    texts, srcs = [], []
    for f in TEXT_FILES:
        col = FILE_SCHEMA[f][0]
        t = load_csv(RAW / f)[col].astype(str).tolist()
        idx = rng.choice(len(t), size=min(args.per_file, len(t)), replace=False)
        texts += [t[i] for i in idx]
        srcs += [f] * len(idx)

    app_enc = [app.encode(t) for t in texts]
    hf_enc = hf(texts, truncation=True, max_length=128, padding="max_length")
    same = [a[0] == h for a, h in zip(app_enc, hf_enc["input_ids"])]
    res["tokenizer_parity"] = {"n": len(texts), "identical_input_ids": int(sum(same)),
                               "by_file": {f: int(sum(s for s, ff in zip(same, srcs) if ff == f)) for f in TEXT_FILES},
                               "app_unk_tokens": int(sum(np.array(a[0]) == app.unk for a in app_enc).sum()),
                               "hf_unk_tokens": int((np.array(hf_enc["input_ids"]) == hf.unk_token_id).sum())}

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    lg_app = run_onnx(sess, [a[0] for a in app_enc], [a[1] for a in app_enc])
    lg_hf = run_onnx(sess, hf_enc["input_ids"], hf_enc["attention_mask"])
    pa, ph = lg_app.argmax(1), lg_hf.argmax(1)
    res["onnx_pred_agreement_app_vs_hf_tokenizer"] = {"n": len(texts), "agree": int((pa == ph).sum()),
                                                       "by_file": {f: int(sum((pa[i] == ph[i]) for i in range(len(texts)) if srcs[i] == f)) for f in TEXT_FILES}}
    res["onnx_pred_distribution_app_tokenizer"] = {f: dict(Counter(LABELS[p] for p, ff in zip(pa, srcs) if ff == f)) for f in TEXT_FILES}

    ck = ROOT / "model_v7_3"
    if (ck / "config.json").exists():
        import torch
        from transformers import AutoModelForSequenceClassification
        cfgj = json.loads((ck / "config.json").read_text())
        res["fp32_config_id2label"] = cfgj.get("id2label")
        model = AutoModelForSequenceClassification.from_pretrained(str(ck)).eval()
        st = ck / "model.safetensors"
        if st.exists():
            res["fp32_size"] = size_units(st.stat().st_size)
        with torch.no_grad():
            lg_fp = []
            ids = np.array(hf_enc["input_ids"]); am = np.array(hf_enc["attention_mask"])
            for i in range(0, len(texts), 64):
                lg_fp.append(model(input_ids=torch.tensor(ids[i:i+64]), attention_mask=torch.tensor(am[i:i+64])).logits.numpy())
        lg_fp = np.concatenate(lg_fp)
        pf = lg_fp.argmax(1)
        def softmax(x):
            e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
        P, Q = softmax(lg_fp), softmax(lg_hf)
        res["fp32_vs_onnx_int8"] = {"n": len(texts), "agree": int((pf == ph).sum()),
                                    "mean_kl_fp32_to_int8": float(np.mean(np.sum(P * (np.log(P + 1e-12) - np.log(Q + 1e-12)), 1))),
                                    "max_abs_logit_diff": float(np.abs(lg_fp - lg_hf).max())}
    else:
        res["fp32_vs_onnx_int8"] = "not run: model_v7_3/ checkpoint not present"
    p = save_result(run, res, "check_onnx.json")
    print(json.dumps(res, indent=1, default=str))
    print("saved", p, run["run_id"])


if __name__ == "__main__":
    main()
