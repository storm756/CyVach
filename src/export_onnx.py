"""Phase 6: export the selected FP32 checkpoint to ONNX (opset 17, dynamic batch + sequence), quantize, verify.

Steps
  1. torch.onnx.export (TorchScript exporter, eager attention), names unchanged: input_ids, attention_mask -> logits.
  2. NO "metadata erasure" hack: onnxruntime quant_pre_process (shape inference + optimisation), then
     quantize_dynamic with QUInt8 (as deployed v7.3) and QInt8 weights.
  3. Checks: onnx.checker, batch 1/4/8 and sequence 16/64/128 run, signature dump.
  4. Parity on VAL (the test set is not touched before Gate 3): PyTorch FP32 (HF tokenizer, dynamic padding, exactly as
     in training eval) vs ONNX FP32 / INT8 fed by the exact port of the APP's JS tokenizer with the app's vocab.txt,
     padded to 128 as the app does. Reports argmax agreement, mean KL(fp32 || int8), macro-F1.
  5. Sizes in bytes, MB and MiB.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from scipy.special import log_softmax, softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS, size_units  # noqa: E402
from common import LABELS, ROOT, new_run, save_result, sha256_file  # noqa: E402
from metrics import evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from subsets import load_split  # noqa: E402


def signature(path):
    m = onnx.load(str(path))
    sig = lambda vs: [{"name": v.name, "type": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                       "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in vs]
    return {"opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "inputs": sig(m.graph.input), "outputs": sig(m.graph.output),
            "producer": f"{m.producer_name} {m.producer_version}"}


def onnx_logits(sess, ids, mask, batch=32):
    out = []
    for i in range(0, len(ids), batch):
        out.append(sess.run(["logits"], {"input_ids": np.asarray(ids[i:i + batch], dtype=np.int64),
                                         "attention_mask": np.asarray(mask[i:i + batch], dtype=np.int64)})[0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="train-20261018-133131", help="model dir under models/ (seed 3, median val F1)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    run = new_run("export_onnx", vars(args))
    mdir = ROOT / "models" / args.run
    out = ROOT / "models" / "onnx"; out.mkdir(parents=True, exist_ok=True)
    fp32, pre = out / "cyvach_fp32.onnx", out / "cyvach_fp32_pre.onnx"
    q = {"QUInt8": out / "cyvach_int8_quint8.onnx", "QInt8": out / "cyvach_int8_qint8.onnx"}
    res: dict = {"model_dir": str(mdir), "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}}

    model = AutoModelForSequenceClassification.from_pretrained(mdir, attn_implementation="eager").eval()
    assert model.config.id2label == dict(enumerate(LABELS)) or [model.config.id2label[i] for i in range(4)] == LABELS
    hf = AutoTokenizer.from_pretrained(mdir)
    dummy = hf(["hello world"] * 2, padding="max_length", max_length=128, return_tensors="pt")

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = Wrap(model).eval()  # a fresh Module is in training mode; export restores that mode afterwards (=> dropout on)
    torch.onnx.export(wrapped, (dummy["input_ids"], dummy["attention_mask"]), str(fp32), dynamo=False,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=args.opset,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}})
    onnx.checker.check_model(str(fp32))
    model.eval(); assert not model.training
    try:
        quant_pre_process(str(fp32), str(pre))
        res["pre_process"] = "default (symbolic shape inference ok)"
    except Exception as e:  # documented ORT option instead of hand-editing graph metadata
        res["pre_process_default_error"] = f"{type(e).__name__}: {e}"
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        res["pre_process"] = "skip_symbolic_shape=True (ONNX shape inference + optimisation still run)"
    for name, path in q.items():
        quantize_dynamic(str(pre), str(path), weight_type=getattr(QuantType, name))
        onnx.checker.check_model(str(path))
    files = {"fp32": fp32, **{f"int8_{k}": v for k, v in q.items()}}
    res["files"] = {k: {"path": str(v.relative_to(ROOT)), "sha256": sha256_file(v), "size": size_units(v.stat().st_size),
                        "signature": signature(v)} for k, v in files.items()}
    res["fp32_checkpoint_size"] = size_units((mdir / "model.safetensors").stat().st_size)

    sess = {k: ort.InferenceSession(str(v), providers=["CPUExecutionProvider"]) for k, v in files.items()}
    shapes = {}
    for k, s in sess.items():
        ok = []
        for b, L in [(1, 128), (4, 64), (8, 16), (1, 7)]:
            ids = np.full((b, L), 101, dtype=np.int64); m = np.ones((b, L), dtype=np.int64)
            ok.append(list(s.run(["logits"], {"input_ids": ids, "attention_mask": m})[0].shape) == [b, 4])
        shapes[k] = all(ok)
    res["dynamic_shapes_ok"] = shapes

    # tokenizer identity: new model's vocab vs the app's vocab.txt
    app_vocab = (APP_ASSETS / "vocab.txt").read_text(encoding="utf-8").splitlines()
    hv = hf.get_vocab()
    res["vocab_identical_to_app"] = len(hv) == len(app_vocab) and all(hv.get(t) == i for i, t in enumerate(app_vocab))

    # parity on val (U1 fixed fill, seed 1234, as in training eval)
    va = load_split(ROOT / "data/val.csv", drop_native=True)
    frng = random.Random(1234)
    texts = [fill(t, frng) for t in va["text"]]
    y = va["final_label"].tolist()
    with torch.no_grad():
        pt = []
        for i in range(0, len(texts), 64):
            enc = hf(texts[i:i + 64], truncation=True, max_length=128, padding=True, return_tensors="pt")
            pt.append(model(**enc).logits.numpy())
    pt = np.concatenate(pt)
    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    enc = [app.encode(t) for t in texts]
    ids, mask = [e[0] for e in enc], [e[1] for e in enc]
    hfe = hf(texts, truncation=True, max_length=128, padding="max_length")
    lg = {"torch_fp32_hf_tok": pt, "onnx_fp32_hf_tok": onnx_logits(sess["fp32"], hfe["input_ids"], hfe["attention_mask"]),
          **{f"onnx_{k}_app_tok": onnx_logits(s, ids, mask) for k, s in sess.items()}}
    res["app_vs_hf_token_ids_identical"] = float(np.mean([a == b for a, b in zip(ids, hfe["input_ids"])]))
    par = {}
    for k, v in lg.items():
        ev = evaluate(y, [LABELS[i] for i in v.argmax(1)])
        p_ref, lp = softmax(pt, 1), log_softmax(v, 1)
        par[k] = {"val_macro_f1": ev["macro_f1"], "val_f1_caution": ev["per_class"]["caution"]["f1"],
                  "fpr_legit_as_scam": ev["fpr_legit_as_scam"],
                  "argmax_agree_vs_torch": float((v.argmax(1) == pt.argmax(1)).mean()),
                  "n_disagree_vs_torch": int((v.argmax(1) == pt.argmax(1)).__invert__().sum()),
                  "mean_kl_torch_to_this": float((p_ref * (log_softmax(pt, 1) - lp)).sum(1).mean()),
                  "max_abs_logit_diff": float(np.abs(v - pt).max())}
    f = lg["onnx_fp32_app_tok"]
    for k in q:
        v = lg[f"onnx_int8_{k}_app_tok"]
        par[f"onnx_int8_{k}_app_tok"]["argmax_agree_vs_onnx_fp32"] = float((v.argmax(1) == f.argmax(1)).mean())
        par[f"onnx_int8_{k}_app_tok"]["mean_kl_onnx_fp32_to_int8"] = float((softmax(f, 1) * (log_softmax(f, 1) - log_softmax(v, 1))).sum(1).mean())
    res["parity_val"] = {"n": len(y), **par}
    save_result(run, res, "export_onnx.json")
    print(json.dumps({k: res.get(k) for k in ("versions", "pre_process", "pre_process_default_error", "app_vs_hf_token_ids_identical", "dynamic_shapes_ok", "vocab_identical_to_app", "fp32_checkpoint_size")}, indent=1))
    for k, v in res["files"].items():
        print(k, v["size"], v["signature"]["opset"], [(i["name"], i["type"], i["shape"]) for i in v["signature"]["inputs"] + v["signature"]["outputs"]])
    print(json.dumps(res["parity_val"], indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
