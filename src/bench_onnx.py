"""Phase 6: DESKTOP CPU latency of ONNX models (not phone latency; the owner measures that on the device).

Protocol: ONNX Runtime CPUExecutionProvider, intra_op threads in {1, 4}, input [1, 128] exactly as the app sends it
(app tokenizer, padded to 128), one fixed realistic SMS; 50 warm-up runs, then 500 timed runs (time.perf_counter_ns).
Reports median and p95 in ms, plus machine info.

On-device protocol (for the owner): same model file and the same message, release build, phone on charger, screen on,
airplane mode, other apps closed; 50 warm-up classifications, then 500 timed calls around session.run only (exclude
tokenisation and UI), report median and p95, and phone model / Android version / ORT-mobile version.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_tokenizer import AppTokenizer  # noqa: E402
from check_onnx import APP_ASSETS, size_units  # noqa: E402
from common import ROOT, new_run, save_result  # noqa: E402

MSG = ("Dear customer, your KYC is pending. Your account will be blocked today. Update now at "
       "https://bit.ly/3xYzAbc or call 9876543210. Ignore if already done.")


def bench(path: Path, threads: int, ids, mask, warm: int, n: int) -> dict:
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = {"input_ids": ids, "attention_mask": mask}
    for _ in range(warm):
        s.run(["logits"], feed)
    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); s.run(["logits"], feed); t.append((time.perf_counter_ns() - t0) / 1e6)
    t = np.array(t)
    return {"median_ms": round(float(np.median(t)), 3), "p95_ms": round(float(np.percentile(t, 95)), 3),
            "mean_ms": round(float(t.mean()), 3), "runs": n, "warmup": warm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    args = ap.parse_args()
    run = new_run("bench_onnx", vars(args))
    models = {"v7.3_deployed_int8": APP_ASSETS / "sms_spam_tiny.onnx",
              "new_fp32": ROOT / "models/onnx/cyvach_fp32.onnx",
              "new_int8_QUInt8": ROOT / "models/onnx/cyvach_int8_quint8.onnx",
              "new_int8_QInt8": ROOT / "models/onnx/cyvach_int8_qint8.onnx"}
    ids, mask = AppTokenizer(APP_ASSETS / "vocab.txt").encode(MSG)
    ids, mask = np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64)
    res = {"setting": "desktop CPU (NOT phone)", "input_shape": [1, 128], "onnxruntime": ort.__version__,
           "machine": {"processor": platform.processor(), "platform": platform.platform(), "python": platform.python_version()},
           "models": {}}
    for k, p in models.items():
        res["models"][k] = {"size": size_units(p.stat().st_size),
                            **{f"threads_{th}": bench(p, th, ids, mask, args.warmup, args.runs) for th in (1, 4)}}
        print(k, res["models"][k], flush=True)
    save_result(run, res, "bench_onnx.json"); print(run["run_id"])


if __name__ == "__main__":
    main()
