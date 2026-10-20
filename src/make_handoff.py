"""Phase 11: build handoff/ for the Android app (model.onnx, the app's own vocab/tokenizer_config, MODEL_INFO.json,
expected_verdicts.csv, INTEGRATION_FOR_ANTIGRAVITY.md), SHA256SUMS, and handoff/cyvach_new_model.zip.
Run src/handoff_check.py first (it writes expected_verdicts.csv and results/handoff_check.json).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_onnx import APP_ASSETS, size_units  # noqa: E402
from common import LABELS, ROOT, new_run, save_result, sha256_file  # noqa: E402

H = ROOT / "handoff"
SHIP = ["model.onnx", "vocab.txt", "tokenizer_config.json", "MODEL_INFO.json", "expected_verdicts.csv", "INTEGRATION_FOR_ANTIGRAVITY.md"]


def r(f):
    return json.loads((ROOT / "results" / f).read_text(encoding="utf-8"))


def main():
    run = new_run("make_handoff", {})
    H.mkdir(exist_ok=True)
    src = ROOT / "models" / "onnx" / "cyvach_int8_quint8.onnx"
    shutil.copyfile(src, H / "model.onnx")
    for f in ("vocab.txt", "tokenizer_config.json"):
        shutil.copyfile(APP_ASSETS / f, H / f)
    chk, st, ph, ex = r("handoff_check.json")["result"], r("stats.json"), r("posthoc.json")["result"], r("export_onnx.json")
    assert chk["names_dtypes_match"] and chk["label_order_match"] and chk["vocab_byte_identical_to_app"]
    p, ci = st["result"]["point"], st["result"]["ci95"]
    info = {
        "deploy_source_file": "cyvach_int8_quint8.onnx (seed 3 = median validation macro-F1 of 5 seeds; dynamic INT8, QUInt8 weights)",
        "shipped_as": "model.onnx (rename to sms_spam_tiny.onnx, or change the filename in the app; see INTEGRATION note)",
        "sha256": sha256_file(H / "model.onnx"), "size": size_units((H / "model.onnx").stat().st_size),
        "onnx_ir_version": chk["new"]["ir_version"], "opset": chk["new"]["opset_ai_onnx"], "node_domains": chk["new"]["node_domains"],
        "inputs": [{"name": "input_ids", "dtype": "int64", "shape_used_by_app": [1, 128], "graph_shape": ["batch", "seq"]},
                   {"name": "attention_mask", "dtype": "int64", "shape_used_by_app": [1, 128], "graph_shape": ["batch", "seq"]}],
        "outputs": [{"name": "logits", "dtype": "float32", "shape_used_by_app": [1, 4], "graph_shape": ["batch", 4]}],
        "label_order": {str(i): l for i, l in enumerate(LABELS)}, "label_meaning": {"spam": "scam / fraud"},
        "max_len": 128, "lowercase": True, "padding": "pad to 128 with [PAD]=0, [CLS]=101 first, [SEP]=102 last (app tokenizer, unchanged)",
        "vocab": "app's existing vocab.txt, byte-identical (30,522 entries)",
        "preprocessing_change_required": False,
        "loaded_and_checked_with_onnxruntime": ["1.31.0 (desktop)", "1.23.0 (desktop; the app's onnxruntime-react-native version)"],
        "test_results_model_only": {
            "test_set": "frozen, 519 real SMS (main subset), labels Claude-proposed with 180 owner-checked; single evaluation",
            "macro_f1": {"new": round(p["new_int8_quint8"]["macro_f1"], 3), "new_ci95": [round(x, 3) for x in ci["new_int8_quint8"]["macro_f1"]],
                         "v7_3_shipped": round(p["v73_int8"]["macro_f1"], 3)},
            "legit_flagged_as_scam": {"new": ph["main_table_counts"]["new_int8_deploy"]["legit_as_scam"], "new_pct": round(100 * p["new_int8_quint8"]["legit_as_scam"], 1),
                                      "v7_3_shipped": ph["main_table_counts"]["v73_int8"]["legit_as_scam"], "v7_3_pct": round(100 * p["v73_int8"]["legit_as_scam"], 1)},
            "with_shipped_rule_layer_macro_f1": {"new": round(p["new_int8_quint8+rules"]["macro_f1"], 3), "v7_3": round(p["v73_int8+rules"]["macro_f1"], 3)}},
        "run_ids": {"export": ex["run_id"], "stats": st["run_id"], "handoff_check": r("handoff_check.json")["run_id"]},
    }
    (H / "MODEL_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    missing = [f for f in SHIP if not (H / f).exists()]
    assert not missing, f"missing in handoff/: {missing}"
    sums = "".join(f"{sha256_file(H / f)}  {f}\n" for f in SHIP)
    (H / "SHA256SUMS").write_bytes(sums.encode("utf-8"))  # LF endings so `sha256sum -c` works
    z = H / "cyvach_new_model.zip"
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in SHIP + ["SHA256SUMS"]:
            info_ = zipfile.ZipInfo(f, date_time=(2026, 10, 19, 0, 0, 0))  # fixed timestamp: reproducible zip
            info_.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info_, (H / f).read_bytes())
    res = {"files": SHIP + ["SHA256SUMS"], "model_sha256": info["sha256"], "zip_sha256": sha256_file(z), "zip_size": size_units(z.stat().st_size)}
    save_result(run, res, "handoff.json")
    print(sums); print(json.dumps(res, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
