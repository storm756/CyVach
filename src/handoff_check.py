"""Phase 11: verify the deploy ONNX is a drop-in replacement for the app's sms_spam_tiny.onnx (v7.3).

- Signature check (names, dtypes, shapes) of new vs v7.3; label order from the checkpoint config.
- vocab: the deploy checkpoint's WordPiece vocab (saved by its tokenizer) vs the app's assets/vocab.txt, byte for byte.
- 20 SYNTHETIC messages (written for this check; no real text): app tokenizer port -> ONNX at batch 1 [1,128];
  prints logits/probabilities and the full app pipeline verdict (src/rules.py) with a synthetic sender.
  The same list (15 of them) is the on-device correctness spot-check for the app team.
- Optional: --ort-only <path> loads the model in whatever onnxruntime is installed and prints version + one output
  (used with onnxruntime 1.23.0 in a throwaway environment).
Outputs: results/handoff_check.json, handoff/expected_verdicts.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SPOT = [  # (synthetic sender, synthetic text) - invented for this check
    ("AX-HDFCBK-S", "482913 is your OTP for HDFC Bank NetBanking login. Do not share it with anyone. Valid for 5 minutes."),
    ("JM-AMAZON-T", "Your Amazon order #402-1188 has been delivered. Thank you for shopping with us."),
    ("VK-FLPKRT-P", "Big Billion Days start tomorrow! Up to 80% off on mobiles and TVs. Shop now on the app."),
    ("9876543210", "Dear customer, your SBI account KYC has expired and will be blocked today. Update now: http://sbi-kyc-update.xyz"),
    ("+919812345678", "Congratulations! You have won Rs 25,00,000 in the KBC lucky draw. Pay Rs 4,999 processing fee to claim."),
    ("BZ-ELECTR-G", "Electricity supply will be disconnected tonight at 9:30 pm as your last bill is unpaid. Call officer 9123456780 immediately."),
    ("TX-JIOINF-S", "Your Jio recharge of Rs 299 is successful. Validity: 28 days. Data: 2GB/day."),
    ("CP-ZOMATO-P", "Flat 50% off on your next order with code TASTY50. Valid till Sunday."),
    ("8899001122", "Hi, reached home safely. Will call you after dinner."),
    ("AD-ICICIB-S", "New login to your ICICI iMobile app from a new device in Pune. Not you? Call 1800 1080 immediately."),
    ("VM-WHTSAP-S", "WhatsApp Web is now active on a new device. If this was not you, log out all devices."),
    ("7766554433", "Part-time job: earn Rs 5,000 per day by liking YouTube videos. WhatsApp now on 7766554433."),
    ("QP-INDPST-S", "Your parcel is held at customs due to an incomplete address. Pay Rs 25 to release it: http://indiapost-redelivery.top"),
    ("BX-KOTAKB-S", "Rs 1,250.00 debited from A/c XX4821 on 12-Oct to SWIGGY. Avl bal Rs 8,402.17."),
    ("9988776655", "I sent an OTP to your number by mistake, please forward it to me quickly."),
    ("VM-PAYTMB-T", "Your UPI PIN was changed successfully. If not done by you, call your bank right away."),
    ("JD-BIGBSK-P", "Fresh fruits at 30% off this weekend only. Order on BigBasket now."),
    ("IR-IRCTCI-S", "PNR 4521873: Train 12951 Coach B3 Berth 41 confirmed. Have a safe journey."),
    ("UNKNWN", "Your Netflix subscription failed. Update payment details within 24 hrs: http://netflx-billing-help.com"),
    ("AX-AIRTEL-S", "Your Airtel bill of Rs 599 has been generated. Pay by 25-Oct to avoid late fees."),
]


def ort_only(path):
    import onnxruntime as ort
    from app_tokenizer import AppTokenizer
    from check_onnx import APP_ASSETS
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ids, mask = AppTokenizer(APP_ASSETS / "vocab.txt").encode(SPOT[0][1])
    out = s.run(None, {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.array([mask], dtype=np.int64)})[0]
    print(json.dumps({"onnxruntime": ort.__version__, "loaded": True, "output_shape": list(out.shape), "logits": [round(float(x), 4) for x in out[0]]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ort-only")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.ort_only:
        return ort_only(args.ort_only)
    import onnx
    import onnxruntime as ort
    import pandas as pd
    from scipy.special import softmax
    from transformers import AutoConfig, AutoTokenizer
    from app_tokenizer import AppTokenizer
    from check_onnx import APP_ASSETS
    from common import LABELS, ROOT, new_run, save_result, sha256_file
    from final_test import SEEDS
    from rules import ALL_LAYERS, pipeline

    run = new_run("handoff_check", {})
    new_p, old_p = ROOT / "models/onnx/cyvach_int8_quint8.onnx", APP_ASSETS / "sms_spam_tiny.onnx"

    def sig(p):
        m = onnx.load(str(p))
        f = lambda vs: [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                         "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in vs]
        return {"inputs": f(m.graph.input), "outputs": f(m.graph.output), "ir_version": m.ir_version,
                "opset_ai_onnx": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
                "node_domains": sorted({n.domain or "ai.onnx" for n in m.graph.node})}
    res = {"new": sig(new_p), "v73": sig(old_p)}
    res["names_dtypes_match"] = ([(i["name"], i["dtype"]) for i in res["new"]["inputs"]] == [(i["name"], i["dtype"]) for i in res["v73"]["inputs"]]
                                 and [(o["name"], o["dtype"]) for o in res["new"]["outputs"]] == [(o["name"], o["dtype"]) for o in res["v73"]["outputs"]])
    cfg = AutoConfig.from_pretrained(ROOT / "models" / SEEDS[3])
    res["label_order_new"] = [cfg.id2label[i] for i in range(4)]
    cfg7 = AutoConfig.from_pretrained(ROOT / "model_v7_3")
    res["label_order_v73"] = [cfg7.id2label[i] for i in range(4)]
    res["label_order_match"] = res["label_order_new"] == res["label_order_v73"] == LABELS
    vocab = AutoTokenizer.from_pretrained(ROOT / "models" / SEEDS[3]).get_vocab()  # token -> id of the deploy checkpoint
    ours = "".join(t + chr(10) for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])).encode("utf-8")
    app_bytes = (APP_ASSETS / "vocab.txt").read_bytes()
    res["vocab_byte_identical_to_app"] = ours == app_bytes
    res["vocab_sizes"] = {"checkpoint": len(vocab), "app_lines": len(app_bytes.decode("utf-8").splitlines())}
    res["deploy_sha256"] = sha256_file(new_p)

    app = AppTokenizer(APP_ASSETS / "vocab.txt")
    sn, so = (ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in (new_p, old_p))
    rows = []
    for i, (sender, text) in enumerate(SPOT, 1):
        ids, mask = app.encode(text)
        feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.array([mask], dtype=np.int64)}
        ln, lo = sn.run(["logits"], feed)[0], so.run(["logits"], feed)[0]
        assert ln.shape == (1, 4)
        pn, po = softmax(ln[0]), softmax(lo[0])
        lab = LABELS[int(pn.argmax())]
        full = pipeline(sender, text, lambda _t: (lab, float(pn.max())), ALL_LAYERS)
        lab7 = LABELS[int(po.argmax())]
        full7 = pipeline(sender, text, lambda _t: (lab7, float(po.max())), ALL_LAYERS)
        rows.append({"id": f"S{i:02d}", "sender": sender, "text": text, "logits": [round(float(x), 3) for x in ln[0]],
                     "model_label": lab, "model_conf": round(float(pn.max()), 4), "app_verdict": full.label, "rule": full.rule,
                     "v73_model_label": lab7, "v73_app_verdict": full7.label})
    res["spot"] = rows
    out = ROOT / "handoff"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows)[["id", "sender", "text", "model_label", "model_conf", "app_verdict", "rule"]].iloc[:15].to_csv(out / "expected_verdicts.csv", index=False)
    save_result(run, res, "handoff_check.json")
    print(json.dumps({k: res[k] for k in ("names_dtypes_match", "label_order_match", "vocab_byte_identical_to_app", "deploy_sha256")}, indent=1))
    print("new", res["new"]); print("v73", res["v73"])
    for r in rows:
        print(r["id"], r["sender"], r["model_label"], r["model_conf"], "->", r["app_verdict"], r["rule"], "| v7.3:", r["v73_model_label"], "->", r["v73_app_verdict"], r["logits"])
    print(run["run_id"])


if __name__ == "__main__":
    main()
