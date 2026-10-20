"""Phase 3: generate the synthetic stress sets (never used for training or model selection).

Outputs (data/stress/, all rows is_synthetic=True):
  stress_spoof.csv        each English base text x 6 sender types (independent of label); scam texts also with camouflage
  stress_obfuscation.csv  each English base text x {none, typos, spacing, homoglyph, leetspeak}
  stress_hinglish.csv     romanized Hindi/Marathi-English messages (needs native-speaker verification)
  digit_probe.csv         OTP/amount messages with realistic random digits vs patterned (112233-style) digits
Spec: docs/stress_spec.md. Seeded and deterministic.
"""
from __future__ import annotations

import argparse
import random
import re
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, normalize, save_result  # noqa: E402
from stress_templates import CAUTION, FAKE_DOMAINS, FILL, HAM, HINGLISH, PROMO, SPAM  # noqa: E402

BASE = {"spam": SPAM, "caution": CAUTION, "promo": PROMO, "ham": HAM}
HEADER_KEY = {"HDFC Bank": "HDFCBK", "SBI": "SBIINB", "ICICI Bank": "ICICIB", "Axis Bank": "AXISBK", "Kotak Bank": "KOTAKB",
              "Bank of Baroda": "BOBTXN", "Canara Bank": "CANBNK", "Amazon": "AMAZON", "Flipkart": "FLPKRT", "Swiggy": "SWIGGY",
              "Zomato": "ZOMATO", "Myntra": "MYNTRA", "Blinkit": "BLNKIT", "Zepto": "ZEPTON", "Jio": "JIONET", "Airtel": "AIRTEL",
              "Vi": "VIINFO", "BSNL": "BSNLIN"}
LOOKALIKE = {"HDFCBK": "HDFCBN", "SBIINB": "SBIIN8", "ICICIB": "lCICIB", "AXISBK": "AXlSBK", "KOTAKB": "K0TAKB", "BOBTXN": "B0BTXN",
             "CANBNK": "CANBNKK", "AMAZON": "AMAZ0N", "FLPKRT": "FLIPKRT", "SWIGGY": "SWlGGY", "ZOMATO": "Z0MATO", "MYNTRA": "MYNTRAA",
             "BLNKIT": "BLINKlT", "ZEPTON": "ZEPT0N", "JIONET": "JI0NET", "AIRTEL": "AIRTELL", "VIINFO": "VIlNFO", "BSNLIN": "BSNLlN"}
PREFIX = ["VM", "AD", "JD", "JM", "AX", "BP", "CP", "VK", "TX"]
HOMO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "i": "і", "y": "у"}  # Cyrillic look-alikes
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
CAMOUFLAGE = [("Security Alert: ", ""), ("", " Do not share your OTP with anyone."), ("", " Beware of fraud calls.")]


def indian_amount(rng):
    v = rng.choice([rng.randint(99, 999), rng.randint(1000, 9999), rng.randint(10000, 99999), rng.randint(100000, 500000)])
    s = str(v)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = ",".join(re.findall(r"\d{1,2}(?=(?:\d{2})*$)", head))
    return f"{head},{tail}"


def render(tpl: str, rng: random.Random, patterned_digits: bool = False) -> tuple[str, dict]:
    used = {}
    def pick(k):
        used[k] = used.get(k) or rng.choice(FILL[k])
        return used[k]
    def otp():
        if patterned_digits:
            d = rng.randint(1, 7); return f"{d}{d}{d+1}{d+1}{d+2}{d+2}"
        n = rng.choice([4, 6, 6, 6]); return "".join(rng.choice(string.digits) for _ in range(n))
    def url():
        k = re.sub(r"[^a-z]", "", (used.get("bank") or used.get("shop") or used.get("tel") or pick("bank")).lower())
        r = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(6))
        return "http://" + rng.choice(FAKE_DOMAINS).format(k=k, r=r)
    def sub(m):
        k = m.group(1)
        if k in FILL:
            return pick(k)
        return {"otp": otp, "amt": lambda: indian_amount(rng), "acct": lambda: f"{rng.randint(1000, 9999)}",
                "date": lambda: f"{rng.randint(1, 28):02d}-{rng.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}",
                "time": lambda: f"{rng.randint(1, 12)}:{rng.choice(['00','15','30','45'])} {rng.choice(['AM','PM'])}",
                "url_fake": url, "phone": lambda: f"{rng.choice('6789')}{''.join(rng.choice(string.digits) for _ in range(9))}"}[k]()
    # fill brand slots first so URLs/headers can reuse the brand
    for k in ("bank", "shop", "tel"):
        if "{" + k + "}" in tpl:
            pick(k)
    return re.sub(r"\{(\w+)\}", sub, tpl), used


def brand_key(used, rng):
    for k in ("bank", "shop", "tel"):
        if k in used:
            return HEADER_KEY[used[k]]
    return rng.choice(list(HEADER_KEY.values()))


def sender_for(kind, used, rng):
    key = brand_key(used, rng)
    pre = rng.choice(PREFIX)
    if kind == "registered_ST":
        return f"{pre}-{key}-{rng.choice('ST')}"
    if kind == "registered_P":
        return f"{pre}-{key}-P"
    if kind == "registered_G":
        return f"{pre}-{rng.choice(['NDMAEW','TRAIND','MAHGOV','UIDAIG'])}-G"
    if kind == "personal_mobile":
        return f"+91{rng.choice('6789')}{''.join(rng.choice(string.digits) for _ in range(9))}"
    if kind == "unknown_alnum":
        return "".join(rng.choice(string.ascii_uppercase) for _ in range(6))
    if kind == "spoof_lookalike":
        return f"{pre}-{LOOKALIKE[key]}-S"
    raise ValueError(kind)


SENDER_KINDS = ["registered_ST", "registered_P", "registered_G", "personal_mobile", "unknown_alnum", "spoof_lookalike"]


def true_label(text_label, sender_kind):
    # taxonomy: OTP-themed texts from personal numbers are scams
    if text_label == "caution" and sender_kind == "personal_mobile":
        return "spam"
    return text_label


def typos(t, rng):
    def f(w):
        if len(w) < 4 or not w.isalpha() or rng.random() > 0.35:
            return w
        i = rng.randrange(1, len(w) - 1)
        op = rng.choice(["swap", "drop", "dup"])
        if op == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return w[:i] + w[i + 1:] if op == "drop" else w[:i] + w[i] + w[i:]
    return " ".join(f(w) for w in t.split(" "))


def spacing(t, rng):
    def f(w):
        if len(w) < 4 or not w.isalpha() or rng.random() > 0.3:
            return w
        return rng.choice([" ", ".", "-"]).join(w)
    return " ".join(f(w) for w in t.split(" "))


def homoglyph(t, rng):
    return "".join(HOMO[c] if c in HOMO and rng.random() < 0.4 else c for c in t)


def leet(t, rng):
    out = []
    for w in t.split(" "):
        if w.startswith("http") or rng.random() > 0.5:
            out.append(w)
        else:
            out.append("".join(LEET.get(c.lower(), c) for c in w))
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    run = new_run("make_stress", vars(args))
    rng = random.Random(args.seed)
    outdir = ROOT / "data" / "stress"
    outdir.mkdir(parents=True, exist_ok=True)

    base = []
    for lab, tpls in BASE.items():
        for j, tpl in enumerate(tpls):
            text, used = render(tpl, rng)
            base.append({"base_id": f"{lab[:2]}{j:02d}", "text_label": lab, "text": text, "used": used})

    spoof = []
    for b in base:
        for kind in SENDER_KINDS:
            variants = [("", "")] + ([rng.choice(CAMOUFLAGE)] if b["text_label"] == "spam" else [])
            for k, (pre, suf) in enumerate(variants):
                spoof.append({"item_id": f"sp_{b['base_id']}_{kind}_{k}", "base_id": b["base_id"], "sender": sender_for(kind, b["used"], rng),
                              "sender_kind": kind, "camouflage": int(k > 0), "text": pre + b["text"] + suf,
                              "text_label": b["text_label"], "final_label": true_label(b["text_label"], kind)})
    obf = []
    for b in base:
        for name, fn in (("none", lambda t, r: t), ("typos", typos), ("spacing", spacing), ("homoglyph", homoglyph), ("leetspeak", leet)):
            obf.append({"item_id": f"ob_{b['base_id']}_{name}", "base_id": b["base_id"], "transform": name,
                        "text": fn(b["text"], rng), "final_label": b["text_label"]})
    hing = []
    for lab, tpls in HINGLISH.items():
        for j, tpl in enumerate(tpls):
            hing.append({"item_id": f"hi_{lab[:2]}{j:02d}", "text": render(tpl, rng)[0], "final_label": lab,
                         "needs_native_verification": True})
    probe = []
    for j, tpl in enumerate(CAUTION + HAM):
        lab = "caution" if j < len(CAUTION) else "ham"
        if "{otp}" not in tpl and "{amt}" not in tpl:
            continue
        for k in range(3):
            probe.append({"item_id": f"dp_{j:02d}_r{k}", "digit_style": "random", "text": render(tpl, rng)[0], "final_label": lab})
        probe.append({"item_id": f"dp_{j:02d}_p", "digit_style": "patterned", "text": render(tpl, rng, patterned_digits=True)[0], "final_label": lab})

    real_norm = set()
    for s in ("train", "val", "test"):
        real_norm |= {normalize(t) for t in pd.read_csv(ROOT / "data" / f"{s}.csv", keep_default_na=False)["text"]}
    summary = {}
    for name, rows in (("stress_spoof", spoof), ("stress_obfuscation", obf), ("stress_hinglish", hing), ("digit_probe", probe)):
        d = pd.DataFrame(rows)
        d["is_synthetic"] = True
        d.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8")
        key = (d["sender"] + "||" + d["text"]) if "sender" in d else d["text"]
        summary[name] = {"rows": int(len(d)), "unique_rows": int(key.nunique()), "unique_texts": int(d["text"].nunique()),
                         "labels": d["final_label"].value_counts().to_dict(),
                         "overlap_with_real_splits_normalized": int(sum(normalize(t) in real_norm for t in d["text"].unique()))}
    summary["stress_spoof"]["by_sender_kind_label"] = {f"{a}/{b}": int(n) for (a, b), n in
                                                      pd.DataFrame(spoof).groupby(["sender_kind", "final_label"]).size().items()}
    save_result(run, summary, "make_stress.json")
    import json
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_sender_kind_label"} for k, v in summary.items()}, indent=1))
    print(run["run_id"])


if __name__ == "__main__":
    main()
