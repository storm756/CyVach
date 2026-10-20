"""Quantify problems in the legacy ultimate_stress_test_400.csv.

The generator (v7_3.ipynb, cell 3) fills every "{}" with a value drawn from a
*random* filler category, independent of the slot. We match each row back to its
template, recover the filler values and check whether each value's category is
plausible for its slot. Slot expectations below are written by hand from the
template wording (one entry per "{}").
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, load_csv, new_run, normalize, save_result  # noqa: E402

FILLERS = {
    "names": ["Ramesh", "Aditya", "Shreya", "Mahesh", "Atharva", "Suresh"],
    "amounts": ["500", "1200", "5000", "25000", "45.00"],
    "times": ["5", "10", "15", "30", "45"],
    "cities": ["Mumbai", "Delhi", "Pune", "Bangalore", "Lonavala"],
    "banks": ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank"],
    "items": ["Milk", "Bread", "iPhone 15", "Pizza", "Medicines"],
    "codes": ["BOGO", "SAVE50", "OTP123", "SAFE99", "WELCOME"],
    "phones": ["9876543210", "8877665544", "7766554433", "9900112233"],
}
NUM = {"amounts", "times", "phones"}

# template -> list of acceptable filler categories per slot
T = {
    "ham": [
        ("Sent Rs.{} from your Bank A/c xx{} to {}. Ref: {}", [{"amounts"}, {"amounts", "times", "phones"}, {"names", "banks"}, {"codes", "amounts", "phones", "times"}]),
        ("Credit Alert: Rs.{} received in your account from {}.", [{"amounts"}, {"names", "banks"}]),
        ("Your order for {} has been delivered. Rate us on the app.", [{"items"}]),
        ("Hey {}, I'll be there in {} minutes. See you soon!", [{"names"}, {"times"}]),
        ("Can you pick up {} from the store on your way home?", [{"items"}]),
        ("Meeting confirmed for {} tomorrow at the office.", [{"names"}]),
        ("Your flight {} to {} is delayed by {} hours.", [{"codes"}, {"cities"}, {"times"}]),
        ("Reminder: Your library book is due on {}.", [set()]),  # expects a date; no date filler exists
        ("Payment of Rs.{} to {} successful. Balance: Rs.{}", [{"amounts"}, {"names", "banks"}, {"amounts"}]),
        ("Dinner is ready, come home fast. Don't forget the keys.", []),
    ],
    "spam": [
        ("URGENT: Your electricity will be disconnected tonight. Call {} now.", [{"phones"}]),
        ("Your SIM card KYC expired. Outgoing calls blocked. Call {} to update.", [{"phones"}]),
        ("Congratulations! You won a cash prize of {} Lakhs in KBC lottery.", [{"amounts", "times"}]),
        ("Earn {} per day working from home. No experience. WhatsApp now.", [{"amounts"}]),
        ("Your bank account is suspended. Verify your PAN card at http://bit.ly/{}", [set(FILLERS)]),
        ("Final Notice: Your insurance policy will lapse. Pay now at {}.", [set()]),  # expects URL
        ("Get {} for only Rs. 999! Limited time deal at http://scam-store.in", [{"items"}]),
        ("Hello from Income Tax dept. Your refund is approved. Click to claim.", []),
        ("Win 1 Crore! Play Rummy now and get a signup bonus of Rs. 500.", []),
        ("Your parcel is held at customs. Pay Rs. 150 to release: http://fake.me", []),
    ],
    "promo": [
        ("Lenskart Sale: Buy 1 Get 1 Free on all eyeglasses today! Use code {}.", [{"codes"}]),
        ("Zomato: Hungry? Get {}% OFF on your next order with code {}.", [{"times"}, {"codes"}]),
        ("Flipkart: Big Billion Days start tomorrow! Lowest prices on {}.", [{"items"}]),
        ("Jio: Recharge with Rs. {} and get extra {}GB data for 30 days.", [{"amounts"}, {"times"}]),
        ("Myntra: End of Reason Sale! 50-80% OFF on top brands like Nike.", []),
        ("Dominos: Free Garlic Bread on orders above Rs. {}. Order now!", [{"amounts"}]),
        ("Apollo: Get {}% discount on medicines + free delivery this week.", [{"times"}]),
        ("Swiggy: Flat Rs. {} off on your first 3 orders. Use code TRY50.", [{"amounts", "times"}]),
        ("Ajio: Style Sale is LIVE! Up to {}% off on 5000+ styles.", [{"times"}]),
        ("Subway: Buy 1 sub and get a free cookie today at all outlets.", []),
    ],
    "caution": [
        ("{} is your OTP for HDFC Netbanking login. Do not share it.", [{"amounts", "times", "phones"}]),
        ("Login alert: A new device accessed your {} account from {}.", [{"banks"}, {"cities"}]),
        ("Wrong mPIN entered. {} attempts left before your UPI is blocked.", [{"times"}]),
        ("Use code {} to verify your mobile number on {}.", [{"amounts", "times", "codes"}, {"banks", "items"}]),
        ("Security Alert: Your password for {} was changed recently.", [{"banks"}]),
        ("Your 2-step verification code for Google is {}. Valid for 5 mins.", [{"amounts", "times", "phones", "codes"}]),
        ("Warning: Unusual activity detected on your Credit Card xx{}.", [{"amounts", "times"}]),
        ("{} is the verification code to link your Aadhaar card.", [{"amounts", "times", "phones", "codes"}]),
        ("A login attempt was blocked on your Binance account. Not you? Tap here.", []),
        ("Your WhatsApp web is now active on a Windows PC.", []),
    ],
}

VAL2CAT = {}
for cat, vals in FILLERS.items():
    for v in vals:
        VAL2CAT.setdefault(v, set()).add(cat)


def compile_tpl(tpl: str) -> re.Pattern:
    parts = [re.escape(p) for p in tpl.split("{}")]
    alts = "|".join(sorted((re.escape(v) for v in VAL2CAT), key=len, reverse=True))
    return re.compile("^" + f"({alts})".join(parts) + "$")


def main():
    run = new_run("audit_stress400", {"file": "ultimate_stress_test_400.csv"})
    df = load_csv(RAW / "ultimate_stress_test_400.csv")
    compiled = [(lab, tpl, compile_tpl(tpl), exp) for lab, lst in T.items() for tpl, exp in lst]
    out = {"rows": len(df), "label_counts": dict(Counter(df["Label"]))}
    out["exact_unique"] = int(df["Body"].nunique())
    out["exact_unique_body_label"] = int(df.drop_duplicates().shape[0])
    out["duplicate_rows"] = len(df) - out["exact_unique"]
    out["duplicate_share"] = out["duplicate_rows"] / len(df)
    out["unique_by_label"] = {k: int(v) for k, v in df.drop_duplicates().groupby("Label").size().items()}
    out["unique_normalized"] = len({normalize(t) for t in df["Body"]})
    matched = unmatched = nonsense_rows = slot_total = slot_bad = 0
    tpl_use = Counter()
    label_mismatch = 0
    bad_examples = []
    for body, lab in zip(df["Body"], df["Label"]):
        hit = None
        for tl, tpl, rx, exp in compiled:
            m = rx.match(body)
            if m:
                hit = (tl, tpl, m, exp)
                break
        if not hit:
            unmatched += 1
            continue
        matched += 1
        tl, tpl, m, exp = hit
        tpl_use[tpl] += 1
        if tl != lab:
            label_mismatch += 1
        row_bad = False
        for val, ok in zip(m.groups(), exp):
            slot_total += 1
            if not (VAL2CAT[val] & ok):
                slot_bad += 1
                row_bad = True
        if row_bad:
            nonsense_rows += 1
            if len(bad_examples) < 15:
                bad_examples.append(body)
    out.update({
        "rows_matched_to_template": matched, "rows_unmatched": unmatched,
        "distinct_templates_used": len(tpl_use), "templates_total": sum(len(v) for v in T.values()),
        "rows_from_slotless_templates": sum(n for t, n in tpl_use.items() if "{}" not in t),
        "slots_total": slot_total, "slots_with_implausible_filler": slot_bad,
        "rows_with_any_implausible_filler": nonsense_rows,
        "share_rows_implausible": nonsense_rows / max(1, matched),
        "template_label_mismatch_rows": label_mismatch,
        "example_implausible_rows": bad_examples,
    })
    p = save_result(run, out, "audit_stress400.json")
    import json
    print(json.dumps({k: v for k, v in out.items()}, indent=1, ensure_ascii=False))
    print("saved", p, run["run_id"])


if __name__ == "__main__":
    main()
