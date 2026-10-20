"""U1: replace scrubber / app-export placeholders with realistic random surface forms.

Why: the training text contains placeholders (<URL>, [OTP], <NAME>, ...) but the deployed app feeds the model raw SMS
with real links, numbers and names. U1 trains (and validates) on realistic values so the model sees deployment-like input;
U0 keeps the placeholders, which would require the app to apply the same masking before inference.
Fills are label-independent: every class draws from the same generators.
"""
from __future__ import annotations

import random
import re
import string

FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Rohan", "Anjali", "Vikram", "Pooja", "Arjun", "Neha", "Karan", "Divya",
         "Suresh", "Kavita", "Manoj", "Meera", "Aditya", "Shreya", "Nikhil", "Ritu", "Sanjay", "Asha", "Deepak", "Nisha"]
LAST = ["Sharma", "Patil", "Kulkarni", "Shah", "Iyer", "Reddy", "Joshi", "Deshmukh", "Mehta", "Nair", "Singh", "Gupta"]
WORDS = ["pay", "info", "track", "offer", "secure", "my", "go", "app", "link", "help", "portal", "update", "service", "web"]
TLDS = [".com", ".in", ".co.in", ".net", ".org", ".info"]
SHORT = ["bit.ly", "tinyurl.com", "t.ly", "cutt.ly", "rb.gy"]
MERCHANTS = ["Swiggy", "Zomato", "Amazon", "Flipkart", "BigBasket", "Reliance Fresh", "DMart", "Uber", "Ola", "PVR"]
VPA = ["okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "paytm", "ibl"]

PH = re.compile(r"<(NAME|URL|PHONE|LONG_DIGITS|AADHAAR|EMAIL|PAN|USERID|SECRET|UKP|CARD)>|"
                r"\[(AMOUNT|PHONE|ACCOUNT|REF_NUM|NAME|OTP|EMAIL|UPI_ID|MERCHANT|INVITE)\]")


def _d(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def _alnum(rng, n):
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n))


def _url(rng):
    r = rng.random()
    if r < 0.35:
        return f"https://{rng.choice(SHORT)}/{_alnum(rng, rng.randint(5, 8))}"
    dom = rng.choice(WORDS) + rng.choice(["", rng.choice(WORDS), _d(rng, 2)]) + rng.choice(TLDS)
    path = "" if rng.random() < 0.3 else "/" + rng.choice(WORDS) + ("/" + _alnum(rng, 6) if rng.random() < 0.5 else "")
    return rng.choice(["https://", "http://", "www.", "https://www."]) + dom + path


def _name(rng):
    return rng.choice(FIRST) + (" " + rng.choice(LAST) if rng.random() < 0.4 else "")


def _amount(rng):
    v = rng.choice([rng.randint(1, 999), rng.randint(100, 9999), rng.randint(1000, 99999)])
    s = f"{v:,}" if rng.random() < 0.5 else str(v)
    return s + (f".{_d(rng, 2)}" if rng.random() < 0.4 else "")


def _fill(kind, rng):
    if kind == "NAME":
        return _name(rng)
    if kind == "URL":
        return _url(rng)
    if kind == "PHONE":
        return rng.choice(["", "+91", "+91 ", "0"]) + rng.choice("6789") + _d(rng, 9)
    if kind == "LONG_DIGITS":
        return _d(rng, rng.randint(9, 16))
    if kind == "AADHAAR":
        return rng.choice(["XXXX XXXX ", "xxxxxxxx"]) + _d(rng, 4)
    if kind == "EMAIL":
        return f"{rng.choice(FIRST).lower()}{_d(rng, rng.randint(0, 4))}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}"
    if kind == "PAN":
        return "".join(rng.choice(string.ascii_uppercase) for _ in range(5)) + _d(rng, 4) + rng.choice(string.ascii_uppercase)
    if kind in ("USERID", "UKP", "INVITE"):
        return _alnum(rng, rng.randint(6, 10))
    if kind == "SECRET":
        return _alnum(rng, 8) + rng.choice("@#$!")
    if kind == "CARD" or kind == "ACCOUNT":
        return rng.choice(["XX", "xx", "XXXX", "*"]) + _d(rng, 4)
    if kind == "AMOUNT":
        return _amount(rng)
    if kind == "REF_NUM":
        return _d(rng, 12)
    if kind == "OTP":
        return _d(rng, rng.choice([4, 6, 6, 6]))
    if kind == "UPI_ID":
        return f"{rng.choice(FIRST).lower()}{_d(rng, rng.randint(0, 3))}@{rng.choice(VPA)}"
    if kind == "MERCHANT":
        return rng.choice(MERCHANTS)
    return _alnum(rng, 6)


def fill(text: str, rng: random.Random) -> str:
    return PH.sub(lambda m: _fill(m.group(1) or m.group(2), rng), text)


if __name__ == "__main__":
    r = random.Random(0)
    for t in ["Dear <NAME>, click <URL> to update KYC. Call <PHONE>", "Rs.[AMOUNT] debited from A/c [ACCOUNT]. OTP [OTP]"]:
        print(fill(t, r))
