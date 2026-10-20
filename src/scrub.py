"""Local, deterministic PII scrubber and audit (no network, no LLM).

Usage:
  python src/scrub.py --audit FILE.csv      # pass 1 (raw) + scrub + pass 2 (residual); prints counts only

Scrub replaces each hit with a typed placeholder: <URL> <EMAIL> <VPA> <PHONE>
<AADHAAR> <PAN> <IFSC> <CARD> <ACCT> <NAME>. Placeholders are chosen so that the
residual detectors below cannot re-match them.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FILE_SCHEMA, RAW, load_csv, new_run, save_result  # noqa: E402

# Order matters: more specific patterns first.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("url", re.compile(
        r"(?:https?://|www\.)\S+"
        r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|in|me|ly|co|org|net|xyz|info|link|app|io|gl|cc|top|site|online|club|biz)\b(?:/\S*)?",
        re.IGNORECASE)),
    ("vpa", re.compile(r"\b[\w.-]{2,}@[a-z][a-z0-9]{1,15}\b", re.IGNORECASE)),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("aadhaar", re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)|(?<!\d)[2-9]\d{11}(?!\d)")),
    ("phone", re.compile(r"(?<![\w])(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
    ("card", re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}(?!\d)|(?<!\d)\d{16}(?!\d)")),
    ("masked_acct", re.compile(r"\b(?:[xX*]{2,}[ -]?\d{3,}|[aA]/?[cC]\.?\s*(?:no\.?\s*)?[xX*]*\d{3,})")),
    ("long_digits", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
]

NAME_TRIGGER = r"(?:dear|hi|hello|hey|to|from|by|mr\.?|mrs\.?|ms\.?|shri|smt\.?|sir|madam|beneficiary|payee|name)"
NAME_RE = re.compile(
    rf"(?<![\w])({NAME_TRIGGER})(\s*[:,]?\s+)((?:[A-Z][a-z]+|[A-Z]{{3,}})(?:\s+(?:[A-Z][a-z]+|[A-Z]{{2,}}))?)",
    re.IGNORECASE,
)
# Capitalized words after a trigger that are NOT names (brands, generic words).
NAME_STOP = {w.lower() for w in """
Your You The Our All Customer Customers User Users Team Sir Madam Bank Account Card Valued Member Members
SBI HDFC ICICI AXIS KOTAK PNB BOB IDFC Paytm Amazon Flipkart Jio Airtel Vi Vodafone BSNL Google UPI OTP KYC
India Indian Mumbai Delhi Pune Bangalore Chennai Kolkata Get Click Call Visit Pay Rs INR Your Please Dear Hi
Hello Thanks Thank This That It We Us Me Your Mobile Number No A An Credit Debit Wallet Mr Mrs Ms Friend Friends
Today Tomorrow Use Avail Enjoy Save Win Free New Now Order Shop Book Recharge Plan Data Pack Offer Sale Code Not
""".split()}


WEAK_TRIGGERS = {"to", "from", "by", "name", "sir", "madam"}
# Words seen in lower case elsewhere in the corpus are common words, not names.
# Filled by set_common_words(); empty means "no corpus evidence" (scrub everything).
COMMON_WORDS: set[str] = set()


def set_common_words(texts, min_count: int = 3) -> int:
    c = Counter(w for t in texts for w in re.findall(r"(?<![A-Za-z])([a-z]{2,})(?![A-Za-z])", str(t)))
    COMMON_WORDS.clear()
    COMMON_WORDS.update(w for w, n in c.items() if n >= min_count)
    return len(COMMON_WORDS)


def _name_sub(m: re.Match, counter: Counter | None) -> str:
    cand = m.group(3)
    first = cand.split()[0]
    if first.lower() in NAME_STOP:
        return m.group(0)
    # Weak triggers ("to Login", "by Amazon") need extra evidence: skip common words.
    if m.group(1).lower() in WEAK_TRIGGERS and first.lower() in COMMON_WORDS:
        return m.group(0)
    if counter is not None:
        counter["name"] += 1
    return f"{m.group(1)}{m.group(2)}<NAME>"


def scrub(text: str, counter: Counter | None = None) -> str:
    t = str(text)
    for kind, pat in PATTERNS:
        def rep(m, kind=kind):
            if counter is not None:
                counter[kind] += 1
            return f"<{kind.upper()}>"
        t = pat.sub(rep, t)
    t = NAME_RE.sub(lambda m: _name_sub(m, counter), t)
    # Final pass: anything the residual detectors would still flag is masked too.
    def fin(kind, repl):
        def f(m):
            if counter is not None:
                counter[kind] += 1
            return repl(m)
        return f
    t = SAL_NAME_RE.sub(fin("name_after_salutation", lambda m: m.group(1) + "<NAME>"), t)
    t = PASSWORD_RE.sub(fin("password", lambda m: m.group(1) + "<SECRET>"), t)
    # capitalised / ALLCAPS words (and single-letter initials) glued to a name placeholder
    t = NEAR_NAME_RE.sub(fin("name_near_placeholder", lambda m: m.group(1) + "<NAME>"), t)
    t = SPACED_ACCT_RE.sub(fin("spaced_acct", lambda m: "<ACCT>"), t)
    kn = _known_names_re()
    if kn is not None:
        t = kn.sub(fin("known_name", lambda m: "<NAME>"), t)
    # second neighbour pass: capitalised words next to any name placeholder that are not
    # corpus-common words (seen in lower case) are treated as names too
    def nb(m):
        w = m.group(2)
        if w.lower() in COMMON_WORDS or w.lower() in NAME_STOP:
            return m.group(0)
        if counter is not None:
            counter["name_neighbour"] += 1
        return m.group(1) + "<NAME>"
    t = NEIGHBOUR_RE.sub(nb, t)
    # payee name in UPI narrations: "UPI/P2A/<ref>/<Payee Name>"
    t = UPI_PAYEE_RE.sub(fin("upi_payee", lambda m: m.group(1) + "<NAME>"), t)
    t = NARRATION_RE.sub(fin("narration_payee", lambda m: m.group(1) + "<NAME>"), t)
    t = BIRTHDAY_RE.sub(fin("greeting_name", lambda m: m.group(1) + "<NAME>"), t)
    t = PASSENGER_RE.sub(fin("passenger_name", lambda m: m.group(1) + "<NAME>"), t)
    # PAN-shaped codes exposed after neighbouring tokens were masked
    t = PAN_LOOSE_RE.sub(fin("pan", lambda m: "<PAN>"), t)
    return t


UPI_PAYEE_RE = re.compile(r"(UPI/P2[AM]/[^/\s]+/)(.+?)(?=\s+Not you|\n|$)")
NARRATION_RE = re.compile(r"((?:ref no|by Transfer)\s*[^,]*,?\s*[^:,]{0,15}:\s*)(.+?)(?=\s+Bal\b|\s+Rs:)")
BIRTHDAY_RE = re.compile(r"((?:happy birthday|happy bday),?\s+)([A-Z][a-z]+)(?=\s*[!,.])", re.IGNORECASE)
PASSENGER_RE = re.compile(r"(Boarding at [^,]+ only,\s*)([^,]+?)(?=(?:\+\d+)?,)")
PAN_LOOSE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![a-z0-9])")


NEIGHBOUR_RE = re.compile(r"((?:\[NAME\]|<NAME>)\s+)([A-Z][A-Za-z]+)")


_KN_CACHE: list = []


def _known_names_re():
    """Names from data/known_names.txt (built by src/build_known_names.py; local only)."""
    if not _KN_CACHE:
        p = Path(__file__).resolve().parents[1] / "data" / "known_names.txt"
        names = [n.strip() for n in p.read_text(encoding="utf-8").splitlines() if n.strip()] if p.exists() else []
        _KN_CACHE.append(re.compile(r"(?<![A-Za-z])(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?![A-Za-z])",
                                    re.IGNORECASE) if names else None)
    return _KN_CACHE[0]


# --- residual audit: broader, independent detectors run on scrubbed text ---
_STOP_ALT = "|".join(sorted(NAME_STOP, key=len, reverse=True))
SAL_NAME_RE = re.compile(
    rf"(\b(?i:dear|hi|hello|mr\.?|mrs\.?|ms\.?|shri|smt\.?)\s+)(?!<NAME>)(?!(?i:{_STOP_ALT})\b)[A-Z][A-Za-z]{{2,}}(?:\s+[A-Z]{{2,}}(?![a-z]))*")
# value must contain a digit or symbol, so "PIN is changed" is left alone
PASSWORD_RE = re.compile(r"(\b(?:pass(?:word|wd|code)?|pwd|pin)\s*(?:is|:|=|-)\s*)(?!<)(?=\S*[\d@#$%&*!])\S+", re.IGNORECASE)
_CAPWORD = r"(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z]\.?)"
NEAR_NAME_RE = re.compile(
    rf"((?:\[NAME\]|<NAME>)\s+)(?!(?i:{_STOP_ALT})\b){_CAPWORD}(?:\s+(?!(?i:{_STOP_ALT})\b){_CAPWORD})*(?![\w])")
SPACED_ACCT_RE = re.compile(
    r"(?<![\d:\-/])(?!\d{2,4}[-/]\d{2}[-/]\d{2,4})(?!\d{4}-\d{2}-\d{2})\d[\d \-]{8,}\d(?![\d:])"
    r"|\b[A-Z]{2,4}\d{2}-\d{6,}[A-Z]*\b")
RESIDUAL = {
    # identifying-risk detectors
    "digit_run_ge9": re.compile(r"(?<!\d)\d{9,}(?!\d)"),
    "spaced_digits_ge8_nondate": re.compile(
        r"(?<![\d:\-/])(?!\d{2,4}[-/]\d{2}[-/]\d{2,4})(?!\d{4}-\d{2}-\d{2})\d[\d \-]{8,}\d(?![\d:])"),
    "at_sign_nonprice": re.compile(r"(?<![\]>])@(?!\s*(?:rs|inr|₹|\d))", re.IGNORECASE),  # "[UPI_ID]@bank" = app placeholder
    "http_or_www": re.compile(r"https?:|www\.", re.IGNORECASE),
    "pan_like": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "salutation_capword": re.compile(
        rf"\b(?:dear|hi|hello|mr\.?|mrs\.?|ms\.?|shri|smt\.?)\s+(?!<NAME>)(?!(?:{_STOP_ALT})\b)[A-Z][A-Za-z]{{2,}}",
        re.IGNORECASE),
    "password_value": re.compile(r"\b(?:pass(?:word|wd)|pwd)\s*(?:is|:|=)\s*(?!<)\S+", re.IGNORECASE),
    "capword_after_name_placeholder": re.compile(r"(?:\[NAME\]|<NAME>)\s+(?!<)[A-Z][A-Za-z]+"),
    # informational: short codes (OTP / reference length), not identifying on their own
    "info_digit_run_6to8": re.compile(r"(?<!\d)\d{6,8}(?!\d)"),
}


def residual_counts(texts) -> dict:
    c = Counter()
    rows = Counter()
    for t in texts:
        hit = False
        for k, pat in RESIDUAL.items():
            n = len(pat.findall(t))
            if n:
                c[k] += n
                rows[k] += 1
                hit = hit or not k.startswith("info_")
        if hit:
            rows["any_identifying"] += 1
    return {"hits": dict(c), "rows_with_hit": dict(rows)}


def audit_file(fname: str) -> dict:
    text_col, _, sender_col = FILE_SCHEMA[fname]
    df = load_csv(RAW / fname)
    texts = df[text_col].tolist()
    pass1 = Counter()
    rows_hit = Counter()
    scrubbed = []
    for t in texts:
        per = Counter()
        scrubbed.append(scrub(t, per))
        pass1.update(per)
        for k in per:
            rows_hit[k] += 1
    res_before = residual_counts(texts)
    res_after = residual_counts(scrubbed)
    out = {
        "file": fname,
        "rows": len(texts),
        "pass1_replacements_by_type": dict(pass1),
        "pass1_rows_with_type": dict(rows_hit),
        "residual_detectors_on_raw": res_before,
        "residual_detectors_after_scrub": res_after,
    }
    if sender_col:
        s = df[sender_col].astype(str)
        out["sender_kinds"] = dict(Counter(sender_kind(x) for x in s))
    return out


def sender_kind(addr: str) -> str:
    a = addr.strip()
    clean = re.sub(r"[\s-]", "", a)
    if not a:
        return "empty"
    if re.fullmatch(r"(\+91|91|0)?[6-9]\d{9}", clean):
        return "mobile_number"
    if re.fullmatch(r"\+?\d+", clean):
        return f"numeric_len{min(len(clean.lstrip('+')), 13)}"
    m = re.fullmatch(r"[A-Z]{2}-[A-Z0-9]{6}(-[PSTG])?", a.upper())
    if m:
        return "dlt_header" + (a.upper()[-2:] if m.group(1) else "_nosuffix")
    if re.fullmatch(r"[A-Za-z0-9]{6}", a):
        return "6char_alnum"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", nargs="+", required=True)
    ap.add_argument("--out", default="pii_audit.json")
    args = ap.parse_args()
    run = new_run("pii_audit", {"files": args.audit})
    results = [audit_file(f) for f in args.audit]
    p = save_result(run, {"files": results}, args.out)
    import json
    print(json.dumps(results, indent=1))
    print("saved", p, run["run_id"])


if __name__ == "__main__":
    main()


def mask_known_names(text: str, counter: Counter | None = None) -> str:
    """Apply only the local known-names mask (for text that was already scrubbed)."""
    kn = _known_names_re()
    if kn is None:
        return text
    def rep(m):
        if counter is not None:
            counter["known_name"] += 1
        return "<NAME>"
    return kn.sub(rep, text)
