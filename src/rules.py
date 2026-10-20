"""Exact Python port of the shipped CyVach rule layer (SMSReaderApp, read 2026-10-17).

Sources: src/ml/scamDetector.ts (classifyText: pre-model heuristics H1, H2) and
src/components/SMSReader.tsx (getFinalClassification: R0..R4). See docs/rule_layer_as_shipped.md.
JS semantics reproduced: String.includes (substring), toLowerCase/toUpperCase, endsWith, the regex
/^(\\+91|91|0)?[6-9]\\d{9}$/ after removing spaces and hyphens, first-matching-rule-wins order.
Not ported: ML Kit language detection/translation (cannot run on desktop) and the 0.5 fallback on errors.

Layer switches (for the Phase 7 ablation):
  pre      H1 missed-call, H2 utility-transaction overrides (before the model)
  trusted  R0 trusted-sender substring whitelist          ("header rules", with suffix)
  safety   R1 safety-advisory veto of spam               ("safety override")
  mobile   R2 mobile-number confidence gate, R3 mobile+keyword -> spam
  suffix   R4 TRAI -P/-G/-S/-T suffix rules              ("header rules")
There is no URL/link rule in the shipped app.
Variant (NOT shipped, Phase 10): "trusted_nc" = R0 that keeps caution as well as promo (use instead of "trusted").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ALL_LAYERS = frozenset({"pre", "trusted", "safety", "mobile", "suffix"})

TRUSTED_SENDERS = [
    "HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YESBNK", "BOB", "IDFC", "PNB",
    "LENSKT", "AMAZON", "FLIPKART", "MYNTRA", "ZOMATO", "SWIGGY",
    "JIO", "MYJIO", "AIRTEL", "AIRINF", "VI", "VODAFONE", "IDEA", "BSNL",
    "NSE", "BSE", "CDSL", "NSDL", "ITDEPT", "GOV",
    "IRCTC", "CRIS", "MSEDCL", "MAHADISCOM", "ADHAR", "UIDAI", "VAHAN", "EPFO",
]
SAFETY_PHRASES = ["beware", "security alert", "do not share", "report to"]
SUSPICIOUS_KEYWORDS = ["otp", "parcel", "verify", "pin", "blocked", "account", "kyc", "pan", "update", "refund"]
MOBILE_RE = re.compile(r"^(\+91|91|0)?[6-9]\d{9}$")


@dataclass
class Result:
    label: str
    confidence: float
    rule: str  # which rule decided ("model" if none fired)


def pre_model_override(text: str, layers=ALL_LAYERS) -> Result | None:
    """scamDetector.ts classifyText heuristics, applied before the model."""
    if "pre" not in layers:
        return None
    t = text.lower()
    if ("missed call from" in t or "last missed call" in t) and any(
            k in t for k in ("jio", "airtel", "vi", "bsnl", "vodafone")):
        return Result("ham", 1.0, "H1_missed_call")
    if ("processed" in t or "bill generated" in t or "payment received" in t) and any(
            k in t for k in ("airtel", "jio", "act fibernet", "tatasky")):
        return Result("ham", 0.98, "H2_utility_txn")
    return None


def is_mobile_number(address: str) -> bool:
    return bool(MOBILE_RE.match(re.sub(r"[\s-]", "", address)))


def final_classification(address: str, body: str, ai_label: str, ai_conf: float, layers=ALL_LAYERS) -> Result:
    """SMSReader.tsx getFinalClassification."""
    text = body.lower()
    sender = address.upper()
    if ("trusted" in layers or "trusted_nc" in layers) and any(b in sender for b in TRUSTED_SENDERS):
        if ai_label == "promo":
            return Result(ai_label, ai_conf, "R0_trusted_keep_promo")
        if "trusted_nc" in layers and "trusted" not in layers and ai_label == "caution":
            # NOT in the shipped app: Phase-10 variant where R0 never overrides caution
            return Result(ai_label, ai_conf, "R0nc_trusted_keep_caution")
        return Result("ham", 0.99, "R0_trusted_veto")
    if "safety" in layers and ai_label == "spam" and any(p in text for p in SAFETY_PHRASES):
        return Result("ham", 0.95, "R1_safety_advisory")
    mobile = is_mobile_number(address)
    if "mobile" in layers:
        if ai_label == "spam" and ai_conf < 0.985 and mobile:
            return Result("caution", ai_conf, "R2_mobile_gate")
        if mobile and any(k in text for k in SUSPICIOUS_KEYWORDS):
            return Result("spam", 1.0, "R3_mobile_keyword")
    if "suffix" in layers:
        if address.endswith("-P") or address.endswith(" P"):
            return Result("promo", 1.0, "R4_suffix_P")
        if address.endswith("-G") or address.endswith(" G"):
            return Result("ham", 1.0, "R4_suffix_G")
        if (address.endswith("-S") or address.endswith("-T")) and ai_label == "spam" and ai_conf < 0.999:
            return Result("ham", 0.95, "R4_suffix_ST_downgrade")
    return Result(ai_label, ai_conf, "model")


def pipeline(address: str, body: str, model_fn, layers=ALL_LAYERS) -> Result:
    """Full app order: pre-model heuristics -> model -> post rules. model_fn(text) -> (label, conf)."""
    # In the app, classifyText returns the heuristic result (if any) instead of the model output, and
    # getFinalClassification is then applied to it like any other AI result.
    pre = pre_model_override(body, layers)
    ai = pre if pre is not None else Result(*model_fn(body), "model")
    out = final_classification(address, body, ai.label, ai.confidence, layers)
    if out.rule == "model":
        out.rule = ai.rule
    return out
