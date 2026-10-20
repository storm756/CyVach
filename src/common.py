"""Shared helpers: paths, CSV loading, text normalization, run logging.

Nothing in this module prints message text. Callers that handle real-inbox
data (balanced_training_data.csv) must only print counts.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

LABELS = ["ham", "spam", "promo", "caution"]  # fixed order, ids 0-3

# Files whose rows are real messages from people's phones: counts only until scrubbed.
PRIVATE_FILES = {"balanced_training_data.csv", "sms_export_2026-01-20.csv", "Friend_sms_export_2026-10-17.csv", "2sms_export_2026-10-18.csv", "3sms_export_2026-10-18.csv"}

# file -> (text column, label column, sender column or None)
FILE_SCHEMA = {
    "balanced_training_data.csv": ("Body", "Label", "Address"),
    "financial_fraud_training_10000.csv": ("text", "label", "sender"),
    "financial_fraud_newlybalanced_10000.csv": ("text", "label", "sender"),
    "Indian_sms_scam.csv": ("Message", "Actual_Label", None),
    "spam.csv": ("Message", "Category", None),
    "ultimate_stress_test_400.csv": ("Body", "Label", None),
    "CEAS_08.csv": ("body", "label", "sender"),
    "sms_export_2026-01-20.csv": ("Body", "Label", "Address"),
    "Friend_sms_export_2026-10-17.csv": ("Body", "Label", "Address"),
    "2sms_export_2026-10-18.csv": ("Body", "Label", "Address"),
    "3sms_export_2026-10-18.csv": ("Body", "Label", "Address"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:com|in|me|ly|co|org|net|xyz|info|link|app|io|gl|cc|top|site|online|club|biz|gov\.in|co\.in)\b(?:/\S*)?",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
DIGITS_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Leakage-check normalization: lowercase, mask URLs/emails/digits, collapse whitespace."""
    t = str(text)
    t = EMAIL_RE.sub(" <email> ", t)
    t = URL_RE.sub(" <url> ", t)
    t = t.lower()
    t = DIGITS_RE.sub("0", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def script_of(text: str) -> str:
    """Coarse script/language-mix bucket by Unicode block."""
    t = str(text)
    has = lambda lo, hi: any(lo <= ord(c) <= hi for c in t)
    dev = has(0x0900, 0x097F)
    guj = has(0x0A80, 0x0AFF)
    other_indic = has(0x0980, 0x0A7F) or has(0x0B00, 0x0DFF)
    latin = any(("a" <= c.lower() <= "z") for c in t)
    if guj:
        return "gujarati" + ("+latin" if latin else "")
    if dev:
        return "devanagari" + ("+latin" if latin else "")
    if other_indic:
        return "other_indic"
    if any(ord(c) > 0x024F and not (0x2000 <= ord(c) <= 0x2BFF) and not (0x1F000 <= ord(c) <= 0x1FFFF) and c != "₹" for c in t):
        return "other_nonlatin"
    return "latin" if latin else "no_letters"


def git_rev() -> str:
    cv = ROOT / "CODE_VERSION"  # written into the Colab bundle (no git there)
    if cv.exists() and not (ROOT / ".git").exists():
        return cv.read_text(encoding="utf-8").strip()
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        rev = out.stdout.strip() or "nocommit"
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def new_run(name: str, config: dict | None = None) -> dict:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return {"run_id": f"{name}-{ts}", "script": name, "started": ts, "code_version": git_rev(), "config": config or {}}


def save_result(run: dict, payload: dict, fname: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    out = {**run, "result": payload}
    p = RESULTS / fname
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with open(LOGS / "runs.csv", "a", encoding="utf-8") as f:
        f.write(f"{run['run_id']},{run['script']},{run['code_version']},{fname},{json.dumps(run['config'])!r}\n")
    return p
