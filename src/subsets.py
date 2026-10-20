"""Pre-registered evaluation subsets and load-time filters (defined 2026-10-18, before any model was run on test).

native_script: the message contains any Indic-script character (U+0900-U+0DFF: Devanagari, Bengali,
Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala). In the deployed app such messages
are translated to English by ML Kit before the model runs, so the model never sees native script:
  - train/val: native_script rows are EXCLUDED from model training and selection.
  - test: the MAIN metric is computed on test minus native_script rows; native_script rows are reported
    separately as "untranslated native-script" (what happens if on-device translation fails/is skipped).
Romanized Hindi/Marathi ("Hinglish") is NOT native_script and stays in all splits (ML Kit may not translate it).
"""
from __future__ import annotations

import re

import pandas as pd

INDIC = re.compile(r"[ऀ-෿]")


def is_native_script(text: str) -> bool:
    return bool(INDIC.search(str(text)))


def load_split(path, drop_native: bool) -> pd.DataFrame:
    d = pd.read_csv(path, keep_default_na=False)
    d["native_script"] = d["text"].map(is_native_script)
    return d[~d["native_script"]].copy() if drop_native else d
