"""Overlap between the UCI SMS Spam Collection and Mishra & Soni (2022): share of Mishra rows whose normalized text
equals a UCI row (exact match after src/common.normalize). Output: results/uci_mishra_overlap.json"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pool import load_sources  # noqa: E402
from common import new_run, normalize, save_result  # noqa: E402

run = new_run("uci_mishra_overlap", {})
s = load_sources()
s["norm"] = [normalize(t) for t in s["raw_text"]]
uci = set(s[s["source"] == "uci"]["norm"])
m = s[s["source"] == "mishra_soni"]
shared = m["norm"].isin(uci)
res = {"uci_rows": int((s["source"] == "uci").sum()), "mishra_rows": int(len(m)), "mishra_exact_in_uci": int(shared.sum()),
       "share": float(shared.mean()), "mishra_unique_not_in_uci": int((~shared).sum())}
save_result(run, res, "uci_mishra_overlap.json")
print(res, run["run_id"])
