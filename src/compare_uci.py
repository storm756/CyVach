"""Compare owner's spam.csv with the UCI SMS Spam Collection original (DOI 10.24432/C5CC84)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, load_csv, new_run, normalize, save_result, sha256_file  # noqa: E402


def load_uci():
    p = RAW / "uci_sms" / "SMSSpamCollection"
    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            lab, _, text = line.partition("\t")
            rows.append((lab, text))
    return rows, p


def main():
    run = new_run("compare_uci", {})
    uci, p = load_uci()
    own = load_csv(RAW / "spam.csv")
    own_rows = list(zip(own["Category"], own["Message"]))
    out = {
        "uci_sha256": sha256_file(p), "uci_rows": len(uci), "uci_labels": dict(Counter(l for l, _ in uci)),
        "own_rows": len(own_rows), "own_labels": dict(Counter(l for l, _ in own_rows)),
    }
    out["exact_same_sequence"] = uci == own_rows
    us, os_ = Counter(uci), Counter(own_rows)
    out["own_pairs_not_in_uci"] = sum((os_ - us).values())
    out["uci_pairs_not_in_own"] = sum((us - os_).values())
    un = Counter((l, normalize(t)) for l, t in uci)
    on = Counter((l, normalize(t)) for l, t in own_rows)
    out["normalized_own_not_in_uci"] = sum((on - un).values())
    out["normalized_uci_not_in_own"] = sum((un - on).values())
    out["position_mismatches"] = sum(1 for a, b in zip(uci, own_rows) if a != b)
    out["uci_exact_dup_texts"] = len(uci) - len({t for _, t in uci})
    out["uci_unique_normalized"] = len({normalize(t) for _, t in uci})
    p2 = save_result(run, out, "compare_uci.json")
    import json
    print(json.dumps(out, indent=1))
    print("saved", p2, run["run_id"])


if __name__ == "__main__":
    main()
