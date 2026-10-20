"""Build data/known_names.txt (never committed, never printed): personal names that the app's
export left next to its [NAME] placeholder, plus capitalised neighbours of those names, plus
the owner's first name (seed). scrub.py masks these tokens anywhere (case-insensitive).
Prints counts only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, ROOT, load_csv  # noqa: E402
from scrub import NAME_STOP  # noqa: E402

# Seed names (owner and close contacts) live in a LOCAL, untracked file: data/name_seeds.txt
SEED_FILE = ROOT / "data" / "name_seeds.txt"
SEED = {l.strip().lower() for l in SEED_FILE.read_text(encoding="utf-8").splitlines() if l.strip()} if SEED_FILE.exists() else set()
FILES = ["sms_export_2026-01-20.csv", "balanced_training_data.csv", "Friend_sms_export_2026-10-17.csv",
         "2sms_export_2026-10-18.csv", "3sms_export_2026-10-18.csv"]
CAP = re.compile(r"^(?:[A-Z][a-z]{2,}|[A-Z]{3,})$")


def main():
    texts = [t for f in FILES for t in load_csv(RAW / f)["Body"]]
    common_lower = {w for t in texts for w in re.findall(r"(?<![A-Za-z])[a-z]{3,}(?![A-Za-z])", t)}
    names = set(SEED)
    for t in texts:
        toks = re.findall(r"\[NAME\]|<NAME>|[A-Za-z]+", t)
        for i, tok in enumerate(toks[:-1]):
            if tok in ("[NAME]", "<NAME>"):
                j = i + 1
                while j < len(toks) and j <= i + 3 and CAP.match(toks[j]) and toks[j].lower() not in NAME_STOP:
                    names.add(toks[j].lower())
                    j += 1
    # neighbours of known names (e.g. surname after a first name), two rounds
    for _ in range(2):
        add = set()
        for t in texts:
            toks = re.findall(r"[A-Za-z]+", t)
            for i, tok in enumerate(toks):
                if tok.lower() in names:
                    for k in (i - 1, i + 1):
                        if 0 <= k < len(toks) and CAP.match(toks[k]) and toks[k].lower() not in NAME_STOP:
                            add.add(toks[k].lower())
        names |= add
    # drop ordinary words: anything seen in lower case in the corpus is not treated as a name
    names = {n for n in names if n not in common_lower or n in SEED}
    # drop brand words (from sender IDs) and ordinary English words (UCI corpus vocabulary)
    senders = {p.lower() for f in FILES for a in load_csv(RAW / f)["Address"] for p in re.split(r"[-\s]", a) if p.isalpha()}
    uci_words = {w.lower() for t in load_csv(RAW / "spam.csv")["Message"] for w in re.findall(r"[A-Za-z]{3,}", t)}
    names = {n for n in names if (n not in senders and n not in uci_words) or n in SEED}
    out = ROOT / "data" / "known_names.txt"
    out.write_text("\n".join(sorted(names)), encoding="utf-8")
    print(f"known names: {len(names)} -> {out}")


if __name__ == "__main__":
    main()
