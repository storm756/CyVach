"""Phase 1: build data/pool.csv from approved sources (scrubbed, normalized, deduped, clustered).

Sources (Gate 1, 2026-10-17): owner inbox file (scrubbed), UCI SMS Spam Collection,
Mishra & Soni 2022 (Mendeley f45bkkt8pr v1). Excluded: financial_fraud_*,
Indian_sms_scam (licence unknown), spam.csv (UCI re-upload), CEAS_08 (email),
ultimate_stress_test_400 (legacy).

Exact duplicates (normalized text) are merged into one row; near-duplicates
(MinHash char-5, t=0.8) share a cluster_id, which is the unit for splitting.
Prints counts only.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_inventory import near_dup_clusters  # noqa: E402
from common import RAW, ROOT, load_csv, new_run, normalize, save_result, script_of  # noqa: E402
from compare_uci import load_uci  # noqa: E402
import scrub as S  # noqa: E402

SOURCE_PRIORITY = {"export2026": 0, "friend2026": 1, "friend2_2026": 2, "friend3_2026": 3, "inbox": 4, "uci": 5, "mishra_soni": 6}
FRIEND_FILES = {"friend2026": "Friend_sms_export_2026-10-17.csv",      # contributor 2, received 2026-10-18
                "friend2_2026": "2sms_export_2026-10-18.csv",         # contributor 3, received 2026-10-18
                "friend3_2026": "3sms_export_2026-10-18.csv"}         # contributor 4, received 2026-10-18 (Label = app prediction)
UNLABELLED_SOURCES = ("export2026", *FRIEND_FILES)
EXPORT_FILE = "sms_export_2026-01-20.csv"  # owner phone export 2026-10-18; Label column = app prediction, not ground truth
# original label -> proposed label in our taxonomy, plus whether it needs review
LABEL_MAP = {
    "inbox": {"ham": ("ham", 0), "spam": ("spam", 1), "promo": ("promo", 0), "caution": ("caution", 0)},
    "uci": {"ham": ("ham", 0), "spam": ("spam", 1)},
    "mishra_soni": {"ham": ("ham", 0), "smishing": ("spam", 0), "spam": ("spam", 1)},
    "export2026": {},  # unlabelled: every row needs a label
    "friend2026": {}, "friend2_2026": {}, "friend3_2026": {},
}
GEN_SENDER = r"^VM-[A-Z0-9]+-[A-Z]$"


def suspected_generated(addr: str) -> bool:
    """Heuristic (00_audit.md section 10): non-DLT 'VM-<WORD>-X' sender with middle != 6 chars or '-C' suffix."""
    import re
    a = str(addr).strip()
    if not re.match(GEN_SENDER, a):
        return False
    return len(a.split("-")[1]) != 6 or a.endswith("-C")


def load_sources() -> pd.DataFrame:
    rows = []
    inbox = load_csv(RAW / "balanced_training_data.csv")
    for i, r in inbox.iterrows():
        rows.append({"source": "inbox", "src_row": i, "raw_text": r["Body"], "raw_sender": r["Address"], "orig_label": r["Label"].strip(),
                     "is_synthetic": suspected_generated(r["Address"]), "app_pred": ""})
    ex = load_csv(RAW / EXPORT_FILE)
    for i, r in ex.iterrows():
        rows.append({"source": "export2026", "src_row": i, "raw_text": r["Body"], "raw_sender": r["Address"], "orig_label": "unlabelled",
                     "is_synthetic": False, "app_pred": r["Label"].strip()})
    for src, fn in FRIEND_FILES.items():
        fr = load_csv(RAW / fn)
        for i, r in fr.iterrows():
            rows.append({"source": src, "src_row": i, "raw_text": r["Body"], "raw_sender": r["Address"], "orig_label": "unlabelled",
                         "is_synthetic": False, "app_pred": r["Label"].strip()})
    uci, _ = load_uci()
    for i, (lab, txt) in enumerate(uci):
        rows.append({"source": "uci", "src_row": i, "raw_text": txt, "raw_sender": "", "orig_label": lab, "is_synthetic": False, "app_pred": ""})
    ms = load_csv(RAW / "mishra_soni" / "Dataset_5971.csv")
    for i, r in ms.iterrows():
        rows.append({"source": "mishra_soni", "src_row": i, "raw_text": r["TEXT"], "raw_sender": "", "orig_label": r["LABEL"].strip().lower(), "is_synthetic": False, "app_pred": ""})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    run = new_run("build_pool", vars(args))
    df = load_sources()
    n_common = S.set_common_words(df["raw_text"])
    cnt = Counter()
    df["text"] = [S.scrub(t, cnt) for t in df["raw_text"]]
    df["sender"] = [S.scrub(s) if s else "" for s in df["raw_sender"]]
    df["sender_kind"] = [S.sender_kind(s) if s else "none" for s in df["raw_sender"]]
    df = df.drop(columns=["raw_text", "raw_sender"])
    res = S.residual_counts(df["text"])
    df["norm"] = [normalize(t) for t in df["text"]]
    df["script"] = [script_of(t) for t in df["text"]]
    df["empty"] = df["norm"].str.len() == 0
    df = df[~df["empty"]].drop(columns=["empty"])

    mapped = [("", 1) if s in UNLABELLED_SOURCES else LABEL_MAP[s].get(l) for s, l in zip(df["source"], df["orig_label"])]
    assert all(m is not None for m in mapped), Counter((s, l) for (s, l), m in zip(zip(df["source"], df["orig_label"]), mapped) if m is None)
    df["label"] = [m[0] for m in mapped]
    df["needs_review"] = [m[1] for m in mapped]

    # exact-duplicate merge on normalized text (keep highest-priority source)
    df["prio"] = df["source"].map(SOURCE_PRIORITY)
    df["src_key"] = df["source"] + ":" + df["src_row"].astype(str)
    df = df.sort_values(["prio", "src_row"]).reset_index(drop=True)
    g = df.groupby("norm", sort=False)
    agg = g.agg(dup_count=("source", "size"),
                dup_sources=("source", lambda s: "|".join(sorted(set(s)))),
                orig_labels=("orig_label", lambda s: "|".join(sorted(set(s)))),
                all_synthetic=("is_synthetic", "all"),
                member_keys=("src_key", lambda s: "|".join(s)))
    first = g.head(1).set_index("norm")
    pool = first.join(agg).reset_index()
    # Mishra & Soni re-annotated UCI spam into smishing vs spam: a smishing vote resolves to scam.
    smish = pool["orig_labels"].str.split("|").apply(lambda s: "smishing" in s)
    pool.loc[smish, "label"] = "spam"
    pool.loc[smish, "needs_review"] = 0
    def mapped_set(row):
        labs = set()
        for src in row["dup_sources"].split("|"):
            for ol in row["orig_labels"].split("|"):
                if ol in LABEL_MAP[src]:
                    labs.add(LABEL_MAP[src][ol][0])
        return labs
    pool["label_conflict"] = [int(len(mapped_set(r)) > 1) for _, r in pool.iterrows()]
    # unlabelled export rows that duplicate a labelled source inherit that label (still needs review)
    for i, r in pool[pool["label"] == ""].iterrows():
        labs = mapped_set(r)
        if len(labs) == 1:
            pool.at[i, "label"] = labs.pop()
    pool["is_synthetic"] = pool["all_synthetic"].astype(bool)  # synthetic only if every duplicate is

    pool["cluster_id"] = near_dup_clusters(pool["norm"].tolist(), args.threshold)
    pool["cluster_id"] = pool["cluster_id"].rank(method="dense").astype(int)
    PFX = {'export2026': 'ex', 'friend2026': 'fr', 'friend2_2026': 'f2', 'friend3_2026': 'f3'}
    pool["member_ids"] = ["|".join(f"{PFX.get(k.split(':')[0], k[:2])}{int(k.split(':')[1]):05d}" for k in mk.split("|")) for mk in pool["member_keys"]]
    pool["item_id"] = [f"{ {'export2026': 'ex', 'friend2026': 'fr', 'friend2_2026': 'f2', 'friend3_2026': 'f3'}.get(s, s[:2]) }{r:05d}" for s, r in zip(pool["source"], pool["src_row"])]
    pool["synthetic_basis"] = pool["is_synthetic"].map({True: "heuristic: non-DLT VM-<WORD>-X sender (00_audit s10)", False: ""})
    pool["seed_id"] = ""
    pool["label_source"] = "original"
    pool["provenance_note"] = pool["source"].map({"inbox": "pseudonymized+balanced; partly possibly generated (unverified)", "uci": "", "mishra_soni": "",
                                                    "export2026": "owner phone export via app, received 2026-10-18; app Label = prediction",
                                                    "friend2026": "friend phone export via app, received 2026-10-18; app Label = prediction",
                                                    "friend2_2026": "friend phone export via app, received 2026-10-18; app Label = prediction",
                                                    "friend3_2026": "friend phone export via app, received 2026-10-18; app Label = prediction"})
    cols = ["item_id", "source", "src_row", "text", "norm", "sender", "sender_kind", "orig_label", "orig_labels", "label",
            "label_source", "needs_review", "label_conflict", "dup_count", "dup_sources", "cluster_id", "is_synthetic",
            "seed_id", "synthetic_basis", "app_pred", "script", "provenance_note", "member_ids"]
    pool = pool[cols]
    out = ROOT / "data" / "pool.csv"
    out.parent.mkdir(exist_ok=True)
    pool.to_csv(out, index=False, encoding="utf-8")

    summary = {
        "rows_loaded": int(len(df)), "common_words": n_common, "scrub_replacements": dict(cnt),
        "residual_after_scrub": res,
        "pool_rows": int(len(pool)), "clusters": int(pool["cluster_id"].nunique()),
        "rows_by_source": pool["source"].value_counts().to_dict(),
        "rows_by_source_label": {f"{s}/{l or 'UNLABELLED'}": int(n) for (s, l), n in pool.groupby(["source", "label"]).size().items()},
        "synthetic_by_source_label": {f"{s}/{l or 'UNLABELLED'}": int(n) for (s, l), n in pool[pool.is_synthetic].groupby(["source", "label"]).size().items()},
        "clusters_by_label": pool.groupby("label")["cluster_id"].nunique().to_dict(),
        "needs_review_rows": int(pool["needs_review"].sum()),
        "cross_source_exact_dups": int((pool["dup_sources"].str.contains(r"\|")).sum()),
        "label_conflict_rows": int(pool["label_conflict"].sum()),
        "clusters_spanning_sources": int((pool.groupby("cluster_id")["source"].nunique() > 1).sum()),
        "clusters_spanning_labels": int((pool.groupby("cluster_id")["label"].nunique() > 1).sum()),
        "script_mix": pool["script"].value_counts().to_dict(),
    }
    p = save_result(run, summary, "build_pool.json")
    import json
    print(json.dumps(summary, indent=1))
    print("saved", out, p, run["run_id"])


if __name__ == "__main__":
    main()
