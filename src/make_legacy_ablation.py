"""POST-HOC same-recipe data ablation: build the legacy training sets (local only, counts printed, no text).

L0 "legacy as-is": every row of data_raw/balanced_training_data.csv (the legacy inbox file behind v7.3), with its
   ORIGINAL legacy label, clones and generated rows kept, scrubbed with the same local scrubber as the pool.
L1 "legacy cleaned": the same messages after the rebuild's cleaning = the inbox-origin rows of the final train
   split (exact duplicates merged, Claude/owner relabels applied, suspected-generated rows dropped).
Leakage guard: a legacy row is dropped from L0 if its message (normalized text) maps to a pool item that is not in
the final train split (i.e. it is in val, test, a test cluster, or excluded). Both sets are trained with the final
recipe (src/train.py, same flags, synthetic rows NOT capped for L0) and evaluated once, post-hoc, on the frozen test.
Writes data/legacy_L0.csv and data/legacy_L1.csv (gitignored) and results/legacy_ablation_data.json.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scrub as S  # noqa: E402
from build_pool import load_sources  # noqa: E402
from common import ROOT, new_run, normalize, save_result  # noqa: E402


def main():
    run = new_run("make_legacy_ablation", {})
    src = load_sources()
    S.set_common_words(src["raw_text"])  # same vocabulary statistics as build_pool
    S.mask_known_names  # noqa: B018 (scrub() applies the known-names mask when present)
    inbox = src[src["source"] == "inbox"].copy()
    cnt = Counter()
    inbox["text"] = [S.scrub(t, cnt) for t in inbox["raw_text"]]
    inbox["norm"] = [normalize(t) for t in inbox["text"]]
    inbox = inbox[inbox["norm"].str.len() > 0]
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)[["item_id", "norm", "cluster_id"]]
    tr = pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False)
    train_ids = set(tr["item_id"])
    m = inbox.merge(pool, on="norm", how="left")
    unmatched = int(m["item_id"].isna().sum())
    m["in_train"] = m["item_id"].isin(train_ids)
    l0 = m[m["in_train"]].copy()
    l0 = pd.DataFrame({"item_id": l0["item_id"] + "_r" + l0["src_row"].astype(str), "cluster_id": l0["cluster_id"],
                       "text": l0["text"], "final_label": l0["orig_label"].str.strip(), "is_synthetic": "False",
                       "label_source": "legacy_original"})
    l1 = tr[tr["source"] == "inbox"].copy()
    l1 = l1[l1["is_synthetic"].astype(str) != "True"]
    out = ROOT / "data"
    l0.to_csv(out / "legacy_L0.csv", index=False)
    l1.to_csv(out / "legacy_L1.csv", index=False)
    res = {"legacy_rows_total": int(len(inbox)), "unmatched_to_pool": unmatched,
           "dropped_not_in_train_split": int((~m["in_train"]).sum()),
           "L0_rows": int(len(l0)), "L0_labels": l0["final_label"].value_counts().to_dict(), "L0_unique_texts": int(l0["text"].nunique()),
           "L1_rows": int(len(l1)), "L1_labels": l1["final_label"].value_counts().to_dict(),
           "note": "L0 keeps clones and generated rows with original legacy labels (is_synthetic forced False so nothing is capped)"}
    save_result(run, res, "legacy_ablation_data.json")
    import json
    print(json.dumps(res, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
