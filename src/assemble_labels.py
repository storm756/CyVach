"""Assemble data/labels.csv: one final label per pool item, with its source.

label_source values:
  human_verified   test item checked by the owner (accept or override) in src/check_app.py
  claude_blind     test item labelled by Claude without seeing original/model labels (not yet human-checked)
  claude_silver    train/val item labelled by Claude (disagreement, needs_review, or 10% agreement sample)
  original         train/val item keeping its source label (original label == v7.3 prediction, not sampled)
Prints counts only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, save_result  # noqa: E402


def main():
    run = new_run("assemble_labels", {})
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)
    gold = pd.read_csv(ROOT / "annotation" / "gold_candidates.csv", keep_default_na=False)
    cg = pd.read_csv(ROOT / "annotation" / "claude_gold_pass1.csv").set_index("item_id")
    cs = pd.read_csv(ROOT / "annotation" / "claude_silver_pass1.csv").set_index("item_id")
    chk_path = ROOT / "annotation" / "check_results.csv"
    chk = pd.read_csv(chk_path).drop_duplicates("item_id", keep="last").set_index("item_id") if chk_path.exists() else None

    gold_ids = set(gold["item_id"])
    # test clusters are recomputed from the CURRENT pool (cluster ids change when the pool is rebuilt)
    gold_clusters = set(pool.loc[pool["item_id"].isin(gold_ids), "cluster_id"])
    rows = []
    for _, r in pool.iterrows():
        i = r["item_id"]
        split_hint = "test" if r["cluster_id"] in gold_clusters else "trainval"
        conf, border = "", ""
        if i in gold_ids:
            lab, src = cg.at[i, "claude_label"], "claude_blind"
            conf, border = cg.at[i, "claude_conf"], cg.at[i, "claude_borderline"]
            if chk is not None and i in chk.index:
                lab, src = chk.at[i, "final_label"], "human_verified"
                border = int(chk.at[i, "borderline"])
        elif split_hint == "test":
            lab, src = "", "test_cluster_member"  # same near-dup cluster as a test item: excluded from train/val
        elif i in cs.index or any(m in cs.index for m in str(r.get("member_ids", "")).split("|")):
            j = i if i in cs.index else next(m for m in str(r["member_ids"]).split("|") if m in cs.index)
            lab, src = cs.at[j, "claude_label"], "claude_silver"
            conf, border = cs.at[j, "claude_conf"], cs.at[j, "claude_borderline"]
        else:
            lab, src = r["label"], "original"
        rows.append({"item_id": i, "cluster_id": r["cluster_id"], "source": r["source"], "is_synthetic": r["is_synthetic"],
                     "split_hint": split_hint, "final_label": lab, "label_source": src, "confidence": conf, "borderline": border})
    out = pd.DataFrame(rows)
    missing = out[(out["final_label"] == "") & (out["label_source"] != "test_cluster_member")]
    assert missing.empty, f"{len(missing)} pool items have no label"
    out.to_csv(ROOT / "data" / "labels.csv", index=False)
    lab = out[out["label_source"] != "test_cluster_member"]
    summary = {
        "rows": int(len(out)),
        "by_split_and_source": {f"{a}/{b}": int(n) for (a, b), n in lab.groupby(["split_hint", "label_source"]).size().items()},
        "test_label_counts": lab[lab.split_hint == "test"]["final_label"].value_counts().to_dict(),
        "trainval_label_counts": lab[lab.split_hint == "trainval"]["final_label"].value_counts().to_dict(),
        "trainval_synthetic_by_label": lab[(lab.split_hint == "trainval") & (lab.is_synthetic.astype(str) == "True")]["final_label"].value_counts().to_dict(),
        "test_cluster_members_excluded": int((out["label_source"] == "test_cluster_member").sum()),
        "silver_labels_not_in_pool": int(len(set(cs.index) - set(pool["item_id"]))),
    }
    save_result(run, summary, "assemble_labels.json")
    import json
    print(json.dumps(summary, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
