"""Gate 2: freeze the test set and make a cluster-aware, stratified train/val split.

- test  = all test candidates (annotation/gold_candidates.csv) with their final labels
          (human_verified where the owner checked them, otherwise claude_blind). Frozen + SHA-256.
- val   = 15% of the REAL train/val clusters, stratified by label (natural distribution, no synthetic rows)
- train = the rest of the real clusters + all synthetic rows (synthetic never in val/test)
Near-duplicate members of test clusters are dropped from train/val. Overlap checks must all be 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, normalize, save_result, sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run = new_run("make_splits", vars(args))
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)
    lab = pd.read_csv(ROOT / "data" / "labels.csv", keep_default_na=False)
    df = pool[["item_id", "text", "sender", "sender_kind", "source", "script"]].merge(lab.drop(columns=["source"]), on="item_id")
    df["is_synthetic"] = df["is_synthetic"].astype(str) == "True"

    frozen = ROOT / "data" / "test.csv"
    FROZEN_SHA = "7fa0c6c4bdaf604a22c7185e39d3c605a266fa7ef0b5072a5c85527ce04e201b"  # Gate 2, make_splits-20261018-022157
    if frozen.exists():
        assert sha256_file(frozen) == FROZEN_SHA, "frozen test.csv has changed!"
        test = pd.read_csv(frozen, keep_default_na=False)
        test["is_synthetic"] = test["is_synthetic"].astype(str) == "True"
        # exclude from train/val anything in the same current-pool cluster as a test item, or with identical normalized text
        tclusters = set(pool.loc[pool["item_id"].isin(test["item_id"]), "cluster_id"])
        tnorm = {normalize(t) for t in test["text"]}
        df = df[~(df["cluster_id"].isin(tclusters) & ~df["item_id"].isin(test["item_id"]))]
        df = df[~df["item_id"].isin(test["item_id"])]
        df = df[~df["text"].map(normalize).isin(tnorm)]
    else:
        test = df[df["split_hint"] == "test"].copy()
        test = test[test["label_source"].isin(["human_verified", "claude_blind"])]
    tv = df[(df["split_hint"] == "trainval")].copy()

    real = tv[~tv["is_synthetic"]]
    # one label per cluster (majority) for stratification
    clab = real.groupby("cluster_id")["final_label"].agg(lambda s: s.value_counts().index[0])
    val_clusters = set()
    for lbl, g in clab.groupby(clab):
        val_clusters |= set(g.sample(frac=args.val_frac, random_state=args.seed).index)
    val = real[real["cluster_id"].isin(val_clusters)].copy()
    train = tv[~tv["cluster_id"].isin(val_clusters)].copy()

    cols = ["item_id", "cluster_id", "source", "sender", "sender_kind", "script", "text", "final_label",
            "label_source", "is_synthetic", "confidence", "borderline"]
    out = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        p = ROOT / "data" / f"{name}.csv"
        if not (name == "test" and frozen.exists()):
            d[cols].sort_values("item_id").to_csv(p, index=False, encoding="utf-8")
        out[name] = {"rows": int(len(d)), "sha256": sha256_file(p), "labels": d["final_label"].value_counts().to_dict(),
                     "synthetic": int(d["is_synthetic"].sum()), "label_source": d["label_source"].value_counts().to_dict(),
                     "sources": d["source"].value_counts().to_dict()}
    norm = {n: set(normalize(t) for t in d["text"]) for n, d in (("train", train), ("val", val), ("test", test))}
    cmap = dict(zip(pool["item_id"], pool["cluster_id"]))  # current-pool cluster ids for every split
    clus = {n: {cmap.get(i) for i in d["item_id"]} - {None} for n, d in (("train", train), ("val", val), ("test", test))}
    ids = {n: set(d["item_id"]) for n, d in (("train", train), ("val", val), ("test", test))}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{a}-{b}"] = {"clusters": len(clus[a] & clus[b]), "normalized_text": len(norm[a] & norm[b]), "item_ids": len(ids[a] & ids[b])}
    assert all(v == 0 for o in overlaps.values() for v in o.values()), overlaps
    out["overlaps"] = overlaps
    p = save_result(run, out, "splits.json")

    md = [f"# Data splits (Gate 2, frozen {run['started']})", "",
          f"Run `{run['run_id']}` (`src/make_splits.py --val-frac {args.val_frac} --seed {args.seed}`). "
          "Test labels: owner-checked where available (`human_verified`), otherwise Claude blind labels (`claude_blind`). "
          "**The test set is frozen: never used for tuning, selection or early stopping; each final model is run on it once.**", "",
          "| split | rows | SHA-256 | synthetic | labels |", "|---|---|---|---|---|"]
    for n in ("train", "val", "test"):
        o = out[n]
        md.append(f"| {n} | {o['rows']} | `{o['sha256']}` | {o['synthetic']} | {o['labels']} |")
    md += ["", "Overlap checks (must all be 0): " + str(overlaps), "",
           "Val is real-only with the natural class distribution; all synthetic rows are in train (cap per class applied at training time)."]
    notes = ROOT / "reports" / "SPLITS_notes.md"  # hand-written notes (re-splits, subset rule) survive regeneration
    if notes.exists():
        md.append(notes.read_text(encoding="utf-8"))
    (ROOT / "data" / "SPLITS.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "reports" / "SPLITS_hashes.md").write_text("\n".join(md), encoding="utf-8")  # committed copy (data/ is ignored)
    import json
    print(json.dumps(out, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
