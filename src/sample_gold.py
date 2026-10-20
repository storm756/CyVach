"""Phase 2: sample test-set candidates (one item per near-dup cluster, stratified by proxy class).

Proxy class = original label, except original `spam` that v7.3 calls `promo` -> proxy `promo`
(UCI/Mishra "spam" mixes marketing and scams). True labels are unknown until labelled.
Each candidate gets an inverse-inclusion weight so metrics can be re-weighted to the
pool's natural prevalence. The whole cluster of every candidate is reserved for test.

Output: annotation/gold_candidates.csv (item_id, cluster_id, source, sender, text, stratum, weight).
It contains NO labels or model predictions: the labelling step must stay blind to them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, save_result, sha256_file  # noqa: E402

TARGET = {"ham": 140, "spam": 142, "promo": 100, "caution": 110}  # caution raised 90->110 on 2026-10-18 (top-up: only 70 confirmed caution)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--keep", default="", help="previous candidates CSV whose (real) items are kept")
    ap.add_argument("--source-quota", nargs="*", default=[], help="e.g. friend2026:40")
    args = ap.parse_args()
    run = new_run("sample_gold", {**vars(args), "target": TARGET})
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)
    preds = pd.read_csv(ROOT / "data" / "v73_preds.csv")
    df = pool.merge(preds[["item_id", "v73_pred"]], on="item_id")
    # real items only (CLAUDE.md rule: synthetic never in test); unlabelled rows use the v7.3 prediction as proxy
    df = df[~df["is_synthetic"].astype(str).str.lower().eq("true")]
    df["stratum"] = df["label"].where(df["label"] != "", df["v73_pred"])
    df.loc[(df["label"] == "spam") & (df["v73_pred"] == "promo"), "stratum"] = "promo"
    rng = np.random.default_rng(args.seed)
    keep_ids = set()
    if args.keep:
        keep_ids = set(pd.read_csv(ROOT / args.keep)["item_id"])
    df["kept"] = df["item_id"].isin(keep_ids)
    # one representative per cluster (a previously sampled item wins), else random member, seeded
    df = df.sample(frac=1, random_state=args.seed).sort_values("kept", ascending=False, kind="stable").drop_duplicates("cluster_id")
    picks = []
    stats = {}
    for s, n in TARGET.items():
        g = df[df["stratum"] == s]
        kept = g[g["kept"]]
        rest = g[~g["kept"]]
        k_new = max(0, min(n - len(kept), len(rest)))
        chosen = pd.concat([kept, rest.sample(n=k_new, random_state=int(rng.integers(1 << 31)))])
        chosen = chosen.assign(weight=len(g) / max(1, len(chosen)))
        picks.append(chosen)
        stats[s] = {"stratum_clusters": int(len(g)), "kept": int(len(kept)), "new": int(k_new), "sampled": int(len(chosen)),
                    "by_source": chosen["source"].value_counts().to_dict()}
    gold = pd.concat(picks)
    # source quota (e.g. a second contributor's phone): extra clusters from that source, proportional to proxy strata
    for spec in args.source_quota:
        src, n = spec.split(":"); n = int(n)
        cand = df[(df["source"] == src) & ~df["cluster_id"].isin(gold["cluster_id"])]
        cand = cand.sample(frac=1, random_state=args.seed)
        extra = pd.concat([g.head(max(1, round(n * len(g) / max(1, len(cand)))))
                           for _, g in cand.groupby("stratum")]) if len(cand) else cand
        gold = pd.concat([gold, extra.head(n)])
        stats[f"quota_{src}"] = {"added": int(min(n, len(extra))), "by_stratum": extra.head(n)["stratum"].value_counts().to_dict()}
    # inverse-inclusion weight per stratum, recomputed on the final sample
    sizes = df.groupby("stratum")["cluster_id"].nunique()
    gold["weight"] = gold["stratum"].map(lambda st: sizes[st] / (gold["stratum"] == st).sum())
    gold = gold.sample(frac=1, random_state=args.seed)  # shuffle: strata not visible by order
    out = gold[["item_id", "cluster_id", "source", "sender", "text", "stratum", "weight"]]
    path = ROOT / "annotation" / "gold_candidates.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    reserved = pool[pool["cluster_id"].isin(set(gold["cluster_id"]))]
    summary = {"candidates": int(len(out)), "strata": stats,
               "reserved_clusters": int(gold["cluster_id"].nunique()), "reserved_pool_rows": int(len(reserved)),
               "script_mix": gold["script"].value_counts().to_dict() if "script" in gold else None,
               "sha256": sha256_file(path)}
    save_result(run, summary, "sample_gold.json")
    import json
    print(json.dumps(summary, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
