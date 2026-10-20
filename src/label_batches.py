"""Make and ingest Claude label batches (silver labels; blind gold proposals).

make:   python src/label_batches.py make --set gold|silver [--batch 1000] [--agree-frac 0.10]
        Writes annotation/batches/<set>_NN.tsv with ONLY: id <TAB> sender <TAB> text.
        No original label and no model prediction is ever written to a batch file.
        silver = non-gold pool items where original label != v7.3 prediction, or needs_review,
                 plus a seeded stratified sample (agree-frac) of the remaining agreements.
ingest: python src/label_batches.py ingest --set gold|silver --pass 1|2
        Reads annotation/batches/<set>_NN.labels.tsv  (id, label, confidence 1-3, borderline 0/1, reason)
        and validates: every id exactly once, label in taxonomy, conf in 1..3, borderline in 0/1.
        Writes annotation/claude_<set>_pass<k>.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402

BATCH_DIR = ROOT / "annotation" / "batches"


def clean(s: str) -> str:
    return " ".join(str(s).split())


def make(args):
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)
    gold = pd.read_csv(ROOT / "annotation" / "gold_candidates.csv", keep_default_na=False)
    gold_clusters = set(gold["cluster_id"])
    if args.set == "gold":
        items = gold.sample(frac=1, random_state=args.seed)
    else:
        preds = pd.read_csv(ROOT / "data" / "v73_preds.csv")
        df = pool.merge(preds[["item_id", "v73_pred"]], on="item_id")
        df = df[~df["cluster_id"].isin(gold_clusters)]
        must = df[(df["label"] != df["v73_pred"]) | (df["needs_review"] == 1)]
        rest = df.drop(must.index)
        sample = rest.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=args.agree_frac, random_state=args.seed))
        items = pd.concat([must, sample]).sample(frac=1, random_state=args.seed)
        print(f"silver: disagreement/needs_review={len(must)}, agreement sample={len(sample)}, non-gold pool={len(df)}")
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    n_batches = int(np.ceil(len(items) / args.batch))
    for b in range(n_batches):
        part = items.iloc[b * args.batch:(b + 1) * args.batch]
        with open(BATCH_DIR / f"{args.set}_{b:02d}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            w.writerow(["id", "sender", "text"])
            for _, r in part.iterrows():
                w.writerow([r["item_id"], clean(r.get("sender", "")), clean(r["text"])])
    run = new_run("label_batches_make", vars(args))
    save_result(run, {"set": args.set, "items": int(len(items)), "batches": n_batches}, f"label_batches_{args.set}.json")
    print(f"{args.set}: {len(items)} items in {n_batches} batches -> {BATCH_DIR}")


def ingest(args):
    expected, got, errors = {}, {}, []
    for bf in sorted(BATCH_DIR.glob(f"{args.set}_[0-9][0-9].tsv")):
        ids = pd.read_csv(bf, sep="\t", keep_default_na=False, quoting=csv.QUOTE_MINIMAL)["id"].tolist()
        for i in ids:
            expected[i] = bf.name
        lf = bf.with_name(bf.stem + (f".labels.tsv" if args.pass_ == 1 else f".labels_pass{args.pass_}.tsv"))
        if not lf.exists():
            errors.append(f"missing {lf.name}")
            continue
        for line in lf.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("id\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                errors.append(f"{lf.name}: bad line {line[:40]!r}"); continue
            i, lab, conf, bl = parts[:4]
            reason = parts[4] if len(parts) > 4 else ""
            if i in got:
                errors.append(f"duplicate id {i}")
            if lab not in LABELS or conf not in {"1", "2", "3"} or bl not in {"0", "1"}:
                errors.append(f"{i}: invalid fields {lab},{conf},{bl}")
            got[i] = (lab, int(conf), int(bl), reason)
    missing = [i for i in expected if i not in got]
    extra = [i for i in got if i not in expected]
    print(f"expected={len(expected)} got={len(got)} missing={len(missing)} extra={len(extra)} errors={len(errors)}")
    for e in errors[:20]:
        print(" ", e)
    if missing or extra or errors:
        sys.exit(1)
    out = pd.DataFrame([(i, *v) for i, v in got.items()], columns=["item_id", "claude_label", "claude_conf", "claude_borderline", "claude_reason"])
    out.to_csv(ROOT / "annotation" / f"claude_{args.set}_pass{args.pass_}.csv", index=False)
    run = new_run("label_batches_ingest", {"set": args.set, "pass": args.pass_})
    save_result(run, {"n": len(out), "label_counts": out["claude_label"].value_counts().to_dict(),
                      "borderline": int(out["claude_borderline"].sum())}, f"claude_{args.set}_pass{args.pass_}.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--set", choices=["gold", "silver"], required=True)
    m.add_argument("--batch", type=int, default=1000); m.add_argument("--agree-frac", type=float, default=0.10)
    m.add_argument("--seed", type=int, default=7)
    g = sub.add_parser("ingest"); g.add_argument("--set", choices=["gold", "silver"], required=True)
    g.add_argument("--pass", dest="pass_", type=int, default=1)
    args = ap.parse_args()
    make(args) if args.cmd == "make" else ingest(args)


if __name__ == "__main__":
    main()
