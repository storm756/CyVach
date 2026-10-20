"""Phase 5 sweep on VAL only (the test set is not in the Colab bundle). Resumable: finished (config, seed) pairs are skipped.

Stage A (design factors, lr 3e-5, 4 epochs, max_len 128, no smoothing):
    balance {S0, S1} x digits {D0, D1, D2} x placeholders {U0, U1} x synthetic-ratio {1.0}  + real-only (ratio 0) for S0/D1/U1
Stage B (hyper-parameters on the Stage-A winner): lr {2e-5, 3e-5, 5e-5} x max_len {64, 128} x label smoothing {0, 0.05}, 5 epochs
    (best epoch <= 5 chosen on val, which covers the brief's "epochs 3-5").
Each config runs with seeds 13 and 21; configs are ranked by mean val macro-F1.

Pre-registered selection rule (written before any sweep result was seen, 2026-10-18):
  1. Rank by mean val macro-F1 over the 2 seeds.
  2. Options that need an app change (digits D2, placeholders U0) are chosen only if they beat the best no-app-change
     config by more than 0.01 mean macro-F1; otherwise the best no-app-change config wins. Either way the gap is reported
     and the owner decides at Gate 3.
  3. Ties within 0.002: prefer the higher mean digit-probe accuracy, then the simpler config (S0, ratio 1.0, D0, U1, lr 3e-5).
Outputs: results/sweep_<stage>.csv (one row per run) and results/sweep_<stage>_summary.csv (per config).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

SEEDS = [13, 21]
BASE = {"balance": "S0", "digits": "D1", "placeholders": "U1", "synthetic_ratio": 1.0,
        "lr": 3e-5, "epochs": 4, "max_len": 128, "label_smoothing": 0.0}


def grid(stage: str, winner: dict | None):
    if stage == "A":
        out = [{**BASE, "balance": b, "digits": d, "placeholders": u}
               for b, d, u in itertools.product(["S0", "S1"], ["D0", "D1", "D2"], ["U0", "U1"])]
        out.append({**BASE, "synthetic_ratio": 0.0})
        return out
    return [{**winner, "lr": lr, "max_len": ml, "label_smoothing": ls, "epochs": 5}
            for lr, ml, ls in itertools.product([2e-5, 3e-5, 5e-5], [64, 128], [0.0, 0.05])]


def key(c: dict) -> str:
    return "|".join(f"{k}={c[k]}" for k in sorted(BASE))


def needs_app_change(c: dict) -> bool:
    return c["digits"] == "D2" or c["placeholders"] == "U0"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("config")
    s = g.agg(val_macro_f1_mean=("val_macro_f1", "mean"), val_macro_f1_std=("val_macro_f1", "std"),
              val_caution_f1_mean=("val_caution_f1", "mean"), probe_acc_mean=("probe_acc", "mean"),
              best_epoch_mean=("best_epoch", "mean"), n=("seed", "count")).reset_index()
    return s.sort_values("val_macro_f1_mean", ascending=False)


def pick(summary: pd.DataFrame) -> tuple[dict, dict]:
    cfgs = {r.config: dict(kv.split("=") for kv in r.config.split("|")) for r in summary.itertuples()}
    for c in cfgs.values():
        for k in ("synthetic_ratio", "lr", "label_smoothing"):
            c[k] = float(c[k])
        for k in ("epochs", "max_len"):
            c[k] = int(c[k])
    s = summary.copy()
    s["app_change"] = [needs_app_change(cfgs[c]) for c in s["config"]]
    best_free = s[~s.app_change].iloc[0]
    best_any = s.iloc[0]
    top = s[(~s.app_change) & (s.val_macro_f1_mean >= best_free.val_macro_f1_mean - 0.002)]
    top = top.sort_values(["probe_acc_mean", "val_macro_f1_mean"], ascending=False)
    chosen = top.iloc[0]
    note = {"best_no_app_change": best_free.config, "best_no_app_change_f1": float(best_free.val_macro_f1_mean),
            "best_overall": best_any.config, "best_overall_f1": float(best_any.val_macro_f1_mean),
            "app_change_gain": float(best_any.val_macro_f1_mean - best_free.val_macro_f1_mean), "chosen": chosen.config}
    if bool(best_any.app_change) and note["app_change_gain"] > 0.01:
        note["owner_decision_needed"] = "an app-change option beats the best no-app-change config by > 0.01"
    return cfgs[chosen.config], note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    args = ap.parse_args()
    res_dir = ROOT / "results"; res_dir.mkdir(exist_ok=True)
    winner = None
    if args.stage == "B":
        sa = pd.read_csv(res_dir / "sweep_A_summary.csv")
        winner, note = pick(sa)
        print("Stage A selection:", json.dumps(note, indent=1))
        (res_dir / "sweep_A_selection.json").write_text(json.dumps(note, indent=1))
    out = res_dir / f"sweep_{args.stage}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["config", "seed"])
    todo = [(c, s) for c in grid(args.stage, winner) for s in SEEDS
            if not ((done["config"] == key(c)) & (done["seed"] == s)).any()]
    print(f"stage {args.stage}: {len(todo)} runs to do", flush=True)
    for i, (c, seed) in enumerate(todo, 1):
        cmd = [sys.executable, str(ROOT / "src/train.py"), "--no-save", "--seed", str(seed),
               "--balance", c["balance"], "--digits", c["digits"], "--placeholders", c["placeholders"],
               "--synthetic-ratio", str(c["synthetic_ratio"]), "--lr", str(c["lr"]), "--epochs", str(c["epochs"]),
               "--max-len", str(c["max_len"]), "--label-smoothing", str(c["label_smoothing"])]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"[{i}/{len(todo)}] {key(c)} seed={seed}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-3000:]); raise SystemExit("train.py failed")
        run_id = p.stdout.strip().splitlines()[-1]
        r = json.loads((res_dir / f"{run_id}.json").read_text(encoding="utf-8"))["result"]
        b = r["best_val"]
        row = {"config": key(c), "seed": seed, "run_id": run_id, "val_macro_f1": b["macro_f1"],
               "val_caution_f1": b["per_class"]["caution"]["f1"], "val_spam_f1": b["per_class"]["spam"]["f1"],
               "val_mcc": b["mcc"], "probe_acc": b.get("probe_acc"), "best_epoch": b["epoch"],
               "train_rows": r["train_rows"], "seconds": sum(h["seconds"] for h in r["history"])}
        print("   ", json.dumps({k: row[k] for k in ("val_macro_f1", "val_caution_f1", "probe_acc", "best_epoch", "seconds")}), flush=True)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out, index=False)
    summ = summarise(done)
    summ.to_csv(res_dir / f"sweep_{args.stage}_summary.csv", index=False)
    print(summ.head(10).to_string(index=False))
    if args.stage == "B":
        final, note = pick(summ)
        (res_dir / "sweep_B_selection.json").write_text(json.dumps({**note, "final_config": final}, indent=1))
        print("Stage B selection:", json.dumps(note, indent=1))


if __name__ == "__main__":
    main()
