"""Phase 5: fine-tune distilbert-base-uncased for [ham, spam, promo, caution]. Runs identically on CPU and Colab GPU.

Selection is on VAL only (best epoch by val macro-F1); the test set is never read here.
Options:
  --balance S0|S1      S0 = natural data + class-weighted CE; S1 = undersample ham/promo to 3x the smallest class
  --synthetic-ratio R  keep at most R * (#real) synthetic rows per class (brief cap: R <= 1.0)
  --placeholders U0|U1 U0 keep scrubber placeholders (<URL>, [OTP], ...; app would have to mask identically);
                       U1 fill them with realistic random values (train: fresh each batch; val/probe: fixed seed 1234)
  --digits D0|D1|D2    D0 as-is; D1 re-randomise every digit each epoch (train only); D2 replace every digit by '0'
                       (D2 requires the same preprocessing in the app: text.replace(/[0-9]/g, '0') before tokenising)
  --label-smoothing, --lr, --epochs, --max-len, --batch, --seed, --limit (smoke tests)
Outputs: models/<run_id>/ (best checkpoint + tokenizer + config with id2label), results/<run_id>.json, logs/<run_id>_epochs.csv
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from metrics import cap_synthetic, evaluate  # noqa: E402
from placeholders import fill  # noqa: E402
from subsets import load_split  # noqa: E402

L2I = {l: i for i, l in enumerate(LABELS)}
DIG = re.compile(r"\d")


def digits(text: str, mode: str, rng: random.Random) -> str:
    if mode == "D1":
        return DIG.sub(lambda m: str(rng.randrange(10)), text)
    if mode == "D2":
        return DIG.sub("0", text)
    return text


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--balance", choices=["S0", "S1"], default="S0")
    ap.add_argument("--synthetic-ratio", type=float, default=1.0)
    ap.add_argument("--train-file", default="data/train.csv", help="training CSV (relative to repo root); default = final train split")
    ap.add_argument("--placeholders", choices=["U0", "U1"], default="U1")
    ap.add_argument("--digits", choices=["D0", "D1", "D2"], default="D0")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N train / N val rows")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    run = new_run("train", vars(args))
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = cap_synthetic(load_split(ROOT / args.train_file, drop_native=True), seed=args.seed, ratio=args.synthetic_ratio)
    va = load_split(ROOT / "data/val.csv", drop_native=True)
    probe_path = ROOT / "data/stress/digit_probe.csv"
    pr = pd.read_csv(probe_path, keep_default_na=False) if probe_path.exists() else None
    if args.placeholders == "U1":  # fixed fill for evaluation sets so every run sees identical val text
        frng = random.Random(1234)
        va = va.assign(text=[fill(t, frng) for t in va["text"]])
    if args.balance == "S1":
        floor = tr["final_label"].value_counts().min()
        tr = pd.concat([g.sample(n=min(len(g), 3 * floor), random_state=args.seed) for _, g in tr.groupby("final_label")])
    if args.limit:
        tr = pd.concat([g.head(max(2, args.limit // 4)) for _, g in tr.groupby("final_label")])
        va = va.head(args.limit)
    counts = tr["final_label"].value_counts().reindex(LABELS).fillna(0).values
    weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float, device=dev)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=4, id2label=dict(enumerate(LABELS)), label2id=L2I).to(dev)
    rng = random.Random(args.seed)
    use_amp = dev.type == "cuda"  # fp16 mixed precision on GPU only; CPU runs stay fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def collate(batch, dmode, fill_now):
        texts = [digits(fill(t, rng) if fill_now else t, dmode, rng) for t, _ in batch]
        enc = tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    tr_items = list(zip(tr["text"], tr["final_label"].map(L2I)))
    va_items = list(zip(va["text"], va["final_label"].map(L2I)))
    val_mode = "D2" if args.digits == "D2" else "D0"  # D1 randomises only during training
    u1 = args.placeholders == "U1"
    tr_dl = DataLoader(tr_items, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate(b, args.digits, u1),
                       generator=torch.Generator().manual_seed(args.seed))
    va_dl = DataLoader(va_items, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, val_mode, False))
    pr_dl = None if pr is None else DataLoader(list(zip(pr["text"], pr["final_label"].map(L2I))), batch_size=64,
                                               shuffle=False, collate_fn=lambda b: collate(b, val_mode, False))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights if args.balance == "S0" else None, label_smoothing=args.label_smoothing)

    def predict(dl):
        model.eval(); loss, preds, ys, lgs = 0.0, [], [], []
        with torch.no_grad():
            for b in dl:
                b = {k: v.to(dev) for k, v in b.items()}
                y = b.pop("labels")
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    lg = model(**b).logits.float()
                loss += torch.nn.functional.cross_entropy(lg, y).item() * len(y)
                preds += lg.argmax(1).tolist(); ys += y.tolist(); lgs.append(lg.cpu().numpy())
        return loss / max(len(ys), 1), ys, preds, np.concatenate(lgs)

    history, best = [], {"macro_f1": -1}
    outdir = ROOT / "models" / run["run_id"]
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tl, nb = 0.0, 0
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            y = b.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**b).logits.float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            tl += loss.item(); nb += 1
        vl, ys, preds, logits = predict(va_dl)
        ev = evaluate([LABELS[i] for i in ys], [LABELS[i] for i in preds])
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "val_macro_f1": ev["macro_f1"],
               "val_acc": ev["accuracy"], "val_caution_f1": ev["per_class"]["caution"]["f1"], "seconds": round(time.time() - t0, 1)}
        if pr_dl is not None:
            _, pys, ppreds, _ = predict(pr_dl)
            rec["probe_acc"] = float(np.mean(np.array(pys) == np.array(ppreds)))
        history.append(rec); print(json.dumps(rec), flush=True)
        if ev["macro_f1"] > best["macro_f1"]:
            best = {**ev, "epoch": ep, "probe_acc": rec.get("probe_acc")}
            if not args.no_save:
                model.save_pretrained(outdir); tok.save_pretrained(outdir)
                np.save(outdir / "val_logits.npy", logits)
                np.save(outdir / "val_labels.npy", np.array(ys))
    (ROOT / "logs").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(ROOT / "logs" / f"{run['run_id']}_epochs.csv", index=False)
    import transformers
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "pandas": pd.__version__,
           "cuda": torch.cuda.get_device_name(0) if use_amp else None, "amp_fp16": use_amp}
    res = {"device": str(dev), "env": env, "train_rows": int(len(tr)), "train_labels": tr["final_label"].value_counts().to_dict(),
           "train_synthetic": int((tr["is_synthetic"].astype(str) == "True").sum()),
           "val_rows": int(len(va)), "history": history, "best_val": best,
           "model_dir": str(outdir) if not args.no_save else None}
    save_result(run, res, f"{run['run_id']}.json")
    print("best epoch", best["epoch"], "val macro-F1", round(best["macro_f1"], 4)); print(run["run_id"])


if __name__ == "__main__":
    main()
