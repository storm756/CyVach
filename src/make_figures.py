"""Phase 9: paper figures (vector PDF, grayscale-safe, fonts >= 7 pt) from saved results only.

fig_confusion.pdf       row-normalised confusion matrices (counts printed), new deploy INT8 vs v7.3 INT8, test main subset
fig_training.pdf        per-epoch train/val loss and val macro-F1 for the 5 final seeds
fig_calibration.pdf     test reliability diagram of the deploy model, before/after val-fitted temperature
fig_tradeoff.pdf        desktop-CPU latency (1 thread, median) vs test macro-F1; file size in the label
fig_risk_coverage.pdf   selective risk vs coverage (deploy model, calibrated max-prob), val and test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import softmax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT  # noqa: E402
from final_test import SEEDS  # noqa: E402

FIG = ROOT / "figures"
NAMES = {"ham": "ham", "spam": "scam", "promo": "promo", "caution": "caution"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2})
W2 = 4.8  # LNNS text width in inches (about 12.2 cm)
LIGHT_GREYS = matplotlib.colors.LinearSegmentedColormap.from_list("light_greys", ["white", "0.55"])


def res(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def confusion():
    st = res("stats.json")["result"]["confusion"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), constrained_layout=True)
    for ax, (key, title) in zip(axs, [("new_int8_quint8", "New model (INT8)"), ("v73_int8", "v7.3 (INT8, shipped)")]):
        cm, rn = np.array(st[key]["counts"]), np.array(st[key]["row_normalised"])
        # White-to-mid-gray scale so every label can be black: white-on-dark text is flagged as
        # "hidden text" by Turnitin's integrity check.
        im = ax.imshow(rn, cmap=LIGHT_GREYS, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{rn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
        ax.set_xticks(range(4), [NAMES[l] for l in LABELS], rotation=30, ha="right"); ax.set_yticks(range(4), [NAMES[l] for l in LABELS])
        ax.set_xlabel("Predicted"); ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("True")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Row share")
    fig.savefig(FIG / "fig_confusion.pdf")


def training():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.0), constrained_layout=True)
    for s, rid in SEEDS.items():
        h = pd.read_csv(ROOT / "logs" / f"{rid}_epochs.csv")
        axs[0].plot(h.epoch, h.train_loss, color="0.25", ls="-", marker="o", ms=3, lw=0.9, label="train" if s == 1 else None)
        axs[0].plot(h.epoch, h.val_loss, color="0.55", ls="--", marker="s", ms=3, lw=0.9, label="val" if s == 1 else None)
        axs[1].plot(h.epoch, h.val_macro_f1, color="0.3", marker="o", ms=3, lw=0.9)
    axs[0].set(xlabel="Epoch", ylabel="Loss (CE)", title="Loss, 5 seeds"); axs[0].legend(frameon=False)
    axs[1].set(xlabel="Epoch", ylabel="Macro-F1", title="Validation macro-F1, 5 seeds")
    for ax in axs:
        ax.set_xticks(range(1, 6)); ax.grid(axis="y", color="0.9", lw=0.5)
    fig.savefig(FIG / "fig_training.pdf")


def reliability(P, y, bins=10):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1); xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append((pred[m] == y[m]).mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns


def calibration():
    pf = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    pf = pf[pf["native_script"].astype(str) != "True"]
    P = pf[[f"seed3_int8_quint8|filled|p_{l}" for l in LABELS]].values.clip(1e-12, 1)
    y = np.array([LABELS.index(t) for t in pf["true"]])
    T = json.loads((ROOT / "models" / SEEDS[3] / "temperature.json").read_text())["temperature"]
    e = res("stats.json")["result"]["ece_test"]["deploy_int8_quint8"]
    fig, ax = plt.subplots(figsize=(2.6, 2.4), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":")
    for Pm, lab, st in [(P, f"uncalibrated (ECE {e['before']:.3f})", dict(color="0.55", ls="--", marker="s")),
                        (softmax(np.log(P) / T, 1), f"T = {T:.2f} from val (ECE {e['after']:.3f})", dict(color="0.1", ls="-", marker="o"))]:
        xs, ys, _ = reliability(Pm, y)
        ax.plot(xs, ys, ms=3.5, lw=1.0, label=lab, **st)
    ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0.25, 1.0), ylim=(0.0, 1.0), title="Test reliability (deploy INT8)")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.27, 0.95, "10 bins; bins with < 5 items omitted", fontsize=7, color="0.4", va="top")
    fig.savefig(FIG / "fig_calibration.pdf")


def tradeoff():
    b = res("bench_onnx.json")["result"]["models"]
    ft = res("final_test.json")["result"]["models"]
    stp = res("stats.json")["result"]["point"]
    pts = [("v7.3 INT8 (shipped)", b["v7.3_deployed_int8"], ft["v73_int8|filled"]["main_raw"]["macro_f1"], "s", (6, -3), True),
           ("DistilBERT FP32", b["new_fp32"], ft["seed3_fp32|filled"]["main_raw"]["macro_f1"], "D", (6, -14), False),
           ("DistilBERT INT8 QUInt8", b["new_int8_QUInt8"], ft["seed3_int8_quint8|filled"]["main_raw"]["macro_f1"], "o", (-58, 9), False),
           ("DistilBERT INT8 QInt8", b["new_int8_QInt8"], ft["seed3_int8_qint8|filled"]["main_raw"]["macro_f1"], "^", (-30, -22), False)]
    tf = ROOT / "results" / "tfidf_extra.json"
    if tf.exists():  # post-hoc (Phase 10): TF-IDF size and latency
        t = res("tfidf_extra.json")["result"]["models"]
        pts += [("TF-IDF LR", {"threads_1": t["lr"]["latency_1thread"], "size": t["lr"]["size"]}, stp["tfidf_lr"]["macro_f1"], "P", (6, 4), False),
                ("TF-IDF SVM", {"threads_1": t["svm"]["latency_1thread"], "size": t["svm"]["size"]}, stp["tfidf_svm"]["macro_f1"], "X", (6, -16), False)]
    fig, ax = plt.subplots(figsize=(3.6, 2.5), constrained_layout=True)
    for lab, bb, yv, mk, off, hollow in pts:
        x = bb["threads_1"]["median_ms"]
        ax.scatter(x, yv, marker=mk, s=30, facecolor="white" if hollow else "0.2", edgecolor="0.1", lw=0.8, zorder=3)
        ax.annotate(lab + "\n" + f"{bb['size']['MB']:.1f} MB", (x, yv), xytext=off, textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set(xlabel="Desktop-CPU latency per message, 1 thread, median (ms, log)", ylabel="Test macro-F1", xlim=(0.6, 400), ylim=(0.6, 0.95),
           title="Accuracy vs latency (desktop CPU, not phone)")
    ax.grid(color="0.92", lw=0.5)
    fig.savefig(FIG / "fig_tradeoff.pdf")


def risk_cov():
    a = res("abstain.json")["result"]["risk_coverage"]
    fig, ax = plt.subplots(figsize=(2.6, 2.2), constrained_layout=True)
    for split, st in [("val", dict(color="0.55", ls="--")), ("test", dict(color="0.1", ls="-"))]:
        c = a[split]["maxprob"]["curve"]
        ax.plot(c["coverage"], c["risk"], label=f"{split} (AURC {a[split]['maxprob']['aurc']:.3f})", **st)
    ax.set(xlabel="Coverage", ylabel="Error rate on accepted", title="Risk-coverage (deploy model)", xlim=(0, 1), ylim=(0, None))
    ax.legend(frameon=False); ax.grid(color="0.92", lw=0.5)
    fig.savefig(FIG / "fig_risk_coverage.pdf")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (confusion, training, calibration, tradeoff, risk_cov):
        f(); print("ok", f.__name__)
