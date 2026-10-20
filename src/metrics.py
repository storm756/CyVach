"""Shared evaluation metrics (labels fixed to [ham, spam, promo, caution])."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support

from common import LABELS


def evaluate(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    legit = [LABELS.index(l) for l in ("ham", "promo", "caution")]
    # false-positive rate on legitimate classes: share of truly legitimate messages flagged as spam(scam)
    legit_rows = cm[legit].sum()
    fpr_legit_as_scam = float(cm[legit, LABELS.index("spam")].sum() / legit_rows) if legit_rows else float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(LABELS)},
        "confusion": cm.tolist(),
        "fpr_legit_as_scam": fpr_legit_as_scam,
    }


def cap_synthetic(df, seed: int = 0, ratio: float = 1.0):
    """Keep at most ratio * (#real) synthetic rows per class (brief: synthetic share <= 50% => ratio 1.0)."""
    import pandas as pd
    syn = df["is_synthetic"].astype(str) == "True"
    keep = [df[~syn]]
    for lab, g in df[syn].groupby("final_label"):
        n_real = int(((df["final_label"] == lab) & ~syn).sum())
        keep.append(g.sample(n=min(len(g), int(ratio * n_real)), random_state=seed))
    return pd.concat(keep).sort_values("item_id")
