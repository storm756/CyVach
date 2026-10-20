"""Phase 11 (paper note): how much of the SCAM decision does the app's header/sender rule layer make on its own?

Computed on the SYNTHETIC spoof stress set (data/stress/stress_spoof.csv), where every scam text is crossed with
every sender type by design, so all shares reflect the generator design, NOT real-inbox prevalence.
For each scam item, the shipped rule layer (src/rules.py, exact port; H1/H2 + R0-R4) is evaluated twice with a
hypothetical model output: "model says scam" (conf 0.99) and "model says ham" (conf 0.99).
  caught_by_rules_alone   : verdict is scam in both cases (the rules force scam whatever the model says)
  diverted_by_rules_alone : verdict is non-scam in both cases (the rules force a non-scam label whatever the model says)
  model_dependent         : the verdict follows the model
Also reported: the rules that actually fired with the deploy model's own predictions (results/preds_stress.csv).
Output: results/header_upstream.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, save_result  # noqa: E402
from rules import ALL_LAYERS, pipeline  # noqa: E402


def verdict(sender, text, model_label):
    return pipeline(sender, text, lambda _t: (model_label, 0.99), ALL_LAYERS)


def main():
    run = new_run("header_upstream", {"data": "stress_spoof (synthetic)", "hypothetical_conf": 0.99})
    d = pd.read_csv(ROOT / "data" / "stress" / "stress_spoof.csv", keep_default_na=False)
    sc = d[d["final_label"] == "spam"].copy()
    a = [verdict(s, t, "spam") for s, t in zip(sc["sender"], sc["text"])]
    b = [verdict(s, t, "ham") for s, t in zip(sc["sender"], sc["text"])]
    sc["if_model_scam"], sc["if_model_ham"] = [x.label for x in a], [x.label for x in b]
    sc["rule_if_model_scam"], sc["rule_if_model_ham"] = [x.rule for x in a], [x.rule for x in b]
    sc["category"] = ["caught_by_rules_alone" if (x == "spam" and y == "spam") else
                      "diverted_by_rules_alone" if (x != "spam" and y != "spam") else "model_dependent"
                      for x, y in zip(sc["if_model_scam"], sc["if_model_ham"])]
    n = len(sc)
    res = {"synthetic": True, "n_scam_items": n, "n_unique_scam_texts": int(sc["text"].nunique()),
           "categories": {k: {"k": int(v), "n": n, "share": float(v / n)} for k, v in sc["category"].value_counts().items()},
           "by_sender_kind": {k: g["category"].value_counts().to_dict() for k, g in sc.groupby("sender_kind")},
           "rules_forcing_non_scam": sc[sc.category == "diverted_by_rules_alone"]["rule_if_model_scam"].value_counts().to_dict(),
           "rules_forcing_scam": sc[sc.category == "caught_by_rules_alone"]["rule_if_model_ham"].value_counts().to_dict()}
    ps = pd.read_csv(ROOT / "results" / "preds_stress.csv", keep_default_na=False)
    for m in ("new_int8_quint8", "v73_int8"):
        p = ps[(ps["set"] == "stress_spoof") & (ps["model"] == m) & (ps["true"] == "spam")]
        res[f"actual_{m}"] = {"n": int(len(p)), "model_says_scam": int((p["raw"] == "spam").sum()),
                              "pipeline_says_scam": int((p["pipeline"] == "spam").sum()),
                              "rules_turned_scam_into_non_scam": int(((p["raw"] == "spam") & (p["pipeline"] != "spam")).sum()),
                              "rules_turned_non_scam_into_scam": int(((p["raw"] != "spam") & (p["pipeline"] == "spam")).sum())}
    save_result(run, res, "header_upstream.json")
    import json
    print(json.dumps(res, indent=1)); print(run["run_id"])


if __name__ == "__main__":
    main()
