"""Phase 10 (POST-HOC): subgroup analyses of SAVED test predictions, data compositions and silver-label checks.

No model is run here and nothing is tuned. Main test subset = test minus native-script rows (as pre-registered).
1. Label-source split: human_verified vs claude_blind test items.
2. Per-source split (UCI, Mishra & Soni, owner export, friends, inbox) and the Indian-origin subset (export+friends+inbox).
   Per-class metrics are reported only for classes with >= 10 items; macro-F1 (and its CI) only if every class has >= 10.
3. Exact counts (k/n) for every rate of the main table.
4. Compositions: test source x class; final training set source x class x label_source; clusters per split per class;
   owner-check overrides by class; borderline test items.
5. Silver-label proxy quality: claude_silver vs the original source label where both exist.
Bootstrap: 1000 item-level resamples (every test item is its own cluster), seed 0, 95% percentile CI.
Outputs: results/posthoc.json, results/tables/tab_posthoc_*.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from final_test import SEEDS  # noqa: E402
from make_tables import NAME, OUT, tex  # noqa: E402
from metrics import evaluate  # noqa: E402
from subsets import load_split  # noqa: E402

B, MIN_N = 1000, 10
LEGIT = ("ham", "promo", "caution")


def models(p, b):
    m = {"new_int8_deploy": p["seed3_int8_quint8|filled|raw"], **{f"new_seed{s}_fp32": p[f"seed{s}_fp32|filled|raw"] for s in SEEDS},
         "tfidf_lr": b["lr"], "tfidf_svm": b["svm"], "v73_int8": p["v73_int8|filled|raw"], "v73_int8_rules": p["v73_int8|filled|pipeline"]}
    return {k: np.asarray(v) for k, v in m.items()}


def counts(y, pr):
    """Exact k/n for every rate."""
    out = {"accuracy": f"{int((pr == y).sum())}/{len(y)}"}
    legit = np.isin(y, LEGIT)
    out["legit_as_scam"] = f"{int((legit & (pr == 'spam')).sum())}/{int(legit.sum())}"
    for l in LABELS:
        out[f"recall_{l}"] = f"{int(((y == l) & (pr == l)).sum())}/{int((y == l).sum())}"
        out[f"precision_{l}"] = f"{int(((y == l) & (pr == l)).sum())}/{int((pr == l).sum())}"
    return out


def block(y, preds, rng):
    n_cls = {l: int((y == l).sum()) for l in LABELS}
    ok_cls = [l for l in LABELS if n_cls[l] >= MIN_N]
    all_ok = len(ok_cls) == 4
    res = {"n": int(len(y)), "n_per_class": n_cls, "skipped_classes_lt10": [l for l in LABELS if l not in ok_cls],
           "macro_f1_skipped": not all_ok, "models": {}}
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    for k, pr in preds.items():
        e = evaluate(y, pr)
        r = {"per_class": {l: {m: e["per_class"][l][m] for m in ("precision", "recall", "f1")} for l in ok_cls},
             "legit_as_scam": e["fpr_legit_as_scam"], "counts": counts(y, pr)}
        if all_ok:
            bs = [evaluate(y[i], pr[i])["macro_f1"] for i in idx]
            r["macro_f1"], r["macro_f1_ci95"] = e["macro_f1"], [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        res["models"][k] = r
    seeds = [f"new_seed{s}_fp32" for s in SEEDS]
    if all_ok:
        f = [res["models"][s]["macro_f1"] for s in seeds]
        res["seeds_macro_f1"] = {"mean": float(np.mean(f)), "std": float(np.std(f, ddof=1))}
    res["seeds_legit_as_scam"] = {"mean": float(np.mean([res["models"][s]["legit_as_scam"] for s in seeds])),
                                  "std": float(np.std([res["models"][s]["legit_as_scam"] for s in seeds], ddof=1))}
    return res


def subgroup_table(groups: dict, fname: str, caption: str, label: str):
    order = [("New INT8 (deployed)", "new_int8_deploy"), ("New FP32, 5 seeds", None), ("TF-IDF LR", "tfidf_lr"),
             ("TF-IDF SVM", "tfidf_svm"), ("v7.3 INT8", "v73_int8"), ("v7.3 INT8 + rules", "v73_int8_rules")]
    rows = []
    for gname, g in groups.items():
        nc = "/".join(str(g["n_per_class"][l]) for l in LABELS)
        rows.append([f"\\textit{{{gname}}} (n={g['n']}; {nc})", "", "", "", "", "", ""])
        for lab, k in order:
            if k is None:
                f1 = f"{g['seeds_macro_f1']['mean']:.3f} $\\pm$ {g['seeds_macro_f1']['std']:.3f}" if "seeds_macro_f1" in g else "skipped"
                rows.append([lab, f1, "", "", "", "", f"{100 * g['seeds_legit_as_scam']['mean']:.1f} $\\pm$ {100 * g['seeds_legit_as_scam']['std']:.1f}"])
                continue
            m = g["models"][k]
            f1 = f"{m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.2f}, {m['macro_f1_ci95'][1]:.2f}]" if "macro_f1" in m else "skipped"
            pcs = [f"{m['per_class'][l]['f1']:.2f}" if l in m["per_class"] else "--" for l in LABELS]
            rows.append([lab, f1, *pcs, f"{m['counts']['legit_as_scam']}"])
    tex(pd.DataFrame(rows, columns=["Model", "Macro-F1 [95\\% CI]"] + [f"F1 {NAME[l]}" for l in LABELS] + ["Legit$\\to$scam (k/n)"]),
        fname, caption, label, "lrrrrrr")


def main():
    run = new_run("posthoc", {"B": B, "min_n": MIN_N, "post_hoc": True})
    rng = np.random.default_rng(0)
    p = pd.read_csv(ROOT / "results" / "preds_final_test.csv", keep_default_na=False)
    b = pd.read_csv(ROOT / "results" / "preds_baselines_test.csv", keep_default_na=False)
    t = pd.read_csv(ROOT / "data" / "test.csv", keep_default_na=False)
    assert (p.item_id.values == b.item_id.values).all() and (p.item_id.values == t.item_id.values).all()
    m = p["native_script"].astype(str) != "True"
    y = p["true"].values
    P = models(p, b)
    src = t["source"].replace({"friend2026": "friends", "friend2_2026": "friends", "friend3_2026": "friends"}).values
    res = {"post_hoc": True}
    # 1. label source
    res["by_label_source"] = {ls: block(y[m & (p.label_source == ls)], {k: v[m & (p.label_source == ls)] for k, v in P.items()}, rng)
                              for ls in ("human_verified", "claude_blind")}
    # 2. per source + Indian-origin
    groups = {s: (m & (src == s)) for s in ("uci", "mishra_soni", "export2026", "friends", "inbox")}
    groups["indian_origin"] = m & np.isin(src, ["export2026", "friends", "inbox"])
    res["by_source"] = {g: block(y[mm], {k: v[mm] for k, v in P.items()}, rng) for g, mm in groups.items()}
    # 3. exact counts for the main table
    full = {**P, "new_int8_qint8": p["seed3_int8_qint8|filled|raw"].values, "v73_fp32": p["v73_fp32|filled|raw"].values,
            "new_int8_deploy_rules": p["seed3_int8_quint8|filled|pipeline"].values}
    res["main_table_counts"] = {k: counts(y[m], v[m]) for k, v in full.items()}
    # 4. compositions
    tm = t[m.values]
    res["test_source_by_class"] = pd.crosstab(tm["source"], tm["final_label"]).reindex(columns=LABELS, fill_value=0).to_dict("index")
    tr = load_split(ROOT / "data/train.csv", drop_native=True)
    tr = tr[tr["is_synthetic"].astype(str) != "True"]
    ct = tr.groupby(["source", "final_label", "label_source"]).size()
    res["train_source_class_labelsource"] = {f"{a}|{c}|{d}": int(n) for (a, c, d), n in ct.items()}
    res["train_labelsource_share_by_class"] = {l: {k: float(v) for k, v in tr[tr.final_label == l]["label_source"].value_counts(normalize=True).items()}
                                               for l in LABELS}
    va = load_split(ROOT / "data/val.csv", drop_native=True)
    res["clusters_per_split_class"] = {nm: {l: int(d[d.final_label == l]["cluster_id"].nunique()) for l in LABELS}
                                       for nm, d in (("train_used", tr), ("val", va), ("test_main", tm))}
    chk = pd.read_csv(ROOT / "annotation" / "check_results.csv").drop_duplicates("item_id", keep="last")
    ov = chk[chk.final_label != chk.proposed]
    res["owner_check"] = {"answered": int(len(chk)), "overrides": int(len(ov)),
                          "overrides_by_proposed_to_final": ov.groupby(["proposed", "final_label"]).size().rename("n").reset_index().to_dict("records"),
                          "answered_by_proposed_class": chk["proposed"].value_counts().to_dict(),
                          "overrides_by_proposed_class": ov["proposed"].value_counts().to_dict(),
                          "note": "200 decisions were logged for 180 distinct items (undo/redo); the last decision per item counts"}
    res["borderline_test"] = {"all": int((t["borderline"].astype(str) == "1").sum()), "main": int((tm["borderline"].astype(str) == "1").sum()),
                              "main_by_class": tm[tm["borderline"].astype(str) == "1"]["final_label"].value_counts().to_dict()}
    # 5. silver proxy agreement
    pool = pd.read_csv(ROOT / "data" / "pool.csv", keep_default_na=False)[["item_id", "label", "source"]]
    lab = pd.read_csv(ROOT / "data" / "labels.csv", keep_default_na=False)
    sv = lab[lab.label_source == "claude_silver"].merge(pool, on="item_id")
    sv = sv[sv["label"].isin(LABELS)]
    res["silver_vs_original"] = {
        "n_both": int(len(sv)), "agree": int((sv.final_label == sv.label).sum()),
        "agreement": float((sv.final_label == sv.label).mean()),
        "confusion_original_rows_silver_cols": pd.crosstab(sv["label"], sv["final_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0).values.tolist(),
        "by_source_agreement": sv.groupby("source_y").apply(lambda d: float((d.final_label == d.label).mean())).to_dict(),
        "caveat": "Silver items were SELECTED mostly where the original label and v7.3 disagreed (plus needs_review and a 10% sample), "
                  "so agreement here is biased low and is not an accuracy estimate. Original labels map UCI/Mishra spam to scam, "
                  "while silver separates scam from promo, which also lowers agreement by design.",
        "owner_silver_spot_check_100": "NOT DONE (the owner's check covered 180 test items instead)",
        "self_kappa_note": "The kappa = 1.00 second pass was not independent (same model, same session, pass-1 labels in context): do not cite."}
    save_result(run, res, "posthoc.json")

    subgroup_table({"Owner-verified": res["by_label_source"]["human_verified"], "Claude-blind only": res["by_label_source"]["claude_blind"]},
                   "tab_posthoc_labelsource.tex", "Post-hoc: test results by label source (main subset). Class counts in order ham/scam/promo/caution.",
                   "tab:ph_labelsource")
    names = {"uci": "UCI", "mishra_soni": "Mishra \\& Soni", "export2026": "Owner export", "friends": "Friends", "inbox": "Inbox file", "indian_origin": "Indian-origin (export+friends+inbox)"}
    subgroup_table({names[k]: v for k, v in res["by_source"].items()}, "tab_posthoc_source.tex",
                   "Post-hoc: test results by source (main subset). Per-class F1 shown only for classes with $\\geq$10 items; macro-F1 only if all four have $\\geq$10.",
                   "tab:ph_source")
    ts = pd.DataFrame(res["test_source_by_class"]).T.reindex(columns=LABELS)
    ts.loc["total"] = ts.sum()
    ts.columns = [NAME[c] for c in ts.columns]
    tex(ts.reset_index().rename(columns={"index": "Source"}), "tab_comp_test.tex",
        "Test set (main subset) by source and class. The test set was sampled stratified by \\emph{proxy} class (original label, v7.3 prediction, rule hints), so these counts are \\emph{not} natural prevalence.", "tab:comp_test")
    rows = []
    for s_ in sorted(tr.source.unique()):
        for ls in ("original", "claude_silver"):
            rows.append([s_.replace("_", "\\_"), ls.replace("_", "\\_")] + [res["train_source_class_labelsource"].get(f"{s_}|{l}|{ls}", 0) for l in LABELS])
    tex(pd.DataFrame(rows, columns=["Source", "Label source"] + [NAME[l] for l in LABELS]), "tab_comp_train.tex",
        "Final training set (5,882 real rows) by source, label source and class.", "tab:comp_train", "llrrrr")
    cl = pd.DataFrame(res["clusters_per_split_class"]).T.reindex(columns=LABELS)
    cl.columns = [NAME[c] for c in cl.columns]
    tex(cl.reset_index().rename(columns={"index": "Split"}), "tab_comp_clusters.tex", "Unique near-duplicate clusters per split and class.", "tab:comp_clusters")
    print(json.dumps({k: res[k] for k in ("main_table_counts", "owner_check", "borderline_test", "train_labelsource_share_by_class")}, indent=1)[:4000])
    sv_ = res["silver_vs_original"]; print("silver vs original", sv_["agree"], "/", sv_["n_both"], round(sv_["agreement"], 3), sv_["by_source_agreement"])
    for grp in ("by_label_source", "by_source"):
        for g, d in res[grp].items():
            mm = d["models"]
            print(grp, g, d["n_per_class"], "skip" if d["macro_f1_skipped"] else
                  {k: round(mm[k]["macro_f1"], 3) for k in ("new_int8_deploy", "tfidf_lr", "tfidf_svm", "v73_int8", "v73_int8_rules")},
                  d.get("seeds_macro_f1"), "L->S", mm["new_int8_deploy"]["counts"]["legit_as_scam"], mm["v73_int8"]["counts"]["legit_as_scam"])
    print(run["run_id"])


if __name__ == "__main__":
    main()
