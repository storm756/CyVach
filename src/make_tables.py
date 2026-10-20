"""Phase 9: LaTeX (booktabs) tables from saved results + data composition (counts only).

Writes results/tables/*.tex and results/data_composition.json.
Data composition counts the rows the final model actually used: train = real, non-native-script rows
(synthetic-ratio 0); val/test main = non-native-script. Clusters = MinHash near-duplicate clusters
(effective unique items).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT, new_run, save_result  # noqa: E402
from subsets import load_split  # noqa: E402

OUT = ROOT / "results" / "tables"
NAME = {"ham": "ham", "spam": "scam", "promo": "promo", "caution": "caution"}


def res(n):
    return json.loads((ROOT / "results" / n).read_text(encoding="utf-8"))["result"]


def tex(df: pd.DataFrame, path: str, caption: str, label: str, colfmt: str | None = None):
    cols = colfmt or "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
             r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def f3(x):
    return f"{x:.3f}"


def composition():
    full_tr = load_split(ROOT / "data/train.csv", drop_native=False)
    tr = full_tr[(~full_tr["native_script"]) & (full_tr["is_synthetic"].astype(str) != "True")]
    va = load_split(ROOT / "data/val.csv", drop_native=True)
    te = load_split(ROOT / "data/test.csv", drop_native=True)
    comp, rows = {}, []
    for name, d in [("train (used)", tr), ("val", va), ("test (main)", te)]:
        comp[name] = {}
        for l in LABELS:
            g = d[d["final_label"] == l]
            comp[name][l] = {"rows": int(len(g)), "real": int((g["is_synthetic"].astype(str) != "True").sum()),
                             "synthetic": int((g["is_synthetic"].astype(str) == "True").sum()),
                             "clusters": int(g["cluster_id"].nunique())}
        comp[name]["sources"] = d["source"].value_counts().to_dict()
        comp[name]["label_source"] = d["label_source"].value_counts().to_dict()
        rows.append([name] + [f"{comp[name][l]['rows']:,} ({comp[name][l]['clusters']:,})" for l in LABELS] + [f"{len(d):,}"])
    syn_tr = full_tr[(~full_tr["native_script"]) & (full_tr["is_synthetic"].astype(str) == "True")]
    comp["train_synthetic_excluded"] = syn_tr["final_label"].value_counts().to_dict()
    comp["train_native_excluded"] = int(full_tr["native_script"].sum())
    tex(pd.DataFrame(rows, columns=["Split"] + [NAME[l] for l in LABELS] + ["Total"]), "tab_data.tex",
        "Data used by the final model: messages per class (near-duplicate clusters = effective unique items). "
        "All rows are real; synthetic rows were excluded from training (they lowered validation F1).", "tab:data")
    src = pd.DataFrame({k: pd.Series(comp[k]["sources"]) for k in ("train (used)", "val", "test (main)")}).fillna(0).astype(int)
    src.index.name = "Source"
    tex(src.reset_index(), "tab_sources.tex", "Messages per source and split.", "tab:sources")
    return comp


def main_results():
    st = res("stats.json")
    rows = []
    order = [("New, INT8 QUInt8 (deployed)", "new_int8_quint8"), ("New, INT8 QInt8", "new_int8_qint8"),
             ("TF-IDF + LogReg", "tfidf_lr"), ("TF-IDF + LinearSVC", "tfidf_svm"),
             ("v7.3, INT8 (shipped)", "v73_int8"), ("v7.3, FP32", "v73_fp32"),
             ("New INT8 + app rules", "new_int8_quint8+rules"), ("v7.3 INT8 + app rules", "v73_int8+rules")]
    for lab, k in order:
        p, ci = st["point"][k], st["ci95"][k]["macro_f1"]
        rows.append([lab, f"{p['macro_f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", f3(p["mcc"])] +
                    [f3(p[f"f1_{l}"]) for l in LABELS] + [f"{100 * p['legit_as_scam']:.1f}"])
    s = st["seeds_macro_f1"]
    rows.insert(2, ["New, FP32, 5 seeds", f"{s['mean']:.3f} $\\pm$ {s['std']:.3f}", "", "", "", "", "", ""])
    tex(pd.DataFrame(rows, columns=["Model", "Macro-F1 [95\\% CI]", "MCC"] + [f"F1 {NAME[l]}" for l in LABELS] + ["Legit$\\to$scam (\\%)"]),
        "tab_main.tex", f"Test results (main subset, n = {st['n_main']}; 95\\% bootstrap CIs, 1000 resamples).", "tab:main", "lrrrrrrr")
    mc = st["mcnemar"]
    rows = [[k.replace("new_int8_quint8_vs_", "vs "), v["new_right_other_wrong"], v["new_wrong_other_right"], f"{v['p_exact']:.2g}"]
            for k, v in mc.items()]
    tex(pd.DataFrame(rows, columns=["Deployed new model", "New right only", "Other right only", "Exact McNemar p"]),
        "tab_mcnemar.tex", "Paired comparison of the deployed new model with other systems on test.", "tab:mcnemar")


def ablation():
    ab = res("ablation.json")["models"]
    cfg = ["raw_model", "+pre", "+trusted", "+safety", "+mobile", "+suffix", "+header(trusted+suffix)", "full", "full-trusted"]
    nice = {"raw_model": "Model only", "+pre": "+ pre-model heuristics (H1, H2)", "+trusted": "+ trusted whitelist (R0)",
            "+safety": "+ safety override (R1)", "+mobile": "+ mobile-number rules (R2, R3)", "+suffix": "+ TRAI suffix rules (R4)",
            "+header(trusted+suffix)": "+ header rules (R0 + R4)", "full": "Full shipped pipeline", "full-trusted": "Full minus R0"}
    rows = []
    for c in cfg:
        a, b = ab["seed3_int8_quint8|filled"][c], ab["v73_int8|filled"][c]
        rows.append([nice[c], f3(a["macro_f1"]), f"{100 * a['legit_as_scam']:.1f}", f"{a['fixed']}/{a['broke']}",
                     f3(b["macro_f1"]), f"{100 * b['legit_as_scam']:.1f}", f"{b['fixed']}/{b['broke']}"])
    rows.append(["+ URL/link rules", "n/a", "", "", "n/a", "", ""])
    tex(pd.DataFrame(rows, columns=["Configuration", "New F1", "New L$\\to$S \\%", "New fixed/broke", "v7.3 F1", "v7.3 L$\\to$S \\%", "v7.3 fixed/broke"]),
        "tab_ablation.tex", "Rule-layer ablation on test (main subset). L$\\to$S: legitimate messages flagged as scam. "
        "fixed/broke: items the rules changed from wrong to right / right to wrong. The app has no URL rule.", "tab:ablation", "lrrrrrr")


def stress():
    s = res("stress_eval.json")
    rows = []
    for st, lab in [("stress_spoof", "Spoofed senders"), ("stress_obfuscation", "Obfuscation"), ("stress_hinglish", "Hinglish"), ("digit_probe", "Digit probe")]:
        d = s[st]
        rows.append([lab, d["n"], d["n_unique_text"], f3(d["models"]["new_int8_quint8"]["raw"]["accuracy"]),
                     f3(d["models"]["v73_int8"]["raw"]["accuracy"])])
        if st == "stress_spoof":
            rows.append(["\\quad with app rules", "", "", f3(d["models"]["new_int8_quint8"]["pipeline"]["accuracy"]),
                         f3(d["models"]["v73_int8"]["pipeline"]["accuracy"])])
            rows.append(["\\quad camouflaged scams only", "", "", f3(d["models"]["new_int8_quint8"]["raw_accuracy_by_camouflage"]["1"]),
                         f3(d["models"]["v73_int8"]["raw_accuracy_by_camouflage"]["1"])])
    tex(pd.DataFrame(rows, columns=["Stress set (synthetic)", "n", "Unique", "New INT8 acc.", "v7.3 INT8 acc."]),
        "tab_stress.tex", "Accuracy on synthetic stress sets (evaluation only).", "tab:stress")


def deploy():
    ex, b = res("export_onnx.json"), res("bench_onnx.json")["models"]
    rows = [["v7.3 INT8 (shipped)", f"{b['v7.3_deployed_int8']['size']['bytes']:,}", f"{b['v7.3_deployed_int8']['size']['MB']:.2f}", f"{b['v7.3_deployed_int8']['size']['MiB']:.2f}", "", ""]]
    for lab, k, bk in [("New FP32", "fp32", "new_fp32"), ("New INT8 QUInt8", "int8_QUInt8", "new_int8_QUInt8"), ("New INT8 QInt8", "int8_QInt8", "new_int8_QInt8")]:
        sz = ex["files"][k]["size"]
        rows.append([lab, f"{sz['bytes']:,}", f"{sz['MB']:.2f}", f"{sz['MiB']:.2f}", "", ""])
    for r, bk in zip(rows, ["v7.3_deployed_int8", "new_fp32", "new_int8_QUInt8", "new_int8_QInt8"]):
        r[4] = f"{b[bk]['threads_1']['median_ms']:.1f} / {b[bk]['threads_1']['p95_ms']:.1f}"
        r[5] = f"{b[bk]['threads_4']['median_ms']:.1f} / {b[bk]['threads_4']['p95_ms']:.1f}"
    tex(pd.DataFrame(rows, columns=["Model file", "Bytes", "MB", "MiB", "1 thread (ms)", "4 threads (ms)"]), "tab_deploy.tex",
        "ONNX model size and desktop-CPU latency (median / p95, input [1,128], 500 runs after 50 warm-ups). Not phone latency.",
        "tab:deploy", "lrrrrr")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run = new_run("make_tables", {})
    comp = composition(); main_results(); ablation(); stress(); deploy()
    save_result(run, comp, "data_composition.json")
    print(json.dumps(comp, indent=1)); print(sorted(p.name for p in OUT.glob("*.tex"))); print(run["run_id"])


if __name__ == "__main__":
    main()
