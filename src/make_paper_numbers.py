"""Phase 9: results/paper_numbers.md - every number intended for the paper, read from saved result JSONs,
with its run id, producing script and result file. Nothing here is typed by hand except labels and the
owner-check numbers, which are recomputed from annotation/check_results.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

ROWS: list[tuple] = []
SCRIPT = {"silver_self_agreement": "agreement", "pii_audit": "scrub"}  # run name -> script file when they differ


def load(f):
    return json.loads((ROOT / "results" / f).read_text(encoding="utf-8"))


def add(section, what, value, f, path=None):
    r = load(f)
    ROWS.append((section, what, value, r["run_id"], (f"src/{SCRIPT.get(r['script'], r['script'])}.py" if not r["script"].startswith("external") else r["script"]) if r.get("script") else "", f"results/{f}"))


def g(f, *keys):
    o = load(f)["result"]
    for k in keys:
        o = o[k]
    return o


def pct(x):
    return f"{100 * x:.1f}%"


def posthoc_section():
    """Phase 10: every value here is tagged post-hoc (analysis of saved predictions, val or stress sets only)."""
    if not (ROOT / "results" / "posthoc.json").exists():
        return
    H = "Post-hoc (Phase 10)"
    ph = g("posthoc.json")
    mods = ("new_int8_deploy", "tfidf_lr", "tfidf_svm", "v73_int8", "v73_int8_rules")

    def grp(prefix, d):
        add(H, f"post-hoc: {prefix} n per class (ham/scam/promo/caution)", "/".join(str(d["n_per_class"][l]) for l in ("ham", "spam", "promo", "caution")), "posthoc.json")
        if d["macro_f1_skipped"]:
            add(H, f"post-hoc: {prefix} macro-F1", f"skipped (classes with < 10 items: {', '.join(d['skipped_classes_lt10'])})", "posthoc.json")
        else:
            add(H, f"post-hoc: {prefix} new FP32 5 seeds macro-F1", f"{d['seeds_macro_f1']['mean']:.3f} ± {d['seeds_macro_f1']['std']:.3f}", "posthoc.json")
        for k in mods:
            m = d["models"][k]
            f1 = f"{m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.3f}, {m['macro_f1_ci95'][1]:.3f}]" if "macro_f1" in m else "macro-F1 skipped"
            pcs = "/".join(f"{m['per_class'][l]['f1']:.2f}" if l in m["per_class"] else "--" for l in ("ham", "spam", "promo", "caution"))
            add(H, f"post-hoc: {prefix} {k}: macro-F1 [CI]; F1 h/s/p/c; legit→scam k/n", f"{f1}; {pcs}; {m['counts']['legit_as_scam']}", "posthoc.json")
    for ls, d in ph["by_label_source"].items():
        grp(f"label source {ls}", d)
    for src, d in ph["by_source"].items():
        grp(f"source {src}", d)
    for k, c in ph["main_table_counts"].items():
        add(H, f"post-hoc: main-table exact counts {k}", ", ".join(f"{a} {b}" for a, b in c.items()), "posthoc.json")
    for l, d in ph["train_labelsource_share_by_class"].items():
        add(H, f"post-hoc: training labels for class {l}: share by label source", ", ".join(f"{k} {100 * v:.1f}%" for k, v in d.items()), "posthoc.json")
    oc = ph["owner_check"]
    add(H, "post-hoc: owner check overrides by proposed class (answered by class)", f"{oc['overrides_by_proposed_class']} of {oc['answered_by_proposed_class']}", "posthoc.json")
    add(H, "post-hoc: borderline test items (all / main; main by class)", f"{ph['borderline_test']['all']} / {ph['borderline_test']['main']}; {ph['borderline_test']['main_by_class']}", "posthoc.json")
    sv = ph["silver_vs_original"]
    add(H, "post-hoc: claude_silver vs original label agreement (biased proxy)", f"{sv['agree']}/{sv['n_both']} = {100 * sv['agreement']:.1f}%", "posthoc.json")
    add(H, "post-hoc: owner 100-item silver spot-check", sv["owner_silver_spot_check_100"], "posthoc.json")
    tf = g("tfidf_extra.json")["models"]
    for k, v in tf.items():
        add(H, f"post-hoc: TF-IDF {k} size", f"{v['size']['bytes']:,} B = {v['size']['MB']} MB = {v['size']['MiB']} MiB", "tfidf_extra.json")
        add(H, f"post-hoc: TF-IDF {k} desktop-CPU latency 1 thread (median/p95 ms)", f"{v['latency_1thread']['median_ms']}/{v['latency_1thread']['p95_ms']}", "tfidf_extra.json")
        st = v["stress"]
        add(H, f"post-hoc: TF-IDF {k} stress acc spoof / camouflaged scams / obfuscation / Hinglish / digit probe",
            f"{st['stress_spoof']['accuracy']:.3f} ({st['stress_spoof']['k']}/{st['stress_spoof']['n']}) / "
            f"{st['stress_spoof']['scam_recall_camouflage_1']['acc']:.3f} ({st['stress_spoof']['scam_recall_camouflage_1']['k']}/{st['stress_spoof']['scam_recall_camouflage_1']['n']}) / "
            f"{st['stress_obfuscation']['accuracy']:.3f} / {st['stress_hinglish']['accuracy']:.3f} / {st['digit_probe']['accuracy']:.3f}", "tfidf_extra.json")
        add(H, f"post-hoc: TF-IDF {k} obfuscation acc by transform", ", ".join(f"{a} {b['acc']:.2f}" for a, b in st["stress_obfuscation"]["accuracy_by_transform"].items()), "tfidf_extra.json")
    if (ROOT / "results" / "tfidf_extra_repeat.json").exists():
        for k, v in g("tfidf_extra_repeat.json", "models").items():
            add(H, f"post-hoc: REPEAT run TF-IDF {k} latency 1 thread (median/p95 ms)", f"{v['latency_1thread']['median_ms']}/{v['latency_1thread']['p95_ms']}", "tfidf_extra_repeat.json")
    av = g("ablation_val.json")
    for m in ("new_int8_deploy_filled", "v73_int8_raw_view"):
        for c in ("raw_model", "full", "full-trusted", "full_R0_keeps_caution"):
            v = av["val"][m][c]
            add(H, f"post-hoc: VAL ablation {m} {c}: macro-F1, F1 caution, legit→scam k/n", f"{v['macro_f1']:.3f}, {v['f1_caution']:.3f}, {v['legit_as_scam_k_n']}", "ablation_val.json")
    for m in ("seed3_int8_quint8|filled", "v73_int8|filled"):
        v = av["test_variant_once"][m]["full_R0_keeps_caution"]
        add(H, f"post-hoc, motivated by the test ablation: TEST {m.split('|')[0]} full pipeline with R0 keeping caution", f"macro-F1 {v['macro_f1']:.3f}, F1 caution {v['f1_caution']:.3f}, legit→scam {v['legit_as_scam_k_n']}, fixed/broke {v['fixed']}/{v['broke']}", "ablation_val.json")


def reviewer_section():
    """Phase 12 post-hoc analyses requested by the v3 reviewers (saved test predictions, val, synthetic stress only)."""
    if not (ROOT / "results" / "legacy_eval.json").exists():
        return
    H = "Post-hoc (Phase 12, reviewer analyses)"
    le = g("legacy_eval.json")["conditions"]
    for c, lab in (("L0", "legacy file as-is (984 rows)"), ("L1", "legacy messages after cleaning (142 rows)")):
        v = le[c]
        add(H, f"post-hoc: same-recipe ablation, {lab}: test macro-F1 mean ± std (3 seeds); legit→scam per seed", f"{v['test_macro_f1_mean']:.3f} ± {v['test_macro_f1_std']:.3f}; " + ", ".join(r["legit_as_scam"] for r in v["runs"]), "legacy_eval.json")
    rx = g("review_extras.json")
    for m, v in rx["camouflage_controls"]["models"].items():
        add(H, f"post-hoc: camouflage controls {m}: scams caught of 150 (model / rules / rules without R1); benign flagged scam of 150",
            f"{v['model_only']['scam_k']} / {v['with_rules']['scam_k']} / {v['rules_without_R1']['scam_k']}; {v['model_only']['legit_flagged_scam_k']}", "review_extras.json")
    for m, v in rx["ppv"].items():
        add(H, f"post-hoc: scam PPV at validation prevalence ({100 * rx['val_scam_prevalence']:.1f}%) {m}", f"{v['ppv_at_val_prevalence']:.2f}", "review_extras.json")
    for m, v in rx["tost"].items():
        add(H, f"post-hoc: TOST deploy INT8 vs {m}: macro-F1 diff, 90% CI, equivalent at ±0.05", f"{v['diff_point']:+.3f}, [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}], {v['equivalent_at_margin']}", "review_extras.json")
    cp = rx["caution_provenance"]
    add(H, "post-hoc: caution items from the author's phone (export2026): train / val / test", f"{cp['train']['by_source'].get('export2026', 0)}/{cp['train']['n']} / {cp['val']['by_source'].get('export2026', 0)}/{cp['val']['n']} / {cp['test']['by_source'].get('export2026', 0)}/{cp['test']['n']}", "review_extras.json")
    ov = g("uci_mishra_overlap.json")
    add(H, "Mishra & Soni rows that duplicate UCI after normalization", f"{ov['mishra_exact_in_uci']}/{ov['mishra_rows']} = {100 * ov['share']:.1f}%", "uci_mishra_overlap.json")
    if (ROOT / "results" / "kappa.json").exists():
        k = g("kappa.json")
        for ref in ("final_label", "claude_label", "owner_label"):
            add(H, f"second annotator vs {ref}: agreement, Cohen's kappa (n={k['n_compared']})", f"{k['vs_' + ref]['k']}/{k['n_compared']}, {k['vs_' + ref]['cohen_kappa']:.3f}", "kappa.json")


def review2_section():
    """Post-hoc analyses requested by the review of v5 / humanised v2 (paper v12)."""
    if not (ROOT / "results" / "review2_extras.json").exists():
        return
    H = "Post-hoc (v12, second review)"
    r = g("review2_extras.json")
    for m, v in r["second_annotator_models"].items():
        if isinstance(v, dict):
            add(H, f"second annotator vs {m} on the 180 checked items: agreement, kappa", f"{v['vs_second']['agree_k']}/180, {v['vs_second']['kappa']:.2f}", "review2_extras.json")
    for k, v in r["per_rule"].items():
        add(H, f"per-rule test changes, {k}: changed / fixed / broke; macro-F1", f"{v['changed']} / {v['fixed']} / {v['broke']}; {v['macro_f1']:.4f}", "review2_extras.json")
    sd = r["single_device_caution"]
    for m, v in sd["models"].items():
        add(H, f"without author-phone caution ({sd['n_removed_author_caution']} removed, {sd['n_kept']} kept), {m}: macro-F1; 3-class; caution caught of {sd['caution_kept']}",
            f"{v['macro_f1_without_author_caution']:.3f}; {v['macro_f1_3class_ham_scam_promo']:.3f}; {v['caution_recall_k']}", "review2_extras.json")
    for m, v in r["ppv_range"].items():
        if isinstance(v, dict):
            add(H, f"scam PPV at 1% / 6.2% / 20% prevalence, {m}", f"{v['ppv@0.01']:.2f} / {v['ppv@0.062']:.2f} / {v['ppv@0.2']:.2f}", "review2_extras.json")
    for m, v in r["camouflage_ci"]["models"].items():
        for k, x in v.items():
            add(H, f"camouflage {m} {k} (of 150) [95% CI, bootstrap over base texts]", f"{x['k']} [{x['ci95_of150'][0]:.0f}, {x['ci95_of150'][1]:.0f}]", "review2_extras.json")
    e = r["ece_check"]
    add(H, "deployed INT8 test ECE before → after T, 15 bins (10 bins)", f"{e['bins15']['before']:.3f} → {e['bins15']['after']:.3f} ({e['bins10']['before']:.3f} → {e['bins10']['after']:.3f})", "review2_extras.json")
    if (ROOT / "results" / "sizematched_eval.json").exists():
        v = g("sizematched_eval.json")["conditions"]["S984"]
        add(H, "post-hoc: size-matched control, 984 rows sampled from new train: test macro-F1 mean ± std (3 seeds); legit→scam per seed",
            f"{v['test_macro_f1_mean']:.3f} ± {v['test_macro_f1_std']:.3f}; " + ", ".join(x["legit_as_scam"] for x in v["runs"]), "sizematched_eval.json")


def device_section():
    """On-device numbers reported by the app team (external; not run in this repo, no raw log)."""
    if not (ROOT / "results" / "device_bench.json").exists():
        return
    D = "On-device (external, reported by app team)"
    d = g("device_bench.json")
    dv = d["device"]
    add(D, "device", f"{dv['model']}; {dv['soc']}; {dv['ram']}; Android {dv['android']}; {dv['onnxruntime']}", "device_bench.json")
    c = d["latency_ms_clean_run"]
    add(D, "full on-device path latency, ms (median / p95 / min; 200 runs after 50 warm-ups)", f"{c['full_path']['median']} / {c['full_path']['p95']} / {c['full_path']['min']}", "device_bench.json")
    add(D, "model-only latency (session.run), ms (median / p95 / min)", f"{c['model_only']['median']} / {c['model_only']['p95']} / {c['model_only']['min']}", "device_bench.json")
    add(D, "true cold start, ms (model load + session + first inference)", c["cold_start_true"], "device_bench.json")
    m = d["memory_mb_first_run"]
    add(D, "app memory, total PSS MB (baseline -> after 200 inferences; first run, background-loaded)", f"{m['total_pss']['baseline']} -> {m['total_pss']['after_200_inferences']} (+{m['total_pss']['delta']})", "device_bench.json")
    sp = d["spot_check"]
    add(D, "on-device vs PC verdict agreement (synthetic spot-check)", f"{sp['verdict_match_with_pc']}/{sp['n']}", "device_bench.json")
    f = d["latency_ms_first_run_superseded"]
    add(D, "SUPERSEDED first run (after ~1,500 background SMS imports): full path median / p95, ms", f"{f['full_path']['median']} / {f['full_path']['p95']}", "device_bench.json")


def main():
    # --- data audit
    A = "Data audit (legacy)"
    s = g("audit_inventory.json", "stepA_cloning_balanced")
    add(A, "balanced_training_data rows per label (ham/spam/caution/promo)", "/".join(str(s["rows_by_label"][k]) for k in ("ham", "spam", "caution", "promo")), "audit_inventory.json")
    add(A, "same, effective unique (MinHash clusters t=0.8, names+digits masked)", "/".join(str(s["minhash_clusters_t0.8"][k]) for k in ("ham", "spam", "caution", "promo")), "audit_inventory.json")
    add(A, "largest ham cluster (rows)", s["max_cluster_size_t0.8"]["ham"], "audit_inventory.json")
    add(A, "patterned surrogate digits vs label, Cramér's V", f"{g('audit_inventory.json', 'files', 'balanced_training_data.csv', 'digit_artifacts', 'label_vs_patterned_test', 'cramers_v'):.2f}", "audit_inventory.json")
    st = g("audit_stress400.json")
    add(A, "ultimate_stress_test_400: rows / exact-unique / normalized-unique / duplicate share", f"{st['rows']} / {st['exact_unique']} / {st['unique_normalized']} / {pct(st['duplicate_share'])}", "audit_stress400.json")
    # --- data used
    D = "Data used"
    c = g("data_composition.json")
    for sp in ("train (used)", "val", "test (main)"):
        add(D, f"{sp}: rows (clusters) ham/scam/promo/caution",
            " / ".join(f"{c[sp][l]['rows']} ({c[sp][l]['clusters']})" for l in ("ham", "spam", "promo", "caution")), "data_composition.json")
        add(D, f"{sp}: label source", ", ".join(f"{k} {v}" for k, v in c[sp]["label_source"].items()), "data_composition.json")
        add(D, f"{sp}: sources", ", ".join(f"{k} {v}" for k, v in c[sp]["sources"].items()), "data_composition.json")
    add(D, "synthetic train rows excluded (by label)", ", ".join(f"{k} {v}" for k, v in c["train_synthetic_excluded"].items()), "data_composition.json")
    add(D, "native-script train rows excluded", c["train_native_excluded"], "data_composition.json")
    ov = g("splits.json", "overlaps")
    add(D, "split overlaps (clusters / normalized text / ids), all pairs", "all 0" if all(v == 0 for p in ov.values() for v in p.values()) else json.dumps(ov), "splits.json")
    add(D, "test.csv SHA-256", g("final_test.json", "test_sha256"), "final_test.json")
    # --- labels
    L = "Labels"
    sa = g("silver_self_agreement.json")
    add(L, "silver pass-2 self-agreement (n, kappa; NOT independent)", f"n={sa['n']}, kappa={sa['cohen_kappa']:.2f}", "silver_self_agreement.json")
    chk = pd.read_csv(ROOT / "annotation" / "check_results.csv").drop_duplicates("item_id", keep="last")
    ROWS.append((L, "owner accept/override check: answered / overrides / override rate", f"{len(chk)} / {(chk.final_label != chk.proposed).sum()} / {pct((chk.final_label != chk.proposed).mean())}",
                 "check tool session 2026-10-18", "src/check_app.py", "annotation/check_results.csv"))
    ROWS.append((L, "total Claude label judgments", "4,485 (gold 673 + silver 3,525 + pass-2 287)", "see docs/AI_USE_LOG.md", "src/label_batches.py", "docs/AI_USE_LOG.md"))
    # --- sweep and seeds
    S = "Training / validation"
    sel = load("sweep_B_selection.json")
    ROWS.append((S, "selected config", json.dumps(sel["final_config"]), "sweep stage B (Colab)", "src/sweep.py", "results/sweep_B_selection.json"))
    ROWS.append((S, "stage-B best mean val macro-F1 (2 seeds, optimistic)", f"{sel['best_no_app_change_f1']:.3f}", "sweep stage B", "src/sweep.py", "results/sweep_B_selection.json"))
    a = pd.read_csv(ROOT / "results" / "sweep_A.csv")
    for k, lab in [("balance=S0", "S0 class-weighted"), ("balance=S1", "S1 undersampled")]:
        ROWS.append((S, f"stage A mean val macro-F1, {lab}", f"{a[a.config.str.contains(k)].val_macro_f1.mean():.3f}", "sweep stage A", "src/sweep.py", "results/sweep_A.csv"))
    r0 = a[a.config.str.contains("synthetic_ratio=0.0")].val_macro_f1
    r1 = a[a.config.str.contains("balance=S0") & a.config.str.contains("digits=D1") & a.config.str.contains("placeholders=U1") & a.config.str.contains("synthetic_ratio=1.0")].val_macro_f1
    ROWS.append((S, "S0/D1/U1 val macro-F1: real-only vs with synthetic (per seed)", f"{', '.join(f'{x:.3f}' for x in r0)} vs {', '.join(f'{x:.3f}' for x in r1)}", "sweep stage A", "src/sweep.py", "results/sweep_A.csv"))
    cal = g("calibrate.json", "summary")
    for k, lab in [("val_macro_f1", "5 seeds val macro-F1"), ("val_f1_caution", "5 seeds val F1 caution"), ("temperature", "temperature T"),
                   ("ece_before", "val ECE before T (in-sample)"), ("ece_after", "val ECE after T (in-sample)")]:
        add(S, lab, f"{cal[k]['mean']:.3f} ± {cal[k]['std']:.3f}", "calibrate.json")
    # --- export
    E = "Export / deployment"
    ex = g("export_onnx.json")
    for k, v in ex["files"].items():
        add(E, f"{k} ONNX size", f"{v['size']['bytes']:,} B = {v['size']['MB']} MB = {v['size']['MiB']} MiB", "export_onnx.json")
    add(E, "FP32 checkpoint size", f"{ex['fp32_checkpoint_size']['bytes']:,} B = {ex['fp32_checkpoint_size']['MB']} MB = {ex['fp32_checkpoint_size']['MiB']} MiB", "export_onnx.json")
    add(E, "quant_pre_process default error", ex.get("pre_process_default_error"), "export_onnx.json")
    pv = ex["parity_val"]
    for k in ("onnx_fp32_app_tok", "onnx_int8_QUInt8_app_tok", "onnx_int8_QInt8_app_tok"):
        add(E, f"val parity {k}: agreement vs PyTorch, mean KL", f"{pct(pv[k]['argmax_agree_vs_torch'])}, {pv[k]['mean_kl_torch_to_this']:.4f}", "export_onnx.json")
    tp = g("final_test.json", "parity_seed3_filled")
    for k, v in tp.items():
        add(E, f"test parity FP32 vs {k}: agreement, KL, ΔF1", f"{pct(v['argmax_agree'])}, {v['mean_kl_fp32_to_int8']:.3f}, {v['macro_f1_diff_main']:+.3f}", "final_test.json")
    b = g("bench_onnx.json", "models")
    for k, v in b.items():
        add(E, f"desktop CPU latency {k} (median/p95 ms; 1 thread ; 4 threads)", f"{v['threads_1']['median_ms']}/{v['threads_1']['p95_ms']} ; {v['threads_4']['median_ms']}/{v['threads_4']['p95_ms']}", "bench_onnx.json")
    if (ROOT / "results" / "bench_onnx_repeat.json").exists():  # reproduction run of make_all.ps1 (timing noise)
        for k, v in g("bench_onnx_repeat.json", "models").items():
            add(E, f"REPEAT run (make_all reproduction) desktop CPU latency {k} (median/p95 ms; 1 thread ; 4 threads)", f"{v['threads_1']['median_ms']}/{v['threads_1']['p95_ms']} ; {v['threads_4']['median_ms']}/{v['threads_4']['p95_ms']}", "bench_onnx_repeat.json")
    v7 = g("check_onnx.json", "onnx_size")
    add(E, "v7.3 deployed ONNX size", f"{v7['bytes']:,} B = {v7['MB']} MB = {v7['MiB']} MiB", "check_onnx.json")
    # --- test
    T = "Test (single run)"
    stt = g("stats.json")
    for k in ("new_int8_quint8", "new_int8_qint8", "tfidf_lr", "tfidf_svm", "v73_int8", "v73_fp32", "new_int8_quint8+rules", "v73_int8+rules"):
        p, ci = stt["point"][k], stt["ci95"][k]["macro_f1"]
        add(T, f"{k}: macro-F1 [95% CI], MCC, legit→scam", f"{p['macro_f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], {p['mcc']:.3f}, {pct(p['legit_as_scam'])}", "stats.json")
        add(T, f"{k}: F1 ham/scam/promo/caution", "/".join(f"{p['f1_' + l]:.3f}" for l in ("ham", "spam", "promo", "caution")), "stats.json")
    sm = stt["seeds_macro_f1"]
    add(T, "5 seeds FP32 test macro-F1", f"{sm['mean']:.3f} ± {sm['std']:.3f} ({', '.join(f'{x:.3f}' for x in sm['values'])})", "stats.json")
    for k, v in stt["diff_vs_deploy_ci95"].items():
        if k in ("v73_int8", "v73_int8+rules", "tfidf_lr", "tfidf_svm", "new_int8_quint8+rules"):
            add(T, f"Δ macro-F1 deploy − {k} [95% CI]", f"{v['macro_f1']['diff']:+.3f} [{v['macro_f1']['ci95'][0]:+.3f}, {v['macro_f1']['ci95'][1]:+.3f}]", "stats.json")
    for k, v in stt["mcnemar"].items():
        add(T, f"McNemar {k} (b/c, p)", f"{v['new_right_other_wrong']}/{v['new_wrong_other_right']}, p={v['p_exact']:.2g}", "stats.json")
    e = stt["ece_test_seed_summary"]
    add(T, "test ECE 5 seeds before → after T", f"{e['before']['mean']:.3f} ± {e['before']['std']:.3f} → {e['after']['mean']:.3f} ± {e['after']['std']:.3f}", "stats.json")
    add(T, "test main n / clusters", f"{stt['n_main']} / {stt['n_clusters']}", "stats.json")
    old = g("eval_v73_test.json", "models")
    add(T, "v7.3 earlier single run (raw view) INT8 / FP32 macro-F1", f"{old['onnx_int8_app_tokenizer']['main_raw']['macro_f1']:.3f} / {old['fp32']['main_raw']['macro_f1']:.3f}", "eval_v73_test.json")
    # --- ablation, stress, abstention, errors
    X = "Ablation / stress / abstention / errors"
    ab = g("ablation.json", "models")
    for m in ("seed3_int8_quint8|filled", "v73_int8|filled"):
        for cfg in ("raw_model", "full", "full-trusted", "+trusted", "+suffix", "+safety"):
            v = ab[m][cfg]
            add(X, f"ablation {m.split('|')[0]} {cfg}: F1, legit→scam, fixed/broke", f"{v['macro_f1']:.3f}, {pct(v['legit_as_scam'])}, {v['fixed']}/{v['broke']}", "ablation.json")
    s = g("stress_eval.json")
    for st in s:
        for m in ("new_int8_quint8", "v73_int8"):
            add(X, f"stress {st} {m} accuracy (n={s[st]['n']}, unique {s[st]['n_unique_text']})", f"{s[st]['models'][m]['raw']['accuracy']:.3f}", "stress_eval.json")
        if st == "stress_spoof":
            for m in ("new_int8_quint8", "v73_int8"):
                add(X, f"spoof {m}: pipeline acc; camouflaged-scam raw acc", f"{s[st]['models'][m]['pipeline']['accuracy']:.3f}; {s[st]['models'][m]['raw_accuracy_by_camouflage']['1']:.3f}", "stress_eval.json")
    ab2 = g("abstain.json")
    add(X, "abstention: tau selected on val", str(ab2["tau_selected_on_val"]), "abstain.json")
    add(X, "abstention val tau=0.95: scam warned rate / legit newly caution", f"{pct(ab2['val_grid']['0.95']['scam_warned_rate'])} / {pct(ab2['val_grid']['0.95']['legit_newly_caution'])}", "abstain.json")
    add(X, "risk-coverage AURC val / test", f"{ab2['risk_coverage']['val']['maxprob']['aurc']:.4f} / {ab2['risk_coverage']['test']['maxprob']['aurc']:.4f}", "abstain.json")
    er = g("error_analysis.json")
    add(X, "deploy-model errors on main test", er["n_errors_main"], "error_analysis.json")
    add(X, "top-50 error categories", "; ".join(f"{k.split(' ')[0]} {v}" for k, v in er["top50_category_counts"].items()), "error_analysis.json")

    posthoc_section()
    device_section()
    reviewer_section()
    review2_section()
    lines = ["# Paper numbers (generated by `src/make_paper_numbers.py`; do not edit by hand)", "",
             "Every number intended for the paper, with the run that produced it. Sizes carry units; latency is **desktop CPU**, not phone.", ""]
    for sec in dict.fromkeys(r[0] for r in ROWS):
        lines += [f"## {sec}", "", "| Quantity | Value | Run id | Script | File |", "|---|---|---|---|---|"]
        lines += [f"| {w} | {v} | `{rid}` | `{sc}` | `{fi}` |" for s_, w, v, rid, sc, fi in ROWS if s_ == sec]
        lines.append("")
    (ROOT / "results" / "paper_numbers.md").write_text("\n".join(lines), encoding="utf-8")
    print(len(ROWS), "numbers written")


if __name__ == "__main__":
    main()
