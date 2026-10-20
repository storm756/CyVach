"""Phase 0 inventory: counts and statistics only (never prints message text).

Covers: rows/columns/labels, script mix, exact + normalized + near-duplicate rates,
is_synthetic share, sender availability, within-file label conflicts, cross-file
overlap and label agreement, cloning check (Step A) and surrogate-digit
artifacts (Step B).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FILE_SCHEMA, RAW, load_csv, new_run, normalize, save_result, script_of  # noqa: E402
from scrub import scrub, sender_kind  # noqa: E402

SEED = 13
SMS_FILES = [
    "balanced_training_data.csv",
    "financial_fraud_training_10000.csv",
    "financial_fraud_newlybalanced_10000.csv",
    "Indian_sms_scam.csv",
    "spam.csv",
    "ultimate_stress_test_400.csv",
]


def shingles(t: str, k: int = 5):
    t = f" {t} "
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def near_dup_clusters(keys: list[str], threshold: float, num_perm: int = 128) -> list[int]:
    """Union-find over MinHash-LSH candidates. Returns a cluster id per item."""
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs = []
    first_of_key: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in first_of_key:  # exact dup: join directly
            parent[find(i)] = find(first_of_key[k])
            sigs.append(None)
            continue
        first_of_key[k] = i
        m = MinHash(num_perm=num_perm, seed=SEED)
        for s in shingles(k):
            m.update(s.encode("utf-8"))
        for j in lsh.query(m):
            parent[find(i)] = find(j)
        lsh.insert(i, m)
        sigs.append(m)
    return [find(i) for i in range(len(keys))]


# ---- Step B: patterned digit runs ----
def digit_pattern(run: str) -> str | None:
    n = len(run)
    if n < 6:
        return None
    if len(set(run)) == 1:
        return "same_digit"
    d = [int(c) for c in run]
    if all((d[i + 1] - d[i]) % 10 == 1 for i in range(n - 1)):
        return "ascending"
    if all((d[i] - d[i + 1]) % 10 == 1 for i in range(n - 1)):
        return "descending"
    if n % 2 == 0 and all(run[i] == run[i + 1] for i in range(0, n, 2)):
        pairs = [d[i] for i in range(0, n, 2)]
        if all(abs(pairs[i + 1] - pairs[i]) % 10 == 1 for i in range(len(pairs) - 1)):
            return "pair_step"  # 112233, 445566, 998877
        return "pair_repeat"
    for p in (2, 3):
        if n >= 2 * p and all(run[i] == run[i % p] for i in range(n)):
            return f"period{p}"
    # long ascending/descending prefix of >=5 (e.g. 9876543xx or 12345678xx)
    best = 1
    cur_a = cur_d = 1
    for i in range(1, n):
        cur_a = cur_a + 1 if (d[i] - d[i - 1]) % 10 == 1 else 1
        cur_d = cur_d + 1 if (d[i - 1] - d[i]) % 10 == 1 else 1
        best = max(best, cur_a, cur_d)
    if best >= 5:
        return "seq_ge5"
    return None


def digit_stats(texts, labels, rng):
    per_label = defaultdict(Counter)
    for t, y in zip(texts, labels):
        runs = re.findall(r"\d{6,}", str(t))
        per_label[y]["rows"] += 1
        if runs:
            per_label[y]["rows_with_run6"] += 1
        pats = [digit_pattern(r) for r in runs]
        if any(pats):
            per_label[y]["rows_with_patterned"] += 1
        for r, p in zip(runs, pats):
            per_label[y]["runs6"] += 1
            if p:
                per_label[y]["patterned_runs"] += 1
                per_label[y][f"pat_{p}"] += 1
    # random baseline: same length distribution, uniformly random digits
    lengths = [len(r) for t in texts for r in re.findall(r"\d{6,}", str(t))]
    sim = 0
    for L in lengths * 20:
        s = "".join(rng.choice("0123456789") for _ in range(L))
        if s[0] in "6789" and L == 10:  # phone-like: leading digit realistic
            pass
        sim += bool(digit_pattern(s))
    baseline = sim / max(1, len(lengths) * 20)
    out = {k: dict(v) for k, v in per_label.items()}
    # chi-square: label vs has_patterned
    try:
        from scipy.stats import chi2_contingency
        labs = sorted(per_label)
        table = [[per_label[l]["rows_with_patterned"], per_label[l]["rows"] - per_label[l]["rows_with_patterned"]] for l in labs]
        if all(sum(r) > 0 for r in table) and sum(r[0] for r in table) > 0:
            chi2, p, dof, _ = chi2_contingency(table)
            n = sum(sum(r) for r in table)
            cramer = (chi2 / n) ** 0.5
            stat = {"chi2": chi2, "p": p, "dof": dof, "cramers_v": cramer}
        else:
            stat = "not computable (no patterned runs or empty label)"
    except Exception as e:  # pragma: no cover
        stat = f"error: {e}"
    return {"per_label": out, "random_baseline_patterned_rate_per_run": baseline,
            "label_vs_patterned_test": stat}


def inventory(fname: str, rng) -> tuple[dict, list[str], list[str]]:
    text_col, label_col, sender_col = FILE_SCHEMA[fname]
    df = load_csv(RAW / fname)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).str.strip().tolist()
    norm = [normalize(t) for t in texts]
    info: dict = {"file": fname, "rows": len(df), "columns": list(df.columns)}
    info["label_counts"] = dict(Counter(labels))
    info["script_mix"] = dict(Counter(script_of(t) for t in texts))
    info["empty_text_rows"] = sum(1 for t in texts if not t.strip())
    info["exact_unique_raw"] = len(set(texts))
    info["exact_dup_rows_raw"] = len(texts) - len(set(texts))
    info["unique_normalized"] = len(set(norm))
    info["dup_rows_normalized"] = len(norm) - len(set(norm))
    lens = np.array([len(t) for t in texts])
    info["char_len"] = {"median": float(np.median(lens)), "p95": float(np.percentile(lens, 95)), "max": int(lens.max())}
    if "is_synthetic" in df.columns:
        info["is_synthetic_counts"] = dict(Counter(df["is_synthetic"]))
        info["is_synthetic_by_label"] = {l: dict(Counter(df.loc[df[label_col] == l, "is_synthetic"])) for l in set(labels)}
    for extra in ("category", "meta", "Type", "Confidence", "Actual"):
        if extra in df.columns:
            vc = Counter(df[extra])
            info[f"{extra}_values"] = dict(vc.most_common(40)) if len(vc) <= 40 else {"n_distinct": len(vc), "top10": dict(vc.most_common(10))}
    if "category" in df.columns:
        info["label_x_category"] = {l: dict(Counter(df.loc[df[label_col] == l, "category"])) for l in set(labels)}
    if sender_col:
        s = df[sender_col].astype(str)
        info["sender_nonempty"] = int((s.str.strip() != "").sum())
        info["sender_kinds"] = dict(Counter(sender_kind(x) for x in s))
        info["sender_kinds_by_label"] = {l: dict(Counter(sender_kind(x) for x in s[df[label_col] == l])) for l in set(labels)}
        info["distinct_senders"] = int(s.nunique())

    # within-file label conflicts on normalized text
    by_key = defaultdict(set)
    for k, y in zip(norm, labels):
        by_key[k].add(y)
    conflicts = {k: v for k, v in by_key.items() if len(v) > 1}
    info["label_conflict_keys"] = len(conflicts)
    info["label_conflict_rows"] = sum(1 for k in norm if k in conflicts)
    info["label_conflict_pairs"] = dict(Counter("|".join(sorted(v)) for v in conflicts.values()))

    # near-duplicates (MinHash on normalized text)
    for thr in (0.8,):
        cl = near_dup_clusters(norm, thr)
        info[f"near_dup_clusters_t{thr}"] = len(set(cl))
        per_label = defaultdict(set)
        for c, y in zip(cl, labels):
            per_label[y].add(c)
        info[f"effective_unique_by_label_t{thr}"] = {y: len(v) for y, v in per_label.items()}
        clabels = defaultdict(set)
        for c, y in zip(cl, labels):
            clabels[c].add(y)
        info[f"clusters_spanning_labels_t{thr}"] = sum(1 for v in clabels.values() if len(v) > 1)
        sizes = Counter(cl)
        info[f"largest_cluster_sizes_t{thr}"] = sorted(sizes.values(), reverse=True)[:10]
    return info, norm, labels


def cloning_check(fname: str) -> dict:
    """Step A: mask names (scrubber) + digits, then count templates per class."""
    text_col, label_col, _ = FILE_SCHEMA[fname]
    df = load_csv(RAW / fname)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).str.strip().tolist()
    k_digits = [normalize(t) for t in texts]
    k_names = [normalize(scrub(t)) for t in texts]
    # additionally mask every capitalized/ALLCAPS token that is not sentence-initial (aggressive name proxy)
    def aggressive(t):
        s = scrub(t)
        s = re.sub(r"(?<=[a-z0-9,;:] )([A-Z][a-z]+|[A-Z]{3,})\b", "<CAP>", s)
        return normalize(s)
    k_aggr = [aggressive(t) for t in texts]
    out = {"rows_by_label": dict(Counter(labels))}
    for name, keys in (("digits_masked", k_digits), ("digits_names_masked", k_names), ("aggressive_caps_masked", k_aggr)):
        per = defaultdict(set)
        for k, y in zip(keys, labels):
            per[y].add(k)
        out[f"exact_templates_{name}"] = {y: len(v) for y, v in per.items()}
    for thr in (0.8, 0.6):
        cl = near_dup_clusters(k_names, thr)
        per = defaultdict(set)
        for c, y in zip(cl, labels):
            per[y].add(c)
        sizes = defaultdict(Counter)
        for c, y in zip(cl, labels):
            sizes[y][c] += 1
        out[f"minhash_clusters_t{thr}"] = {y: len(v) for y, v in per.items()}
        out[f"rows_in_clusters_size_ge3_t{thr}"] = {y: sum(n for n in sizes[y].values() if n >= 3) for y in sizes}
        out[f"max_cluster_size_t{thr}"] = {y: max(sizes[y].values()) for y in sizes}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit_inventory.json")
    args = ap.parse_args()
    rng = random.Random(SEED)
    run = new_run("audit_inventory", {"seed": SEED, "files": SMS_FILES, "minhash": "char5, 128 perm, LSH t=0.8"})
    files, norms = {}, {}
    for f in SMS_FILES:
        info, norm, labels = inventory(f, rng)
        info["digit_artifacts"] = digit_stats(load_csv(RAW / f)[FILE_SCHEMA[f][0]].tolist(), labels, rng)
        files[f] = info
        norms[f] = (norm, labels)
        print(f"done {f}: rows={info['rows']}")

    # cross-file overlap on normalized text
    cross = {}
    key_sets = {f: set(n) for f, (n, _) in norms.items()}
    for i, a in enumerate(SMS_FILES):
        for b in SMS_FILES[i + 1:]:
            inter = key_sets[a] & key_sets[b]
            if not inter:
                continue
            la = defaultdict(set)
            for k, y in zip(*norms[a]):
                if k in inter:
                    la[k].add(y)
            lb = defaultdict(set)
            for k, y in zip(*norms[b]):
                if k in inter:
                    lb[k].add(y)
            pairs = Counter(f"{'/'.join(sorted(la[k]))} -> {'/'.join(sorted(lb[k]))}" for k in inter)
            cross[f"{a} & {b}"] = {"shared_normalized_texts": len(inter), "label_pairs": dict(pairs.most_common(20))}

    clone = cloning_check("balanced_training_data.csv")
    payload = {"files": files, "cross_file_overlap": cross, "stepA_cloning_balanced": clone}
    p = save_result(run, payload, args.out)
    print("saved", p, run["run_id"])


if __name__ == "__main__":
    main()
