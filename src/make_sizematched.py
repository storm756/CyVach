"""POST-HOC size-matched data ablation (v4 review): is the gain from data quality or just quantity?

For seeds 1-3, draws 984 rows (= size of the legacy training set L0) from the REAL rows of the final train split,
stratified by class at the train split's natural proportions, with random_state = seed. Each set is trained with the
final recipe (src/train.py, same flags as the five final seeds) and evaluated once, post hoc, on the frozen test set
(src/sizematched_eval.py). Writes data/sizematched_s{1,2,3}.csv (gitignored) and results/sizematched_data.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, new_run, save_result  # noqa: E402
from subsets import load_split  # noqa: E402

N = 984


def main():
    run = new_run("make_sizematched", {"n": N, "seeds": [1, 2, 3]})
    tr = load_split(ROOT / "data" / "train.csv", drop_native=True)
    real = tr[tr["is_synthetic"].astype(str) != "True"]
    share = real["final_label"].value_counts(normalize=True)
    quota = (share * N).round().astype(int)
    quota[quota.idxmax()] += N - quota.sum()  # rounding remainder goes to the largest class
    res = {"post_hoc": True, "n_real_train": int(len(real)), "quota": quota.to_dict(), "sets": {}}
    for s in (1, 2, 3):
        d = pd.concat([real[real["final_label"] == c].sample(n=int(k), random_state=s) for c, k in quota.items()])
        d = d.sample(frac=1, random_state=s)
        d.to_csv(ROOT / "data" / f"sizematched_s{s}.csv", index=False)
        res["sets"][f"s{s}"] = {"n": int(len(d)), "by_label": d["final_label"].value_counts().to_dict(),
                                "clusters": int(d["cluster_id"].nunique())}
    save_result(run, res, "sizematched_data.json")
    print(res); print(run["run_id"])


if __name__ == "__main__":
    main()
