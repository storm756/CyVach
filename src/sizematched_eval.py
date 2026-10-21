"""POST-HOC size-matched data ablation, evaluation: each model trained on 984 rows sampled from the new train split
(src/make_sizematched.py; seeds 1-3, final recipe) is scored ONCE on the frozen test set, with exactly the protocol of
src/legacy_eval.py. Usage: python src/sizematched_eval.py RUN_ID_SEED1 RUN_ID_SEED2 RUN_ID_SEED3
Output: results/sizematched_eval.json (post-hoc).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_eval import main  # noqa: E402

if __name__ == "__main__":
    assert len(sys.argv) == 4, __doc__
    main({"S984": sys.argv[1:]}, "sizematched_eval")
