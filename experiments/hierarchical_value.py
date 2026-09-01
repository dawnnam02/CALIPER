"""Is per-target calibration worth it, and after how many wells?

Validates (or kills) caliper/hierarchical.py, which was written on the strength
of a literature finding -- that precision at a fixed confidence threshold ranges
0.1 to 1.0 across targets -- and then never tested, because until the Overath
data arrived there was only one target to test against.

The question has to be posed correctly
--------------------------------------
Plain leave-one-target-out cannot evaluate per-target calibration: a held-out
target has no labels, so there is nothing to fit a target-specific curve on.
Pooled calibration is the only option there, and "hierarchical" degenerates to
it by construction.

The question a campaign actually faces is different:

    A new target has had one round. There are k wells of outcome data.
    Is it better to use them, or to trust the pooled curve from other targets?

So: hold out a target, reveal k of its labels, fit
  * POOLED     -- other targets only, ignoring the k
  * TARGET     -- the k labels only, ignoring other targets
  * HIERARCHICAL -- the k shrunk toward the pooled curve
and score all three on the target's remaining, unseen designs.

The measured per-target spread on this data is large -- AUC 0.573 (Mdm2) to
1.000 (LTK), hit rate 2.1% to 57.3% -- so if partial pooling is ever going to
earn its place, it should be here.

Data: Overath et al. 2025, Zenodo 10.5281/zenodo.15722219, CC-BY-4.0
Run:  python experiments/hierarchical_value.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.calibrate import brier_score, expected_calibration_error
from caliper.hierarchical import HierarchicalCalibrator
from caliper.smallsample import PlattCalibrator, average_precision
from caliper.stats import bootstrap_ci, equal_mass_ece, paired_bootstrap
from caliper.types import stable_hash

DATA = Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"
SCORE = "af3_ipSAE_min"           # best single metric on this data, AUC 0.786
REVEAL = [5, 10, 20, 40, 80]      # wells of outcome data on the new target
N_REPEATS = 20                    # random reveals per (target, k)
MIN_TEST = 40                     # need enough held-back designs to score on


def load() -> pd.DataFrame:
    if not DATA.exists():
        print(f"dataset not found: {DATA}\n"
              "  curl -L -o data/overath/final_dataset.csv \\\n"
              "    https://zenodo.org/api/records/15722219/files/"
              "final_dataset.csv/content", file=sys.stderr)
        raise SystemExit(2)
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df.binder.notna()].copy()
    df["y"] = df.binder.astype(bool).astype(int)
    df[SCORE] = pd.to_numeric(df[SCORE], errors="coerce")
    return df.dropna(subset=[SCORE]).reset_index(drop=True)


def scale(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map scores to [0,1] using TRAINING bounds only, so the test target's
    own range never leaks into the transform."""
    if hi <= lo:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def main() -> int:
    df = load()
    targets = sorted(df.target_id.unique())
    print("=" * 78)
    print("Is per-target calibration worth it? | Overath et al. 2025")
    print("=" * 78)
    print(f"  {len(df):,} designs | {len(targets)} targets | "
          f"score = {SCORE} | {N_REPEATS} random reveals per point")

    out: dict = defaultdict(lambda: defaultdict(list))
    per_target: dict = defaultdict(lambda: defaultdict(list))

    for held in targets:
        te_all = df[df.target_id == held]
        tr = df[df.target_id != held]
        if len(te_all) < MIN_TEST + max(REVEAL) // 2 or te_all.y.sum() < 5:
            continue

        lo, hi = float(tr[SCORE].min()), float(tr[SCORE].max())
        s_tr = scale(tr[SCORE].to_numpy(float), lo, hi)
        y_tr = tr.y.to_numpy()
        s_all = scale(te_all[SCORE].to_numpy(float), lo, hi)
        y_all = te_all.y.to_numpy()
        n = len(te_all)

        # pooled curve: fitted once on the other 14 targets
        pooled = PlattCalibrator().fit(s_tr, y_tr)

        for k in REVEAL:
            if n - k < MIN_TEST:
                continue
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(
                    int(stable_hash([held, k, rep]), 16) % 2**31)
                idx = rng.permutation(n)
                rev, test = idx[:k], idx[k:]
                if len(set(y_all[rev].tolist())) < 2:
                    continue        # revealed wells all one class: skip
                sr, yr = s_all[rev], y_all[rev]
                st, yt = s_all[test], y_all[test]

                preds = {"pooled": pooled.predict(st)}
                try:
                    preds["target_only"] = PlattCalibrator().fit(sr, yr).predict(st)
                except Exception:
                    continue

                h = HierarchicalCalibrator(shrink_k=25.0)
                h.fit(np.concatenate([np.full(len(s_tr), "pool"),
                                      np.full(k, str(held))]),
                      np.concatenate([s_tr, sr]),
                      np.concatenate([y_tr, yr]))
                preds["hierarchical"] = np.asarray(h.predict(str(held), st))

                for name, p in preds.items():
                    out[k][f"{name}_ece"].append(equal_mass_ece(p, yt))
                    out[k][f"{name}_brier"].append(brier_score(p, yt))
                    per_target[held][f"{name}_k{k}"].append(brier_score(p, yt))

    print()
    print("Out-of-sample Brier score on the unseen part of the held-out target")
    print("(lower is better; base rate differs per target so read the columns "
          "against each other, not across rows)")
    hdr = (f"  {'wells revealed':<16}{'pooled':>24}{'target only':>24}"
           f"{'hierarchical':>24}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in REVEAL:
        if k not in out or not out[k]["pooled_brier"]:
            continue
        row = f"  k = {k:<12}"
        for name in ("pooled", "target_only", "hierarchical"):
            row += f"{str(bootstrap_ci(out[k][f'{name}_brier'], seed=1)):>24}"
        print(row)

    print()
    print("Paired tests, hierarchical vs the alternatives (lower Brier is better)")
    for k in REVEAL:
        if k not in out or not out[k]["pooled_brier"]:
            continue
        for rival in ("pooled", "target_only"):
            t = paired_bootstrap(out[k]["hierarchical_brier"],
                                 out[k][f"{rival}_brier"],
                                 "hierarchical", rival, seed=4)
            print(f"  k={k:<4} vs {rival:<12} {t.verdict(lower_is_better=True)}")

    print()
    print("Calibration error (equal-mass ECE), same runs")
    hdr2 = (f"  {'wells revealed':<16}{'pooled':>24}{'target only':>24}"
            f"{'hierarchical':>24}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for k in REVEAL:
        if k not in out or not out[k]["pooled_ece"]:
            continue
        row = f"  k = {k:<12}"
        for name in ("pooled", "target_only", "hierarchical"):
            row += f"{str(bootstrap_ci(out[k][f'{name}_ece'], seed=1)):>24}"
        print(row)

    res = {str(k): {m: [float(x) for x in v] for m, v in d.items()}
           for k, d in out.items()}
    p = Path(__file__).resolve().parent / "hierarchical_results.json"
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nraw results -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
