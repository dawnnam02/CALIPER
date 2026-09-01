"""The experiment that decides whether CALIPER is worth anything.

Addresses CRITIQUE A1, A2, A3, A7, C1, C2 in one place:

* every scheduler sees the SAME pre-computed score matrix, so differences are
  attributable to the scheduling policy and nothing else;
* many seeds, reported with bootstrap intervals rather than as point estimates;
* the exploration quota is ABLATED (0% vs 5% vs 25%) so the claim that it helps
  calibration is tested rather than asserted;
* calibration quality is out-of-fold throughout.

Run:  python experiments/validate.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.baselines import compare_schedulers, quantile_thresholds
from caliper.backends.simulator import (SimAssay, SimDesigner, SimScorer,
                                        noise_for_auc, roc_auc, true_affinity)
from caliper.calibrate import Calibrator
from caliper.stats import bootstrap_ci, cross_validated_calibration, wilson
from caliper.types import Target

# ---- knobs -----------------------------------------------------------------
N_SEEDS = 12
N_START = 3000
N_FINAL = 24
BASE_RATE = 0.116          # Overath 2025: 436/3,766
STAGE_AUCS = [("seq", 0.62, 0.5), ("fold", 0.68, 20.0), ("refold", 0.75, 60.0)]
EXPLORE_FRACTIONS = [0.0, 0.05, 0.25]
TARGET = Target(
    "RBD",
    "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVY",
    hotspots=(5, 12, 30),
)
# ----------------------------------------------------------------------------


def build_world(seed: int):
    """One independent campaign world: designs, truth, scores, outcomes."""
    designer = SimDesigner()
    pool = designer.design(TARGET, N_START, seed)
    seqs = [c.sequence for c in pool]

    assay = SimAssay.for_base_rate(TARGET, seqs, BASE_RATE)
    outcomes = np.array(assay.run(TARGET, pool, seed), dtype=int)
    truth = np.array([true_affinity(TARGET, s) for s in seqs], dtype=float)

    scores = {}
    realised_auc = {}
    for stage, auc, _cost in STAGE_AUCS:
        gain, bias = 0.9, 0.1
        sigma = noise_for_auc(TARGET, seqs, outcomes, auc, gain=gain, bias=bias,
                              seed=seed)
        sc = SimScorer(stage, noise=sigma, unit_cost=1.0, bias=bias, gain=gain)
        v = np.array(sc.score(TARGET, pool, seed), dtype=float)
        scores[stage] = v
        # CRITIQUE B5: verify the AUC we asked for is the AUC we got.
        realised_auc[stage] = roc_auc(v, outcomes)
    return pool, seqs, truth, scores, outcomes, realised_auc


def main() -> int:
    stages = [s for s, _, _ in STAGE_AUCS]
    costs = [c for _, _, c in STAGE_AUCS]

    per_sched = defaultdict(lambda: defaultdict(list))
    auc_check = defaultdict(list)
    explore_cal = defaultdict(list)

    for seed in range(N_SEEDS):
        pool, seqs, truth, scores, outcomes, rauc = build_world(seed)
        for k, v in rauc.items():
            auc_check[k].append(v)

        thresholds = quantile_thresholds(scores, stages, keep_fraction=1 / 3)
        results = compare_schedulers(scores, stages, costs, N_START, N_FINAL,
                                     truth=truth, thresholds=thresholds, seed=seed)

        true_top = set(np.argsort(-truth, kind="mergesort")[:N_FINAL].tolist())
        for r in results:
            kept = r.kept
            if not kept:
                per_sched[r.name]["hits"].append(0.0)
                per_sched[r.name]["recall"].append(0.0)
                per_sched[r.name]["cost"].append(r.cost)
                per_sched[r.name]["n_kept"].append(0)
                continue
            per_sched[r.name]["hits"].append(float(outcomes[kept].mean()))
            per_sched[r.name]["recall"].append(
                len(true_top & set(kept)) / len(true_top))
            per_sched[r.name]["cost"].append(r.cost)
            per_sched[r.name]["n_kept"].append(len(kept))

        # ---- exploration ablation (CRITIQUE A7) --------------------------
        sh = next(r for r in results if r.name == "successive_halving")
        rejected = [i for i in range(N_START) if i not in set(sh.kept)]
        rng = np.random.default_rng(seed)
        for frac in EXPLORE_FRACTIONS:
            capacity = 48
            n_ex = int(round(capacity * frac))
            n_ex = min(n_ex, len(rejected))
            idx = list(sh.kept[:capacity - n_ex])
            if n_ex:
                idx += [rejected[int(i)] for i in
                        rng.choice(len(rejected), size=n_ex, replace=False)]
            x = scores[stages[-1]][idx]
            y = outcomes[idx]
            cv = cross_validated_calibration(x, y, seed=seed)
            if "error" not in cv:
                explore_cal[frac].append(cv["ece_calibrated_oof"])

    # ---- report --------------------------------------------------------
    print("=" * 78)
    print(f"CALIPER validation — {N_SEEDS} seeds, {N_START} designs, "
          f"shortlist {N_FINAL}, base rate {BASE_RATE:.3f}")
    print("=" * 78)

    print("\nSimulator fidelity check (asked vs realised ROC AUC):")
    for stage, auc, _ in STAGE_AUCS:
        got = bootstrap_ci(auc_check[stage], seed=0)
        print(f"  {stage:<8} asked {auc:.2f}   realised {got}")

    print("\nScheduler comparison  (mean [95% bootstrap CI] over seeds)")
    print(f"  {'scheduler':<22}{'hit rate':>26}{'top-k recall':>26}"
          f"{'kept':>7}{'cost':>11}")
    print("  " + "-" * 92)
    order = ["oracle", "successive_halving", "fixed_threshold_topk",
             "full_sweep", "fixed_threshold", "random"]
    for name in order:
        if name not in per_sched:
            continue
        d = per_sched[name]
        h = bootstrap_ci(d["hits"], seed=1)
        r = bootstrap_ci(d["recall"], seed=1)
        c = float(np.mean(d["cost"]))
        nk = float(np.mean(d["n_kept"]))
        print(f"  {name:<22}{str(h):>26}{str(r):>26}{nk:>7.0f}{c:>11,.0f}")

    print("\nExploration-quota ablation — out-of-fold calibration ECE (lower better)")
    print(f"  {'explore fraction':<22}{'OOF ECE':>26}")
    print("  " + "-" * 48)
    for frac in EXPLORE_FRACTIONS:
        if explore_cal[frac]:
            print(f"  {frac:<22.2f}{str(bootstrap_ci(explore_cal[frac], seed=2)):>26}")
        else:
            print(f"  {frac:<22.2f}{'no usable folds':>26}")

    out = {
        "config": {"n_seeds": N_SEEDS, "n_start": N_START, "n_final": N_FINAL,
                   "base_rate": BASE_RATE,
                   "stage_aucs": {s: a for s, a, _ in STAGE_AUCS}},
        "realised_auc": {k: float(np.mean(v)) for k, v in auc_check.items()},
        "schedulers": {k: {m: [float(x) for x in v] for m, v in d.items()}
                       for k, d in per_sched.items()},
        "explore_ablation": {str(k): [float(x) for x in v]
                             for k, v in explore_cal.items()},
    }
    p = Path(__file__).resolve().parent / "validation_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nraw results -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
