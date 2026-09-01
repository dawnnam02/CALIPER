"""The experiment that decides whether CALIPER is worth anything.

Every scheduler sees the SAME pre-computed score matrix, so any difference is
attributable to the scheduling policy and nothing else.

What changed after the second adversarial pass:

* **40 seeds, not 12.**  At n=12 the headline recall difference gave a paired
  t of 2.16 against a critical value of 2.20 -- it did not reach significance,
  and the README claimed it anyway.
* **Better metrics.**  Top-k set overlap treats rank 25 and rank 3,000 as
  equally wrong.  It returned 0.042 for both a 3-stage cascade and a
  single-stage sweep whose shortlists differed in mean true quality by
  0.785 versus 0.710.  The metric was destroying the signal it existed to
  measure.
* **Paired significance tests.**  No comparative claim goes in the README
  unless it passes one, and every non-significant result reports how many
  seeds would actually be needed.

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

from caliper.backends.simulator import (SimAssay, SimDesigner, SimScorer,
                                        noise_for_auc, roc_auc, true_affinity)
from caliper.baselines import compare_schedulers, quantile_thresholds
from caliper.metrics import mean_quality, normalised_quality, top_decile_rate
from caliper.stats import (bootstrap_ci, cross_validated_calibration,
                           paired_bootstrap, seeds_needed)
from caliper.types import Target

# ---- knobs -----------------------------------------------------------------
N_SEEDS = 40
N_START = 3000
N_FINAL = 24
BASE_RATE = 0.116          # Overath 2025: 436/3,766
STAGE_AUCS = [("seq", 0.62, 0.5), ("fold", 0.68, 20.0), ("refold", 0.75, 60.0)]
EXPLORE_FRACTIONS = [0.0, 0.05, 0.25]
QUALITY_METRICS = ("quality", "top_decile", "normalised")
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

    scores, realised_auc = {}, {}
    for stage, auc, _cost in STAGE_AUCS:
        gain, bias = 0.9, 0.1
        sigma = noise_for_auc(TARGET, seqs, outcomes, auc, gain=gain, bias=bias,
                              seed=seed)
        v = np.array(SimScorer(stage, noise=sigma, unit_cost=1.0, bias=bias,
                               gain=gain).score(TARGET, pool, seed), dtype=float)
        scores[stage] = v
        realised_auc[stage] = roc_auc(v, outcomes)   # asked-vs-got check
    return pool, seqs, truth, scores, outcomes, realised_auc


def main() -> int:
    stages = [s for s, _, _ in STAGE_AUCS]
    costs = [c for _, _, c in STAGE_AUCS]

    per_sched: dict = defaultdict(lambda: defaultdict(list))
    auc_check: dict = defaultdict(list)
    explore_cal: dict = defaultdict(list)
    base_rates: list[float] = []

    for seed in range(N_SEEDS):
        pool, seqs, truth, scores, outcomes, rauc = build_world(seed)
        base_rates.append(float(outcomes.mean()))
        for k, v in rauc.items():
            auc_check[k].append(v)

        thresholds = quantile_thresholds(scores, stages, keep_fraction=1 / 3)
        results = compare_schedulers(scores, stages, costs, N_START, N_FINAL,
                                     truth=truth, thresholds=thresholds,
                                     seed=seed)

        true_top = set(np.argsort(-truth, kind="mergesort")[:N_FINAL].tolist())
        for r in results:
            d = per_sched[r.name]
            d["cost"].append(r.cost)
            d["n_kept"].append(len(r.kept))
            if not r.kept:
                d["hits"].append(0.0)
                d["recall"].append(0.0)
                for m in QUALITY_METRICS:
                    d[m].append(float("nan"))
                continue
            d["hits"].append(float(outcomes[r.kept].mean()))
            d["recall"].append(len(true_top & set(r.kept)) / len(true_top))
            d["quality"].append(mean_quality(r.kept, truth))
            d["top_decile"].append(top_decile_rate(r.kept, truth))
            d["normalised"].append(normalised_quality(r.kept, truth, N_FINAL))

        # ---- exploration ablation ----------------------------------------
        sh = next(r for r in results if r.name == "successive_halving")
        chosen = set(sh.kept)
        rejected = [i for i in range(N_START) if i not in chosen]
        rng = np.random.default_rng(seed)
        for frac in EXPLORE_FRACTIONS:
            capacity = 48
            n_ex = min(int(round(capacity * frac)), len(rejected))
            idx = list(sh.kept[:capacity - n_ex])
            if n_ex:
                idx += [rejected[int(i)] for i in
                        rng.choice(len(rejected), size=n_ex, replace=False)]
            cv = cross_validated_calibration(scores[stages[-1]][idx],
                                             outcomes[idx], seed=seed)
            if "error" not in cv:
                explore_cal[frac].append(cv["ece_calibrated_oof"])

    # ---- report ---------------------------------------------------------
    def ci(vals, seed=1):
        return bootstrap_ci([v for v in vals if v == v], seed=seed)

    print("=" * 78)
    print(f"CALIPER validation | {N_SEEDS} seeds | {N_START} designs | "
          f"shortlist {N_FINAL}")
    print("=" * 78)

    print()
    print("Simulator fidelity (does the harness match the literature it cites?)")
    print(f"  base rate       asked {BASE_RATE:.3f}   realised {ci(base_rates, 0)}")
    for stage, auc, _ in STAGE_AUCS:
        print(f"  AUC {stage:<11} asked {auc:.3f}   realised "
              f"{ci(auc_check[stage], 0)}")

    print()
    print("Scheduler comparison (mean [95% bootstrap CI] over seeds)")
    hdr = (f"  {'scheduler':<21}{'hit rate':>25}{'normalised quality':>25}"
           f"{'top-decile share':>25}{'kept':>6}{'cost':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    order = ["oracle", "successive_halving", "fixed_threshold_topk",
             "full_sweep", "fixed_threshold", "random"]
    for name in order:
        if name not in per_sched:
            continue
        d = per_sched[name]
        print(f"  {name:<21}{str(ci(d['hits'])):>25}"
              f"{str(ci(d['normalised'])):>25}{str(ci(d['top_decile'])):>25}"
              f"{np.mean(d['n_kept']):>6.0f}{np.mean(d['cost']):>10,.0f}")

    print()
    print("Paired tests vs the standard fixed-threshold policy (same seeds)")
    ref = "fixed_threshold_topk"
    tests = {}
    if ref in per_sched:
        for metric in ("hits", "normalised", "top_decile", "recall"):
            a = [v for v in per_sched["successive_halving"][metric] if v == v]
            b = [v for v in per_sched[ref][metric] if v == v]
            t = paired_bootstrap(a, b, "successive_halving", ref, seed=7)
            tests[metric] = t
            print(f"  {metric:<12} {t.verdict()}")
            if not t.significant:
                print(f"  {'':<12}   would need about {seeds_needed(a, b):,.0f} "
                      f"seeds to detect a difference this small")

    print()
    print("Exploration-quota ablation | out-of-fold calibration ECE "
          "(lower is better)")
    for frac in EXPLORE_FRACTIONS:
        print(f"  explore {frac:<5.2f}{str(ci(explore_cal[frac], 2)):>32}")
    if explore_cal[0.0] and explore_cal[0.25]:
        t = paired_bootstrap(explore_cal[0.0], explore_cal[0.25],
                             "explore_0pct", "explore_25pct", seed=8)
        tests["exploration"] = t
        # ECE: lower is better, so the polarity must be stated explicitly.
        print(f"  {t.verdict(lower_is_better=True)}")

    out = {
        "config": {"n_seeds": N_SEEDS, "n_start": N_START, "n_final": N_FINAL,
                   "base_rate_asked": BASE_RATE,
                   "stage_aucs": {s: a for s, a, _ in STAGE_AUCS}},
        "base_rate_realised": base_rates,
        "realised_auc": {k: [float(x) for x in v] for k, v in auc_check.items()},
        "schedulers": {k: {m: [float(x) for x in v] for m, v in d.items()}
                       for k, d in per_sched.items()},
        "explore_ablation": {str(k): [float(x) for x in v]
                             for k, v in explore_cal.items()},
        "paired_tests": {k: {"mean_diff": t.mean_diff, "ci": [t.ci_lo, t.ci_hi],
                             "significant": t.significant, "d": t.effect_size}
                         for k, t in tests.items()},
    }
    p = Path(__file__).resolve().parent / "validation_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nraw results -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
