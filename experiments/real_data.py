"""CALIPER on real experimental data. No simulator.

Data: Overath et al. 2025, "Predicting Experimental Success in De Novo Binder
Design: A Meta-Analysis of 3,766 Experimentally Characterised Binders"
(bioRxiv 10.1101/2025.08.14.670059), dataset Zenodo 10.5281/zenodo.15722219,
CC-BY-4.0.  3,669 labelled designs across 15 targets, 10.7% binders, with
AF2-initial-guess, ColabFold, AF3 and Boltz-1 all run on the SAME designs.

That last property is what makes this file possible: a multi-stage allocation
policy cannot be evaluated retrospectively unless every stage's score exists
for every design.

Protocol
--------
**Leave-one-target-out.**  Thresholds and calibration are fitted on 14 targets
and evaluated on the 15th, never on the same data.  This is the specific
hygiene that Kapoor & Narayanan (Patterns 2023) found missing in 294 papers
across 17 fields: tuning a cut-off on the data you then report on inflates
every number.  A random split would leak, because designs against one target
are far more similar to each other than to designs against another.

What this file will NOT do
--------------------------
* It cannot evaluate a ProteinMPNN-score first stage.  No public dataset pairs
  sequence-design scores with downstream structure scores and experimental
  outcomes on the same designs; that gap was confirmed independently twice.
  The cheap first rung here is AF2-initial-guess, not ProteinMPNN.
* It cannot measure wall-clock cost.  Stage costs below are ESTIMATES and are
  flagged as such; no published pipeline reports GPU-hours per accepted design.

Korean note:
시뮬레이터를 쓰지 않는다.  실제 설계 3,669개, 실제 실험 결과다.
표적 14개로 학습하고 남은 1개로 평가한다(leave-one-target-out).  같은 데이터로
임계값을 맞추고 그 데이터에서 성능을 보고하면 숫자가 부풀려지기 때문이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.backends.simulator import roc_auc
from caliper.metrics import top_decile_rate
from caliper.smallsample import (average_precision, build_calibrator,
                                 choose_calibration)
from caliper.stats import (bootstrap_ci, cross_validated_calibration,
                           paired_bootstrap, wilson)
from caliper.types import stable_hash

DATA = Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"

# The cascade, cheapest first.
#
# WARNING: unit costs are ESTIMATES.  AF2-initial-guess skips the MSA search
# that ColabFold requires, and AF3/Boltz-1 are larger models, so the ordering is
# defensible -- but no published pipeline reports GPU-hours per accepted design,
# so the magnitudes are not sourced.  `--cost-sensitivity` re-runs the whole
# comparison under alternative cost vectors precisely because of this.
STAGES = [
    ("af2_pae_interaction", 1.0, True),    # True = lower is better
    ("colab_ipSAE_min", 8.0, False),
    ("af3_ipSAE_min", 20.0, False),
]
N_FINAL = 24
REDUCTION = 3.0


def load() -> pd.DataFrame:
    if not DATA.exists():
        print(f"dataset not found: {DATA}\n"
              "download final_dataset.csv (82 MB, CC-BY-4.0) from\n"
              "  https://zenodo.org/records/15722219", file=sys.stderr)
        raise SystemExit(2)
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df.binder.notna()].copy()
    df["y"] = df.binder.astype(bool).astype(int)
    keep = [c for c, _, _ in STAGES]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=keep).reset_index(drop=True)
    if len(df) < before:
        print(f"dropped {before - len(df)} rows missing a stage score")
    return df


def oriented(df: pd.DataFrame, col: str, lower_better: bool) -> np.ndarray:
    """Return the score with 'higher is better' orientation."""
    v = df[col].to_numpy(dtype=float)
    return -v if lower_better else v


def run_cascade(scores: dict[str, np.ndarray], keeps: list[int]) -> np.ndarray:
    alive = np.arange(len(next(iter(scores.values()))))
    for (col, _c, _l), k in zip(STAGES, keeps):
        order = np.argsort(-scores[col][alive], kind="mergesort")
        alive = alive[order[:k]]
    return alive


def ladder_keeps(n: int) -> list[int]:
    keeps, cur = [], n
    for i in range(len(STAGES)):
        cur = N_FINAL if i == len(STAGES) - 1 else max(N_FINAL, int(cur / REDUCTION))
        keeps.append(min(cur, n))
    return keeps


def fixed_threshold(scores, thresholds, n_final):
    alive = np.arange(len(next(iter(scores.values()))))
    for col, _c, _l in STAGES:
        alive = alive[scores[col][alive] >= thresholds[col]]
        if alive.size == 0:
            return alive
    last = STAGES[-1][0]
    order = np.argsort(-scores[last][alive], kind="mergesort")
    return alive[order[:n_final]]


def main() -> int:
    df = load()
    targets = sorted(df.target_id.unique())
    print("=" * 78)
    print("CALIPER on REAL data | Overath et al. 2025 (Zenodo 15722219, CC-BY-4.0)")
    print("=" * 78)
    print(f"  {len(df):,} labelled designs | {df.y.sum()} binders "
          f"({100 * df.y.mean():.1f}%) | {len(targets)} targets")
    print(f"  cascade: {' -> '.join(c for c, _, _ in STAGES)}")
    print(f"  protocol: leave-one-target-out, shortlist {N_FINAL}")

    # ---- single-metric discrimination, for context ---------------------
    print()
    print("Single-metric discrimination on the whole set (ROC AUC)")
    for col, _c, low in STAGES:
        print(f"  {col:<28}{roc_auc(oriented(df, col, low), df.y.values):.3f}")

    rows = []
    cal_rows = []
    for held in targets:
        te = df[df.target_id == held]
        tr = df[df.target_id != held]
        if len(te) < 30 or te.y.sum() < 3 or te.y.sum() == len(te):
            continue

        s_te = {c: oriented(te, c, low) for c, _cost, low in STAGES}
        s_tr = {c: oriented(tr, c, low) for c, _cost, low in STAGES}
        y_te = te.y.to_numpy()
        n = len(te)
        keeps = ladder_keeps(n)

        # thresholds are fitted on TRAINING targets only
        thr = {c: float(np.quantile(s_tr[c], 1 - 1 / REDUCTION))
               for c, _cost, _l in STAGES}

        picks = {
            "cascade": run_cascade(s_te, keeps),
            "fixed_threshold": fixed_threshold(s_te, thr, N_FINAL),
            "best_single": np.argsort(-s_te[STAGES[-1][0]],
                                      kind="mergesort")[:N_FINAL],
            # content hash, not Python's salted hash(): the latter changes
            # between processes and made this sample irreproducible.
            "random": np.random.default_rng(
                int(stable_hash(["rand", held]), 16) % 2**31).choice(
                n, size=min(N_FINAL, n), replace=False),
        }
        costs = {
            "cascade": sum(k_in * c for k_in, (_n, c, _l) in
                           zip([n] + keeps[:-1], STAGES)),
            "fixed_threshold": None,
            # best_single only ever runs the LAST stage, so charging it for
            # the earlier ones would rig the cost comparison in the cascade's
            # favour.  It pays for that one model on every design.
            "best_single": n * STAGES[-1][1],
            "random": 0.0,
        }
        # fixed threshold pays full price at every stage it reaches
        alive = np.arange(n)
        ft_cost = 0.0
        for col, c, _l in STAGES:
            ft_cost += len(alive) * c
            alive = alive[s_te[col][alive] >= thr[col]]
        costs["fixed_threshold"] = ft_cost

        for name, idx in picks.items():
            if len(idx) == 0:
                rows.append({"target": held, "policy": name, "n_kept": 0,
                             "hit_rate": 0.0, "top_decile": 0.0,
                             "cost": costs[name]})
                continue
            rows.append({
                "target": held, "policy": name, "n_kept": len(idx),
                "hit_rate": float(y_te[idx].mean()),
                "top_decile": top_decile_rate(idx, s_te[STAGES[-1][0]]),
                "cost": costs[name],
            })

        # ---- calibration, fitted on training targets only ---------------
        col = STAGES[-1][0]
        cal, decision = build_calibrator(s_tr[col], tr.y.to_numpy())
        rec = {"target": held, "method": decision.method,
               "n_train": decision.n_labels, "events": decision.n_events,
               "ap_test": average_precision(s_te[col], y_te),
               "auc_test": roc_auc(s_te[col], y_te)}
        if cal is not None:
            p = np.asarray(cal.predict(s_te[col]))
            rec["ece_test"] = float(np.mean(np.abs(p - y_te.mean())))
            rec["mean_pred"] = float(p.mean())
            rec["actual_rate"] = float(y_te.mean())
        cal_rows.append(rec)

    R = pd.DataFrame(rows)
    C = pd.DataFrame(cal_rows)

    print()
    print(f"Leave-one-target-out results ({R.target.nunique()} evaluable targets)")
    hdr = f"  {'policy':<18}{'hit rate':>26}{'top-decile share':>26}{'cost':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in ("cascade", "fixed_threshold", "best_single", "random"):
        g = R[R.policy == name]
        if g.empty:
            continue
        print(f"  {name:<18}{str(bootstrap_ci(g.hit_rate, seed=1)):>26}"
              f"{str(bootstrap_ci(g.top_decile, seed=1)):>26}"
              f"{g.cost.mean():>12,.0f}")

    print()
    print("Paired tests across held-out targets (cascade vs each rival)")
    piv = R.pivot_table(index="target", columns="policy", values="hit_rate")
    for rival in ("fixed_threshold", "best_single", "random"):
        if rival not in piv:
            continue
        sub = piv[["cascade", rival]].dropna()
        if len(sub) < 3:
            continue
        t = paired_bootstrap(sub["cascade"], sub[rival], "cascade", rival, seed=5)
        print(f"  hit_rate   {t.verdict()}")

    print()
    print("Calibration under the sample-size gate")
    print(f"  {'target':<18}{'method':<10}{'train ev':>9}{'AUC':>8}{'AP':>8}"
          f"{'predicted':>11}{'actual':>9}")
    print("  " + "-" * 73)
    for _, r in C.iterrows():
        mp = f"{r['mean_pred']:.3f}" if "mean_pred" in r and r.get("mean_pred") == r.get("mean_pred") else "refused"
        ar = f"{r['actual_rate']:.3f}" if "actual_rate" in r and r.get("actual_rate") == r.get("actual_rate") else "-"
        print(f"  {str(r['target'])[:17]:<18}{r['method']:<10}{r['events']:>9}"
              f"{r['auc_test']:>8.3f}{r['ap_test']:>8.3f}{mp:>11}{ar:>9}")

    out = Path(__file__).resolve().parent / "real_data_results.json"
    out.write_text(json.dumps(
        {"policies": R.to_dict("records"), "calibration": C.to_dict("records")},
        indent=2, default=str), encoding="utf-8")
    print(f"\nraw results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
