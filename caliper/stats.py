"""Intervals and out-of-sample calibration.

Fixes CRITIQUE C1 (no confidence intervals), C2 (single seed), C4/C5 (unstable
ECE), and A4 (calibration measured in-sample).

The in-sample point is the important one.  Isotonic regression fitted and then
evaluated on the same labels can drive ECE to nearly zero by construction, so
an in-sample ECE improvement is not evidence that the calibration generalises.
Everything here reports out-of-sample or with an interval.

Korean note:
등장회귀는 자기가 배운 데이터로 평가하면 ECE를 0에 가깝게 만들 수 있다. 구조상 그렇다.
그래서 그 숫자는 아무것도 증명하지 못한다. 여기 있는 건 전부 교차검증이거나 구간이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .calibrate import Calibrator, brier_score, expected_calibration_error


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def wilson(successes: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a proportion.

    Chosen over the normal approximation because n is small (tens of wells) and
    the proportion is often near 0 or 1, exactly where the normal interval
    produces impossible bounds.
    """
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n)


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Interval:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = statistic(rng.choice(v, size=v.size, replace=True))
    return Interval(float(statistic(v)),
                    float(np.quantile(stats, alpha / 2)),
                    float(np.quantile(stats, 1 - alpha / 2)),
                    int(v.size))


def equal_mass_ece(prob, outcome, n_bins: int = 5) -> float:
    """ECE with equal-population bins (fixes C5).

    Equal-width bins leave most bins empty when scores cluster, which makes ECE
    look small for the wrong reason.  Equal-mass bins put the same number of
    samples in each bin, so every bin actually tests something.
    """
    p = np.asarray(prob, dtype=float)
    o = np.asarray(outcome, dtype=float)
    if p.size == 0:
        return float("nan")
    n_bins = max(1, min(n_bins, p.size))
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        total += (chunk.size / p.size) * abs(p[chunk].mean() - o[chunk].mean())
    return float(total)


def cross_validated_calibration(scores, outcomes, *, k: int = 5, seed: int = 0,
                                prior_strength: float = 10.0) -> dict:
    """Honest calibration quality: fit on k-1 folds, score the held-out fold.

    Returns raw and calibrated ECE/Brier computed entirely out of sample, plus
    the in-sample values so the gap between them is visible.  A large gap is
    the signature of overfitting the calibration curve.
    """
    s = np.asarray(scores, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = s.size
    if n == 0:
        raise ValueError("nothing to cross-validate")
    if len(set(o.tolist())) < 2:
        return {"error": "only one outcome class present; calibration undefined",
                "n": int(n), "base_rate": float(o.mean())}

    k = max(2, min(k, n))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)

    oof = np.full(n, np.nan)
    usable = 0
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        if train.size == 0 or len(set(o[train].tolist())) < 2:
            # Degenerate fold: fall back to the training base rate rather than
            # silently dropping the fold, so coverage stays honest.
            oof[test] = o[train].mean() if train.size else o.mean()
            continue
        c = Calibrator(prior_strength=prior_strength).fit(s[train], o[train])
        oof[test] = c.predict(s[test])
        usable += 1

    in_sample = Calibrator(prior_strength=prior_strength).fit(s, o).predict(s)
    return {
        "n": int(n),
        "k": int(k),
        "usable_folds": usable,
        "base_rate": float(o.mean()),
        "ece_raw": expected_calibration_error(s, o),
        "ece_raw_equalmass": equal_mass_ece(s, o),
        "ece_calibrated_insample": expected_calibration_error(in_sample, o),
        "ece_calibrated_oof": expected_calibration_error(oof, o),
        "ece_calibrated_oof_equalmass": equal_mass_ece(oof, o),
        "brier_raw": brier_score(s, o),
        "brier_calibrated_insample": brier_score(in_sample, o),
        "brier_calibrated_oof": brier_score(oof, o),
        "overfit_gap": float(expected_calibration_error(oof, o)
                             - expected_calibration_error(in_sample, o)),
    }


def summarise_runs(values_by_metric: dict[str, list[float]], seed: int = 0
                   ) -> dict[str, Interval]:
    """Across-seed summary (fixes C2)."""
    return {k: bootstrap_ci(v, seed=seed) for k, v in values_by_metric.items()
            if len(v) > 0}
