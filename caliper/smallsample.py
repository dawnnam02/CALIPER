"""Calibration that is defensible at the sample sizes a wet lab actually has.

Why this file replaces the isotonic path
----------------------------------------
CALIPER v0.2 fitted isotonic regression to 26-48 assay labels.  Every relevant
source says that is the worst available choice:

* Niculescu-Mizil & Caruana (ICML 2005): isotonic matches or beats Platt only
  "when there are 1000 or more points in the calibration set"; below that "it
  is easier for it to overfit when the calibration set is small".
* Kull, Silva Filho & Flach (AISTATS 2017): isotonic "is prone to overfitting
  on smaller datasets"; beta calibration is "a good alternative ... where
  isotonic calibration might overfit".
* Riley et al. (Stat Med 2021): a *flexible* calibration curve needs at least
  200 events AND 200 non-events; a calibration slope alone needs 100+ events.
* Manokhin (2026) small-data benchmark: isotonic ranks last of six calibrators
  below n=250 and is "harmful as a default at this scale".
* Guo et al. (ICML 2017): well-calibrated models sit near ECE 0.01-0.05.

Isotonic is the most flexible calibrator; a wet-lab campaign has the least
data.  That pairing is indefensible, and no amount of shrinkage fixes it.

What this module does instead
-----------------------------
1. **Platt scaling** (2 parameters) with the Lin et al. label smoothing that
   exists precisely to stop small samples driving the fit to 0/1.
2. **Beta calibration** (3 parameters), the recommended small-sample choice.
3. **A refusal gate.**  Below the sample size at which *any* post-hoc
   calibration method has been validated, the honest output is not a worse
   probability -- it is no probability at all, plus a discrimination metric.

Korean note:
등장회귀는 가장 유연한 교정법이고, 실험실은 데이터가 가장 적다.  이 조합이 최악이다.
그래서 (1) 파라미터 2~3개짜리 방법으로 바꾸고, (2) 표본이 문헌 기준에 못 미치면
**확률을 아예 내놓지 않는다.**  나쁜 확률을 주는 것보다 모른다고 하는 게 정직하다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize

# --- sample-size thresholds, all from the literature ------------------------
MIN_EVENTS_FOR_FLEXIBLE = 200     # Riley 2021: flexible calibration curve
MIN_EVENTS_FOR_SLOPE = 100        # Riley 2021: calibration slope
MIN_EVENTS_FOR_ANY = 20           # below this nothing has ever been validated
MIN_N_FOR_ISOTONIC = 1000         # Niculescu-Mizil & Caruana 2005

EPS = 1e-6


def _logistic_nll(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    z = np.clip(X @ w, -35.0, 35.0)
    # log(1+exp(z)) computed stably
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def _fit_logistic(X: np.ndarray, y: np.ndarray,
                  bounds=None) -> np.ndarray:
    w0 = np.zeros(X.shape[1])
    res = minimize(_logistic_nll, w0, args=(X, y), method="L-BFGS-B",
                   bounds=bounds)
    if not res.success and not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"calibration fit failed: {res.message}")
    return res.x


def platt_targets(y: np.ndarray) -> np.ndarray:
    """Lin/Platt label smoothing.

    Replaces hard 0/1 targets with (N+ + 1)/(N+ + 2) and 1/(N- + 2).  This is
    not cosmetic: with a handful of labels, hard targets push the fit toward
    infinite slope, which is exactly the small-sample failure mode.
    """
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    return np.where(y > 0.5, hi, lo)


@dataclass(slots=True)
class PlattCalibrator:
    """Two-parameter sigmoid: logit(p) = a * s + b."""

    a: float = 1.0
    b: float = 0.0
    n_labels: int = 0
    fitted: bool = False

    def fit(self, scores, outcomes) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if s.size == 0:
            raise ValueError("nothing to fit")
        t = platt_targets(y)
        X = np.column_stack([s, np.ones_like(s)])
        w = _fit_logistic(X, t)
        self.a, self.b = float(w[0]), float(w[1])
        self.n_labels = int(s.size)
        self.fitted = True
        return self

    def predict(self, scores):
        s = np.asarray(scores, dtype=float)
        if not self.fitted:
            return np.full(s.shape, 0.5)
        return 1.0 / (1.0 + np.exp(-np.clip(self.a * s + self.b, -35, 35)))


@dataclass(slots=True)
class BetaCalibrator:
    """Three-parameter beta calibration (Kull et al. 2017).

    logit(p) = a * log(s) - b * log(1 - s) + c, with a, b >= 0 so the map stays
    monotone.  Strictly more expressive than Platt while still being a
    three-number fit, which is what makes it usable on tens of labels.
    """

    a: float = 1.0
    b: float = 1.0
    c: float = 0.0
    n_labels: int = 0
    fitted: bool = False

    def _features(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(s, EPS, 1 - EPS)
        return np.column_stack([np.log(s), -np.log(1 - s), np.ones_like(s)])

    def fit(self, scores, outcomes) -> "BetaCalibrator":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if s.size == 0:
            raise ValueError("nothing to fit")
        if s.min() < 0 or s.max() > 1:
            raise ValueError("beta calibration needs scores in [0, 1]")
        t = platt_targets(y)
        X = self._features(s)
        w = _fit_logistic(X, t, bounds=[(0, None), (0, None), (None, None)])
        self.a, self.b, self.c = (float(w[0]), float(w[1]), float(w[2]))
        self.n_labels = int(s.size)
        self.fitted = True
        return self

    def predict(self, scores):
        s = np.asarray(scores, dtype=float)
        if not self.fitted:
            return np.full(s.shape, 0.5)
        z = self._features(s) @ np.array([self.a, self.b, self.c])
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
Method = Literal["none", "platt", "beta", "isotonic"]


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    """What CALIPER is willing to claim, given how many labels exist."""

    method: Method
    n_labels: int
    n_events: int
    n_nonevents: int
    reason: str
    report_probabilities: bool

    def __str__(self) -> str:
        head = (f"n={self.n_labels} ({self.n_events} events / "
                f"{self.n_nonevents} non-events) -> {self.method}")
        return f"{head}\n    {self.reason}"


def choose_calibration(outcomes) -> CalibrationDecision:
    """Pick a method, or refuse.

    Refusing is a feature.  Reporting a probability fitted to 5 positive labels
    invites someone to spend a plate on it.
    """
    y = np.asarray(outcomes, dtype=float)
    n = int(y.size)
    ev = int(y.sum())
    ne = n - ev

    if n == 0 or ev == 0 or ne == 0:
        return CalibrationDecision(
            "none", n, ev, ne,
            "only one outcome class present; a calibration map is undefined. "
            "Report the base rate and discrimination only.",
            False)
    if ev < MIN_EVENTS_FOR_ANY or ne < MIN_EVENTS_FOR_ANY:
        return CalibrationDecision(
            "none", n, ev, ne,
            f"fewer than {MIN_EVENTS_FOR_ANY} events or non-events. No post-hoc "
            "calibration method has been validated at this scale (the smallest "
            "stratum in the small-data benchmark literature starts at n=100). "
            "Reporting a probability here would be a guess wearing a decimal "
            "point. Report ranking quality (AP / AUC) instead.",
            False)
    if ev < MIN_EVENTS_FOR_SLOPE:
        return CalibrationDecision(
            "platt", n, ev, ne,
            f"{ev} events is below the {MIN_EVENTS_FOR_SLOPE} that Riley et al. "
            "(2021) require even for a calibration slope. Using the most frugal "
            "2-parameter fit with Lin label smoothing. Treat the output as "
            "indicative ordering, not a trustworthy probability.",
            True)
    if ev < MIN_EVENTS_FOR_FLEXIBLE:
        return CalibrationDecision(
            "beta", n, ev, ne,
            f"{ev} events supports a calibration slope but not the "
            f"{MIN_EVENTS_FOR_FLEXIBLE} events plus {MIN_EVENTS_FOR_FLEXIBLE} "
            "non-events Riley et al. require for a flexible curve. Beta "
            "calibration (3 parameters) is the recommended small-sample choice.",
            True)
    if n < MIN_N_FOR_ISOTONIC:
        return CalibrationDecision(
            "beta", n, ev, ne,
            f"enough events for a flexible curve, but n={n} is still below the "
            f"{MIN_N_FOR_ISOTONIC} points at which isotonic stops overfitting "
            "(Niculescu-Mizil & Caruana 2005). Staying parametric.",
            True)
    return CalibrationDecision(
        "isotonic", n, ev, ne,
        f"n={n} with {ev} events clears every published threshold; the "
        "non-parametric fit is now the better choice.",
        True)


def build_calibrator(scores, outcomes):
    """Return (calibrator_or_None, decision).

    ``None`` means CALIPER declined to produce probabilities.  Callers must
    handle that rather than substituting a default -- the point of the gate is
    that the absence of a number is the finding.
    """
    d = choose_calibration(outcomes)
    if not d.report_probabilities:
        return None, d
    if d.method == "platt":
        return PlattCalibrator().fit(scores, outcomes), d
    if d.method == "beta":
        return BetaCalibrator().fit(scores, outcomes), d
    from .calibrate import Calibrator
    return Calibrator().fit(scores, outcomes), d


def average_precision(scores, outcomes) -> float:
    """Area under the precision-recall curve.

    What to report when the gate refuses probabilities: it measures ranking,
    needs no calibration, and is meaningful at small n with a low base rate --
    exactly the regime binder campaigns live in.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if y.size == 0 or y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / y.sum())
