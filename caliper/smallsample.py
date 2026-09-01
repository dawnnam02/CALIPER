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
* Riley et al. (Stat Med 2021): four criteria, take the maximum.  At an event
  fraction of 0.107 the O/E criterion needs ~346 events for a tight interval
  and ~31 for a relaxed one; the familiar "100/200 events" figures are called
  minimum targets there, not sufficient conditions.
* Manokhin (2026) small-data benchmark: isotonic ranks last of six calibrators
  below n=250 and is "harmful as a default at this scale".
* Guo et al. (ICML 2017): well-calibrated models sit near ECE 0.01-0.05.

Isotonic is the most flexible calibrator; a wet-lab campaign has the least
data.  That pairing is indefensible, and no amount of shrinkage fixes it.

What this module does instead
-----------------------------
1. **Venn-Abers**, which leads on mean log-loss below n=1000 and returns an
   INTERVAL rather than a point.  On a 24-well plate, "somewhere in [0.1, 0.6]"
   is more useful than "0.35" with no error bar.
2. **Beta calibration** (3 parameters), statistically tied with Venn-Abers on
   average rank in the same benchmark.
3. **Platt scaling** (2 parameters), kept as the most frugal option; the same
   benchmark calls it "much of a coin toss", so it is no longer the default.
4. **A refusal gate.**  Below the sample size at which *any* post-hoc method has
   been validated, the honest output is not a worse probability -- it is no
   probability at all, plus a discrimination metric.

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
# Riley et al. 2021 give four criteria and take the maximum.  Evaluated at an
# event fraction of 0.107 (the binder rate in the Overath data) the O/E
# criterion needs ~346 events for a tight interval (CI width 0.2) and ~31 for a
# relaxed one (CI width 0.7).  The commonly quoted "100 events" and "200 events"
# rules of thumb are described in that paper as minimum targets, not sufficient
# conditions.  These constants are the relaxed / slope / strict points.
MIN_EVENTS_FOR_ANY = 31           # Riley 2021 relaxed O/E at phi=0.107
MIN_EVENTS_FOR_SLOPE = 100        # Riley 2021: calibration slope rule of thumb
MIN_EVENTS_FOR_FLEXIBLE = 346     # Riley 2021 strict O/E at phi=0.107
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
Method = Literal["none", "platt", "beta", "venn_abers", "isotonic"]


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
            "venn_abers", n, ev, ne,
            f"{ev} events is below the {MIN_EVENTS_FOR_SLOPE} that Riley et al. "
            "(2021) require even for a calibration slope. Using Venn-Abers, "
            "which leads on mean log-loss below n=1000 and returns an interval "
            "rather than a point -- the width IS the information at this size. "
            "Treat the output as indicative ordering, not a trustworthy "
            "probability.",
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
    if d.method == "venn_abers":
        return VennAbersCalibrator().fit(scores, outcomes), d
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


# ---------------------------------------------------------------------------
# Venn-Abers
#
# Added after a literature check reported that, for n <= 1000, "Venn-Abers
# leads by mean improvement and beta calibration leads by average rank, with
# the two statistically tied", while "Platt appears to be much of a coin toss"
# and holdout isotonic "is harmful at this scale" (Manokhin 2026).
#
# The construction is neat: to score a test point, fit the isotonic map TWICE
# on the calibration set plus that point labelled 0, then plus that point
# labelled 1.  The two answers bracket the truth, and their disagreement is an
# honest width rather than a fabricated confidence.  That width is the reason
# to prefer it here: on a plate of 24 wells, knowing the probability is
# somewhere in [0.1, 0.6] is more useful than being told 0.35 with no error bar.
#
# Korean note:
# 이 방법은 예측을 하나로 주지 않고 [p0, p1] 구간으로 준다.  라벨이 적을 때 그 구간이
# 넓어지는데, 그게 정직한 신호다.  "0.35" 라고 단정하는 것보다 "0.1~0.6 사이" 가 낫다.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class VennAbersCalibrator:
    """Inductive Venn-Abers predictor.

    ``predict`` returns the standard point summary p1 / (1 - p0 + p1);
    ``predict_interval`` returns the (p0, p1) bracket that motivates using it.
    """

    x: np.ndarray | None = None
    y: np.ndarray | None = None
    n_labels: int = 0
    fitted: bool = False

    def fit(self, scores, outcomes) -> "VennAbersCalibrator":
        s = np.asarray(scores, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        if s.size == 0:
            raise ValueError("nothing to fit")
        if s.shape != o.shape:
            raise ValueError(f"shape mismatch {s.shape} vs {o.shape}")
        order = np.argsort(s, kind="mergesort")
        self.x, self.y = s[order], o[order]
        self.n_labels = int(s.size)
        self.fitted = True
        return self

    def _one(self, v: float, label: float) -> float:
        from .calibrate import pava
        xs = np.concatenate([self.x, [v]])
        ys = np.concatenate([self.y, [label]])
        order = np.argsort(xs, kind="mergesort")
        xs, ys = xs[order], ys[order]
        fit = pava(ys, np.ones_like(ys))
        # position of the inserted point after sorting
        j = int(np.searchsorted(xs, v, side="left"))
        j = min(j, fit.size - 1)
        return float(fit[j])

    def predict_interval(self, scores):
        s = np.atleast_1d(np.asarray(scores, dtype=float))
        if not self.fitted:
            return np.full(s.shape, 0.0), np.full(s.shape, 1.0)
        p0 = np.array([self._one(v, 0.0) for v in s])
        p1 = np.array([self._one(v, 1.0) for v in s])
        return p0, p1

    def predict(self, scores):
        s = np.atleast_1d(np.asarray(scores, dtype=float))
        if not self.fitted:
            return np.full(s.shape, 0.5)
        p0, p1 = self.predict_interval(s)
        denom = 1.0 - p0 + p1
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom > 0, p1 / denom, 0.5)
        return np.clip(out, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Detecting a calibration that has inverted
#
# Found while testing whether an exploration quota is worth its wells.  Across
# 19 (target, budget) cells the quota made no average difference -- but one cell
# blew up spectacularly without it, and the post-mortem named the mechanism.
#
# EGFR, top 24 of 434 designs:
#   the labelled scores span 0.650-0.723, which is 10% of the full range.
#   Inside that narrow band the score happens to correlate NEGATIVELY with
#   outcome, so Platt fits a slope of -17.1.  The curve now says "higher
#   confidence means less likely to bind", and predicts 0.962 for the 410
#   unassayed designs whose true rate is 0.046.  ECE 0.916.
#   At N=48 and N=96 the same target spans 17% and 33% of the range, the slope
#   comes out +5.7 and +9.1, and ECE falls to 0.017.
#
# The important part: this is detectable from the labelled data alone.  A
# negative slope is not a subtle miscalibration, it is a curve pointing the
# wrong way, and no amount of extra wells is needed to notice.  Checking costs
# nothing; an exploration quota costs a quarter of the plate.
#
# Korean note:
# 라벨 표본이 점수 범위의 좁은 띠만 덮으면 기울기가 잡음에 끌려 부호가 뒤집힌다.
# 그러면 "점수가 높을수록 안 붙는다"는 곡선이 나와 미실험 설계를 전부 96%로 예측한다.
# 중요한 건 이걸 라벨만 보고 알 수 있다는 점이다.  웰을 더 쓸 필요가 없다.
# ---------------------------------------------------------------------------
MIN_SCORE_SPAN = 0.15      # fraction of the full range the labels must cover


@dataclass(frozen=True, slots=True)
class CalibrationHealth:
    ok: bool                 # False means: do not act on this curve
    warning: str | None      # usable, but with a caveat worth printing
    slope: float
    span_fraction: float
    reason: str

    def __str__(self) -> str:
        head = "BROKEN" if not self.ok else ("USABLE" if self.warning else "HEALTHY")
        out = f"{head} — {self.reason}"
        return out + (f"\n    warning: {self.warning}" if self.warning else "")


def check_calibration(calibrator, labelled_scores, full_scores,
                      min_span: float = MIN_SCORE_SPAN) -> CalibrationHealth:
    """Is this calibration curve safe to act on?

    Two checks, both computable before a single extra well is spent:

    1. **Slope sign** (hard veto).  A calibrator that maps higher scores to
       lower probabilities has inverted.  Whatever it says about unassayed
       designs is worse than useless, because it ranks the plate backwards.
    2. **Score span** (warning only).  If the labels cover a sliver of the
       range, most predictions are extrapolation.  Worth saying, not worth
       refusing over: on the 19 cells measured, the slope rule alone caught the
       single catastrophic fit and nothing else, while a 15% span veto also
       rejected five usable ones.

    ``ok=False`` means do not use this curve -- fall back to the pooled curve
    from other targets, or assay across a wider score range.
    """
    ls = np.asarray(labelled_scores, dtype=float)
    fs = np.asarray(full_scores, dtype=float)
    if ls.size == 0 or fs.size == 0:
        return CalibrationHealth(False, None, float("nan"), 0.0,
                                 "no scores supplied")
    full_range = float(fs.max() - fs.min())
    span = float(ls.max() - ls.min())
    frac = 1.0 if full_range <= 0 else span / full_range

    # empirical slope of the fitted curve across the full score range
    grid = np.linspace(float(fs.min()), float(fs.max()), 64)
    p = np.asarray(calibrator.predict(grid), dtype=float)
    slope = float(np.polyfit(grid, p, 1)[0]) if np.ptp(grid) > 0 else 0.0

    if slope < 0:
        return CalibrationHealth(
            False, None, slope, frac,
            f"the fitted curve has a NEGATIVE slope ({slope:.2f}): it claims a "
            "higher confidence score means a lower chance of binding. The "
            "labelled scores span only "
            f"{100 * frac:.0f}% of the range, which is how a slope inverts. "
            "Do not act on this curve -- fall back to the pooled one.")
    # A narrow span is a warning, not a veto.  Measured on 19 cells, the slope
    # rule alone flagged exactly the one catastrophic fit (ECE 0.916) and
    # nothing else; adding a 15% span veto rejected five more cells whose ECE
    # was 0.026-0.134, all perfectly usable.  A detector that cries wolf five
    # times per real catch will be switched off.
    warn = None
    if frac < min_span:
        warn = (f"the labelled scores cover only {100 * frac:.0f}% of the score "
                f"range, so the curve is extrapolating over most of it. It is "
                "not inverted, but treat probabilities far outside the "
                "labelled band as weakly supported.")
    return CalibrationHealth(
        True, warn, slope, frac,
        f"slope {slope:+.2f} over {100 * frac:.0f}% of the score range.")
