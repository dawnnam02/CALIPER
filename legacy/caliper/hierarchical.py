"""Per-target calibration with partial pooling, and IPS-corrected labels.

Why this file exists
--------------------
A 2025 meta-analysis of 3,766 experimentally characterised de novo binders
found that confidence metrics (AF2/AF3 ipTM, interface PAE, ipSAE) are only
weak-to-moderate classifiers -- ROC AUC roughly 0.64 to 0.77 -- and, far more
damaging for a pipeline, that **precision at any fixed threshold ranges from
0.1 to 1.0 depending on the target**.  The Adaptyv EGFR competition
independently reports ipTM AUC 0.64 and pLDDT AUC 0.66 for predicting whether
a design binds at all.

A single global score-to-probability curve is therefore not merely imprecise,
it is wrong in a way that varies by target: the same ipTM cut-off that yields
a 90% hit rate on one target yields 10% on another.  Fitting one curve per
target is the correct model, but most targets have far too few labels to fit
anything.  Partial pooling is the standard answer: each target's curve is
shrunk toward the pooled curve by an amount that depends on how many labels
that target has.

The second half of the file handles a subtler problem.  Labels only exist for
designs the pipeline chose to assay, and that choice depended on the score.
Fitting on those labels without correction is the classic *selective labels*
problem (Lakkaraju et al. 2017; Kleinberg et al. 2018).  Inverse propensity
scoring (Rosenbaum & Rubin 1983; Swaminathan & Joachims 2015) reweights each
label by 1 / P(it was selected), which is knowable here because the pipeline
itself chose, and therefore knows, the selection probability.

Korean note:
표적마다 같은 점수가 다른 뜻을 갖는다.  ipTM 0.8이 어떤 표적에선 90%, 다른 표적에선
10%다.  그래서 곡선을 표적별로 따로 만들되, 라벨이 적은 표적은 전체 곡선 쪽으로
끌어당긴다(부분 풀링).  그리고 실험한 후보는 무작위로 고른 게 아니라 점수가 높아서
고른 것이므로, 그대로 학습하면 편향된다.  뽑힐 확률의 역수로 가중치를 준다(IPS).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibrate import Calibrator, pava


def ips_weights(propensity, *, clip: float = 10.0) -> np.ndarray:
    """Inverse propensity weights, clipped to bound the variance.

    Unclipped IPS is unbiased but has unbounded variance: one design that was
    almost never going to be assayed gets an enormous weight and dominates the
    fit.  Clipping trades a little bias for a lot of variance, which is the
    standard practical choice.
    """
    p = np.asarray(propensity, dtype=float)
    if p.size == 0:
        return p
    if np.any(p <= 0):
        raise ValueError(
            "propensity must be strictly positive; a design with propensity 0 "
            "could never have been assayed, so its label cannot be reweighted"
        )
    if np.any(p > 1):
        raise ValueError("propensity must be <= 1")
    return np.clip(1.0 / p, 0.0, clip)


def weighted_isotonic(scores, outcomes, weights) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic fit honouring sample weights.  Returns (knot_x, knot_p)."""
    s = np.asarray(scores, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(s, kind="mergesort")
    s, o, w = s[order], o[order], w[order]
    fitted = pava(o, w)
    x, first = np.unique(s, return_index=True)
    return x, np.clip(fitted[first], 1e-6, 1 - 1e-6)


@dataclass(slots=True)
class HierarchicalCalibrator:
    """One calibration curve per target, shrunk toward the pooled curve.

    ``shrink_k`` is the number of labels at which a target is trusted half on
    its own data and half on the pool.  With the observed per-target spread in
    the literature (precision 0.1 to 1.0), being slow to trust a target is the
    safe direction: an over-confident per-target curve sends designs to the
    bench that will not bind.
    """

    pooled: Calibrator = field(default_factory=Calibrator)
    per_target: dict[str, Calibrator] = field(default_factory=dict)
    shrink_k: float = 25.0
    ips_clip: float = 10.0

    def fit(self, target_ids, scores, outcomes, propensity=None
            ) -> "HierarchicalCalibrator":
        t = np.asarray(target_ids)
        s = np.asarray(scores, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        if not (t.shape == s.shape == o.shape):
            raise ValueError(
                f"target_ids {t.shape}, scores {s.shape}, outcomes {o.shape} "
                "must have the same shape"
            )
        if s.size == 0:
            raise ValueError("nothing to fit")
        w = (np.ones_like(s) if propensity is None
             else ips_weights(propensity, clip=self.ips_clip))

        # Pooled curve uses every label, IPS-weighted.
        px, pp = weighted_isotonic(s, o, w)
        self.pooled = Calibrator()
        self.pooled.x, self.pooled.p = px, pp
        self.pooled.base_rate = float(np.average(o, weights=w))
        self.pooled.n_labels = int(o.size)

        self.per_target = {}
        for tid in np.unique(t):
            m = t == tid
            if m.sum() < 2 or len(set(o[m].tolist())) < 2:
                continue                       # degenerate; rely on the pool
            tx, tp = weighted_isotonic(s[m], o[m], w[m])
            c = Calibrator()
            c.x, c.p = tx, tp
            c.base_rate = float(np.average(o[m], weights=w[m]))
            c.n_labels = int(m.sum())
            self.per_target[str(tid)] = c
        return self

    def _lam(self, n: int) -> float:
        return n / (n + self.shrink_k)

    def predict(self, target_id: str, scores):
        s = np.asarray(scores, dtype=float)
        pooled = self.pooled.predict(s)
        c = self.per_target.get(str(target_id))
        if c is None:
            return pooled
        lam = self._lam(c.n_labels)
        return lam * c.predict(s) + (1.0 - lam) * pooled

    def threshold_for(self, target_id: str, target_precision: float,
                      grid: np.ndarray | None = None) -> float:
        """Lowest score whose *shrunk* probability reaches target_precision.

        Solved on a grid because the shrunk curve is a blend of two step
        functions and has no closed form.  Returns inf when unreachable, which
        callers must read as "send nothing", never "send everything".
        """
        if not 0.0 < target_precision < 1.0:
            raise ValueError("target_precision must be in (0, 1)")
        if grid is None:
            lo, hi = 0.0, 1.0
            if self.pooled.fitted:
                lo, hi = float(self.pooled.x[0]), float(self.pooled.x[-1])
            grid = np.linspace(lo, hi, 2001)
        p = self.predict(target_id, grid)
        ok = np.nonzero(p >= target_precision)[0]
        return float(grid[ok[0]]) if ok.size else float("inf")

    def coverage(self) -> dict:
        return {
            "pooled_labels": self.pooled.n_labels,
            "targets": {k: v.n_labels for k, v in self.per_target.items()},
            "shrinkage": {k: round(self._lam(v.n_labels), 3)
                          for k, v in self.per_target.items()},
        }


@dataclass(slots=True)
class GateChain:
    """Independent gates a design must pass before affinity is even measurable.

    The literature is emphatic that these are NOT predictable from folding
    confidence: in the Adaptyv EGFR competition, ipTM and pLDDT predicted
    expression at AUC 0.58 and 0.55 -- essentially chance.  Meanwhile roughly
    65% of de novo monomer failures are attributed to insolubility and
    aggregation, and 25-30% of designs never express at all.

    Folding those losses into a single "probability of binding" is therefore a
    modelling error, not a simplification: it attributes to affinity a failure
    that had nothing to do with affinity, and it corrupts the affinity
    calibration with labels from designs that were never measured.

    Korean note:
    발현·용해도 실패는 ipTM으로 예측이 안 된다(AUC 0.58, 사실상 동전던지기).
    그런데 단백질 설계 실패의 65%가 그쪽이다.  이걸 "결합 확률"에 섞으면
    결합과 무관한 실패를 결합 탓으로 돌리게 된다.  그래서 따로 센다.
    """

    names: tuple[str, ...] = ("expression", "solubility")
    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=lambda: {
        # Literature baselines, used until the lab has its own labels.
        "expression": 0.73,    # Adaptyv EGFR round 1: 146/201 expressed
        "solubility": 0.55,    # ~65% of monomer failures are solubility/aggregation
    })

    def probability(self, gate: str, scores=None) -> np.ndarray | float:
        c = self.calibrators.get(gate)
        if c is None or not c.fitted or scores is None:
            return self.priors.get(gate, 0.5)
        return c.predict(scores)

    def joint_reachability(self) -> float:
        """P(a design reaches the binding assay at all), from priors."""
        p = 1.0
        for n in self.names:
            p *= self.priors.get(n, 1.0)
        return p
