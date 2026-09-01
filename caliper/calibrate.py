"""Turn a raw design score into an honest probability of experimental success.

This is the reason CALIPER exists.  Public binder pipelines filter on raw
confidence numbers (ipTM > 0.8, pAE < 10) as if those were probabilities.
They are not.  A 2026 meta-analysis of experimentally characterised binders
found there is still no standard criterion for prioritising designs, and the
active-learning literature repeatedly notes that uncertainty estimates in this
domain are poorly calibrated.

CALIPER therefore keeps every score it ever computed -- including the scores of
candidates it killed -- and fits a monotone map

    raw score  ->  P(binds in the assay)

from whatever wet-lab outcomes the lab has accumulated.  Thresholds are then
set in probability space, where "spend 20 wells" is a decision a person can
actually reason about.

Korean note:
ipTM 0.8 is not a probability, yet everyone uses it like one.  Here we recompute
"in MY lab, what fraction of ipTM 0.8 designs actually bound?" from data.  Once
that exists, a threshold becomes a decision about how many wells to spend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Isotonic regression via Pool Adjacent Violators (no sklearn dependency)
# ---------------------------------------------------------------------------
def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) fit of ``y``.

    Classic pool-adjacent-violators: walk left to right maintaining a stack of
    blocks; whenever a block mean drops below its left neighbour, merge them.
    Runs in O(n) and returns the fitted value at each input position.
    """
    n = len(y)
    if n == 0:
        return np.asarray(y, dtype=float).copy()
    vals = np.empty(n, dtype=float)
    wts = np.empty(n, dtype=float)
    span = np.empty(n, dtype=int)  # how many original points each block covers
    k = 0
    for i in range(n):
        vals[k] = y[i]
        wts[k] = w[i]
        span[k] = 1
        k += 1
        while k > 1 and vals[k - 1] < vals[k - 2]:
            tw = wts[k - 2] + wts[k - 1]
            vals[k - 2] = (vals[k - 2] * wts[k - 2] + vals[k - 1] * wts[k - 1]) / tw
            wts[k - 2] = tw
            span[k - 2] += span[k - 1]
            k -= 1
    out = np.empty(n, dtype=float)
    pos = 0
    for b in range(k):
        out[pos:pos + span[b]] = vals[b]
        pos += span[b]
    return out


@dataclass(slots=True)
class Calibrator:
    """Monotone score -> probability map with a shrinkage fallback.

    ``fit`` needs (score, outcome) pairs where outcome is 0/1.  With very few
    labels an isotonic fit is degenerate, so the curve is blended toward the
    base rate with weight n / (n + prior_strength).  That stops an early
    campaign from acting on a curve fitted to nine wells.
    """

    x: np.ndarray | None = None      # sorted knot scores
    p: np.ndarray | None = None      # calibrated probability at each knot
    base_rate: float = 0.5
    n_labels: int = 0
    prior_strength: float = 10.0     # pseudo-observations pulling toward base rate

    @property
    def fitted(self) -> bool:
        return self.x is not None and len(self.x) > 0

    def fit(self, scores, outcomes) -> "Calibrator":
        s = np.asarray(scores, dtype=float)
        o = np.asarray(outcomes, dtype=float)
        if s.shape != o.shape:
            raise ValueError(f"scores {s.shape} and outcomes {o.shape} differ in shape")
        if s.size == 0:
            raise ValueError("Calibrator.fit needs at least one labelled example")
        if not np.all(np.isfinite(s)):
            raise ValueError("Calibrator.fit: scores contain NaN or inf")
        seen = set(np.unique(o).tolist())
        if not seen <= {0.0, 1.0}:
            raise ValueError(f"outcomes must be 0/1, saw {sorted(seen)}")

        order = np.argsort(s, kind="mergesort")
        s, o = s[order], o[order]
        self.base_rate = float(o.mean())
        self.n_labels = int(o.size)

        raw = pava(o, np.ones_like(o))
        lam = self.n_labels / (self.n_labels + self.prior_strength)
        shrunk = lam * raw + (1.0 - lam) * self.base_rate

        # Collapse duplicate scores so interpolation is well defined.
        self.x, first = np.unique(s, return_index=True)
        self.p = np.clip(shrunk[first], 1e-6, 1 - 1e-6)
        return self

    def predict(self, scores):
        s = np.asarray(scores, dtype=float)
        if not self.fitted:
            # No labels yet: return the prior.  Honest, and it makes the
            # "we do not know yet" case visible downstream instead of silent.
            return np.full(s.shape, self.base_rate, dtype=float)
        return np.interp(s, self.x, self.p, left=self.p[0], right=self.p[-1])

    def threshold_for(self, target_precision: float) -> float:
        """Lowest raw score whose calibrated probability >= target_precision.

        Returns ``inf`` when nothing reaches it.  Callers must read that as
        "send nothing to the bench", never as "send everything".
        """
        if not 0.0 < target_precision < 1.0:
            raise ValueError("target_precision must be in (0, 1)")
        if not self.fitted:
            return math.inf
        ok = np.nonzero(self.p >= target_precision)[0]
        return float(self.x[ok[0]]) if ok.size else math.inf

    def to_dict(self) -> dict:
        return {
            "x": None if self.x is None else self.x.tolist(),
            "p": None if self.p is None else self.p.tolist(),
            "base_rate": self.base_rate,
            "n_labels": self.n_labels,
            "prior_strength": self.prior_strength,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        c = cls(
            base_rate=d["base_rate"],
            n_labels=d["n_labels"],
            prior_strength=d.get("prior_strength", 10.0),
        )
        if d.get("x") is not None:
            c.x = np.asarray(d["x"], dtype=float)
            c.p = np.asarray(d["p"], dtype=float)
        return c


# ---------------------------------------------------------------------------
# Diagnostics -- how wrong is the map?
# ---------------------------------------------------------------------------
def expected_calibration_error(prob, outcome, n_bins: int = 10) -> float:
    """Mean |confidence - accuracy|, weighted by bin population."""
    p = np.asarray(prob, dtype=float)
    o = np.asarray(outcome, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - o[m].mean())
    return float(total)


def brier_score(prob, outcome) -> float:
    p = np.asarray(prob, dtype=float)
    o = np.asarray(outcome, dtype=float)
    return float(np.mean((p - o) ** 2)) if p.size else float("nan")


def reliability_table(prob, outcome, n_bins: int = 10) -> list[dict]:
    """Per-bin predicted vs observed rate, for the reliability diagram."""
    p = np.asarray(prob, dtype=float)
    o = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        rows.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "predicted": float(p[m].mean()) if m.any() else None,
            "observed": float(o[m].mean()) if m.any() else None,
        })
    return rows
