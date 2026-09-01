"""A multi-fidelity simulator with known ground truth.

This is NOT a protein model and does not pretend to be one.  It is a test
harness: every candidate carries a hidden true affinity, and each stage returns
a noisy, monotonically-related observation of it.  Cheaper stages are noisier.

Why bother: the thing CALIPER actually claims to improve is *scheduling and
calibration*, not structure prediction.  Claims about a scheduler are only
checkable against known ground truth, exactly as a CPU scheduler is validated
on synthetic workloads before it meets a real one.  With this backend the
questions "did the ladder keep the true top-k?" and "is the calibrated
probability honest?" have exact answers.

Korean note:
이건 단백질 모델이 아니다.  성능을 검증하기 위한 시험대다.  후보마다 "진짜 친화도"를
숨겨두고, 각 단계는 그걸 잡음 섞어 관측한다.  싼 단계일수록 잡음이 크다.
정답을 알고 있으니 "사다리가 진짜 상위 k개를 남겼나"를 정확히 잴 수 있다.
"""

from __future__ import annotations

import numpy as np

from ..types import Candidate, Target, stable_hash

AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy, used only to give the simulator a plausible,
# sequence-dependent signal.  Not a binding model.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def _rng(*parts) -> np.random.Generator:
    """Deterministic generator keyed by content, not by call order.

    Using a content hash rather than a running counter means a candidate gets
    the same noise whether it is scored first or thousandth, so caching and
    re-runs agree.  This matters: order-dependent seeding is the classic way
    pipelines become irreproducible.
    """
    h = stable_hash(list(parts))
    return np.random.default_rng(int(h, 16) % (2**63))


def true_affinity(target: Target, sequence: str) -> float:
    """The hidden ground truth, in [0, 1].  Deterministic in the inputs.

    Three terms are combined *multiplicatively*, not additively.  That is the
    important modelling choice: binding requires several things to be right at
    once, so a product of [0, 1] factors reproduces the heavy right skew of
    real affinity landscapes -- most designs are bad, good ones are rare, and
    you cannot compensate for a fatal flaw by being excellent elsewhere.
    An additive score would make a mediocre-everywhere design look as good as
    a nearly-perfect one, which is exactly the failure mode that makes naive
    composite scores untrustworthy.

    Terms:
      * hydropathy complementarity against the target hotspot residues
      * a length preference (very short and very long binders are penalised)
      * a pseudo-random 'epistatic' term keyed by the sequence

    Korean note:
    곱으로 묶은 게 핵심이다.  결합은 여러 조건이 동시에 맞아야 하므로,
    한 군데가 치명적이면 다른 데서 아무리 잘해도 못 살린다.  더하기로 만들면
    "다 그저 그런" 후보가 "거의 완벽한" 후보와 같은 점수를 받아버린다.
    """
    if not sequence:
        return 0.0
    hot = [target.sequence[i] for i in target.hotspots] or list(target.sequence[:8])
    t_h = float(np.mean([HYDROPATHY.get(a, 0.0) for a in hot]))
    b_h = float(np.mean([HYDROPATHY.get(a, 0.0) for a in sequence]))
    # complementary hydropathy: opposite signs score well
    comp = 1.0 - min(abs(t_h + b_h) / 9.0, 1.0)

    n = len(sequence)
    length_pref = float(np.exp(-((n - 60) ** 2) / (2 * 18.0 ** 2)))

    epistatic = float(_rng("epistasis", target.uid, sequence).uniform(0.0, 1.0))

    score = comp * length_pref * (0.15 + 0.85 * epistatic)
    return float(np.clip(score, 0.0, 1.0))


class SimDesigner:
    """Samples random binder sequences, biased toward plausible lengths."""

    name = "sim-designer"
    version = "1.0"
    unit_cost = 1.0

    def __init__(self, length_range: tuple[int, int] = (40, 80)) -> None:
        lo, hi = length_range
        if not 1 <= lo <= hi:
            raise ValueError(f"bad length_range {length_range}")
        self.length_range = length_range

    def design(self, target: Target, n: int, seed: int) -> list[Candidate]:
        if n <= 0:
            return []
        rng = _rng("design", target.uid, seed, n, self.length_range)
        lo, hi = self.length_range
        out: list[Candidate] = []
        for i in range(n):
            length = int(rng.integers(lo, hi + 1))
            seq = "".join(rng.choice(list(AA), size=length))
            out.append(Candidate(
                cid=f"{target.name}-{seed}-{i:06d}",
                target_uid=target.uid,
                sequence=seq,
                origin=self.name,
            ))
        return out


class SimScorer:
    """Observes ``true_affinity`` through stage-specific noise.

    ``noise`` is the standard deviation added before clipping.  A cheap stage
    should be given large noise and a small unit cost; that is the whole point
    of a multi-fidelity ladder.

    ``bias`` shifts the observation, which is what makes calibration necessary:
    a stage can be informative (high rank correlation) while its absolute
    numbers mean nothing.
    """

    def __init__(self, stage: str, noise: float, unit_cost: float,
                 bias: float = 0.0, gain: float = 1.0) -> None:
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if unit_cost < 0:
            raise ValueError("unit_cost must be non-negative")
        self.stage = stage
        self.noise = noise
        self.unit_cost = unit_cost
        self.bias = bias
        self.gain = gain
        self.name = f"sim-{stage}"
        self.version = "1.0"

    def score(self, target: Target, candidates: list[Candidate],
              seed: int) -> list[float]:
        out = []
        for c in candidates:
            t = true_affinity(target, c.sequence)
            eps = _rng("score", self.stage, target.uid, c.sequence, seed).normal(
                0.0, self.noise)
            observed = self.gain * t + self.bias + eps
            out.append(float(np.clip(observed, 0.0, 1.0)))
        return out


def roc_auc(scores, labels) -> float:
    """ROC AUC via the Mann-Whitney statistic, ties counted as half."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def noise_for_auc(target: Target, sequences: list[str], labels,
                  auc: float, *, gain: float = 1.0, bias: float = 0.0,
                  seed: int = 0) -> float:
    """Solve for the observation noise that yields a given ROC AUC.

    Published discrimination for structure-prediction confidence metrics is
    only moderate -- ROC AUC roughly 0.64 (ipTM, Adaptyv EGFR) to 0.72
    (ESMFold pLDDT on monomers), rising to 0.86 for VHH antibodies and
    collapsing to chance for peptide binders.  Hand-picking a noise value
    produces a simulator whose stages are far sharper than any real tool, and
    every downstream number then flatters the pipeline.

    Solving for the noise that reproduces a *published* AUC makes the harness
    defensible: each stage is exactly as informative as the literature says
    the corresponding real tool is.

    Korean note:
    잡음값을 손으로 정하면 시뮬레이터가 실제 도구보다 똑똑해진다.  그러면 그 뒤 숫자가
    전부 후하게 나온다.  발표된 AUC를 그대로 재현하는 잡음을 역산해서 쓴다.
    """
    if not 0.5 < auc < 1.0:
        raise ValueError("auc must be in (0.5, 1.0)")
    t = np.array([true_affinity(target, s) for s in sequences], dtype=float)
    y = np.asarray(labels, dtype=int)
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("labels must contain both classes to solve for AUC")
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=t.size)   # fixed draw: monotone in sigma

    def auc_at(sigma: float) -> float:
        obs = np.clip(gain * t + bias + sigma * base, 0.0, 1.0)
        return roc_auc(obs, y)

    lo, hi = 1e-4, 5.0
    if auc_at(lo) < auc:          # even noiseless cannot reach it
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if auc_at(mid) > auc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class SimAssay:
    """The wet-lab step: a Bernoulli draw whose rate is the true affinity.

    Returns 0/1 outcomes, which is what the calibrator consumes.  ``threshold``
    and ``steepness`` shape how affinity turns into a binding call, standing in
    for assay sensitivity.
    """

    name = "sim-assay"
    version = "1.0"

    def __init__(self, unit_cost: float = 500.0, threshold: float = 0.42,
                 steepness: float = 30.0, p_express: float = 0.73,
                 p_soluble: float = 0.55) -> None:
        # p_express 0.73  : Adaptyv EGFR round 1, 146/201 expressed
        # p_soluble 0.55  : ~65% of de novo monomer failures are
        #                   insolubility/aggregation (Garcia & Dixit 2026)
        self.unit_cost = unit_cost
        self.threshold = threshold
        self.steepness = steepness
        self.p_express = p_express
        self.p_soluble = p_soluble
        self.stage = "assay"

    @classmethod
    def for_base_rate(cls, target: Target, sequences: list[str],
                      base_rate: float, *, unit_cost: float = 500.0,
                      steepness: float = 30.0, p_express: float = 0.73,
                      p_soluble: float = 0.55) -> "SimAssay":
        """Pick the threshold that makes a random design succeed ``base_rate``.

        Real de novo binder campaigns report success in the low single-digit
        percent.  Hard-coding a threshold would silently make the harness easy
        or impossible as the landscape changes, so it is solved for instead.
        """
        if not 0.0 < base_rate < 1.0:
            raise ValueError("base_rate must be in (0, 1)")
        if not sequences:
            raise ValueError("need sequences to calibrate the assay against")
        t = np.array([true_affinity(target, s) for s in sequences], dtype=float)
        # base_rate is the OVERALL success rate, but the affinity term only
        # governs designs that already expressed and stayed soluble.  Solve
        # for the affinity rate that yields the requested overall rate.
        # Earlier this divided base_rate by p_express * p_soluble, assuming the
        # gates are independent of each other AND of affinity. They are not:
        # both gates depend on length and hydropathy, which also drive affinity.
        # That assumption made the realised base rate a consistent +10-14% too
        # high. Solving against the ACTUAL joint outcome removes the bias
        # instead of modelling it.
        probe = cls(unit_cost=unit_cost, steepness=steepness,
                    p_express=p_express, p_soluble=p_soluble)
        gates = np.array([
            probe.gate_probabilities(sq)["expression"]
            * probe.gate_probabilities(sq)["solubility"] for sq in sequences])

        lo, hi = 0.0, 1.0
        for _ in range(60):  # bisection on a monotone function
            mid = 0.5 * (lo + hi)
            affinity = 1.0 / (1.0 + np.exp(-steepness * (t - mid)))
            rate = float(np.mean(gates * affinity))
            if rate > base_rate:
                lo = mid
            else:
                hi = mid
        return cls(unit_cost=unit_cost, threshold=0.5 * (lo + hi),
                   steepness=steepness, p_express=p_express,
                   p_soluble=p_soluble)

    def probability(self, target: Target, sequence: str) -> float:
        """P(binds | it reached the binding measurement at all)."""
        t = true_affinity(target, sequence)
        return float(1.0 / (1.0 + np.exp(-self.steepness * (t - self.threshold))))

    # -- non-affinity gates ------------------------------------------------
    # A design must express, stay soluble, and only then can it bind.  These
    # are modelled as LARGELY INDEPENDENT of affinity, because that is what the
    # data says: in the Adaptyv EGFR competition ipTM and pLDDT predicted
    # expression at ROC AUC 0.58 and 0.55 -- essentially chance -- while
    # roughly 65% of de novo monomer failures are attributed to insolubility
    # and aggregation.
    #
    # Omitting these gates is not a harmless simplification.  With affinity as
    # the only failure mode, any score correlated with affinity eventually
    # reaches a 100% hit rate, which is both impossible in practice and
    # flattering to the pipeline being tested.
    #
    # Korean note:
    # 발현 실패와 응집은 친화도와 거의 무관하다 (ipTM의 발현 예측 AUC 0.58 = 동전던지기).
    # 이 게이트가 없으면 점수가 좋은 후보는 전부 성공해 적중률 100%가 나온다.
    # 실제로는 불가능한 숫자이고, 파이프라인을 실제보다 좋아 보이게 만든다.
    def gate_probabilities(self, sequence: str) -> dict[str, float]:
        n = len(sequence)
        # hydrophobic, long designs aggregate more -- a weak, real trend that
        # is deliberately NOT a function of true_affinity
        hyd = float(np.mean([HYDROPATHY.get(a, 0.0) for a in sequence])) if sequence else 0.0
        aggregation_risk = float(np.clip(0.5 + 0.12 * hyd + 0.004 * (n - 60), 0.05, 0.95))
        return {
            "expression": float(np.clip(self.p_express - 0.002 * (n - 60), 0.05, 0.99)),
            "solubility": float(np.clip(1.0 - aggregation_risk * (1.0 - self.p_soluble) / 0.5,
                                        0.05, 0.99)),
        }

    def run(self, target: Target, candidates: list[Candidate],
            seed: int) -> list[int]:
        """1 only if the design expresses AND stays soluble AND binds."""
        out = []
        for c in candidates:
            g = self.gate_probabilities(c.sequence)
            r = _rng("assay", target.uid, c.sequence, seed)
            ok = 1
            for name in ("expression", "solubility"):
                if r.uniform() >= g[name]:
                    ok = 0
                    break
            if ok and r.uniform() >= self.probability(target, c.sequence):
                ok = 0
            out.append(int(ok))
        return out

    def outcome_detail(self, target: Target, candidates: list[Candidate],
                       seed: int) -> list[dict]:
        """Per-candidate attrition record: where exactly each design died.

        No published de novo binder campaign reports a complete waterfall
        (designed -> expressed -> soluble -> tested -> bound).  The harness
        emits one so the pipeline can be evaluated against the failure mode
        that actually dominates, rather than only against affinity.
        """
        rows = []
        for c in candidates:
            g = self.gate_probabilities(c.sequence)
            r = _rng("assay", target.uid, c.sequence, seed)
            rec = {"cid": c.cid, "expressed": None, "soluble": None,
                   "bound": None, "died_at": None}
            rec["expressed"] = int(r.uniform() < g["expression"])
            if not rec["expressed"]:
                rec["died_at"] = "expression"
                rows.append(rec)
                continue
            rec["soluble"] = int(r.uniform() < g["solubility"])
            if not rec["soluble"]:
                rec["died_at"] = "solubility"
                rows.append(rec)
                continue
            rec["bound"] = int(r.uniform() < self.probability(target, c.sequence))
            rec["died_at"] = None if rec["bound"] else "affinity"
            rows.append(rec)
        return rows
