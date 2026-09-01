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

from ...metrics import roc_auc
from ...types import Candidate, Target, stable_hash

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


# ---------------------------------------------------------------------------
# Correlated stages
#
# The independent-noise assumption was wrong, and measurably so.  On the
# Overath dataset the three cascade stages have Spearman correlations of
# 0.550 (AF2 vs AF3), 0.657 (ColabFold vs AF3) and 0.574 (AF2 vs ColabFold).
# They are all structure predictors looking at the same complex, so of course
# their errors move together.
#
# Independent noise is the single most favourable assumption a cascade can be
# given: it makes each rung an ensemble member contributing fresh information.
# Under that assumption the simulator said the cascade beat every rival.  On
# real data with real correlation it lost, because the cheap stage discards
# 29% of the binders the expensive stage would have found.
#
# Korean note:
# "단계 잡음이 독립"이라는 가정이 틀렸다.  실측 상관은 0.55~0.66이다.
# 같은 복합체를 보는 예측기들이니 당연히 오차가 같이 움직인다.
# 독립 가정은 캐스케이드에 가장 유리한 가정이고, 그래서 시뮬레이터가 나를 편들었다.
# ---------------------------------------------------------------------------
def correlated_noise(target: Target, sequences: list[str], corr: np.ndarray,
                     seed: int = 0) -> np.ndarray:
    """Standard normals of shape (n_sequences, n_stages) with the given
    correlation, generated deterministically per sequence.

    Cholesky of the correlation matrix turns independent draws into correlated
    ones.  Seeding per sequence (not per call) keeps the harness order- and
    cache-independent, which is the property the whole store depends on.
    """
    corr = np.asarray(corr, dtype=float)
    k = corr.shape[0]
    if corr.shape != (k, k):
        raise ValueError(f"correlation matrix must be square, got {corr.shape}")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("correlation matrix must have unit diagonal")
    eig = np.linalg.eigvalsh(corr)
    if eig.min() < -1e-8:
        raise ValueError(
            f"correlation matrix is not positive semi-definite "
            f"(smallest eigenvalue {eig.min():.4f}); it cannot describe any "
            "set of random variables"
        )
    L = np.linalg.cholesky(corr + 1e-10 * np.eye(k))
    out = np.empty((len(sequences), k), dtype=float)
    for i, s in enumerate(sequences):
        z = _rng("corrnoise", target.uid, s, seed).normal(size=k)
        out[i] = L @ z
    return out


class CorrelatedScorers:
    """A set of stage scorers whose errors are correlated as specified.

    Each stage still observes ``true_affinity`` with its own sigma, but the
    normal draws are shared through a Cholesky factor rather than being
    independent.  Setting ``corr`` to the identity reproduces the old
    behaviour, which is what makes the two regimes directly comparable.
    """

    def __init__(self, stages: list[str], sigmas: list[float],
                 unit_costs: list[float], corr: np.ndarray,
                 gain: float = 0.9, bias: float = 0.1) -> None:
        if not (len(stages) == len(sigmas) == len(unit_costs)):
            raise ValueError("stages, sigmas and unit_costs must be the same length")
        self.stages = stages
        self.sigmas = np.asarray(sigmas, dtype=float)
        self.unit_costs = unit_costs
        self.corr = np.asarray(corr, dtype=float)
        self.gain = gain
        self.bias = bias

    def score_all(self, target: Target, sequences: list[str],
                  seed: int = 0) -> dict[str, np.ndarray]:
        t = np.array([true_affinity(target, s) for s in sequences], dtype=float)
        z = correlated_noise(target, sequences, self.corr, seed)
        out = {}
        for j, stage in enumerate(self.stages):
            v = self.gain * t + self.bias + self.sigmas[j] * z[:, j]
            out[stage] = np.clip(v, 0.0, 1.0)
        return out


def noise_corr_for_observed(target: Target, sequences: list[str],
                            sigmas: list[float], observed: np.ndarray,
                            *, gain: float = 0.9, bias: float = 0.1,
                            seed: int = 0) -> np.ndarray:
    """Solve for the NOISE correlation that produces a target OBSERVED
    correlation between stage scores.

    Two stages correlate for two reasons: they share the latent ``true_affinity``
    signal, and their errors may move together.  Setting the noise correlation
    to the observed value therefore overshoots -- the shared signal is already
    contributing.  Asking for 0.574 and getting 0.628 is exactly that.

    Bisects on a single scalar applied to the off-diagonals, which is enough
    because the whole point is to reproduce one measured correlation level, not
    an arbitrary matrix.

    Korean note:
    두 단계가 닮은 이유는 둘이다 — 같은 진짜 신호를 보기 때문, 그리고 오차가 같이
    움직이기 때문.  잡음 상관에 관측값을 그대로 넣으면 신호 몫만큼 초과한다.
    그래서 역산한다.
    """
    obs = np.asarray(observed, dtype=float)
    k = obs.shape[0]
    off = obs[np.triu_indices(k, 1)].mean()

    def realised(scale: float) -> float:
        c = np.eye(k) + scale * (obs - np.eye(k))
        c = np.clip(c, -0.999, 0.999)
        np.fill_diagonal(c, 1.0)
        try:
            s = CorrelatedScorers([f"s{i}" for i in range(k)], sigmas,
                                  [1.0] * k, c, gain=gain,
                                  bias=bias).score_all(target, sequences, seed)
        except np.linalg.LinAlgError:
            return 1.0
        from ...metrics import spearman
        vals = [spearman(s[f"s{i}"], s[f"s{j}"])
                for i in range(k) for j in range(i + 1, k)]
        return float(np.mean(vals))

    lo, hi = 0.0, 1.0
    if realised(0.0) > off:
        # even with independent noise the shared signal already exceeds the
        # target: nothing to solve, use independence and say so.
        return np.eye(k)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if realised(mid) < off:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    c = np.eye(k) + scale * (obs - np.eye(k))
    np.fill_diagonal(c, 1.0)
    return c
