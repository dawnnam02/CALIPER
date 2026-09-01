"""Tests for CALIPER (fixes CRITIQUE D18).

These are not smoke tests.  Each one pins a property the project actually
claims, so that breaking a claim breaks a test.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.allocate import (budget_to_start, naive_cost, plan_cost,
                              successive_halving)
from caliper.backends.simulator import (SimAssay, SimDesigner, SimScorer,
                                        noise_for_auc, roc_auc, true_affinity)
from caliper.calibrate import (Calibrator, expected_calibration_error, pava)
from caliper.hierarchical import HierarchicalCalibrator, ips_weights
from caliper.metrics import spearman, topk_recall
from caliper.stats import cross_validated_calibration, equal_mass_ece, wilson
from caliper.store import RunDir, Store
from caliper.types import Candidate, Target, stable_hash


# --------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------
def test_target_rejects_nonstandard_residues():
    with pytest.raises(ValueError, match="non-standard"):
        Target("bad", "ACDXZ")


def test_target_rejects_out_of_range_hotspot():
    with pytest.raises(ValueError, match="hotspot"):
        Target("t", "ACDEF", hotspots=(99,))


def test_candidate_scores_are_append_only():
    c = Candidate("c", "t", "ACDE", "test").with_score("fold", 0.5)
    with pytest.raises(ValueError, match="already scored"):
        c.with_score("fold", 0.9)


def test_stable_hash_is_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


# --------------------------------------------------------------------------
# isotonic / calibration
# --------------------------------------------------------------------------
def test_pava_output_is_monotone():
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    out = pava(y, np.ones_like(y))
    assert np.all(np.diff(out) >= -1e-12)


def test_calibration_reduces_error_on_a_biased_score():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 800)
    p = 1 / (1 + np.exp(-(8 * s - 5)))
    y = (rng.uniform(size=800) < p).astype(float)
    before = expected_calibration_error(s, y)
    after = expected_calibration_error(Calibrator().fit(s, y).predict(s), y)
    assert after < before


def test_calibrator_without_labels_returns_the_prior():
    """Claiming to know nothing is the honest behaviour with zero labels."""
    assert np.allclose(Calibrator().predict([0.1, 0.9]), 0.5)


def test_calibrator_shrinks_toward_base_rate_with_few_labels():
    few = Calibrator().fit([0.1, 0.5, 0.9], [0, 1, 1]).predict([0.9])[0]
    assert few < 1.0, "3 labels must not produce a confident 1.0"


def test_threshold_for_returns_inf_when_unreachable():
    c = Calibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])
    assert math.isinf(c.threshold_for(0.99))


def test_calibrator_roundtrips_through_dict():
    c = Calibrator().fit([0.1, 0.5, 0.9], [0, 1, 1])
    back = Calibrator.from_dict(c.to_dict())
    assert np.allclose(c.predict([0.3, 0.7]), back.predict([0.3, 0.7]))


def test_calibrator_rejects_non_binary_outcomes():
    with pytest.raises(ValueError, match="0/1"):
        Calibrator().fit([0.1, 0.5], [0.0, 0.5])


# --------------------------------------------------------------------------
# the headline honesty claim
# --------------------------------------------------------------------------
def test_in_sample_calibration_is_optimistic():
    """The defect that CRITIQUE A4 caught must stay visible.

    In-sample ECE after isotonic regression is near zero by construction.  If
    this test ever fails it means the out-of-fold number stopped being
    computed, and the project is back to reporting a meaningless figure.
    """
    rng = np.random.default_rng(3)
    s = rng.uniform(0, 1, 60)
    y = (rng.uniform(size=60) < s).astype(float)
    cv = cross_validated_calibration(s, y, seed=0)
    assert cv["ece_calibrated_insample"] <= cv["ece_calibrated_oof"] + 1e-9
    assert cv["overfit_gap"] >= 0.0


def test_equal_mass_ece_handles_clustered_scores():
    p = np.concatenate([np.full(50, 0.95), np.array([0.1])])
    o = np.concatenate([np.ones(50), np.zeros(1)])
    assert not math.isnan(equal_mass_ece(p, o, n_bins=5))


# --------------------------------------------------------------------------
# IPS / hierarchical
# --------------------------------------------------------------------------
def test_ips_weights_are_clipped():
    w = ips_weights([1.0, 0.001], clip=10.0)
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(10.0)


def test_ips_rejects_zero_propensity():
    with pytest.raises(ValueError, match="strictly positive"):
        ips_weights([0.0, 0.5])


def test_hierarchical_falls_back_to_pool_for_unseen_target():
    t = ["a"] * 40 + ["b"] * 40
    s = list(np.linspace(0, 1, 40)) * 2
    y = [int(v > 0.5) for v in s]
    h = HierarchicalCalibrator().fit(t, s, y)
    assert np.allclose(h.predict("never-seen", [0.7]), h.pooled.predict([0.7]))


def test_hierarchical_shrinkage_increases_with_labels():
    h = HierarchicalCalibrator(shrink_k=25.0)
    assert h._lam(5) < h._lam(100)


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------
def test_ladder_is_monotone_and_cheaper_than_brute_force():
    r = successive_halving(1000, ["a", "b", "c"], [1.0, 10.0, 100.0], n_final=8)
    assert all(x.n_out <= x.n_in for x in r)
    assert plan_cost(r) < naive_cost(1000, [1.0, 10.0, 100.0])


def test_ladder_never_returns_empty():
    r = successive_halving(3, ["a", "b", "c"], [1.0, 1.0, 1.0], n_final=8)
    assert r[-1].n_out >= 1


def test_budget_to_start_respects_the_budget():
    stages, costs = ["a", "b"], [1.0, 10.0]
    n = budget_to_start(5000, stages, costs, n_final=4)
    assert plan_cost(successive_halving(n, stages, costs, n_final=4)) <= 5000
    over = successive_halving(n + 1, stages, costs, n_final=4)
    assert plan_cost(over) > 5000


def test_budget_to_start_returns_zero_when_broke():
    assert budget_to_start(0.0, ["a"], [100.0]) == 0


def test_allocate_rejects_bad_input():
    with pytest.raises(ValueError):
        successive_halving(10, ["a"], [1.0, 2.0])
    with pytest.raises(ValueError):
        successive_halving(10, ["a"], [1.0], reduction=0.5)


# --------------------------------------------------------------------------
# simulator fidelity
# --------------------------------------------------------------------------
TGT = Target("T", "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTK",
             hotspots=(5, 12, 30))


def test_scores_are_order_independent():
    """Order-dependent seeding is the classic irreproducibility bug."""
    pool = SimDesigner().design(TGT, 60, seed=2)
    sc = SimScorer("fold", noise=0.2, unit_cost=1.0)
    a = sc.score(TGT, pool, 2)
    b = list(reversed(sc.score(TGT, list(reversed(pool)), 2)))
    assert a == pytest.approx(b)


def test_true_affinity_is_deterministic():
    v = [true_affinity(TGT, "ACDEFGHIKLMNPQRSTVWY") for _ in range(3)]
    assert len(set(v)) == 1


def test_assay_base_rate_is_solved_not_guessed():
    pool = SimDesigner().design(TGT, 1500, seed=4)
    seqs = [c.sequence for c in pool]
    a = SimAssay.for_base_rate(TGT, seqs, 0.116)
    got = float(np.mean(a.run(TGT, pool, 4)))
    assert 0.06 < got < 0.20, f"requested 0.116, realised {got}"


def test_noise_solver_hits_the_requested_auc():
    pool = SimDesigner().design(TGT, 1200, seed=5)
    seqs = [c.sequence for c in pool]
    y = SimAssay.for_base_rate(TGT, seqs, 0.15).run(TGT, pool, 5)
    sigma = noise_for_auc(TGT, seqs, y, 0.70, seed=5)
    got = roc_auc(SimScorer("s", noise=sigma, unit_cost=1.0).score(TGT, pool, 5), y)
    assert abs(got - 0.70) < 0.05, f"asked 0.70, got {got}"


def test_gates_make_perfect_hit_rates_impossible():
    """Without non-affinity gates the pipeline reached a 100% hit rate.

    Real campaigns do not, because ~27% of designs never express and
    aggregation kills more.  This test pins the fix in place.
    """
    pool = SimDesigner().design(TGT, 400, seed=6)
    a = SimAssay.for_base_rate(TGT, [c.sequence for c in pool], 0.5)
    best = sorted(pool, key=lambda c: -true_affinity(TGT, c.sequence))[:40]
    assert float(np.mean(a.run(TGT, best, 6))) < 0.95


# --------------------------------------------------------------------------
# metrics / stats
# --------------------------------------------------------------------------
def test_topk_recall_bounds():
    truth = {f"c{i}": float(i) for i in range(10)}
    assert topk_recall([f"c{i}" for i in range(7, 10)], truth, 3) == 1.0
    assert topk_recall(["c0"], truth, 3) == 0.0


def test_spearman_detects_monotone_relationships():
    x = np.arange(20.0)
    assert spearman(x, 2 * x + 1) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_wilson_interval_brackets_the_point_estimate():
    i = wilson(3, 24)
    assert i.lo < i.point < i.hi and 0 <= i.lo and i.hi <= 1


def test_wilson_handles_zero_successes():
    i = wilson(0, 10)
    assert i.lo == 0.0 and i.hi > 0.0


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
def test_cache_roundtrip_and_stats():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d)
        k = s.key("fold", "b", "1", {"x": 1}, ["seq"])
        assert s.get(k) is None
        s.put(k, 0.42)
        assert s.get(k) == 0.42
        assert s.stats["hits"] == 1


def test_corrupt_cache_entry_is_survivable():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d)
        k = s.key("f", "b", "1", {}, [])
        (Path(d) / "cache" / f"{k}.json").write_text("{not json", encoding="utf-8")
        assert s.get(k) is None       # must not raise


def test_disabled_cache_never_hits():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, enabled=False)
        k = s.key("f", "b", "1", {}, [])
        s.put(k, 1.0)
        assert s.get(k) is None


def test_manifest_is_append_only():
    with tempfile.TemporaryDirectory() as d:
        r = RunDir(d, "run")
        r.log("a", n=1)
        r.log("b", n=2)
        assert [m["kind"] for m in r.read_manifest()] == ["a", "b"]


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
def test_campaign_runs_and_is_reproducible():
    from caliper.pipeline import Campaign

    def once(tmp):
        pool_designer = SimDesigner()
        seqs = [c.sequence for c in pool_designer.design(TGT, 800, seed=99)]
        assay = SimAssay.for_base_rate(TGT, seqs, 0.116)
        scorers = [SimScorer("seq", 0.5, 0.5), SimScorer("fold", 0.3, 20.0)]
        camp = Campaign(TGT, pool_designer, scorers, assay,
                        store=Store(tmp + "/s", enabled=False),
                        rundir=RunDir(tmp + "/r", "x"),
                        explore_fraction=0.25, seed=1)
        return camp.run(600, n_final=12, assay_capacity=24)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a, b = once(d1), once(d2)
    assert [c.cid for c in a.shortlist] == [c.cid for c in b.shortlist]
    # CRITIQUE D15: the exploration sample used Python's salted hash() and was
    # NOT reproducible across processes.  Pin it.
    assert [c.cid for c in a.explored] == [c.cid for c in b.explored]
    # CRITIQUE C10: propensities must exist for every candidate.
    assert len(a.propensity) == len(a.candidates)
    assert all(v > 0 for c in a.assayed for v in [a.propensity[c.cid]])


def test_campaign_uses_its_full_assay_capacity():
    """CRITIQUE D5: 22 of 48 wells used to be silently wasted."""
    from caliper.pipeline import Campaign

    with tempfile.TemporaryDirectory() as d:
        designer = SimDesigner()
        seqs = [c.sequence for c in designer.design(TGT, 800, seed=9)]
        assay = SimAssay.for_base_rate(TGT, seqs, 0.116)
        camp = Campaign(TGT, designer, [SimScorer("fold", 0.3, 1.0)], assay,
                        store=Store(d + "/s", enabled=False),
                        rundir=RunDir(d + "/r", "y"),
                        explore_fraction=0.25, seed=0)
        r = camp.run(500, n_final=10, assay_capacity=40)
    assert r.diagnostics["n_assayed"] == 40
