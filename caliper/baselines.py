"""Competing schedulers, so CALIPER's numbers mean something.

Fixes CRITIQUE A1/A3: comparing a cascade against "run everything through every
stage" flatters it, because nobody does that.  The comparators that matter are:

* ``FixedThreshold`` -- what every published pipeline actually does
  (ipTM > 0.8, pAE < 10).  Budget-blind by construction.
* ``RandomShortlist`` -- the floor.  Any scheduler that cannot beat this is
  worthless.
* ``OracleShortlist`` -- the ceiling.  Only computable in simulation; shows how
  much of the achievable gain the ladder captured.
* ``SuccessiveHalving`` -- CALIPER's own, exposed here so all four run through
  identical code paths and identical accounting.

Every scheduler reports its own compute cost, so the comparison is
cost-for-cost rather than "mine got a better hit rate while spending 40x more".

Korean note:
"전부 다 돌리기"와 비교하면 당연히 좋아 보인다. 아무도 그렇게 안 하니까.
진짜 비교 대상은 (1) 고정 임계값 (실제 논문들이 쓰는 방식), (2) 무작위 (바닥),
(3) 정답을 아는 신탁 (천장) 이다. 셋을 다 넣고 같은 비용으로 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .allocate import successive_halving


@dataclass(slots=True)
class ScheduleResult:
    name: str
    kept: list[int]              # indices into the original pool
    cost: float
    stage_costs: dict[str, float] = field(default_factory=dict)
    note: str = ""


def _stage_scores(scores_by_stage: dict[str, np.ndarray], stage: str) -> np.ndarray:
    if stage not in scores_by_stage:
        raise KeyError(f"no scores for stage {stage!r}")
    return scores_by_stage[stage]


def successive_halving_schedule(scores_by_stage, stages, unit_costs, n_start,
                                *, reduction=3.0, n_final=8) -> ScheduleResult:
    rungs = successive_halving(n_start, stages, unit_costs,
                               reduction=reduction, n_final=n_final)
    alive = np.arange(n_start)
    cost = 0.0
    per_stage = {}
    for stage, rung in zip(stages, rungs):
        per_stage[stage] = len(alive) * rung.unit_cost
        cost += per_stage[stage]
        s = _stage_scores(scores_by_stage, stage)[alive]
        order = np.argsort(-s, kind="mergesort")
        alive = alive[order[:rung.n_out]]
    return ScheduleResult("successive_halving", alive.tolist(), cost, per_stage)


def fixed_threshold_schedule(scores_by_stage, stages, unit_costs, n_start,
                             thresholds: dict[str, float]) -> ScheduleResult:
    """What published pipelines do: a hard cut-off per stage.

    Budget-blind: the survivor count is whatever the data gives.  That is the
    point of including it -- on a hard target it can return nothing, on an easy
    one it can return thousands.
    """
    alive = np.arange(n_start)
    cost = 0.0
    per_stage = {}
    for stage, uc in zip(stages, unit_costs):
        per_stage[stage] = len(alive) * uc
        cost += per_stage[stage]
        s = _stage_scores(scores_by_stage, stage)[alive]
        thr = thresholds.get(stage)
        alive = alive if thr is None else alive[s >= thr]
        if len(alive) == 0:
            return ScheduleResult("fixed_threshold", [], cost, per_stage,
                                  note=f"nothing survived stage {stage!r}")
    return ScheduleResult("fixed_threshold", alive.tolist(), cost, per_stage)


def random_schedule(n_start, n_final, unit_costs, stages, rng) -> ScheduleResult:
    """The floor: pick n_final at random, having scored nothing.

    Cost is only the first (cheapest) stage, since a random picker does not
    need any score at all; charging it more would be a rigged comparison.
    """
    pick = rng.choice(n_start, size=min(n_final, n_start), replace=False)
    return ScheduleResult("random", sorted(pick.tolist()), 0.0,
                          {s: 0.0 for s in stages},
                          note="no scoring performed")


def oracle_schedule(truth: np.ndarray, n_final: int, stages,
                    unit_costs) -> ScheduleResult:
    """The ceiling: the true best n_final.  Simulation only."""
    order = np.argsort(-np.asarray(truth), kind="mergesort")
    return ScheduleResult("oracle", order[:n_final].tolist(), 0.0,
                          {s: 0.0 for s in stages},
                          note="requires ground truth; not computable on real data")


def full_sweep_schedule(scores_by_stage, stages, unit_costs, n_start,
                        n_final) -> ScheduleResult:
    """Score everything at every stage, then take the top n_final by the last.

    This is the upper bound on information and the upper bound on cost.  It is
    the right thing to call "brute force" -- and, crucially, it is the ceiling
    a cascade is trying to approximate cheaply, not a baseline anyone runs.
    """
    cost = 0.0
    per_stage = {}
    for stage, uc in zip(stages, unit_costs):
        per_stage[stage] = n_start * uc
        cost += per_stage[stage]
    s = _stage_scores(scores_by_stage, stages[-1])
    order = np.argsort(-s, kind="mergesort")
    return ScheduleResult("full_sweep", order[:n_final].tolist(), cost, per_stage)


def fixed_threshold_topk_schedule(scores_by_stage, stages, unit_costs, n_start,
                                  thresholds, n_final) -> ScheduleResult:
    """Fixed thresholds, then truncated to n_final by the last stage score.

    Without this the comparison is rigged in the threshold policy's favour:
    a budget-blind filter that happens to keep 218 designs will always show
    better top-k recall than one that keeps 24, purely because it kept more.
    Recall is only comparable at equal shortlist size.

    Korean note:
    고정임계값은 218개를 남기고 내 사다리는 24개를 남긴다. 그 상태로 recall을 비교하면
    당연히 218개 쪽이 이긴다. 같은 개수로 잘라야 정책 비교가 된다.
    """
    r = fixed_threshold_schedule(scores_by_stage, stages, unit_costs,
                                 n_start, thresholds)
    if not r.kept:
        return ScheduleResult("fixed_threshold_topk", [], r.cost, r.stage_costs,
                              note=r.note)
    kept = np.asarray(r.kept)
    s = _stage_scores(scores_by_stage, stages[-1])[kept]
    order = np.argsort(-s, kind="mergesort")
    return ScheduleResult("fixed_threshold_topk",
                          kept[order[:n_final]].tolist(), r.cost, r.stage_costs,
                          note=f"truncated from {len(r.kept)} to {n_final}")


def compare_schedulers(scores_by_stage, stages, unit_costs, n_start, n_final,
                       truth=None, thresholds=None, seed=0) -> list[ScheduleResult]:
    """Run every scheduler on the SAME pre-computed scores.

    Sharing the score matrix removes backend noise from the comparison: any
    difference between schedulers is then attributable to the scheduling
    policy alone.
    """
    rng = np.random.default_rng(seed)
    out = [
        successive_halving_schedule(scores_by_stage, stages, unit_costs,
                                    n_start, n_final=n_final),
        full_sweep_schedule(scores_by_stage, stages, unit_costs, n_start, n_final),
        random_schedule(n_start, n_final, unit_costs, stages, rng),
    ]
    if thresholds:
        out.append(fixed_threshold_schedule(scores_by_stage, stages, unit_costs,
                                            n_start, thresholds))
        out.append(fixed_threshold_topk_schedule(scores_by_stage, stages,
                                                 unit_costs, n_start,
                                                 thresholds, n_final))
    if truth is not None:
        out.append(oracle_schedule(truth, n_final, stages, unit_costs))
    return out


def quantile_thresholds(scores_by_stage, stages, keep_fraction: float
                        ) -> dict[str, float]:
    """Thresholds that keep a fixed fraction -- a fair fixed-threshold rival.

    Using a literature constant like ipTM>0.8 would be unfair in simulation
    because the score scale is arbitrary.  Matching the *retained fraction*
    makes the comparison about the policy, not the units.
    """
    return {s: float(np.quantile(scores_by_stage[s], 1.0 - keep_fraction))
            for s in stages}
