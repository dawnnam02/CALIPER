"""Where to spend the next unit of compute.

Idea borrowed from two neighbouring fields:

* AutoML / hyper-parameter search uses *successive halving*: give every
  configuration a small budget, keep the best fraction, multiply the budget,
  repeat.  Total spend stays near the cost of evaluating one configuration at
  full budget, but the survivor is nearly always the true best.
* High-throughput screening in drug discovery uses the same shape by another
  name -- a cheap primary screen of millions, then a confirmatory screen of
  thousands.

Binder pipelines already have the tiers (design -> sequence -> fold -> refold)
but almost always drive them with *fixed thresholds* (ipTM > 0.8).  Fixed
thresholds are budget-blind: if a target is hard and nothing clears 0.8 you get
zero designs, and if a target is easy you get ten thousand and blow the GPU
budget.  Allocating by *rank under a budget* fixes both ends.

Korean note:
"점수가 0.8 넘으면 통과"는 예산을 모른다.  어려운 표적이면 0개가 남고, 쉬운 표적이면
1만 개가 남아 GPU가 죽는다.  "예산 안에서 상위 몇 개"로 바꾸면 양쪽이 다 풀린다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rung:
    """One tier of the ladder."""

    name: str
    n_in: int
    n_out: int
    unit_cost: float          # cost of evaluating ONE candidate at this rung

    @property
    def cost(self) -> float:
        return self.n_in * self.unit_cost


def successive_halving(
    n_start: int,
    stage_names: list[str],
    unit_costs: list[float],
    *,
    reduction: float = 3.0,
    n_final: int = 1,
) -> list[Rung]:
    """Plan the ladder: how many candidates survive each stage.

    ``reduction`` is the classic eta.  eta=3 keeps the top third at each rung,
    which is the value Hyperband settled on as a good speed/robustness trade.

    The last rung is pinned to at least ``n_final`` so the plan can never
    return an empty shortlist -- a pipeline that proudly spends nothing is not
    useful.
    """
    if n_start <= 0:
        raise ValueError("n_start must be positive")
    if reduction <= 1.0:
        raise ValueError("reduction (eta) must be > 1")
    if len(stage_names) != len(unit_costs):
        raise ValueError(
            f"{len(stage_names)} stages but {len(unit_costs)} unit costs"
        )
    if not stage_names:
        raise ValueError("need at least one stage")
    if n_final < 1:
        raise ValueError("n_final must be >= 1")
    if any(c < 0 for c in unit_costs):
        raise ValueError("unit costs must be non-negative")

    rungs: list[Rung] = []
    n = n_start
    last = len(stage_names) - 1
    for i, (name, cost) in enumerate(zip(stage_names, unit_costs)):
        if i == last:
            n_out = max(n_final, 1)
        else:
            n_out = max(n_final, int(math.floor(n / reduction)))
        n_out = min(n_out, n)          # a stage can never emit more than it got
        rungs.append(Rung(name, n, n_out, cost))
        n = n_out
    return rungs


def plan_cost(rungs: list[Rung]) -> float:
    return sum(r.cost for r in rungs)


def naive_cost(n_start: int, unit_costs: list[float]) -> float:
    """Cost of running every candidate through every stage (no filtering)."""
    return n_start * sum(unit_costs)


def speedup(rungs: list[Rung]) -> float:
    """How much cheaper the ladder is than the brute-force sweep."""
    if not rungs:
        return float("nan")
    naive = naive_cost(rungs[0].n_in, [r.unit_cost for r in rungs])
    planned = plan_cost(rungs)
    return float("inf") if planned == 0 else naive / planned


def budget_to_start(
    budget: float,
    stage_names: list[str],
    unit_costs: list[float],
    *,
    reduction: float = 3.0,
    n_final: int = 1,
    hi: int = 1_000_000,
) -> int:
    """Largest ``n_start`` whose ladder fits inside ``budget``.

    Binary search rather than a closed form, because ``successive_halving``
    applies floors and clamps that make the cost curve only piecewise smooth.
    Returns 0 when even a single candidate does not fit.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    one = plan_cost(successive_halving(1, stage_names, unit_costs,
                                       reduction=reduction, n_final=n_final))
    if one > budget:
        return 0
    lo, best = 1, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        cost = plan_cost(successive_halving(mid, stage_names, unit_costs,
                                            reduction=reduction, n_final=n_final))
        if cost <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def describe(rungs: list[Rung]) -> str:
    """Human-readable ladder, for logs and the run report."""
    lines = [
        f"{'stage':<14}{'in':>8}{'out':>8}{'keep%':>8}{'unit':>9}{'cost':>11}",
        "-" * 58,
    ]
    for r in rungs:
        keep = 0.0 if r.n_in == 0 else 100.0 * r.n_out / r.n_in
        lines.append(
            f"{r.name:<14}{r.n_in:>8}{r.n_out:>8}{keep:>7.1f}%"
            f"{r.unit_cost:>9.2f}{r.cost:>11.1f}"
        )
    lines.append("-" * 58)
    total = plan_cost(rungs)
    naive = naive_cost(rungs[0].n_in, [r.unit_cost for r in rungs]) if rungs else 0.0
    lines.append(f"{'total':<14}{'':>8}{'':>8}{'':>8}{'':>9}{total:>11.1f}")
    lines.append(f"{'brute force':<14}{'':>8}{'':>8}{'':>8}{'':>9}{naive:>11.1f}")
    lines.append(f"speedup: {speedup(rungs):.2f}x")
    return "\n".join(lines)
