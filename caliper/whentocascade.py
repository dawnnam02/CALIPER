"""When is a cascade worth it, and when should you just run the good model?

This module exists because CALIPER's own evaluation produced a negative result
and the investigation into it produced something more useful than the original
claim.

The finding
-----------
On real data (Overath et al. 2025, 10 evaluable targets, leave-one-target-out)
the cascade beat a single strong filter on some targets and lost badly on
others.  Two properties predict which:

  * **AUC gap** between the cheap first stage and the final stage.
    Rank correlation with cascade advantage: **-0.652**.  This dominates.
    VirB8 (AF2 0.619 vs AF3 0.810, gap 0.191) -> cascade -0.208.
    Mdm2  (AF2 0.580 vs AF3 0.573, gap -0.007) -> cascade +0.083.
  * **Pool size.**  Rank correlation +0.518.  A bigger pool gives the cascade
    more to save on, which is the whole reason it exists.

Stage correlation matters too but explains less than expected.  Injecting the
measured correlations (0.55-0.66) into the simulator moved the cascade from
"significantly better" to "no significant difference" -- it did not reproduce
the outright loss seen on real data.  The AUC gap did.

The rule
--------
Do not add a cheap stage to a cascade because it is cheap.  Add it only if its
discrimination is close enough to the final stage that the designs it discards
would mostly have been discarded anyway.  A stage that is much worse at the
same job is not a filter, it is noise applied early.

Korean note:
싸다는 이유로 단계를 앞에 붙이면 안 된다.  그 단계의 판별력이 최종 단계와 비슷할 때만
의미가 있다.  같은 일을 훨씬 못하는 단계를 먼저 걸면, 거르는 게 아니라 잡음을 먼저
집어넣는 것이다.  실측에서 이게 가장 큰 요인이었다 (상관 -0.652).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured on the Overath dataset, 10 evaluable targets.  Small n; treat the
# coefficients as indicative, not precise.
EMPIRICAL = {
    "n_targets": 10,
    "corr_auc_gap_vs_advantage": -0.652,
    "corr_pool_size_vs_advantage": 0.518,
    "corr_cheap_auc_vs_advantage": 0.146,
    "source": "Overath et al. 2025, Zenodo 10.5281/zenodo.15722219",
}


@dataclass(frozen=True, slots=True)
class CascadeVerdict:
    should_cascade: bool
    auc_gap: float
    pool_size: int
    cost_saving: float
    reason: str

    def __str__(self) -> str:
        head = "CASCADE" if self.should_cascade else "SINGLE STAGE"
        return (f"{head}  (AUC gap {self.auc_gap:+.3f}, pool {self.pool_size:,}, "
                f"cost saving {self.cost_saving:.1f}x)\n    {self.reason}")


def cost_saving(pool: int, unit_costs: list[float], reduction: float = 3.0,
                n_final: int = 24) -> float:
    """How many times cheaper the ladder is than running the last stage on all."""
    if not unit_costs:
        raise ValueError("need at least one stage")
    single = pool * unit_costs[-1]
    n, total = pool, 0.0
    for i, c in enumerate(unit_costs):
        total += n * c
        n = n_final if i == len(unit_costs) - 1 else max(n_final, int(n / reduction))
    return float("inf") if total == 0 else single / total


def should_cascade(cheap_auc: float, final_auc: float, pool: int,
                   unit_costs: list[float], *, reduction: float = 3.0,
                   n_final: int = 24, max_gap: float = 0.05,
                   min_saving: float = 1.5) -> CascadeVerdict:
    """Decide whether the cheap stage earns its place.

    ``max_gap`` defaults to 0.05 because in the measured data every target with
    a gap above roughly 0.08 lost, and the two that gained had gaps at or below
    zero.  With ten targets that boundary is soft; it is a starting default,
    not a constant of nature.

    The saving must also be worth having: filtering that halves nothing is pure
    downside, since it can only discard good designs.
    """
    if not 0.0 <= cheap_auc <= 1.0 or not 0.0 <= final_auc <= 1.0:
        raise ValueError("AUCs must be in [0, 1]")
    if pool <= 0:
        raise ValueError("pool must be positive")

    gap = final_auc - cheap_auc
    saving = cost_saving(pool, unit_costs, reduction, n_final)

    if gap > max_gap:
        return CascadeVerdict(
            False, gap, pool, saving,
            f"the cheap stage is {gap:.3f} AUC worse than the final one. It "
            "looks at the same thing less accurately, so filtering on it first "
            "discards designs the good stage would have kept. Measured effect: "
            "on the Overath data the AUC gap correlates -0.65 with cascade "
            "advantage, and a 0.19 gap cost 0.21 hit rate.")
    if saving < min_saving:
        return CascadeVerdict(
            False, gap, pool, saving,
            f"the ladder is only {saving:.2f}x cheaper. Filtering can only lose "
            "true positives, so it has to buy a real saving to be worth it. "
            "With a pool this small, run the good model on everything.")
    return CascadeVerdict(
        True, gap, pool, saving,
        f"the cheap stage is within {max_gap:.2f} AUC of the final one and the "
        f"ladder is {saving:.1f}x cheaper. Under a fixed compute budget that "
        f"buys roughly {saving:.1f}x more designs screened, which is where the "
        "cascade wins.")


def explain_stage_order(aucs: list[float], unit_costs: list[float],
                        names: list[str] | None = None) -> str:
    """Diagnose a proposed cascade stage by stage.

    Prints, for each rung, whether it is pulling its weight relative to the
    final stage -- the check that would have caught this pipeline's own bad
    configuration before it was measured.
    """
    if len(aucs) != len(unit_costs):
        raise ValueError("aucs and unit_costs must be the same length")
    names = names or [f"stage{i}" for i in range(len(aucs))]
    final = aucs[-1]
    lines = [f"{'stage':<24}{'AUC':>8}{'gap':>9}{'cost':>9}   verdict",
             "-" * 72]
    for nm, a, c in zip(names, aucs, unit_costs):
        gap = final - a
        if nm == names[-1]:
            v = "final stage"
        elif gap > 0.05:
            v = f"DROP -- {gap:.3f} worse at the same job"
        else:
            v = "keep"
        lines.append(f"{nm:<24}{a:>8.3f}{gap:>+9.3f}{c:>9.1f}   {v}")
    return "\n".join(lines)
