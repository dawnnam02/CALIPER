"""Using round-1 outcomes to pick a better ranking for round 2.

Why this is the only thing round-1 labels can change
----------------------------------------------------
A calibration curve is a **monotone** map from score to probability.  Monotone
maps do not reorder, so recalibrating cannot change *which* designs a round
selects -- only *how many* are worth sending.  Anyone expecting a second round
to pick different designs because the calibration improved has mistaken a
threshold for a ranking.

What round-1 labels CAN change is which score you rank by.  On the Overath data
the best single metric differs by target: six different metrics win across ten
targets, and the global favourite (AF3 ipSAE_min) is best for only four of them.
Picking per-target is worth +0.028 AUC on average and +0.092 at the extreme
(TrkA, where Boltz-1 ipSAE_min scores 0.846 against AF3's 0.754).

The trap
--------
Choosing the argmax of twelve AUCs estimated from twenty wells is a winner's
curse: the metric that looks best on a small sample is partly just lucky.  The
selection rule here therefore requires a *margin* before it switches, and the
margin is scaled by how noisy an AUC estimate is at that sample size.  A rule
that switches on any advantage will underperform never switching at all.

Korean note:
교정 곡선은 단조 변환이라 순위를 바꾸지 못한다.  즉 라운드 2가 "다른 후보"를 고르게
만들 수 있는 건 교정이 아니라 **어떤 점수로 줄을 세우느냐**다.
문제는 웰 20개로 지표 12개 중 최고를 고르면 운 좋은 놈이 뽑힌다는 것(승자의 저주).
그래서 여유(margin)를 넘어야만 갈아탄다.

VERDICT AFTER MEASUREMENT: DO NOT USE THIS
------------------------------------------
experiments/multiround.py tested all of the above on real data, and the answer
is that none of it pays:

    round-2 hit rate     oracle 0.239 | static 0.225 | switch 0.220 | guarded 0.215

* **The ceiling is tiny.**  Even an oracle that knows the truly best metric for
  the target beats never-switching by only +0.015 (d=0.52).  There is very
  little to win here in the first place.
* **The ceiling is unreachable.**  A 24-well round reveals a median of 9
  positives.  The naive argmax picks the true best metric 27.8% of the time and
  switches on 65.7% of rounds -- it is mostly reacting to noise.
* **The guard did not save it.**  Requiring a one-standard-error margin cut
  switching from 65.7% to 25.0%, but **0 of those 27 switches chose the truly
  best metric**, and guarded ended up slightly WORSE than never switching
  (d=-0.21).  Filtering on a large observed gap selects precisely the estimates
  that overfit the revealed wells hardest.

So this module is kept as the record of a measured negative result, and as the
implementation to reuse if a future setting supplies far more round-1 labels.
It is deliberately NOT wired into any pipeline.  The honest advice is: rank by
the metric that was best across your other targets, and do not let one round of
24 wells talk you out of it.

측정 결과: 쓰지 마라.  천장이 +0.015 로 작고, 웰 24개(양성 중앙값 9개)로는 거기
도달할 수 없다.  여유를 둔 규칙조차 27번 전환 중 0번만 진짜 최고를 골랐고,
아예 안 바꾸는 것보다 나빴다.  이 모듈은 그 부정 결과의 기록으로 남긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUC estimate.

    Used to size the switching margin.  With 20 wells at a 10% hit rate there
    are about 2 positives, and the standard error is enormous -- which is the
    quantitative reason a naive argmax rule fails at small k.
    """
    if n_pos <= 0 or n_neg <= 0:
        return float("inf")
    a = float(np.clip(auc, 1e-6, 1 - 1e-6))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


@dataclass(frozen=True, slots=True)
class MetricChoice:
    metric: str
    switched: bool
    observed_auc: float
    incumbent_auc: float
    margin: float
    n_pos: int
    n_neg: int
    reason: str

    def __str__(self) -> str:
        head = "SWITCH" if self.switched else "KEEP"
        return (f"{head} -> {self.metric} (observed {self.observed_auc:.3f} vs "
                f"incumbent {self.incumbent_auc:.3f}, margin {self.margin:.3f}, "
                f"{self.n_pos} pos / {self.n_neg} neg)\n    {self.reason}")


def choose_metric(scores_by_metric: dict[str, np.ndarray], outcomes,
                  incumbent: str, *, z: float = 1.0,
                  min_positives: int = 3) -> MetricChoice:
    """Pick the ranking metric for the next round from round-1 outcomes.

    ``z`` sets how many standard errors of advantage a challenger needs.  z=1 is
    deliberately lenient-but-not-free: at twenty wells the standard error is so
    large that almost nothing clears it, which is the correct behaviour.

    Returns the incumbent unchanged whenever the evidence is too thin, and says
    why.  Refusing to switch is the common case and is not a failure.
    """
    from .backends.simulator import roc_auc

    y = np.asarray(outcomes, dtype=int)
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    if incumbent not in scores_by_metric:
        raise KeyError(f"incumbent metric {incumbent!r} not among the candidates")

    inc_auc = roc_auc(scores_by_metric[incumbent], y)
    if n_pos < min_positives or n_neg < min_positives:
        return MetricChoice(
            incumbent, False, inc_auc, inc_auc, float("inf"), n_pos, n_neg,
            f"only {n_pos} positives and {n_neg} negatives revealed. An AUC "
            f"estimated from this cannot distinguish twelve candidates; "
            "switching here would be picking the luckiest, not the best.")

    se = auc_standard_error(inc_auc, n_pos, n_neg)
    margin = z * se

    best, best_auc = incumbent, inc_auc
    for name, s in scores_by_metric.items():
        a = roc_auc(s, y)
        if a == a and a > best_auc:
            best, best_auc = name, a

    if best == incumbent:
        return MetricChoice(incumbent, False, inc_auc, inc_auc, margin,
                            n_pos, n_neg,
                            "the incumbent is still the best on the revealed "
                            "wells.")
    if best_auc - inc_auc < margin:
        return MetricChoice(
            incumbent, False, best_auc, inc_auc, margin, n_pos, n_neg,
            f"{best} leads by {best_auc - inc_auc:.3f}, inside the {margin:.3f} "
            "standard error of the estimate. Not enough evidence to switch.")
    return MetricChoice(
        best, True, best_auc, inc_auc, margin, n_pos, n_neg,
        f"{best} leads by {best_auc - inc_auc:.3f}, clearing the {margin:.3f} "
        "margin. Switching for the next round.")


def next_batch(scores: np.ndarray, already_done: set[int], k: int) -> np.ndarray:
    """Top-k of what has not been assayed yet, by the chosen metric."""
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    out = [int(i) for i in order if int(i) not in already_done]
    return np.array(out[:k], dtype=int)
