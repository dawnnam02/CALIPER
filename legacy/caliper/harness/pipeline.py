"""The campaign runner.

Flow
----
1. Plan a successive-halving ladder from the compute budget.
2. Design ``n_start`` candidates.
3. Walk the rungs.  Each rung scores the survivors and keeps the top ``n_out``.
   Killed candidates keep their scores; nothing is deleted.
4. Spend the assay budget in TWO pots:
     * exploit -- the ladder's shortlist
     * explore -- a random sample of candidates the ladder REJECTED
5. Fit the calibrator on the union, and report how honest the scores were.

Why step 4 has two pots
-----------------------
If you only ever assay the winners, the labelled set is truncated at the top of
the score range.  A calibrator fitted on that data cannot see what happens at
low scores, so it cannot tell you the difference between "0.4 means 3% success"
and "0.4 means 30% success".  This is plain selection bias, and it is why
published confidence cut-offs do not transfer between labs: everyone fits on
their own winners.

The exploration pot is the cheapest possible fix -- a handful of deliberately
"bad" wells per round buys the negative labels that make every future threshold
meaningful.  Multi-fidelity screening work in drug discovery makes the same
complaint from the other direction: the millions of primary-screen measurements
are thrown away instead of being used jointly.

Korean note:
승자만 실험하면 낮은 점수대에 라벨이 없다.  그러면 "0.4점은 몇 %인가"를 영원히 모른다.
그래서 매 라운드 실험 예산의 일부를 일부러 탈락한 후보에 쓴다.  버리는 돈처럼 보이지만,
그게 있어야 임계값이 의미를 갖는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .allocate import Rung, describe, naive_cost, plan_cost, successive_halving
from ..calibrate import (Calibrator, brier_score, expected_calibration_error,
                        reliability_table)
from ..metrics import cost_summary, hit_rate
from ..stats import cross_validated_calibration, wilson
from .store import RunDir, Store
from ..types import Candidate, StageReport, Target, stable_hash


@dataclass(slots=True)
class CampaignResult:
    target: str
    rungs: list[Rung]
    reports: list[StageReport]
    candidates: list[Candidate]
    shortlist: list[Candidate]
    explored: list[Candidate]
    outcomes: dict[str, int]
    propensity: dict[str, float]
    calibrator: Calibrator
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def assayed(self) -> list[Candidate]:
        return self.shortlist + self.explored


class Campaign:
    """One target, one budget, one run."""

    def __init__(
        self,
        target: Target,
        designer,
        scorers: Sequence,
        assay,
        *,
        store: Store,
        rundir: RunDir,
        explore_fraction: float = 0.25,
        seed: int = 0,
    ) -> None:
        if not scorers:
            raise ValueError("need at least one scoring stage")
        if not 0.0 <= explore_fraction < 1.0:
            raise ValueError("explore_fraction must be in [0, 1)")
        names = [s.stage for s in scorers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")
        self.target = target
        self.designer = designer
        self.scorers = list(scorers)
        self.assay = assay
        self.store = store
        self.rundir = rundir
        self.explore_fraction = explore_fraction
        self.seed = seed

    # -- internals ---------------------------------------------------------
    def _score_stage(self, scorer, cands: list[Candidate]) -> tuple[list[float], int]:
        """Score with per-candidate caching.  Returns (scores, cache_hits)."""
        scores: list[float | None] = [None] * len(cands)
        misses: list[int] = []
        hits = 0
        for i, c in enumerate(cands):
            key = self.store.key(scorer.stage, scorer.name, scorer.version,
                                 {"seed": self.seed}, c.sequence)
            got = self.store.get(key)
            if got is None:
                misses.append(i)
            else:
                scores[i] = float(got)
                hits += 1
        if misses:
            fresh = scorer.score(self.target, [cands[i] for i in misses], self.seed)
            if len(fresh) != len(misses):
                raise RuntimeError(
                    f"backend {scorer.name} returned {len(fresh)} scores for "
                    f"{len(misses)} candidates"
                )
            for i, v in zip(misses, fresh):
                v = float(v)
                if not np.isfinite(v):
                    raise RuntimeError(
                        f"backend {scorer.name} produced non-finite score for "
                        f"candidate {cands[i].cid}"
                    )
                scores[i] = v
                key = self.store.key(scorer.stage, scorer.name, scorer.version,
                                     {"seed": self.seed}, cands[i].sequence)
                self.store.put(key, v)
        return [float(s) for s in scores], hits  # type: ignore[arg-type]

    # -- public ------------------------------------------------------------
    def run(self, n_start: int, *, reduction: float = 3.0, n_final: int = 8,
            assay_capacity: int | None = None) -> CampaignResult:
        rungs = successive_halving(
            n_start,
            [s.stage for s in self.scorers],
            [s.unit_cost for s in self.scorers],
            reduction=reduction,
            n_final=n_final,
        )
        self.rundir.log("plan", target=self.target.name, n_start=n_start,
                        reduction=reduction, n_final=n_final,
                        planned_cost=plan_cost(rungs),
                        naive_cost=naive_cost(n_start,
                                              [s.unit_cost for s in self.scorers]))

        # --- design -------------------------------------------------------
        t0 = time.perf_counter()
        pool = self.designer.design(self.target, n_start, self.seed)
        if len(pool) != n_start:
            raise RuntimeError(
                f"designer returned {len(pool)} candidates, expected {n_start}"
            )
        reports = [StageReport("design", self.designer.name, 0, len(pool),
                               len(pool) * self.designer.unit_cost,
                               time.perf_counter() - t0)]

        by_id: dict[str, Candidate] = {c.cid: c for c in pool}
        alive = list(pool)

        # --- ladder -------------------------------------------------------
        for scorer, rung in zip(self.scorers, rungs):
            t0 = time.perf_counter()
            scores, hits = self._score_stage(scorer, alive)
            alive = [c.with_score(scorer.stage, s) for c, s in zip(alive, scores)]
            for c in alive:
                by_id[c.cid] = c

            order = np.argsort(-np.asarray(scores), kind="mergesort")
            keep_idx = set(order[:rung.n_out].tolist())
            survivors, killed = [], []
            for i, c in enumerate(alive):
                (survivors if i in keep_idx else killed).append(
                    c if i in keep_idx else c.killed(scorer.stage))
            for c in killed:
                by_id[c.cid] = c

            reports.append(StageReport(
                scorer.stage, scorer.name, len(alive), len(survivors),
                len(alive) * scorer.unit_cost, time.perf_counter() - t0,
                cache_hits=hits,
                params={"keep": rung.n_out},
            ))
            self.rundir.log("stage", stage=scorer.stage, n_in=len(alive),
                            n_out=len(survivors), cache_hits=hits)
            alive = survivors

        shortlist = alive

        # --- assay: exploit + explore ------------------------------------
        capacity = assay_capacity if assay_capacity is not None else len(shortlist)
        if capacity < 0:
            raise ValueError("assay_capacity must be non-negative")
        n_explore = int(round(capacity * self.explore_fraction))
        n_exploit = max(0, capacity - n_explore)

        # CRITIQUE D5: back-fill unused wells instead of silently wasting them.
        # capacity 48 with a 24-design shortlist used to leave 22 wells idle.
        exploit = shortlist[:n_exploit]
        rejected = sorted((c for c in by_id.values() if not c.alive),
                          key=lambda c: c.cid)          # D6: deterministic order
        # Back-fill unused wells into the exploration pot -- but ONLY if the
        # caller asked for exploration at all. An earlier version back-filled
        # unconditionally, so explore_fraction=0 with a shortlist smaller than
        # the plate silently spent 160 wells on rejected designs. A zero means
        # zero; the fix must not quietly override the caller.
        spare = n_exploit - len(exploit)
        if self.explore_fraction > 0.0:
            n_explore = min(n_explore + spare, len(rejected))
        else:
            n_explore = 0

        # CRITIQUE D15: Python's hash() is salted per process, so the previous
        # seeding made the exploration sample IRREPRODUCIBLE across runs -- in a
        # project whose selling point is reproducibility. Use the content hash.
        rng = np.random.default_rng(
            int(stable_hash(["explore", self.target.uid, self.seed]), 16) % (2**63))

        # CRITIQUE C10: record the selection probability for every candidate so
        # the labels can be IPS-corrected later. Without this the selection-bias
        # correction is not computable from what the run logs.
        propensity: dict[str, float] = {}
        p_explore = (n_explore / len(rejected)) if rejected else 0.0
        for c in by_id.values():
            propensity[c.cid] = 1.0 if c.alive else p_explore

        if rejected and n_explore > 0:
            pick = rng.choice(len(rejected), size=n_explore, replace=False)
            explored = [rejected[int(i)] for i in pick]
        else:
            explored = []

        t0 = time.perf_counter()
        to_assay = exploit + explored
        results = self.assay.run(self.target, to_assay, self.seed) if to_assay else []
        outcomes = {c.cid: int(y) for c, y in zip(to_assay, results)}
        reports.append(StageReport(
            "assay", self.assay.name, len(to_assay), int(sum(results)),
            len(to_assay) * self.assay.unit_cost, time.perf_counter() - t0,
            params={"exploit": len(exploit), "explore": len(explored)},
        ))
        self.rundir.log("assay", n=len(to_assay), hits=int(sum(results)),
                        exploit=len(exploit), explore=len(explored))

        # --- calibrate ----------------------------------------------------
        last_stage = self.scorers[-1].stage
        cal_x, cal_y = [], []
        for c in to_assay:
            if last_stage in c.scores:
                cal_x.append(c.scores[last_stage])
                cal_y.append(outcomes[c.cid])
        calibrator = Calibrator()
        if cal_x and len(set(cal_y)) > 1:
            calibrator.fit(cal_x, cal_y)
        elif cal_x:
            # All outcomes identical: isotonic is degenerate but the base rate
            # is still information.  Record it and say so.
            calibrator.base_rate = float(np.mean(cal_y))
            calibrator.n_labels = len(cal_y)

        diagnostics: dict[str, Any] = {
            "ladder": describe(rungs),
            "cost": cost_summary(reports),
            "n_designed": len(pool),
            "n_shortlist": len(shortlist),
            "n_assayed": len(to_assay),
            "n_explore": len(explored),
            # named for what it actually measures: the assayed part of the
            # shortlist, which is not the whole shortlist when capacity binds.
            "hit_rate_exploit": hit_rate([outcomes[c.cid] for c in exploit]) if exploit else None,
            "n_exploit": len(exploit),
            "hit_rate_explore": hit_rate([outcomes[c.cid] for c in explored]) if explored else None,
            "calibrator_labels": calibrator.n_labels,
            "calibrator_base_rate": calibrator.base_rate,
        }
        if cal_x:
            probs = calibrator.predict(cal_x)
            diagnostics["ece_raw"] = expected_calibration_error(cal_x, cal_y)
            diagnostics["ece_calibrated"] = expected_calibration_error(probs, cal_y)
            diagnostics["brier_raw"] = brier_score(cal_x, cal_y)
            diagnostics["brier_calibrated"] = brier_score(probs, cal_y)
            diagnostics["reliability"] = reliability_table(probs, cal_y)
            # CRITIQUE A4: the numbers above are IN-SAMPLE. Isotonic regression
            # can drive in-sample ECE to ~0 by construction, so they prove
            # nothing on their own. The cross-validated block is the honest one.
            diagnostics["calibration_cv"] = cross_validated_calibration(
                cal_x, cal_y, seed=self.seed)

        # CRITIQUE C1: hit rates on tens of wells need intervals, not points.
        if exploit:
            w = wilson(sum(outcomes[c.cid] for c in exploit), len(exploit))
            diagnostics["hit_rate_exploit_ci"] = [w.lo, w.hi]
        if explored:
            w = wilson(sum(outcomes[c.cid] for c in explored), len(explored))
            diagnostics["hit_rate_explore_ci"] = [w.lo, w.hi]

        self.rundir.log("done", **{k: v for k, v in diagnostics.items()
                                   if not isinstance(v, (list, dict, str))})

        return CampaignResult(
            target=self.target.name,
            rungs=rungs,
            reports=reports,
            candidates=list(by_id.values()),
            shortlist=shortlist,
            explored=explored,
            outcomes=outcomes,
            propensity=propensity,
            calibrator=calibrator,
            diagnostics=diagnostics,
        )
