"""One entry point for every check this project earned.

What CALIPER turned out to be
-----------------------------
It began as an attempt to build a better binder-design pipeline.  Measured
against real data, most of it did not survive:

    cascade scheduling      conditional -- loses on a fixed pool, wins on budget
    exploration quota       rejected -- no benefit, replaced by a free check
    multi-round learning    rejected -- ceiling +0.015 and unreachable
    IPS bias correction     rejected -- consistently worse
    probability reporting   narrowed -- refused below the validated sample size
    hierarchical calibration  SURVIVED -- switch to target-specific near 20 wells
    inverted-curve detection  SURVIVED -- and came out of a rejected claim

So the useful artefact is not a pipeline that wins.  It is a set of checks that
say when the standard way of running one is about to fail, each earned by
killing a plausible-sounding idea.  This module is where they meet.

Give it what a campaign actually has -- stage scores, a candidate pool size,
whatever outcome labels exist -- and it reports which failure modes apply.

Korean note:
이 저장소의 결론은 "내 파이프라인이 이긴다"가 아니라 "표준적인 방식이 언제 실패하는지"다.
검증에서 살아남은 검사들을 여기 모아, 실제 캠페인 데이터를 넣으면 어떤 실패 양상에
해당하는지 알려준다.  각 규칙은 그럴듯한 아이디어를 하나씩 죽여서 얻어졌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .smallsample import (PlattCalibrator, average_precision,
                          check_calibration, choose_calibration)
from .stats import wilson
from .whentocascade import cost_saving, should_cascade

# The point at which a target's own labels beat borrowing from other targets.
# Measured on the Overath data: below this the target-only fit overfits and
# pooling rescues it; above it the target's own data takes over.
HIERARCHICAL_SWITCH_WELLS = 20


@dataclass(slots=True)
class Finding:
    level: str          # "blocker" | "warning" | "ok"
    check: str
    message: str
    evidence: str = ""

    def __str__(self) -> str:
        mark = {"blocker": "STOP", "warning": "WARN", "ok": "OK  "}[self.level]
        out = f"[{mark}] {self.check}\n       {self.message}"
        return out + (f"\n       evidence: {self.evidence}" if self.evidence else "")


@dataclass(slots=True)
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "blocker"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    def add(self, level: str, check: str, message: str, evidence: str = "") -> None:
        self.findings.append(Finding(level, check, message, evidence))

    def __str__(self) -> str:
        head = (f"{len(self.blockers)} blocking, {len(self.warnings)} warning, "
                f"{len(self.findings)} checks run")
        body = "\n".join(str(f) for f in self.findings)
        return f"CALIPER audit -- {head}\n{'-' * 72}\n{body}"


def audit(
    *,
    stage_aucs: dict[str, float] | None = None,
    stage_costs: dict[str, float] | None = None,
    pool_size: int | None = None,
    scores: Sequence[float] | None = None,
    outcomes: Sequence[float] | None = None,
    all_scores: Sequence[float] | None = None,
    n_target_wells: int | None = None,
    shortlist_size: int = 24,
) -> AuditReport:
    """Run every surviving check against one campaign's situation.

    Every argument is optional; each check runs only if it has what it needs and
    stays silent otherwise, so a campaign with partial information still gets
    whatever can be said.

    Parameters
    ----------
    stage_aucs, stage_costs, pool_size
        For the cascade check: is each cheap stage earning its place?
    scores, outcomes
        Labelled designs so far -- the wells that came back.
    all_scores
        Scores for the whole pool, used to see how much of the range the labels
        actually cover.
    n_target_wells
        How many labelled wells exist for THIS target, for the pooling decision.
    """
    r = AuditReport()

    # ---- 1. is the cascade worth it? ------------------------------------
    if stage_aucs and stage_costs and pool_size:
        names = list(stage_aucs)
        final = names[-1]
        costs = [stage_costs[n] for n in names]
        saving = cost_saving(pool_size, costs, n_final=shortlist_size)
        for name in names[:-1]:
            v = should_cascade(stage_aucs[name], stage_aucs[final], pool_size,
                               costs, n_final=shortlist_size)
            if not v.should_cascade:
                r.add("warning", f"cascade stage: {name}",
                      f"drop this stage. {v.reason}",
                      f"AUC gap {v.auc_gap:+.3f}, ladder {saving:.1f}x cheaper")
            else:
                r.add("ok", f"cascade stage: {name}",
                      f"keep. {v.reason}")
        r.add("ok", "cascade economics",
              f"the ladder is {saving:.1f}x cheaper than running {final} on "
              f"everything. On real data a cascade LOSES on a fixed pool and "
              f"WINS at equal budget, so this number is the whole argument: it "
              f"is what lets you screen {saving:.1f}x more designs.",
              "measured: fixed pool d=-0.73, equal budget d=+1.17 (9/10 targets)")

    # ---- 2. may we report probabilities at all? -------------------------
    if outcomes is not None and len(outcomes) > 0:
        y = np.asarray(outcomes, dtype=float)
        d = choose_calibration(y)
        if not d.report_probabilities:
            r.add("blocker", "calibration sample size",
                  f"do not report probabilities. {d.reason}",
                  f"n={d.n_labels} ({d.n_events} events)")
            if scores is not None:
                ap = average_precision(scores, y)
                w = wilson(int(y.sum()), int(y.size))
                r.add("ok", "what to report instead",
                      f"ranking quality is still meaningful: average precision "
                      f"{ap:.3f}, hit rate {w}.")
        else:
            r.add("ok", "calibration sample size",
                  f"{d.method} calibration is appropriate. {d.reason}",
                  f"n={d.n_labels} ({d.n_events} events)")

    # ---- 3. does the score point the right way on THIS target? ----------
    #
    # Deliberately independent of the sample-size gate above.  The gate decides
    # whether a probability may be reported; this decides whether the score is
    # even ordered correctly here, and that matters most exactly when labels are
    # scarce.  Running it only when the gate passes would have made it
    # unreachable: at a 10% hit rate, 96 wells still yield only ~20 events.
    if (scores is not None and outcomes is not None and all_scores is not None
            and len(scores) > 1 and len(set(np.asarray(outcomes).tolist())) > 1):
        diag = PlattCalibrator().fit(scores, outcomes)
        h = check_calibration(diag, scores, all_scores)
        if not h.ok:
            r.add("blocker", "score direction",
                  "on your labelled wells a HIGHER score went with a WORSE "
                  "outcome, so this score is ordering the pool backwards for "
                  "this target. Do not rank by it, and do not trust any "
                  "threshold derived from it. Fall back to the ordering that "
                  "worked on your other targets. " + h.reason,
                  f"slope {h.slope:+.2f}, labels cover "
                  f"{100 * h.span_fraction:.0f}% of the score range")
        else:
            r.add("ok", "score direction", h.reason)
            if h.warning:
                r.add("warning", "score coverage", h.warning)

    # ---- 4. pooled or target-specific calibration? ----------------------
    if n_target_wells is not None:
        if n_target_wells < HIERARCHICAL_SWITCH_WELLS:
            r.add("warning", "which calibration curve",
                  f"{n_target_wells} wells on this target is below the "
                  f"{HIERARCHICAL_SWITCH_WELLS} at which a target-specific fit "
                  "starts beating a borrowed one. Use HierarchicalCalibrator so "
                  "the curve leans on your other targets.",
                  "measured: at 5-10 wells a target-only fit is the WORST of "
                  "three strategies; partial pooling is never the worst")
        else:
            r.add("ok", "which calibration curve",
                  f"{n_target_wells} wells is past the ~{HIERARCHICAL_SWITCH_WELLS}-well "
                  "switch-over, so this target's own data can carry the fit. "
                  "Partial pooling still costs nothing and never hurt.")

    # ---- 5. things measured and found not to work -----------------------
    r.add("ok", "ideas already ruled out",
          "an exploration quota (no benefit; the slope check above replaces "
          "it), multi-round metric switching (ceiling +0.015, unreachable from "
          "24 wells), and IPS reweighting (consistently worse) were all "
          "measured on real data and rejected. Do not spend wells on them.")

    return r
