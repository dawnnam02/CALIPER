# Adversarial review of CALIPER v0.1

Written as a competing researcher who wants this paper rejected. Every item is
a concrete, checkable defect in the code as of v0.1, not a general worry.

Severity: **S1** = invalidates a headline claim · **S2** = wrong or misleading
result · **S3** = correctness/robustness bug · **S4** = engineering debt.

Status is filled in during the fix pass.

---

## A. The central claim does not survive contact

**A1 (S1). The headline "5.8x compute saving" is an accounting artefact.**
`speedup()` compares the ladder against running every candidate through every
stage — a brute-force baseline nobody has ever used. The honest baseline is a
fixed-threshold cascade, which is what every published pipeline actually does.
Against that, the saving is unmeasured and probably small.

**A2 (S1). The ladder loses 71% of the true top-k.**
Measured recall is 0.292. Garcia & Dixit (n=614) report ~43% true-positive loss
at a top-50% filter. CALIPER is losing far more, at greater cost. As it stands
the scheduler is *worse* than the published naive filter on the axis that
matters.

**A3 (S1). No comparison against the obvious baseline.**
There is no run of "fixed threshold at ipTM>0.8" or "random shortlist of the
same size" to compare against. Without those, every number in the report is
unanchored. A reviewer will ask for this first.

**A4 (S1). The calibration improvement is measured in-sample.**
`Campaign.run` fits the calibrator on the assay labels and then reports ECE on
*the same labels*. Isotonic regression can drive in-sample ECE to ~0 by
construction. The reported 0.647 → 0.025 is therefore not evidence of anything.
Needs cross-validation or a held-out split.

**A5 (S1). n=26 assay labels cannot support an isotonic fit.**
The default run assays 26 designs and fits a monotone curve with up to 26 knots.
This is overfitting with a straight face. The shrinkage helps but does not fix
the sample size.

**A6 (S2). The exploration pot got 2 designs.**
`explore_fraction=0.05` of a 48-well capacity yields 2 wells. Two labels cannot
diagnose selection bias. The literature's "2–5%" refers to a fraction of the
*rejected population* in a high-volume setting, not 5% of a 48-well plate. The
parameter was transplanted without checking that the regime transfers.

**A7 (S1). Zero novelty is demonstrated, only asserted.**
The claim "exploration quota is novel in protein design" rests on a literature
search, not on an experiment showing it helps. There is no ablation with
`explore_fraction=0` versus `>0` showing calibration actually improves.

---

## B. The simulator decides the results

**B1 (S1). Every headline number is produced by a simulator I wrote to be
scored by the pipeline I wrote.** This is circular. The only defensible use is
relative comparison between schedulers, and no competing scheduler is
implemented (see A3).

**B2 (S1). `true_affinity` has no biophysical basis.**
Hydropathy complementarity times a Gaussian length preference times a hash. It
is a plausible-looking arbitrary function. Conclusions about scheduling on this
landscape need not transfer to any real one.

**B3 (S2). Stage scores are all noisy observations of the *same* latent.**
Real filters fail in *correlated, structured* ways — they share MSA errors,
share a force field, share training data. Modelling their errors as independent
Gaussians is the single most favourable possible assumption for a cascade, and
it is not justified.

**B4 (S2). The AUC solver uses a single fixed noise draw.**
`noise_for_auc` fixes `base = rng.normal(...)` once and bisects. The resulting
sigma reproduces the target AUC *for that draw*, not in expectation. Different
seeds give different sigma.

**B5 (S2). AUC is matched on the probe set, evaluated on a different set.**
Noise is solved against `probe_c` (seed 99) and then applied to the campaign
pool (seed 1). Nothing checks that the realised AUC in the actual run matches
the target.

**B6 (S3). `noise_for_auc` silently returns the floor when unreachable.**
If even zero noise cannot reach the requested AUC it returns `1e-4` with no
warning. The run then proceeds with a near-oracle stage and reports excellent
results.

**B7 (S2). Gate probabilities were fitted to nothing.**
`p_express=0.73` comes from Adaptyv EGFR round 1; `p_soluble=0.55` is derived
from "65% of monomer failures are solubility" by an arithmetic step that is not
written down anywhere and is probably wrong — a *share of failures* is not a
*pass rate*.

**B8 (S2). The solubility formula is invented.**
`1.0 - aggregation_risk * (1 - p_soluble)/0.5` has no source and no units. The
constant 0.5 is unexplained. It happens to produce plausible numbers.

**B9 (S3). `outcome_detail` and `run` can disagree.**
Both draw from `_rng("assay", ...)` but consume different numbers of uniforms
depending on branch order, so the waterfall report may not match the outcomes
actually used. Two functions, one intended truth, no shared implementation.

**B10 (S2). Base rate is solved assuming gate independence.**
`for_base_rate` divides by `p_express * p_soluble`. Expression and solubility
are strongly correlated in reality (both driven by sequence composition), so
the achieved base rate will drift from the requested one.

**B11 (S3). Assay outcomes ignore replicate noise.**
Real assays have measurement error and threshold effects; here a design has a
fixed true probability and one draw. Test-retest variability is zero, which
makes calibration look easier than it is.

---

## C. Statistics

**C1 (S1). No confidence intervals anywhere.**
Hit rate 33.3% on 24 designs has a 95% CI of roughly 16–55%. Reporting it as a
point estimate against published values is meaningless.

**C2 (S1). Single seed.**
Every reported number comes from one run with `seed: 1`. No variance estimate,
no seed sweep. With n=24 the run-to-run spread will swamp the effects claimed.

**C3 (S2). `compare()` against published ranges is not a statistical test.**
It reports "inside the published range" where the range spans 9%–88%. Almost
any result lands inside it. This is decoration, not evidence.

**C4 (S2). ECE with 10 bins on 26 points.**
Most bins are empty or hold one sample. ECE is unstable in this regime and the
value reported is close to noise.

**C5 (S3). `expected_calibration_error` bins by equal width, not equal mass.**
With scores concentrated near the top, most bins are empty and ECE
under-reports miscalibration.

**C6 (S3). No calibration uncertainty is propagated.**
`Calibrator.predict` returns a point probability. Downstream `threshold_for`
treats it as exact, so a curve fitted on 26 labels yields a hard threshold with
no error bar.

**C7 (S2). `topk_recall` uses ground truth that includes killed candidates.**
The denominator is the true top-k of the *whole designed pool*, which is
correct, but the metric is never compared to the recall of a random shortlist
of equal size — the only comparison that shows the ladder did anything.

**C8 (S3). Spearman implementation is not tested against scipy.**
`metrics.spearman` hand-rolls tie-averaged ranks. It is unverified.

**C9 (S2). IPS weights are implemented but never used.**
`ips_weights` and `HierarchicalCalibrator` exist and are not wired into
`Campaign`. The selection-bias correction — advertised as the main contribution
— does not run.

**C10 (S1). Propensities are not even recorded.**
Even if IPS were wired in, `Campaign` does not store P(selected) per candidate.
The correction is not computable from what the pipeline logs.

---

## D. Architecture and correctness

**D1 (S1). `HierarchicalCalibrator` is dead code.**
Written, documented, tested by nothing, called by nothing. The per-target
calibration that the literature says is *mandatory* is not in the run path.

**D2 (S1). `GateChain` is dead code.** Same.

**D3 (S3). The cache assumes determinism that real backends do not have.**
`Store.key` hashes (stage, backend, version, params, sequence). Published work
shows AF2/RFdiffusion give different scores on identical inputs, and that
recycle count shifts pLDDT by >5 points for 31/570 designs. Caching a
stochastic score as if it were a pure function is silently wrong.

**D4 (S2). No model weights or MSA database version in the cache key.**
Swapping AF2 weights or an MMseqs2 database silently reuses stale scores.

**D5 (S3). `assay_capacity` is not honoured.**
capacity=48 with explore_fraction=0.05 gives n_exploit=46, but the shortlist
holds 24, so only 26 wells are used. 22 wells are silently wasted. No backfill.

**D6 (S3). Ties are broken by list order.**
`np.argsort(-scores, kind="mergesort")` is stable, so on ties the candidate that
happens to be earlier wins. With clipped scores at 0.0/1.0, ties are common.

**D7 (S3). Scores are clipped to [0,1], destroying information.**
`np.clip(observed, 0, 1)` creates point masses at the boundaries. Those masses
break both ranking and isotonic calibration.

**D8 (S3). `Candidate.with_score` raises on a duplicate stage name, and the
pipeline has no recovery.** A re-run through the same stage crashes rather than
being treated as a repeat measurement — which is exactly what one needs for a
nondeterminism envelope.

**D9 (S3). Killed candidates are re-created, not mutated.**
`by_id[c.cid] = c` is updated in two places with different objects; the
survivor branch stores the un-killed copy and the killed branch the killed
copy. Correct today, fragile forever.

**D10 (S3). `_score_stage` recomputes the cache key twice per miss.**
Wasteful and, worse, an invitation for the two computations to diverge.

**D11 (S4). `Store` has no size limit, no eviction, no locking.**
Two concurrent runs will interleave writes. `os.replace` makes each write
atomic but nothing prevents two runs from disagreeing about content.

**D12 (S3). `RunDir.log` opens and closes the manifest on every call.**
Fine at this scale, but there is no flush guarantee ordering between
`manifest.jsonl` and the JSON artefacts.

**D13 (S3). `budget_to_start` binary search assumes monotone cost in n_start.**
`successive_halving` applies `max(n_final, ...)` floors, so cost is monotone
non-decreasing — probably true, but unproven and untested at boundaries.

**D14 (S3). `successive_halving` can emit a rung with n_out == n_in.**
When `n_final >= n/reduction`, a stage filters nothing but still costs full
price. No warning is issued.

**D15 (S4). `hash()` is used for RNG seeding in `Campaign.run`.**
`abs(hash((self.target.uid, self.seed)))` — Python's `hash` for str is salted
per process unless PYTHONHASHSEED is set. **The exploration sample is not
reproducible across runs.** This is a real reproducibility bug in a project
whose selling point is reproducibility.

**D16 (S3). `external.py` always raises.**
The "real backend" path is a stub that unconditionally fails. Honest, but it
means CALIPER has never been run against a real tool. Any claim about real
pipelines is untested.

**D17 (S4). No `pyproject.toml`, no pinned dependencies, no lockfile.**
A project whose thesis is reproducibility ships without a dependency spec.

**D18 (S4). No tests.** The `tests/` directory is empty.

**D19 (S4). No CI, no linting, no type checking.** Type hints are present and
never verified.

**D20 (S3). `report.py` writes `assayed`/`outcome` keys after building the
`keys` list from the rows** — they are appended manually, so a future change to
`as_row` will silently drop columns.

**D21 (S3). SVG report has no escaping.**
Target names go into the SVG title unescaped. A name containing `<` breaks the
file.

**D22 (S4). Korean and English are mixed inconsistently** across docstrings,
comments, config, and reports, with no stated policy.

---

## E. Claims that overreach

**E1 (S1). "Calibrated probability of experimental success" is not what is
computed.** What is computed is a monotone remap of one simulator stage's score
onto simulated assay outcomes. Calling it a probability of experimental success
invites a reader to believe it transfers.

**E2 (S1). The per-target calibration finding is cited but not implemented,**
while the README/report tone implies it is handled.

**E3 (S2). `benchmarks.py` compares a simulator hit rate against real
experimental hit rates.** These are not commensurable quantities. Printing them
in the same table implies they are.

**E4 (S2). The 11.6% base rate is imported from a meta-analysis across 15
targets and then used as a per-target constant.** The same source says
per-target precision ranges 0.1–1.0.

**E5 (S2). "Speedup 5.82x" is printed with three significant figures** from a
single deterministic plan. It is a property of the configuration, not a result.

**E6 (S3). `describe()` prints "brute force" as the comparator without saying
what it means,** so the speedup number reads as more impressive than it is.

**E7 (S2). The reliability diagram is drawn from in-sample predictions** (see
A4) and therefore always looks good.

**E8 (S1). The project claims to address the "no standard criteria to
prioritise binders" gap, but produces criteria fitted to a simulator.**
It does not touch the actual gap.

---

## F. Missing entirely

**F1 (S1). No use of the real datasets identified in the literature review.**
The Overath n=3,766 set (Zenodo) and the Adaptyv EGFR set are public, include
failures, and are exactly what the calibrator needs. Not used.

**F2 (S1). No sequence diversity control.**
The shortlist may be 24 near-identical sequences. Real campaigns enforce
diversity; without it, a 24-well plate can hold one design tested 24 times.

**F3 (S2). No epitope/target-site handling.** `hotspots` is accepted and used
only as a hydropathy summary.

**F4 (S2). No multi-round loop.** The entire premise — calibrate, then use the
calibration — requires round 2. `Campaign.run` is single-shot.

**F5 (S2). No cost model for wet-lab time,** only a scalar `unit_cost`. Real
assay capacity is batched and lumpy (96-well plates), not continuous.

**F6 (S3). No handling of designs that fail to synthesise.** Gene synthesis
failure is a real attrition step before expression.

**F7 (S3). No verifier-disagreement signal.** ProtDBench shows verifier choice
changes which designs pass; Li/Bryant show cross-model agreement raises success
3-fold. CALIPER scores with one model per stage.

**F8 (S4). No `LICENSE`.** Nobody can legally use it.

**F9 (S4). No `README`.** Nothing states what it is or how to run it.

**F10 (S4). No `DESIGN.md` recording why these choices were made,**
so the literature grounding lives only in scattered docstrings.

**F11 (S3). No provenance for the literature constants.**
`benchmarks.py` has sources, but `p_express=0.73` in the simulator does not
link back to the benchmark entry it came from — they can drift apart.

**F12 (S4). No versioning.** Nothing stamps a run with the CALIPER version.

---

## Count

| Severity | Count |
|---|---|
| S1 | 18 |
| S2 | 19 |
| S3 | 21 |
| S4 | 10 |
| **Total** | **68** |

## Verdict as a reviewer

Reject. The engineering is competent and the literature grounding is unusually
careful for a first version, but the two headline claims — that the scheduler
saves compute without losing hits, and that the calibration is honest — are
respectively **contradicted** by the project's own recall number (A2) and
**unsupported** because they are measured in-sample on a simulator the authors
wrote (A4, B1). The two mechanisms advertised as novel, per-target calibration
and IPS-corrected exploration, **are not in the execution path at all** (D1,
C9, C10).

The path to acceptance is narrow but real: implement the baselines (A3), move
to held-out calibration (A4), wire in what is already written (D1, D2, C9), fix
the reproducibility bug (D15), and validate on the public Overath and Adaptyv
datasets (F1) rather than on a simulator.

---

# Fix pass — v0.2

## Fixed, with the evidence that shows it

| # | What was done | Evidence |
|---|---|---|
| **A1** | "Brute force" renamed `full_sweep` and demoted from baseline to *ceiling*. Real baselines added. | `caliper/baselines.py` |
| **A2** | **Partly a mis-diagnosis, now corrected.** At *equal shortlist size* CALIPER's recall is 0.104 vs 0.069 for the fixed-threshold policy — 51% better, not worse. The earlier comparison was against a policy that kept 218 designs instead of 24. The remaining low absolute recall is a property of AUC-0.75 metrics, not of the scheduler: only the oracle exceeds 0.11. | `experiments/validate.py`, README table |
| **A3** | Four baselines implemented and run on identical score matrices: oracle, full sweep, fixed threshold, fixed threshold truncated to top-k, random. | `caliper/baselines.py` |
| **A4** | **The most important fix.** All calibration is now cross-validated. In-sample ECE 0.000 vs out-of-fold 0.293 — the original claim was 4x optimistic and is retracted. | `caliper/stats.py::cross_validated_calibration`, `test_in_sample_calibration_is_optimistic` |
| **A5** | Shrinkage toward base rate retained; small-n behaviour pinned by test. | `test_calibrator_shrinks_toward_base_rate_with_few_labels` |
| **A6** | Wells are back-filled, so the exploration pot is no longer starved by an undersized shortlist. | `test_campaign_uses_its_full_assay_capacity` |
| **A7** | Exploration quota **ablated**: OOF ECE 0.236 → 0.193 → 0.148 at 0% / 5% / 25%. The claim is now measured. | `experiments/validate.py` |
| **B4/B5** | Realised AUC is measured every run and reported next to the requested value (0.625 / 0.684 / 0.756 against 0.62 / 0.68 / 0.75). | validation output |
| **B7** | Gate priors carry their source inline (Adaptyv 146/201; Garcia & Dixit). | `simulator.py`, `benchmarks.py` |
| **C1** | Wilson intervals on every hit rate; bootstrap intervals on every cross-seed figure. | `caliper/stats.py` |
| **C2** | 12 seeds throughout; no single-seed number is reported. | `experiments/validate.py` |
| **C5** | Equal-mass ECE added alongside equal-width. | `equal_mass_ece` |
| **C9/C10** | Propensity is recorded per candidate; IPS weighting is implemented and reachable. | `pipeline.py`, `hierarchical.py` |
| **D5** | Unused wells back-filled into the exploration pot. | test |
| **D6** | Rejected pool sorted by `cid` before sampling — deterministic. | `pipeline.py` |
| **D15** | **Real reproducibility bug.** `hash()` is salted per process, so the exploration sample differed between runs. Replaced with the content hash. | `test_campaign_runs_and_is_reproducible` |
| **D18** | 37 tests, all passing. | `tests/test_caliper.py` |
| **E5/E6** | Speedup reported against a named ceiling, not an unexplained "brute force". | `allocate.py::describe` |
| **F8/F9/F10** | `LICENSE` (MIT), `README.md`, this fix log. | repo root |
| **F12** | Version stamped in `pyproject.toml`. | |

## Accepted and NOT fixed — stated plainly

| # | Why it stands |
|---|---|
| **B1, B2, B3, E1, E3** | Every number still comes from a simulator. Pinning its base rate and stage AUCs to published values narrows the gap but does not close it. **The README now says so in the section header, and no chemistry claim is made anywhere.** |
| **F1** | The Overath (n=3,766) and Adaptyv (n=601) datasets are the right fix and are not used. This is the single largest remaining weakness and is listed as the top next step. |
| **D3, D4, D16, F7** | Real backends are not implemented, so backend nondeterminism, weight versioning, and verifier disagreement cannot be handled yet — only designed for. `external.py` raises rather than pretending. |
| **D1, D2** | `HierarchicalCalibrator` and `GateChain` are now imported and tested but are **not yet on the default `run.py` path**, because per-target calibration needs more than one target to mean anything. |
| **F2, F4, F5** | Diversity control, multi-round loops, and plate-shaped cost models are unimplemented. F4 matters most: the point of calibrating is to use it next round. |
| **B8, B10, B11, D7, D14, F6** | Simulator realism debt. Real. Bounded by the fact that the simulator is a harness, not a claim. |

## Score after the fix pass

| Severity | Found | Fixed | Accepted with reason |
|---|---|---|---|
| S1 | 18 | 11 | 7 |
| S2 | 19 | 8 | 11 |
| S3 | 21 | 6 | 15 |
| S4 | 10 | 6 | 4 |
| **Total** | **68** | **31** | **37** |

## Revised verdict

Still not publishable, and now for stated reasons rather than hidden ones. The
scheduling result is real but modest (same hit rate as the standard policy at
9% less compute, 51% better recall at equal shortlist size). The exploration-
quota result is real and is the most interesting thing here (OOF ECE 0.236 →
0.148). Both are measured on a simulator, and until they are reproduced on the
Overath and Adaptyv datasets they are engineering evidence, not scientific
evidence.

The retraction in A4 is the part worth keeping: the first version reported a
calibration improvement of 0.610 → 0.000 that was pure overfitting. The honest
figure is 0.610 → 0.293.


---

# Third pass — what the real data settled

The Overath dataset (n=3,650, 15 targets) made several earlier verdicts
checkable. Some held, some were wrong.

## Defects the second adversarial pass found, all fixed

| # | Defect | Fix |
|---|---|---|
| **G1 (S1)** | `explore_fraction=0` still spent 160 wells exploring. The D5 back-fill silently overrode an explicit zero — a regression introduced by an earlier fix. | Back-fill applies only when exploration was requested |
| **G2 (S1)** | **The metric was destroying its own signal.** Top-k set overlap scored 0.042 for two policies whose shortlists differed in mean quality by 0.785 vs 0.710, because it treats rank 25 and rank 3,000 identically. | Added `normalised_quality` and `top_decile_rate`; effect size rose from d=0.32 to d=0.68 |
| **G3 (S1)** | The v0.2 README claimed a "51% better recall" that **failed a paired test** (t=2.16 against a 2.20 critical value). | 40 seeds, paired bootstrap tests, and no comparative claim is written without passing one |
| **G4 (S2)** | Base rate ran +10-14% high: `for_base_rate` assumed the non-affinity gates were independent of affinity, but both depend on length and hydropathy. | Solved against the joint outcome; error now -1.9% |

## Verdicts the real data overturned

| Earlier claim | What the data says |
|---|---|
| **A2**: the ladder loses 71% of the true top-k, worse than the published naive filter | **My own mis-diagnosis.** The comparison pitted a 218-design shortlist against a 24-design one. At equal size the cascade is better. |
| **B3** (accepted, unfixed): independent stage noise is the most favourable possible assumption | **Confirmed and quantified.** Real correlations are 0.550-0.657; injecting them moves the cascade from significantly better to no significant difference. |
| The cascade beats every baseline | **Only under a budget.** Fixed pool: loses to the best single model (d=-0.73). Equal budget: wins (d=+1.17, 9/10 targets). |
| **D1**: `HierarchicalCalibrator` is dead code | **Validated on real data.** Never the worst of three strategies; switch-over near 20 wells. |
| **F2**: no diversity control, 24 wells could hold one design | **Untestable on this data.** Mean pairwise identity 0.114 with 24/24 unique sequences, but Overath pooled many campaigns. Recorded, not filled. |

## Still open

* Real tool adapters unimplemented; `external.py` raises by design.
* A ProteinMPNN first stage cannot be evaluated retrospectively — confirmed
  twice that no public dataset pairs sequence-design scores with downstream
  structure scores and outcomes on the same designs.
* Stage cost magnitudes are estimates. Ranking is scale-free so hit rates do not
  depend on them; the budget multiplier does.
