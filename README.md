# CALIPER

**Cal**ibrated **i**terative **p**rotein **e**ngineering **r**uns — a budget-aware
allocation layer for de novo protein binder design.

> **A multi-fidelity cascade improves hit rates under an equal compute budget
> but not on a fixed candidate pool.** The crossover between those two regimes
> is the finding, and it follows from correlated errors between structure
> predictors.

CALIPER does not predict structures. It decides **which designs get the next
unit of compute** and **which get a well on the plate**.

---

## Both results are real. The crossover is the point

I built a multi-fidelity cascade (AF2-initial-guess → ColabFold → AF3) that
allocates by rank under a compute budget instead of using fixed confidence
thresholds. On a simulator it beat every baseline. Then I evaluated it on real
experimental data and it lost.

**Real data: 3,650 designs, 15 targets, 10.7% binders**
([Overath et al. 2025](https://zenodo.org/records/15722219), CC-BY-4.0; all four
structure predictors run on the *same* designs). Protocol: leave-one-target-out,
shortlist of 24, thresholds fitted only on training targets.

| policy | hit rate | relative cost |
|---|---|---|
| **run AF3 on everything** | **0.400 [0.300, 0.517]** | 6,820 |
| 2-stage cascade (ColabFold → AF3) | 0.358 [0.246, 0.479] | 5,004 |
| 3-stage cascade (as designed) | 0.338 [0.204, 0.479] | 2,199 |
| random | 0.163 [0.101, 0.253] | 0 |

Both cascades lose to the single best model, significantly
(paired d = −0.71 and −0.73 over 10 targets).

**On a fixed candidate pool, cascading does not pay.**

This is not a bug. Two-stage screening theory (Tang, *Naval Research Logistics*
1988) shows the benefit of a screening stage falls continuously as its
correlation with the final measure rises, converging to the single-stage
procedure at ρ = 1. Our stages correlate at 0.55–0.66. Losing here is what the
theory predicts.

---

## Why it lost — and the rule that came out of it

### 1. The simulator was flattering the method

It modelled stage errors as **independent**. Measured Spearman on real data:

| pair | correlation |
|---|---|
| AF2 ↔ AF3 | 0.550 |
| ColabFold ↔ AF3 | 0.657 |
| AF2 ↔ ColabFold | 0.574 |

They are all structure predictors looking at the same complex, so of course
their errors move together. Independence is the single most favourable
assumption a cascade can be given.

Injecting the measured correlation into the simulator moved the cascade from
**significantly better** (+0.061, d = 0.48) to **no significant difference**
(+0.017). So correlation explains part of the gap — not all of it.

### 2. The mechanism

**The cheap AF2 rung discards 29% of the real binders that AF3's top-24 would
have found.** It correlates 0.55 with AF3 while scoring 0.066 lower AUC: it
looks at the same thing, less accurately. Filtering on it first does not add
information, it adds noise early.

### 3. What actually predicts whether a cascade helps

Across 10 evaluable targets:

| factor | rank correlation with cascade advantage |
|---|---|
| **AUC gap between cheap and final stage** | **−0.652** ← dominant |
| pool size | +0.518 |
| absolute AUC of the cheap stage | +0.146 ← irrelevant |

VirB8: AF2 0.619 vs AF3 0.810 (gap 0.191) → cascade **−0.208**.
Mdm2: AF2 0.580 vs AF3 0.573 (gap −0.007) → cascade **+0.083**.

### 4. The rule

> **Do not add a stage to a cascade because it is cheap.** Add it only if its
> discrimination is close to the final stage's. A stage that is much worse at
> the same job is not a filter — it is noise applied early.

Implemented in `caliper/whentocascade.py`. It agrees with the measured outcome
on 7 of 10 targets. Applied to CALIPER's own configuration it says: **drop the
AF2 stage**. Doing so helps (0.358 vs 0.338) but not significantly, and does
not flip the conclusion.

---

## Where the cascade does win

Its whole purpose is to screen more candidates for the same spend. Given an
**equal compute budget** rather than an equal candidate pool:

| policy | hit rate |
|---|---|
| **cascade (screens 3.0× more designs)** | **0.338 [0.204, 0.479]** |
| best single metric, budget-matched | 0.246 [0.142, 0.358] |

Paired **d = 1.17**, significant, and the cascade wins on **9 of 10 targets**.

**The hit-rate comparison does not depend on the cost estimates.** Ranking is
scale-free, so changing the assumed stage costs (1 : 5.2 : 20 from measured
runtimes, or 1 : 8 : 20, or 1 : 5.2 : 40) leaves every hit rate unchanged. Cost
only sets the multiplier — how many more designs the same money buys.

**The crossover.** The cascade gives up 0.062 hit rate on a fixed pool and gains
0.092 at equal budget, because it screens **3.0×** more designs. That
multiplier is the design parameter: if you can afford to screen roughly 3× more
candidates with cheap early stages, the cascade is worth running. If compute is
free, run the good model on everything.

A reviewer's objection to budget-matching — *"it lets a weak method win by brute
force"* — does not apply here. Under brute force the per-design precision would
stay flat or fall; instead the cascade's hit rate is **higher** than the
budget-matched single filter, so the selection itself is doing work.

---

## The calibration layer, and why its claims were withdrawn

The original plan was to convert confidence scores into calibrated
probabilities using labels from designs the filter *rejected* — correcting the
selection bias that makes published cut-offs non-transferable.

Two things went wrong, both now fixed in the open:

**The reported improvement was overfitting.** v0.1 reported calibration error
falling from 0.610 to 0.000. That was measured in-sample; isotonic regression
drives in-sample ECE to zero by construction. Cross-validated, the honest
figure is **0.610 → 0.293**. Every calibration number is now out-of-fold, and
`test_in_sample_calibration_is_optimistic` fails if that regresses.

**The method was indefensible at the sample size.** Fitting isotonic regression
to 26–48 labels contradicts every published threshold — Niculescu-Mizil &
Caruana (2005) show it overfits below 1,000 points; Riley et al. (2021) require
200 events *and* 200 non-events for a flexible calibration curve. So
`caliper/smallsample.py` now uses **Venn-Abers** (which leads on mean log-loss
below n = 1000 and returns an *interval*, so the width itself carries the
uncertainty), with beta and Platt available, and below the validated regime it
**refuses to emit probabilities at all** and reports average precision instead.
The sample-size gate uses Riley's criteria evaluated at this data's 10.7% event
fraction: ~31 events relaxed, ~346 strict. On this harness Platt beats isotonic
at n = 40, 120 and 400.

What the real data shows about calibration is itself the finding: **it does not
transfer across targets.** Per-target AUC ranges 0.573 (Mdm2) to 1.000 (LTK),
hit rate 2.1% to 57.3%, and a curve fitted on 14 targets mispredicts the 15th
badly (TrkA: predicted 0.307, actual 0.071).

### Per-target calibration, and when to switch to it

That non-transfer is what `caliper/hierarchical.py` exists for, and it is the
one component here that clearly works on real data. The question a campaign
faces is concrete: *a new target has had one round, there are k wells of data —
use them, or trust the pooled curve?*

Out-of-sample Brier on the unseen part of a held-out target, 10 targets,
20 random reveals per point:

| wells revealed | pooled only | target only | **hierarchical** |
|---|---|---|---|
| 5 | 0.137 | 0.160 | **0.135** |
| 10 | 0.134 | 0.140 | **0.130** |
| 20 | 0.121 | **0.114** | 0.115 |
| 40 | 0.118 | **0.108** | 0.110 |
| 80 | 0.096 | **0.086** | 0.087 |

Partial pooling is **never the worst of the three**. With few wells the
target-only fit overfits and pooling saves it; past roughly 20 wells the
target's own data takes over and hierarchical follows it. Against pooled-only it
is significantly better at every k >= 10 (d = -0.22 to -0.51); against
target-only it is significantly better at k = 5 and 10 and ties thereafter.

**The switch-over is about 20 wells.** Below that, borrow from other targets.

### The exploration quota

Spending a few wells on rejected designs measurably improves calibration on the
simulator — out-of-fold ECE 0.220 → 0.160 at 25% exploration, paired d = 0.64.

This is *reject inference*, and a thirty-year credit-scoring literature finds it
usually adds little. The one version that literature endorses is Hand & Henley's:
**actually test a sample of the rejected cases and observe the outcome**, rather
than inferring their labels. That is what this does. It has not been reproduced
on real data.

---

## Install and run

```bash
git clone <this repo> && cd CALIPER
pip install -e .

python experiments/real_data.py         # the main result (needs the CSV below)
python experiments/why_cascade_lost.py  # the post-mortem and the rule
python experiments/budget_matched.py    # where the cascade wins
python experiments/hierarchical_value.py # is per-target calibration worth it?
python experiments/diversity_check.py   # do shortlists contain near-duplicates?
python experiments/validate.py          # simulator, 40 seeds, 4 baselines
pytest                                  # 37 tests
```

Python 3.11+, numpy, scipy, PyYAML (+ pandas for the real-data experiments).
**No GPU and no model weights needed.**

The real-data experiments need `final_dataset.csv` (82 MB, CC-BY-4.0):

```bash
curl -L -o data/overath/final_dataset.csv \
  https://zenodo.org/api/records/15722219/files/final_dataset.csv/content
```

---

## Honest status

| | |
|---|---|
| Allocation layer | **loses on a fixed pool (d=−0.73), wins at equal budget (d=1.17, 9/10 targets)** |
| Decision rule for when to cascade | derived from data, 7/10 agreement, n=10 |
| Calibration | **claims withdrawn** below the validated sample size |
| Exploration quota | works on the simulator, unvalidated on real data |
| Provenance and caching | content-addressed, atomic writes, run manifests |
| **Real tool adapters** | **not implemented** — `external.py` raises rather than silently substituting the simulator |
| Per-target calibration | **validated on real data** — never worse than either alternative, switch-over ~20 wells |
| Multi-round campaigns | not implemented |
| Sequence diversity control | not implemented; measured as a non-problem in this data, but this data cannot test it |

**On comparability.** Two independent literature searches confirmed that no
published work benchmarks allocation policies on a shared binder pool, and no
binder pipeline reports GPU-hours per accepted design. So this does not beat a
published number — there is none. The baselines here were built for the
comparison, and that is the correct way to read every table above.

**On diversity.** A 24-well plate could in principle hold one design tested 24
times. Measured here it does not: all ten targets give 24 distinct sequences,
mean pairwise identity 0.114, one pair in ~2,760 above 90%. But Overath's
designs were pooled from many separate campaigns and are diverse by
construction, so **this dataset cannot test the failure mode** — a single
RFdiffusion run emitting thousands of backbones would look very different. The
gap is recorded rather than filled blind.

`CRITIQUE.md` is an adversarial review of this repository: 68 defects with
severities, 31 fixed, 37 accepted with stated reasons, plus a second pass that
found four more. Read it before trusting anything here. `NEXT.md` is what
remains.

---

## License

MIT. See `LICENSE`.
