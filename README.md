# CALIPER

**Cal**ibrated **i**terative **p**rotein **e**ngineering **r**uns — a budget-aware
allocation layer for de novo protein binder design.

> **Most of what I built did not survive contact with real data.** What is left
> is a set of checks that say when a confidence-score-driven design campaign is
> about to fail — each one earned by killing an idea that sounded good.

## The result worth reading first

Confidence scores sometimes point the **wrong way** on a target: the higher the
score, the *worse* the design does. A campaign that fits a calibration curve on
its own top-N wells can end up predicting 0.96 for a pool whose true hit rate is
0.05. It costs nothing to detect this, and no published pipeline checks.

Validated on **two independent datasets** — different targets, different people
generating the designs, different labs running the assay:

| | |
|---|---|
| cells tested | 41 (29 Overath + 12 Adaptyv) |
| **sensitivity** — catastrophes caught | **0.917** [0.646, 0.985] |
| **precision** — firings that were real | **0.846** [0.578, 0.957] |
| specificity | 0.931 [0.780, 0.981] |
| mean out-of-sample ECE when it fires | **0.664** |
| mean when it stays silent | **0.076** |

An order of magnitude between the two groups, from arithmetic on data the
campaign already has. `python experiments/detector.py`

*Sanity check first: on the Adaptyv data this code measures ipTM AUC 0.636 and
pLDDT AUC 0.656, against 0.64 and 0.66 as published — so the files are being
read the way their authors intended.*

CALIPER does not predict structures, and it is no longer trying to be a pipeline
that wins. It answers four questions about a campaign you are already running:

| question | answer | how it was earned |
|---|---|---|
| Is this cheap cascade stage worth keeping? | `whentocascade` | measured: AUC gap dominates (ρ=−0.65) |
| Is my score pointing the **wrong way** on this target? | `check_calibration` | sensitivity 0.917 on two independent datasets |
| Do I have enough wells to quote a probability? | `choose_calibration` | Riley's criteria at this data's 10.7% event rate |
| Pooled curve, or this target's own? | `HierarchicalCalibrator` | measured: switch near 20 wells |

```python
from caliper.audit import audit
print(audit(stage_aucs=..., stage_costs=..., pool_size=...,
            scores=..., outcomes=..., all_scores=..., n_target_wells=...))
```

### The scoreboard

| idea | verdict |
|---|---|
| cascade scheduling | ⚠️ **conditional** — loses on a fixed pool, wins at equal budget |
| exploration quota *(the novel one)* | ❌ **rejected** — no benefit; a free check replaced it |
| multi-round metric switching | ❌ **rejected** — ceiling +0.015, unreachable |
| IPS bias correction | ❌ **rejected** — consistently worse |
| reporting probabilities | ⚠️ **narrowed** — refused below the validated sample size |
| hierarchical calibration | ✅ **survived** |
| inverted-curve detection | ✅ **survived** — and is the strongest result here |

Five killed, two standing. One of the survivors exists only because chasing a
dead claim turned up something cheaper.

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

### The exploration quota — withdrawn, and what replaced it

The most novel claim here was that spending wells on designs the filter
*rejected* buys better calibration, by supplying the low-score labels a
winners-only campaign never sees. On the simulator it worked: out-of-fold ECE
0.220 → 0.160 at 25% exploration, d = 0.64.

**On real data it does not.** Aggregating by (target, budget) cell — the exploit
arm is deterministic, so counting its repeats separately would inflate one
failure into forty — gives 19 cells:

| | exploit | explore 25% |
|---|---|---|
| median ECE | 0.065 | 0.061 |
| **maximum ECE** | **0.916** | **0.167** |
| paired test | no significant difference (diff +0.041, CI [−0.017, +0.142]) | |

In 18 of 19 cells it is a wash. What the quota did do was eliminate a single
catastrophic failure — and chasing that failure produced something better than
the quota itself.

**The catastrophe.** EGFR, top 24 of 434 designs. The labelled scores span
0.650–0.723, which is 10% of the full range. Inside that narrow band the score
correlates *negatively* with outcome by chance, so Platt fits a slope of −17.1.
The curve then predicts 0.962 for the 410 unassayed designs whose true rate is
0.046. ECE 0.916. At N=48 and N=96 the same target spans 17% and 33%, the slope
comes out positive, and ECE falls to 0.017.

**The free fix.** An inverted curve is detectable from the labelled data alone —
checking the sign of the fitted slope costs nothing, while an exploration quota
costs a quarter of the plate. `caliper.smallsample.check_calibration` does it,
and across those 19 cells it rejected **exactly** the catastrophic fit and
nothing else. The worst curve it let through had ECE 0.138.

So: drop the exploration quota, keep the slope check.

For the record, this claim was *reject inference*, which a thirty-year
credit-scoring literature finds usually adds little. The one version that
literature endorses is Hand & Henley's — actually testing a sample of rejected
cases rather than inferring their labels — which is what was tested here. The
literature was right.

---

## Install and run

```bash
git clone <this repo> && cd CALIPER
pip install -e .

python experiments/real_data.py         # the main result (needs the CSV below)
python experiments/why_cascade_lost.py  # the post-mortem and the rule
python experiments/budget_matched.py    # where the cascade wins
python experiments/hierarchical_value.py # is per-target calibration worth it?
python experiments/detector.py          # the headline result, two datasets
python experiments/exploration_verdict.py # is an exploration quota worth its wells?
python experiments/multiround.py        # does round 2 learn from round 1?
python experiments/diversity_check.py   # do shortlists contain near-duplicates?
python experiments/validate.py          # simulator, 40 seeds, 4 baselines
pytest                                  # 37 tests
```

Python 3.11+, numpy, scipy, PyYAML (+ pandas for the real-data experiments).
**No GPU and no model weights needed.**

The real-data experiments need two public datasets:

```bash
# Overath et al. 2025 -- 3,650 designs, 15 targets (82 MB, CC-BY-4.0)
curl -L -o data/overath/final_dataset.csv \
  https://zenodo.org/api/records/15722219/files/final_dataset.csv/content

# Adaptyv EGFR competition round 2 -- 380 designs, independent (ODbL)
curl -L -o data/adaptyv/round2.csv \
  https://raw.githubusercontent.com/adaptyvbio/egfr_competition_2/main/results/result_summary.csv
```

They are independent in every way that matters: different targets, different
teams generating the designs, different labs running the assay. The first
version of this project rested on Overath alone, which was its weakest
feature.

---

## Honest status

| | |
|---|---|
| Allocation layer | **loses on a fixed pool (d=−0.73), wins at equal budget (d=1.17, 9/10 targets)** |
| Decision rule for when to cascade | derived from data, 7/10 agreement, n=10 |
| Calibration | **claims withdrawn** below the validated sample size |
| Exploration quota | **withdrawn** — no benefit on real data; replaced by a free slope check |
| Provenance and caching | content-addressed, atomic writes, run manifests |
| **Real tool adapters** | **not implemented** — `external.py` raises rather than silently substituting the simulator |
| Per-target calibration | **validated on real data** — never worse than either alternative, switch-over ~20 wells |
| Multi-round campaigns | **measured and rejected** — see below |
| Sequence diversity control | not implemented; measured as a non-problem in this data, but this data cannot test it |

**On comparability.** Two independent literature searches confirmed that no
published work benchmarks allocation policies on a shared binder pool, and no
binder pipeline reports GPU-hours per accepted design. So this does not beat a
published number — there is none. The baselines here were built for the
comparison, and that is the correct way to read every table above.

**On multi-round campaigns.** A second round informed by the first sounds
obviously right, so it was measured before being built. Two findings killed it.

First, *calibration cannot change a ranking* — a calibration curve is a monotone
map, so it changes how many designs are worth sending, never which ones. The
only thing round-1 labels can change is which score you rank by, and the best
metric genuinely does vary by target (six different metrics win across ten
targets; the global favourite is best for only four).

Second, exploiting that is not possible at this scale:

| round-2 policy | hit rate |
|---|---|
| oracle (knows the truly best metric) | 0.239 |
| static (never switch) | 0.225 |
| switch (naive argmax on revealed wells) | 0.220 |
| guarded (switch only past one standard error) | 0.215 |

The ceiling is +0.015 and nobody reaches it. A 24-well round reveals a median of
9 positives; the naive rule picks the truly best metric 27.8% of the time, and
the guarded rule — which was meant to be the safe version — made 27 switches of
which **zero** chose the best metric, ending up slightly worse than never
switching at all. Requiring a large observed gap selects exactly the estimates
that overfit hardest.

`caliper/multiround.py` is kept as the record and is deliberately not wired into
anything. The practical advice it produced: rank by the metric that was best
across your other targets, and do not let one round of 24 wells talk you out of
it.

**On diversity.** A 24-well plate could in principle hold one design tested 24
times. Measured here it does not: all ten targets give 24 distinct sequences,
mean pairwise identity 0.114, one pair in ~2,760 above 90%. But Overath's
designs were pooled from many separate campaigns and are diverse by
construction, so **this dataset cannot test the failure mode** — a single
RFdiffusion run emitting thousands of backbones would look very different. The
gap is recorded rather than filled blind.

## What is in here, and what is only history

```
caliper/                what a campaign should actually use
  audit.py              every surviving check, one entry point
  smallsample.py        calibration that refuses when the data is too thin
  hierarchical.py       per-target calibration with partial pooling
  whentocascade.py      whether a cheap stage earns its place
  stats.py metrics.py   paired tests, intervals, out-of-fold calibration
  benchmarks.py         published numbers this measures itself against
  multiround.py         a measured negative result, wired to nothing

caliper/harness/        how the above was arrived at. Not the contribution.
  backends/simulator.py the simulator that proved itself wrong
  pipeline.py store.py  the campaign runner built around it
  baselines.py          the schedulers raced before real data existed
```

The harness is kept rather than deleted for one reason: **the simulator is what
proved itself wrong.** It modelled stage errors as independent, declared the
cascade a winner, and was contradicted by real data. Chasing that contradiction
produced the correlation measurement, the AUC-gap rule, and eventually the
detector at the top of this page. A record of a method flattering its author is
worth keeping somewhere it can be re-run.

`CRITIQUE.md` is an adversarial review of this repository: 68 defects with
severities, 31 fixed, 37 accepted with stated reasons, plus a second pass that
found four more. Read it before trusting anything here. `NEXT.md` is what
remains.

---

## License

MIT. See `LICENSE`.
