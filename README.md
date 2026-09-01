# CALIPER

**Cal**ibrated **i**terative **p**rotein **e**ngineering **r**uns — a scheduling
and calibration layer for de novo protein binder design.

CALIPER does not predict structures. It decides **which designs are worth the
next unit of compute, and which are worth a well on the plate** — and it turns
raw confidence scores into probabilities you can actually budget against.

---

## The problem it addresses

A 2025 meta-analysis of 3,766 experimentally characterised de novo binders
found that structure-prediction confidence metrics are only weak-to-moderate
classifiers of experimental success — ROC AUC roughly 0.64–0.77 — and, more
awkwardly, that **precision at any fixed threshold ranges from 0.1 to 1.0
depending on the target**.

So the standard practice — filter on `ipTM > 0.8`, send the survivors to the
bench — has two failure modes:

1. **It is budget-blind.** On a hard target nothing clears the bar; on an easy
   one ten thousand designs do.
2. **The number is not a probability.** `ipTM = 0.8` does not mean 80%, does
   not mean the same thing on two targets, and does not transfer between labs.

CALIPER replaces the fixed threshold with a compute-budget ladder, and replaces
the raw score with a calibrated probability fitted on the lab's own outcomes.

---

## Three ideas, borrowed on purpose

| Idea | Taken from | What it does here |
|---|---|---|
| **Successive halving** | AutoML (Hyperband/ASHA), HTS screening cascades | Spend the budget by rank, not by threshold |
| **Keep the losers' data** | Multi-fidelity HTS benchmarks, which complain that primary-screen measurements get discarded | Killed candidates keep their scores; they are the negative labels calibration needs |
| **Exploration quota** | Reject inference in credit scoring; post-rejection sampling in trading; the selective-labels literature | Spend a few wells on designs the filter rejected, so the calibration is not fitted only on winners |

The third is the one that is **not standard in protein design** — the concept
is mature in credit scoring, recommender systems, and causal inference, but a
literature search found no binder-design pipeline that does it.

---

## Measured results

12 seeds, 3,000 designs per campaign, shortlist of 24, base rate 11.6%
(matching the meta-analysis). Every scheduler sees the **same** score matrix,
so differences are attributable to policy alone.

| scheduler | hit rate | true top-24 recall | kept | cost |
|---|---|---|---|---|
| oracle *(ceiling)* | 0.413 [0.358, 0.465] | 1.000 | 24 | 0 |
| **successive halving** | **0.406 [0.361, 0.455]** | **0.104 [0.083, 0.128]** | 24 | **41,480** |
| fixed threshold, top-24 | 0.413 [0.365, 0.465] | 0.069 [0.042, 0.101] | 24 | 45,575 |
| full sweep | 0.375 [0.347, 0.403] | 0.059 [0.031, 0.090] | 24 | 241,500 |
| random *(floor)* | 0.153 [0.111, 0.194] | 0.003 [0.000, 0.010] | 24 | 0 |

**Read it honestly:**

- Against the fixed-threshold policy at equal shortlist size, CALIPER gets the
  **same hit rate (CIs overlap almost entirely) for 9% less compute, with 51%
  better recall of the true top-24**. That is a modest win, not a large one.
- Against scoring everything at every stage, it is better on both axes at
  **1/6 the cost**.
- **Every** scheduler has poor recall (0.06–0.10). That is not a scheduling
  failure: with stage AUCs of 0.62/0.68/0.75, no ranking policy can find the
  true top-24. Only the oracle reaches 1.0. **The bottleneck is metric quality,
  not allocation.**

### Does the exploration quota actually help?

Out-of-fold calibration error, same 12 seeds:

| exploration fraction | out-of-fold ECE |
|---|---|
| 0% | 0.236 [0.183, 0.282] |
| 5% | 0.193 [0.150, 0.232] |
| 25% | 0.148 [0.108, 0.194] |

Monotone, and the 0% and 25% intervals barely overlap. Spending wells on
rejected designs measurably improves calibration.

Note the divergence from the credit-scoring literature: 2–5% is enough to
*diagnose* a feedback loop, but *fitting a usable calibration curve* keeps
improving well past that.

---

## The most important number in this repo

| | ECE |
|---|---|
| raw score used as a probability | 0.610 |
| after calibration, **in-sample** | **0.000** |
| after calibration, **cross-validated** | **0.293** |

Isotonic regression can drive in-sample calibration error to zero by
construction. The first version of this project reported the 0.000. It was
meaningless. Calibration does help — 0.610 → 0.293 is a 52% reduction — but it
is four times worse than the in-sample number suggested.

Every calibration number CALIPER reports is now out-of-fold.

---

## Install and run

```bash
git clone <this repo> && cd CALIPER
pip install -e .
python run.py                      # one campaign, writes runs/<timestamp>/
python experiments/validate.py     # the scheduler comparison above
pytest                             # tests
```

Requires Python 3.11+, numpy, scipy, PyYAML. **No GPU and no model weights are
needed**, because the default backend is a simulator.

Edit `config.yaml`; you should not need to touch anything else.

---

## About that simulator

The default backend is **not a protein model and does not pretend to be one.**
It is a test harness with known ground truth, in the sense that a CPU scheduler
is validated on synthetic workloads before it meets a real one.

It is pinned to the literature in two ways:

- the assay threshold is **solved** so the base rate matches a published one;
- each stage's noise is **solved** so its ROC AUC matches a published value
  (0.62/0.68/0.75, checked against the realised AUC every run).

That makes claims about *scheduling* checkable. It makes claims about
*chemistry* impossible, and CALIPER does not make any.

**Real backends are not implemented.** `caliper/backends/external.py` raises
with installation instructions rather than silently substituting the simulator —
a report full of real-looking numbers from a fallback backend would be the worst
failure this project could have.

---

## Honest status

| | |
|---|---|
| Scheduling layer | works, validated against 4 baselines over 12 seeds |
| Calibration layer | works, reported out-of-fold |
| Exploration quota | works, ablated |
| Provenance and caching | works (content-addressed, atomic writes, run manifests) |
| **Real tool adapters** | **not implemented** |
| **Validation on real data** | **not done** — see below |
| Per-target hierarchical calibration | written, wired, under-tested |
| Multi-round campaigns | not implemented |
| Sequence diversity control | not implemented |

The single most valuable next step is to refit the calibrator on the public
[Overath et al. dataset](https://zenodo.org/doi/10.5281/zenodo.15722219)
(n=3,766, includes failures) and the Adaptyv EGFR competition data (n=601, with
Kd values), replacing the simulator for the calibration claims entirely.

`CRITIQUE.md` is an adversarial review of this repo listing 68 defects with
severities and fix status. Read it before trusting anything above.

---

## License

MIT. See `LICENSE`.
