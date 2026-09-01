# DESIGN — why CALIPER is built this way

Every non-obvious choice, with the evidence behind it. If a number appears in
the code, its source should appear here.

---

## 1. What problem this is, precisely

Not "design better binders". The generative models are not the bottleneck.
The bottleneck is **deciding what to do next with a fixed budget**, and the
literature says so with numbers:

| Finding | Source |
|---|---|
| Confidence metrics classify experimental success at ROC AUC **0.64–0.77** | Overath et al. 2025 (n=3,766); Adaptyv EGFR 2025 (n=400) |
| Precision at a fixed threshold ranges **0.1 to 1.0 across targets** | Overath et al. 2025 |
| "There are **no standard criteria** to prioritise binders for experimental testing" | Overath et al. 2025 |
| Median co-folding confidence **does not separate** productive from unproductive campaigns | Golden 2026 (n=1,320) |
| For peptide binders, pLDDT/PAE have **no meaningful correlation** with Kd | Li, Vlachos & Bryant 2024 |

A threshold on a number with AUC 0.66 whose meaning changes per target is a
weak instrument. CALIPER's thesis is that the *allocation policy* and the
*score-to-probability map* are the parts worth engineering.

---

## 2. Three borrowed ideas, and why each transfers

### 2.1 Successive halving — from AutoML

Hyperband keeps the top 1/η and multiplies the budget, repeating. η=3 is the
convention (Li et al., JMLR 2018: "default η = 3 … works well for most
deep-learning settings").

**Why it transfers:** binder pipelines already have fidelity tiers (sequence
score → single-sequence fold → MSA fold → assay) with costs spanning three
orders of magnitude. That is exactly the structure successive halving assumes.

**Why it might not, and what we did about it:** η=3 has **never been validated
in a wet-lab setting** — confirmed as a gap by two independent literature
searches. Wet-lab "budget" is lumpy (96-well plates) and its noise structure is
nothing like training epochs. So `reduction` is a config parameter, not a
constant, and `budget_to_start` inverts the ladder from an actual budget rather
than assuming one.

### 2.2 Keep the losers' scores — from multi-fidelity HTS

The MF-PCBA benchmark work makes the complaint directly: in high-throughput
screening "the millions of measurements performed as part of primary screening
are typically discarded after the initial filtering step … the different tiers
or fidelities generated during HTS are not jointly leveraged".

**In CALIPER:** `Candidate.with_score` is append-only and `killed()` does not
delete anything. A candidate that dies at stage 1 keeps its stage-1 score, and
those scores are the low end of the calibration curve.

### 2.3 Exploration quota — from credit scoring, not from biology

This is the one that is **not standard in protein design**. Two independent
literature searches found the concept mature elsewhere and absent here:

| Field | Work |
|---|---|
| Credit scoring | Scarone & Baeza-Yates 2026 — "controlled exploration", **2–5% of rejected applicants is enough to diagnose the feedback loop at near-zero cost** |
| Algorithmic trading | Kamat 2026 — post-rejection follow-up sampling |
| Causal inference | Selective labels (Lakkaraju 2017; Kleinberg 2018); IPS (Rosenbaum & Rubin 1983); CRM (Swaminathan & Joachims 2015) |
| ML-guided design (theory) | Fannjiang & Park 2025; "selection tax" 2026 |
| **Protein binder design** | **nothing found** |

**The mechanism:** if you only assay winners, your labels are truncated at the
top of the score range. A calibration curve fitted there cannot distinguish
"0.4 means 3%" from "0.4 means 30%", which is why published cut-offs do not
transfer between labs — everyone fits on their own winners.

**Measured effect** (12 seeds, out-of-fold ECE, lower better):

| exploration | OOF ECE |
|---|---|
| 0% | 0.236 [0.183, 0.282] |
| 5% | 0.193 [0.150, 0.232] |
| 25% | 0.148 [0.108, 0.194] |

Note the divergence from the source field: 2–5% suffices to *detect* bias, but
*fitting a usable curve* keeps improving past 25%. The credit-scoring number
was transplanted and then checked, not assumed.

---

## 3. Design decisions and their reasons

### 3.1 Calibration is monotone (isotonic), not parametric

Platt scaling assumes a sigmoid. Nothing justifies that shape here. The only
property worth assuming is **monotonicity** — a higher confidence score should
not mean a lower success probability — so isotonic regression via PAVA is the
minimal assumption. Implemented directly (~25 lines) rather than pulling in
scikit-learn, which is otherwise unused.

### 3.2 Shrinkage toward the base rate

With few labels an isotonic fit is a step function through noise. The curve is
blended toward the base rate with weight `n / (n + prior_strength)`. With 3
labels a perfect-looking design gets 0.74, not 1.0.

### 3.3 Every calibration number is out-of-fold

**The single most important decision in the repo.** Isotonic regression fitted
and evaluated on the same labels drives ECE to ~0 by construction. v0.1 reported
0.610 → 0.000 and it was meaningless. The honest figure is 0.610 → 0.293.

`test_in_sample_calibration_is_optimistic` fails if this ever regresses.

### 3.4 Non-affinity gates are modelled separately

Folding expression and solubility failures into "probability of binding"
attributes to affinity a failure that had nothing to do with it.

| Gate | Value | Source |
|---|---|---|
| Expression rate | 0.73 | Adaptyv EGFR round 1, 146/201 |
| ipTM predicts expression at | AUC 0.58 | Adaptyv EGFR — essentially chance |
| Share of monomer failures from insolubility/aggregation | ~65% | Garcia & Dixit 2026 (n=614) |

**Why it mattered in practice:** before these gates the pipeline reported a
**100% hit rate**, which is impossible. `test_gates_make_perfect_hit_rates_impossible`
pins the fix.

### 3.5 The simulator is pinned to published values, not hand-tuned

Two solvers, because hand-picked constants make a harness that flatters
whatever it tests:

- `SimAssay.for_base_rate` solves the assay threshold so the base rate matches
  a published one (default 11.6% = 436/3,766, Overath et al.).
- `noise_for_auc` solves each stage's noise so its ROC AUC matches a published
  value (0.62/0.68/0.75).

The first version used hand-picked noise of 0.30/0.15/0.06. The solver says the
literature-consistent values are **0.74/0.47/0.39** — the original harness was
2–6× too generous, and every downstream number was inflated.

Realised AUC is measured every run and printed next to the requested value.

### 3.6 `true_affinity` is a product, not a sum

Binding needs several things right at once. A product of [0,1] factors
reproduces the heavy right skew of real landscapes: most designs bad, good ones
rare, no compensating for a fatal flaw. An additive score would rank a
mediocre-everywhere design alongside a nearly-perfect one.

### 3.7 Content-addressed cache with atomic writes

From build systems. Key = hash(stage, backend, version, params, sequence).
Writes go through `os.replace` so a crash cannot leave a half-written entry, and
a corrupt entry is dropped and recomputed rather than raising.

**Known limit:** this assumes backend determinism, which real tools do not have.
Published evidence: 31/570 designs shift pLDDT by >5 points across recycle
settings (Garcia & Dixit), and verifier choice systematically changes which
designs pass (ProtDBench 2026). Caching a stochastic score as a pure function is
wrong, and is recorded as an accepted defect (CRITIQUE D3/D4) rather than
silently ignored.

### 3.8 Content-hash seeding, never `hash()`

Python's `hash()` for strings is salted per process. v0.1 seeded the exploration
sample with it, so **the sample was not reproducible across runs** — in a project
whose selling point is reproducibility. Now seeded from `stable_hash`, pinned by
`test_campaign_runs_and_is_reproducible`.

### 3.9 Missing real backends fail loudly

`external.py` raises `BackendUnavailable` with installation instructions. It
never falls back to the simulator. A report full of real-looking numbers
produced by a silent fallback is the worst failure this project could have.

---

## 4. What is deliberately not here

| Not implemented | Why |
|---|---|
| Real tool adapters | Would need RFdiffusion/ProteinMPNN/Boltz + GPU. The orchestration layer is the contribution and is testable without them. |
| Validation on real data | **The top priority.** Overath (n=3,766, Zenodo) and Adaptyv EGFR (n=601) are public and include failures. |
| Multi-round campaigns | The point of calibrating is to use it next round. Largest functional gap. |
| Sequence diversity control | A 24-well plate can currently hold 24 near-identical designs. |
| Per-target calibration on the default path | `HierarchicalCalibrator` is written and tested but needs more than one target to mean anything. |

---

## 5. Sources

- Overath et al. 2025, *Predicting Experimental Success in De Novo Binder Design*, bioRxiv 10.1101/2025.08.14.670059 — n=3,766, data on Zenodo 10.5281/zenodo.15722219
- Garcia & Dixit 2026, *Evaluating Zero-Shot Prediction of Monomeric Protein Design Success*, Protein Science 10.1002/pro.70453 — n=614
- Adaptyv EGFR competition 2025, bioRxiv 10.1101/2025.04.17.648362 — n=601
- Chow et al. 2025, bioRxiv 10.64898/2025.12.12.694033 — ipSAE ≥ 0.85, 4.35% → 30.0%
- Watson et al. 2023, *RFdiffusion*, Nature 10.1038/s41586-023-06415-8
- Pacesa et al. 2025, *BindCraft*, Nature 10.1038/s41586-025-09429-6
- Bennett et al. 2023, Nature Communications 10.1038/s41467-023-38328-5 — pAE_interaction < 10
- Li, Jamieson et al. 2018, *Hyperband*, JMLR — η = 3
- Scarone & Baeza-Yates 2026, arXiv:2606.18479 — 2–5% exploration
- Rosenbaum & Rubin 1983, *Biometrika* — inverse propensity scoring
- Lakkaraju et al. 2017 — selective labels
- ProtDBench 2026, arXiv:2605.04118 — verifier-dependent bias
