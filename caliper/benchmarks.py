"""Published numbers CALIPER measures itself against.

Every entry is a real reported result with its sample size and source, so a
run report can say "your hit rate is 30% against a published range of 9-88%"
instead of leaving the user to guess whether their number is good.

This file is deliberately data, not code.  When a number here is wrong or
superseded, it should be edited here and nowhere else.

Korean note:
내 파이프라인 숫자가 좋은지 나쁜지는 혼자서는 알 수 없다.  그래서 발표된 수치를
여기 모아두고 리포트에서 자동으로 비교한다.  숫자가 바뀌면 여기만 고치면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Benchmark:
    key: str
    label: str
    value: float | tuple[float, float]
    n: int | None
    source: str
    note: str = ""

    def contains(self, x: float) -> bool:
        if isinstance(self.value, tuple):
            return self.value[0] <= x <= self.value[1]
        return False


# --- experimental hit rates (fraction of tested designs that bind) ---------
HIT_RATES = [
    Benchmark("meta3766", "de novo binder meta-analysis", 0.116, 3766,
              "Overath et al. 2025, bioRxiv 10.1101/2025.08.14.670059",
              "436 binders / 3,766 designs across 15 targets"),
    Benchmark("rfdiffusion", "RFdiffusion campaign", 0.19, 475,
              "Watson et al. 2023, Nature 10.1038/s41586-023-06415-8",
              "95 designs x 5 targets, BLI at 10 uM"),
    Benchmark("bindcraft", "BindCraft", (0.10, 1.00), 212,
              "Pacesa et al. 2025, Nature 10.1038/s41586-025-09429-6",
              "per-target range; ~30.7% overall from supplementary CSV"),
    Benchmark("alphaproteo", "AlphaProteo", (0.09, 0.88), None,
              "Zambaldi et al. 2024, arXiv:2409.08022",
              "7 targets; failed entirely on TNFa"),
    Benchmark("monomers614", "de novo monomers", 0.43, 614,
              "Garcia & Dixit 2026, Protein Science 10.1002/pro.70453",
              "monomer folding success, not binding"),
    Benchmark("golden1320", "pre-synthesis predictor study", 0.27, 1320,
              "Golden 2026, OSF (preprint)",
              "354 binders / 1,320 designs across 15 targets"),
]

# --- how well confidence metrics classify (ROC AUC) -----------------------
DISCRIMINATION = [
    Benchmark("egfr_iptm", "ipTM, EGFR competition", 0.64, 400,
              "Adaptyv EGFR competition 2025, 10.1101/2025.04.17.648362"),
    Benchmark("egfr_plddt", "pLDDT, EGFR competition", 0.66, 400,
              "Adaptyv EGFR competition 2025"),
    Benchmark("monomer_esm", "ESMFold pLDDT, monomers", 0.72, 614,
              "Garcia & Dixit 2026", "best single metric; range 0.60-0.72"),
    Benchmark("vhh_iptm", "AF3 ipTM, VHH antibodies", 0.86, None,
              "Bennett et al. 2025, Nature 10.1038/s41586-025-09721-5",
              "much higher than binder/monomer AUCs -- design class matters"),
    Benchmark("peptide", "pLDDT/PAE vs Kd, peptides", 0.50, 40,
              "Li, Vlachos & Bryant 2024, 10.1101/2024.06.20.599739",
              "NO meaningful correlation reported -- metrics uninformative here"),
]

# --- true positives lost to filtering -------------------------------------
RECALL_LOSS = [
    Benchmark("monomer_top50", "top-50% pLDDT filter", 0.43, 614,
              "Garcia & Dixit 2026",
              "success 39% -> 57% in-filter, implying ~43% of true "
              "positives fall in the discarded half"),
    Benchmark("cd22_ipsae", "ipSAE >= 0.85 on CD22", 0.00, 95,
              "Chow et al. 2025, 10.64898/2025.12.12.694033",
              "4.35% (3/69) -> 30.0% (3/10); all 3 known binders retained"),
]

# --- exploration quota needed to detect selection bias --------------------
EXPLORATION = [
    Benchmark("credit", "controlled exploration, credit scoring", (0.02, 0.05),
              None, "Scarone & Baeza-Yates 2026, arXiv:2606.18479",
              "2-5% deliberately-approved rejects suffices to diagnose the "
              "feedback loop at near-zero cost"),
]

# --- non-binding attrition -------------------------------------------------
ATTRITION = [
    Benchmark("expression", "designs that express", 0.73, 201,
              "Adaptyv EGFR round 1", "146/201; ipTM/pLDDT do NOT predict this "
              "(AUC 0.58/0.55)"),
    Benchmark("solubility_failshare", "share of monomer failures from "
              "insolubility/aggregation", 0.65, 614,
              "Garcia & Dixit 2026"),
    Benchmark("cart_lift", "biology-informed filter stack, CAR-T CD20",
              (0.138, 0.386), 11984,
              "Bozkurt 2026, 10.64898/2026.04.13.718094",
              "2.8x enrichment lift; aggregation propensity most robust signal"),
]

# --- reproducibility -------------------------------------------------------
NONDETERMINISM = [
    Benchmark("recycle_shift", "designs shifting pLDDT > 5 pts across recycle "
              "settings", 31 / 570, 570, "Garcia & Dixit 2026",
              "3 designs shifted >15 pts -- enough to flip a pass/fail at 80"),
]


def compare(value: float, pool: list[Benchmark]) -> str:
    """One-line verdict placing ``value`` against a set of published numbers."""
    if value != value:
        return "no comparison (value is NaN)"
    lows, highs = [], []
    for b in pool:
        v = b.value
        lo, hi = (v if isinstance(v, tuple) else (v, v))
        lows.append(lo)
        highs.append(hi)
    lo, hi = min(lows), max(highs)
    if value < lo:
        return f"{value:.3f} is BELOW the published range {lo:.3f}-{hi:.3f}"
    if value > hi:
        return f"{value:.3f} is ABOVE the published range {lo:.3f}-{hi:.3f}"
    return f"{value:.3f} sits inside the published range {lo:.3f}-{hi:.3f}"


def table(pool: list[Benchmark]) -> str:
    w = max(len(b.label) for b in pool) + 2
    lines = []
    for b in pool:
        v = (f"{b.value[0]:.3f}-{b.value[1]:.3f}" if isinstance(b.value, tuple)
             else f"{b.value:.3f}")
        n = f"n={b.n:,}" if b.n else "n=?"
        lines.append(f"  {b.label:<{w}}{v:>14}  {n:>10}   {b.source}")
    return "\n".join(lines)
