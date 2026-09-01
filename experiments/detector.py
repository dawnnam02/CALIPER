"""The strongest result here: catching a calibration that points the wrong way.

Two independent datasets, one detector, no wells spent.

Data
----
* **Overath et al. 2025** (Zenodo 15722219, CC-BY-4.0): 3,650 designs, 15
  targets, designs pooled from many published campaigns, AF3 ipSAE_min.
* **Adaptyv EGFR competition round 2** (ODbL, github.com/adaptyvbio): 380
  labelled designs, one target, crowdsourced from many teams, assayed in one
  lab, with ipTM / pLDDT / pAE-interaction.

These differ in target, in who generated the designs, and in who ran the assay.
That independence is the point: the first version of this project rested
entirely on one dataset, which was its weakest feature.

Sanity check before anything else: on the Adaptyv data this code measures
ipTM AUC 0.636 and pLDDT AUC 0.656, against 0.64 and 0.66 as published. The
reader can trust that the file is being parsed the way its authors intended.

How the evidence is counted, and why it was recounted
-----------------------------------------------------
An earlier version of this file reported "41 cells" and quoted a sensitivity
taken from them. That number was inflated. Each SITUATION -- one (dataset,
target, metric) -- is tested at four budgets, so the same target was counted up
to four times. Measuring one person's height four times does not give you four
people.

There are 13 independent situations, not 41 independent observations, and only
6 of them contain a catastrophe. The point estimates barely move under the
correction; the intervals widen a lot. That widening is the honest cost.

Three views are printed, most trustworthy first:

  by situation   one row per (dataset, target, metric). No nesting. PRIMARY.
  by budget      within a single budget the situations are independent too,
                 so this shows whether the detector holds at each plate size.
  by cell        every (situation, budget) pair. Budgets are nested inside
                 situations here, so this is descriptive only.

What is being detected
----------------------
A campaign assays its top-N designs and fits a calibration curve. If the
labelled scores sit in a narrow band, the fitted slope can invert -- the curve
then claims a higher confidence score means a lower chance of binding, and every
prediction it makes about the unassayed pool is backwards.

The check is one line of arithmetic on data you already have. No extra wells.

Korean note:
독립적인 데이터셋 둘로 검증한다. 표적도 다르고, 설계를 만든 사람도 다르고, 실험한
곳도 다르다. 이전 판이 데이터셋 하나에만 기대고 있던 게 가장 약한 지점이었다.
감지하는 것은 "라벨 표본이 좁은 띠만 덮어 교정 곡선의 기울기가 뒤집힌 상태"다.
이미 가진 데이터로 계산만 하면 되고, 웰을 더 쓰지 않는다.

세는 단위를 정정했다. 예전에는 "셀 41개"로 셌지만 예산 4개가 표적 하나 안에
중첩돼 있었다. 같은 표적을 네 번 잰 것이라 표본 수가 부풀려진 것이다. 실제
독립 단위는 상황 13개이고 그중 파국은 6개뿐이다. 점추정은 거의 그대로지만
신뢰구간은 훨씬 넓어진다.

Run:  python experiments/detector.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.metrics import roc_auc
from caliper.smallsample import PlattCalibrator, check_calibration
from caliper.stats import bootstrap_ci, equal_mass_ece, wilson

ROOT = Path(__file__).resolve().parents[1]
OVERATH = ROOT / "data" / "overath" / "final_dataset.csv"
ADAPTYV = ROOT / "data" / "adaptyv" / "round2.csv"

BUDGETS = (12, 24, 48, 96)
CATASTROPHE = 0.20          # out-of-sample ECE above which a curve is useless


def cells(scores: np.ndarray, y: np.ndarray, label: str) -> list[dict]:
    """One row per (situation, budget): did the detector fire, and was it right?"""
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    out = []
    for N in BUDGETS:
        if n < N * 2:
            continue
        idx = order[:N]
        if len(set(y[idx].tolist())) < 2:
            continue
        cal = PlattCalibrator().fit(scores[idx], y[idx])
        health = check_calibration(cal, scores[idx], scores)
        test = np.array([i for i in range(n) if i not in set(idx.tolist())])
        ece = equal_mass_ece(cal.predict(scores[test]), y[test])
        out.append({
            "situation": label, "N": N,
            "fired": not health.ok,
            "slope": health.slope,
            "span": health.span_fraction,
            "ece": ece,
            "catastrophe": ece > CATASTROPHE,
        })
    return out


def load_overath() -> list[dict]:
    if not OVERATH.exists():
        return []
    df = pd.read_csv(OVERATH, low_memory=False)
    df = df[df.binder.notna()].copy()
    df["y"] = df.binder.astype(bool).astype(int)
    df["af3_ipSAE_min"] = pd.to_numeric(df["af3_ipSAE_min"], errors="coerce")
    df = df.dropna(subset=["af3_ipSAE_min"])
    rows = []
    for t, g in df.groupby("target_id"):
        if len(g) < 60 or g.y.sum() < 5:
            continue
        rows += cells(g["af3_ipSAE_min"].to_numpy(float), g.y.to_numpy(),
                      f"Overath/{t}")
    return rows


def load_adaptyv() -> tuple[list[dict], dict]:
    if not ADAPTYV.exists():
        return [], {}
    df = pd.read_csv(ADAPTYV)
    df = df[df.binding.astype(str).str.lower() != "unknown"].copy()
    df["y"] = (df.binding.astype(str).str.lower() == "true").astype(int)
    rows, aucs = [], {}
    for col, lower_better in (("iptm", False), ("plddt", False),
                              ("pae_interaction", True)):
        v = pd.to_numeric(df[col], errors="coerce")
        m = v.notna().to_numpy()
        s = (-v[m] if lower_better else v[m]).to_numpy(float)
        y = df.y.to_numpy()[m]
        aucs[col] = roc_auc(s, y)
        rows += cells(s, y, f"Adaptyv/{col}")
    return rows, aucs


def confusion(fired, catastrophe) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN -- fired-and-bad, fired-and-fine, missed, rightly silent."""
    f = np.asarray(list(fired), dtype=bool)
    c = np.asarray(list(catastrophe), dtype=bool)
    return (int((f & c).sum()), int((f & ~c).sum()),
            int((~f & c).sum()), int((~f & ~c).sum()))


def by_situation(D: pd.DataFrame) -> pd.DataFrame:
    """Collapse the budgets inside each situation, so every row is independent.

    A situation counts as a catastrophe if ANY budget blew up, and as a firing
    if the detector spoke up at ANY budget. That pairing is deliberate: it is
    the rule a real campaign would follow. You run the check at whatever plate
    size you have, and one warning is a warning.
    """
    return (D.groupby("situation")
             .agg(fired=("fired", "any"),
                  catastrophe=("catastrophe", "any"),
                  worst_ece=("ece", "max"),
                  budgets=("N", "count"))
             .reset_index())


def report(tp: int, fp: int, fn: int, tn: int, indent: str = "  ") -> None:
    print(f"{indent}{'':<22}{'catastrophe':>14}{'fine':>10}")
    print(f"{indent}{'detector fired':<22}{tp:>14}{fp:>10}")
    print(f"{indent}{'detector silent':<22}{fn:>14}{tn:>10}")
    print()
    if tp + fn:
        print(f"{indent}sensitivity  {wilson(tp, tp + fn)}")
    if tp + fp:
        print(f"{indent}precision    {wilson(tp, tp + fp)}")
    if tn + fp:
        print(f"{indent}specificity  {wilson(tn, tn + fp)}")


def main() -> int:
    over = load_overath()
    adap, adap_aucs = load_adaptyv()
    if not over and not adap:
        print("no data found. Run: python scripts/get_data.py", file=sys.stderr)
        return 2
    D = pd.DataFrame(over + adap)
    S = by_situation(D)

    loaded = [n for n, r in (("Overath", over), ("Adaptyv", adap)) if r]
    print("=" * 78)
    print("Inverted-calibration detector | "
          + (" + ".join(loaded) if len(loaded) > 1
             else f"{loaded[0]} only" if loaded else "no data"))
    print("=" * 78)
    print(f"  {len(S)} independent situations (dataset x target x metric), "
          f"tested at up to {len(BUDGETS)} budgets -> {len(D)} cells")
    if len(loaded) < 2:
        print("  NOTE: the README numbers come from BOTH datasets. "
              "Run scripts/get_data.py to fetch the other one.")
    if adap_aucs:
        print("\n  Sanity check against the Adaptyv paper's own numbers:")
        for k, v in adap_aucs.items():
            pub = {"iptm": 0.64, "plddt": 0.66}.get(k)
            note = f"   published {pub:.2f}" if pub else ""
            print(f"    {k:<18}{v:.3f}{note}")

    # -- PRIMARY: one row per situation. Nothing is nested inside a row. ----
    print()
    print(f"BY SITUATION  ({len(S)} independent units)  <-- the number to quote")
    print(f"  a catastrophe is out-of-sample ECE > {CATASTROPHE} at any budget;")
    print("  a firing is the detector speaking up at any budget")
    print()
    report(*confusion(S.fired, S.catastrophe))

    # -- also unnested, and shows whether plate size matters ---------------
    print()
    print("BY BUDGET  (within one budget the situations are independent too)")
    print(f"  {'wells':<10}{'situations':>12}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'TN':>5}   sensitivity")
    print("  " + "-" * 72)
    for N, g in D.groupby("N"):
        tp, fp, fn, tn = confusion(g.fired, g.catastrophe)
        sens = (str(wilson(tp, tp + fn)) if tp + fn
                else "no catastrophes")
        print(f"  N={N:<8}{len(g):>12}{tp:>5}{fp:>5}{fn:>5}{tn:>5}   {sens}")

    # -- descriptive only ---------------------------------------------------
    tp, fp, fn, tn = confusion(D.fired, D.catastrophe)
    print()
    print(f"BY CELL  ({len(D)} rows)  -- descriptive only: budgets are NESTED")
    print("  inside situations, so these intervals are too narrow to quote. An")
    print("  earlier version of this file used them as the headline. They are")
    print("  kept here so that the correction stays visible.")
    print()
    report(tp, fp, fn, tn, indent="    ")

    print()
    print("Out-of-sample ECE, split by what the detector said (all cells)")
    print(f"  fired   {bootstrap_ci(D[D.fired].ece, seed=1)}")
    print(f"  silent  {bootstrap_ci(D[~D.fired].ece, seed=1)}")
    print(f"  worst curve the detector let through: {D[~D.fired].ece.max():.3f}")
    print(f"  worst curve it caught:                {D[D.fired].ece.max():.3f}")

    print()
    print("Every cell (a star marks the detector firing)")
    print(f"  {'situation':<26}{'N':>5}{'fired':>8}{'slope':>9}"
          f"{'span':>8}{'ECE':>9}")
    print("  " + "-" * 65)
    for _, r in D.sort_values(["situation", "N"]).iterrows():
        print(f"  {r.situation[:26]:<26}{r.N:>5}{'*' if r.fired else '-':>8}"
              f"{r.slope:>9.2f}{100 * r.span:>7.0f}%{r.ece:>9.3f}")

    print()
    print("Reading it")
    print(f"  {len(S)} situations is a thin evidence base and the intervals say")
    print("  so. One claim survives all three views: the curves the detector")
    print("  let through were an order of magnitude better calibrated than the")
    print("  ones it caught.")
    print()
    print("  The clean sheet on false positives does NOT survive. At situation")
    print("  level it never fired on a healthy target; by cell it did so twice,")
    print("  both at the larger budgets, where a wider score span makes a flat")
    print("  fit look inverted. Quote the situation-level number, but know that")
    print("  it is the kinder of the two.")
    print()
    print("  Narrowing those intervals needs more TARGETS. More metrics on the")
    print("  same designs would add cells without adding situations, which is")
    print("  the very mistake this file was written to correct.")
    print()
    print("  This costs nothing to run. Published pipelines do not check.")

    out = ROOT / "experiments" / "detector_results.json"
    out.write_text(D.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\nraw results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
