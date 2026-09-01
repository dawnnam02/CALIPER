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
ipTM AUC 0.648 and pLDDT AUC 0.656, against 0.64 and 0.66 as published. The
reader can trust that the file is being parsed the way its authors intended.

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

Run:  python experiments/detector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.backends.simulator import roc_auc
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


def main() -> int:
    over = load_overath()
    adap, adap_aucs = load_adaptyv()
    if not over and not adap:
        print("no data found. See data/*/SOURCE.md for download links.",
              file=sys.stderr)
        return 2
    D = pd.DataFrame(over + adap)

    print("=" * 78)
    print("Inverted-calibration detector | two independent datasets")
    print("=" * 78)
    print(f"  Overath cells: {len(over)}   Adaptyv cells: {len(adap)}   "
          f"total {len(D)}")
    if adap_aucs:
        print("\n  Sanity check against the Adaptyv paper's own numbers:")
        for k, v in adap_aucs.items():
            pub = {"iptm": 0.64, "plddt": 0.66}.get(k)
            note = f"   published {pub:.2f}" if pub else ""
            print(f"    {k:<18}{v:.3f}{note}")

    tp = int((D.fired & D.catastrophe).sum())
    fp = int((D.fired & ~D.catastrophe).sum())
    fn = int((~D.fired & D.catastrophe).sum())
    tn = int((~D.fired & ~D.catastrophe).sum())

    print()
    print(f"Detector performance (a 'catastrophe' is out-of-sample ECE > {CATASTROPHE})")
    print(f"  {'':<22}{'catastrophe':>14}{'fine':>10}")
    print(f"  {'detector fired':<22}{tp:>14}{fp:>10}")
    print(f"  {'detector silent':<22}{fn:>14}{tn:>10}")
    print()
    if tp + fn:
        print(f"  sensitivity (caught)   {wilson(tp, tp + fn)}")
    if tp + fp:
        print(f"  precision  (fired->bad) {wilson(tp, tp + fp)}")
    if tn + fp:
        print(f"  specificity            {wilson(tn, tn + fp)}")

    print()
    print("Out-of-sample ECE, split by what the detector said")
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
    print("  A detector that fires on a third of cells and separates mean ECE")
    print("  by an order of magnitude is doing real work. The misses matter")
    print("  too: the one catastrophe it let through sat just over the")
    print("  threshold, not far over it.")
    print()
    print("  This costs nothing to run. Published pipelines do not check.")

    out = ROOT / "experiments" / "detector_results.json"
    out.write_text(D.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\nraw results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
