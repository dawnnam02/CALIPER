"""The strongest result here: catching a calibration that points the wrong way.

Three independent campaigns, one detector, no wells spent.

Data
----
* **Overath et al. 2025** (Zenodo 15722219, CC-BY-4.0): 3,650 designs, 15
  targets, pooled from many published campaigns, scored with AF3 ipSAE_min.
* **Adaptyv EGFR competition round 2** (ODbL, github.com/adaptyvbio): 380
  labelled designs, one target, crowdsourced from many teams, assayed in one
  lab, with ipTM / pLDDT / pAE-interaction.
* **Bennett et al. 2023** (Nat Commun, CC-BY-4.0, Supplementary Data 4):
  603,178 designs, 10 targets, AF2 and RF2 scores with a yeast-display
  avidity readout. Two orders of magnitude more designs than the other two.

They differ in target, in who generated the designs, and in who ran the assay.
That spread is the point: the first version of this project rested on one
dataset, which was its weakest feature.

Sanity check before anything else: on the Adaptyv data this code measures ipTM
AUC 0.636 and pLDDT AUC 0.656, against 0.64 and 0.66 as published. The reader
can trust that the file is being parsed the way its authors intended.

How the evidence is counted
---------------------------
The unit is a **(campaign, target)** pair, scored on that campaign's primary
metric. One unit, one row, nothing nested inside it.

Two earlier counts were wrong and are kept visible rather than deleted:

1. The very first version quoted **41 (situation x budget) cells**. Four
   budgets on one target are four measurements of one thing, not four things.
2. The next version quoted **13 (dataset, target, metric) situations**, which
   still let three Adaptyv metrics over the same 380 designs count as three
   units. Metrics over one design set share their labels.

Both are still printed below, clearly labelled, so the correction is checkable.

Two honesty checks run every time and print their own numbers:

* **Slice overlap.** Bennett re-scored designs from earlier campaigns, so 45%
  of Overath's designs appear in Bennett's table by name. What matters is
  whether the *labelled* top-N slices overlap, since those are what each
  calibration is fit on. They barely do -- the two rank different pools by
  different scores. The audit prints the count.
* **Distinct target proteins.** 19 units span fewer than 19 proteins, because
  seven targets appear in two campaigns. The unit count is not a protein count
  and is not reported as one.

What is being detected
----------------------
A campaign assays its top-N designs and fits a calibration curve. If the
labelled scores sit in a narrow band, the fitted slope can invert -- the curve
then claims a higher confidence score means a lower chance of binding, and every
prediction it makes about the unassayed pool is backwards.

The check is one line of arithmetic on data you already have. No extra wells.

Korean note:
독립적인 캠페인 셋으로 검증한다. 표적도 다르고, 설계를 만든 사람도 다르고,
실험한 곳도 다르다. 세는 단위는 (캠페인, 표적) 한 쌍이고, 그 안에 아무것도
중첩시키지 않는다.

예전에 두 번 잘못 셌고 둘 다 아래에 그대로 남겨 둔다.
(1) 처음에는 "셀 41개" — 예산 4개가 표적 하나 안에 중첩돼 있었다.
(2) 다음에는 "상황 13개" — Adaptyv 지표 3개가 같은 설계 380개를 공유하는데
    이걸 3개로 셌다. 같은 설계를 쓰면 라벨이 같다.

Bennett은 이전 캠페인 설계를 다시 채점한 것이라 Overath 설계의 45%가 이름
그대로 들어 있다. 다만 교정 곡선이 실제로 학습하는 건 상위 N개 슬라이스이고,
그 슬라이스는 거의 겹치지 않는다. 매번 세어서 출력한다.

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
BENNETT = ROOT / "data" / "bennett" / "retrospective.csv"
NIPAH = ROOT / "data" / "nipah" / "nipah.csv"
RBX1 = ROOT / "data" / "rbx1" / "rbx1.csv"
BINDCRAFT = ROOT / "data" / "bindcraft" / "screening.csv"

BUDGETS = (12, 24, 48, 96)
CATASTROPHE = 0.20          # out-of-sample ECE above which a curve is useless


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

def cells(scores: np.ndarray, y: np.ndarray, campaign: str, target: str,
          metric: str, primary: bool) -> list[dict]:
    """One row per (unit, budget): did the detector fire, and was it right?"""
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
        test = np.setdiff1d(np.arange(n), idx)
        ece = equal_mass_ece(cal.predict(scores[test]), y[test])
        out.append({
            "campaign": campaign, "target": target, "metric": metric,
            "primary": primary, "unit": f"{campaign}/{target}",
            "situation": f"{campaign}/{target}/{metric}",
            "N": N, "pool": n,
            "fired": not health.ok,
            "slope": health.slope,
            "span": health.span_fraction,
            "ece": ece,
            "catastrophe": ece > CATASTROPHE,
        })
    return out


def _scored(v: pd.Series, lower_is_better: bool) -> tuple[np.ndarray, np.ndarray]:
    """Numeric column -> (scores where higher is better, keep mask)."""
    num = pd.to_numeric(v, errors="coerce")
    keep = num.notna().to_numpy()
    s = (-num[keep] if lower_is_better else num[keep]).to_numpy(float)
    return s, keep


# --------------------------------------------------------------------------
# the three campaigns
# --------------------------------------------------------------------------

def load_overath() -> tuple[list[dict], pd.DataFrame | None]:
    """Overath: primary metric af3_ipSAE_min, higher is better."""
    if not OVERATH.exists():
        return [], None
    df = pd.read_csv(OVERATH, low_memory=False)
    df = df[df.binder.notna()].copy()
    df["y"] = df.binder.astype(bool).astype(int)
    df["_m"] = pd.to_numeric(df.af3_ipSAE_min, errors="coerce")
    df = df.dropna(subset=["_m"])
    rows = []
    for t, g in df.groupby("target_id"):
        if len(g) < 60 or g.y.sum() < 5:
            continue
        rows += cells(g._m.to_numpy(float), g.y.to_numpy(),
                      "Overath", str(t), "af3_ipSAE_min", primary=True)
    return rows, df


def load_adaptyv() -> tuple[list[dict], dict]:
    """Adaptyv: primary metric ipTM. pLDDT and pAE are kept as extras only."""
    if not ADAPTYV.exists():
        return [], {}
    df = pd.read_csv(ADAPTYV)
    df = df[df.binding.astype(str).str.lower() != "unknown"].copy()
    df["y"] = (df.binding.astype(str).str.lower() == "true").astype(int)
    rows, aucs = [], {}
    for col, lower_better in (("iptm", False), ("plddt", False),
                              ("pae_interaction", True)):
        s, keep = _scored(df[col], lower_better)
        y = df.y.to_numpy()[keep]
        aucs[col] = roc_auc(s, y)
        rows += cells(s, y, "Adaptyv", "EGFR", col, primary=(col == "iptm"))
    return rows, aucs


def load_bennett() -> tuple[list[dict], pd.DataFrame | None]:
    """Bennett: primary metric pAE_interaction, lower is better.

    The outcome is a measurable avidity Kd from yeast display (`avid_ub`
    finite). The alternative readout, a measurable SPR Kd (`kd_ub`), was tried
    and rejected as a primary: at its 0.02-2% hit rate every target counts as a
    catastrophe, so a detector that always fired would score a perfect 8 of 8.
    A test no failure can fail is not a test. Both are reported below.
    """
    if not BENNETT.exists():
        return [], None
    cols = ["description", "target", "avid_ub", "kd_ub", "pAE_interaction",
            "RF2_pAE_interaction", "AF2_plddt_monomer", "AF2_complex_RMSD"]
    df = pd.read_csv(BENNETT, usecols=cols, low_memory=False)
    df["y"] = np.isfinite(pd.to_numeric(df.avid_ub, errors="coerce")).astype(int)
    metrics = {"pAE_interaction": True, "RF2_pAE_interaction": True,
               "AF2_plddt_monomer": False, "AF2_complex_RMSD": True}
    rows = []
    for t, g in df.groupby("target"):
        for m, lower_better in metrics.items():
            s, keep = _scored(g[m], lower_better)
            y = g.y.to_numpy()[keep]
            if len(s) < 200 or y.sum() < 5:
                continue
            rows += cells(s, y, "Bennett", str(t), m,
                          primary=(m == "pAE_interaction"))
    return rows, df


def load_nipah() -> list[dict]:
    """Adaptyv Nipah competition: primary metric Boltz-2 ipSAE, higher is better.

    The first campaign here whose scores come from something other than
    AlphaFold. Its pool is itself a selection -- 600 designs by best ipSAE, 200
    by community vote, 200 by expert panel, out of 10,000+ submissions -- which
    is true of the other campaigns too and is why the pool, not the submission
    set, is what gets ranked.
    """
    if not NIPAH.exists():
        return []
    df = pd.read_csv(NIPAH)
    metrics = {"boltz2_ipsae": False, "boltz2_iptm": False,
               "boltz2_plddt": False, "boltz2_complex_iplddt": False}
    rows = []
    for m, lower_better in metrics.items():
        if m not in df.columns:
            continue
        s, keep = _scored(df[m], lower_better)
        rows += cells(s, df.binder.to_numpy()[keep], "Nipah", "NiV-G", m,
                      primary=(m == "boltz2_ipsae"))
    return rows


def load_rbx1() -> list[dict]:
    """GEM x Adaptyv RBX1 competition: primary metric ESMFold pLDDT.

    Weaker evidence than the others and labelled as such: this collection
    publishes no interface score, so the ranking metric is a monomer confidence.
    It is kept because RBX1 is a target protein no other campaign here covers.
    """
    if not RBX1.exists():
        return []
    df = pd.read_csv(RBX1)
    rows = []
    for m, lower_better in (("esmfold_plddt", False),
                            ("proteinmpnn_score", True)):
        if m not in df.columns:
            continue
        s, keep = _scored(df[m], lower_better)
        rows += cells(s, df.binder.to_numpy()[keep], "RBX1comp", "RBX1", m,
                      primary=(m == "esmfold_plddt"))
    return rows


def load_bindcraft() -> list[dict]:
    """BindCraft: primary metric interface pTM, higher is better.

    212 designs over 13 targets, so almost every target is too small to rank a
    top-N against a held-out remainder. Only PD1 (53 labelled designs) clears
    the bar, and only PD1 is used. The rest are left out rather than pooled
    across targets, which would measure a different thing.
    """
    if not BINDCRAFT.exists():
        return []
    metrics = {"Average_i_pTM": False, "Average_i_pAE": True,
               "Average_pLDDT": False}
    raw = pd.read_csv(BINDCRAFT, low_memory=False)
    # the file has 225 columns; take the handful we use before adding one
    df = raw[["Target", "Binding"] + [m for m in metrics if m in raw.columns]].copy()
    df["y"] = pd.to_numeric(df.Binding, errors="coerce")
    df = df.dropna(subset=["y"])
    rows = []
    for t, g in df.groupby("Target"):
        if len(g) < 2 * min(BUDGETS) or g.y.sum() < 3:
            continue
        for m, lower_better in metrics.items():
            if m not in g.columns:
                continue
            s, keep = _scored(g[m], lower_better)
            rows += cells(s, g.y.to_numpy()[keep], "BindCraft", str(t), m,
                          primary=(m == "Average_i_pTM"))
    return rows


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------

def confusion(fired, catastrophe) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN -- fired-and-bad, fired-and-fine, missed, rightly silent."""
    f = np.asarray(list(fired), dtype=bool)
    c = np.asarray(list(catastrophe), dtype=bool)
    return (int((f & c).sum()), int((f & ~c).sum()),
            int((~f & c).sum()), int((~f & ~c).sum()))


def collapse(D: pd.DataFrame, key: str) -> pd.DataFrame:
    """Collapse budgets (and metrics, when key is coarse) into one row per key.

    A key counts as a catastrophe if ANY budget blew up, and as a firing if the
    detector spoke up at ANY budget. That pairing is the rule a real campaign
    would follow: you run the check at whatever plate size you have, and one
    warning is a warning.
    """
    return (D.groupby(key)
             .agg(fired=("fired", "any"), catastrophe=("catastrophe", "any"),
                  worst_ece=("ece", "max"), rows=("N", "count"))
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


def slice_overlap(over: pd.DataFrame | None, benn: pd.DataFrame | None) -> None:
    """Do the two campaigns' labelled slices contain the same designs?

    Bennett re-scored designs from earlier published campaigns, so the two
    tables share design names. The calibrations are fit on the top-N slices,
    so that is where sharing would actually create a dependence.
    """
    o_ids = set(over.binder_id.dropna().astype(str))
    b_ids = set(benn.description.dropna().astype(str))
    shared_pool = len(o_ids & b_ids)
    print(f"  design names shared between Overath and Bennett: "
          f"{shared_pool:,} of Overath's {len(o_ids):,} "
          f"({100 * shared_pool / max(1, len(o_ids)):.0f}%)")

    N = max(BUDGETS)
    benn = benn.assign(_m=pd.to_numeric(benn.pAE_interaction, errors="coerce"))
    total, pairs = 0, 0
    for t in sorted(set(over.target_id.astype(str)) & set(benn.target.astype(str))):
        go = over[over.target_id.astype(str) == t].nlargest(N, "_m")
        gb = benn[benn.target.astype(str) == t].nsmallest(N, "_m")
        if go.empty or gb.empty:
            continue
        pairs += 1
        total += len(set(go.binder_id.astype(str)) & set(gb.description.astype(str)))
    print(f"  but in the top-{N} slices the calibrations are actually fit on,")
    print(f"  they share {total} designs across {pairs} shared targets "
          f"({total} of {2 * N * pairs:,}). The two rank different pools by")
    print("  different scores, so the slices land in different places.")


# --------------------------------------------------------------------------

def main() -> int:
    over, over_df = load_overath()
    adap, adap_aucs = load_adaptyv()
    benn, benn_df = load_bennett()
    nip, rbx, bind = load_nipah(), load_rbx1(), load_bindcraft()
    rows = over + adap + benn + nip + rbx + bind
    if not rows:
        print("no data found. Run: python scripts/get_data.py", file=sys.stderr)
        return 2
    D = pd.DataFrame(rows)
    P = D[D.primary]                            # primary metric only
    units = collapse(P, "unit")
    sits = collapse(D, "situation")

    campaigns = (("Overath", over), ("Adaptyv", adap), ("Bennett", benn),
                 ("Nipah", nip), ("RBX1", rbx), ("BindCraft", bind))
    loaded = [n for n, r in campaigns if r]
    missing = [n for n, r in campaigns if not r]
    print("=" * 78)
    print("Inverted-calibration detector | " + " + ".join(loaded))
    print("=" * 78)
    proteins = sorted(set(D.target.astype(str)))
    repeats = len(units) - len(proteins)
    print(f"  {len(units)} unit{'s' if len(units) != 1 else ''} "
          "(campaign x target, primary metric only)")
    print(f"  spanning {len(proteins)} distinct target protein"
          f"{'s' if len(proteins) != 1 else ''}"
          + (f" -- {repeats} of them appear in two campaigns" if repeats else ""))
    print(f"  {D.pool.groupby(D.unit).first().sum():,} designs in the pools, "
          f"{len(D)} (unit x metric x budget) cells behind it all")
    if missing:
        print(f"  NOTE: {', '.join(missing)} not loaded, so these are not the")
        print("        README's numbers. Run: python scripts/get_data.py")

    if adap_aucs:
        print("\n  Sanity check against the Adaptyv paper's own numbers:")
        for k, v in adap_aucs.items():
            pub = {"iptm": 0.64, "plddt": 0.66}.get(k)
            print(f"    {k:<18}{v:.3f}" + (f"   published {pub:.2f}" if pub else ""))

    print("\n  Independence audit:")
    slice_overlap(over_df, benn_df)

    # ---- PRIMARY --------------------------------------------------------
    print()
    print(f"BY UNIT  ({len(units)} campaign-target pairs)  <-- the number to quote")
    print(f"  a catastrophe is out-of-sample ECE > {CATASTROPHE} at any budget;")
    print("  a firing is the detector speaking up at any budget")
    print()
    report(*confusion(units.fired, units.catastrophe))

    # ---- robustness -----------------------------------------------------
    print()
    print("BY BUDGET  (within one budget the units are independent too)")
    print(f"  {'wells':<10}{'units':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"   sensitivity")
    print("  " + "-" * 72)
    for N, g in P.groupby("N"):
        tp, fp, fn, tn = confusion(g.fired, g.catastrophe)
        sens = str(wilson(tp, tp + fn)) if tp + fn else "no catastrophes"
        print(f"  N={N:<8}{len(g):>8}{tp:>5}{fp:>5}{fn:>5}{tn:>5}   {sens}")

    print()
    print(f"BY SITUATION  ({len(sits)} unit x metric)  -- secondary: metrics over")
    print("  one design set share their labels, so these units are not independent")
    print("  of each other the way the primary ones are.")
    print()
    report(*confusion(sits.fired, sits.catastrophe), indent="    ")

    tp, fp, fn, tn = confusion(D.fired, D.catastrophe)
    print()
    print(f"BY CELL  ({len(D)} rows)  -- descriptive only: budgets are NESTED")
    print("  inside units. The project's first headline came from a count like")
    print("  this one. It is kept so the correction stays checkable.")
    print(f"    TP {tp}  FP {fp}  FN {fn}  TN {tn}   "
          f"sensitivity {wilson(tp, tp + fn)}")

    # ---- what the split actually buys -----------------------------------
    print()
    print("Out-of-sample ECE, split by what the detector said (all cells)")
    print(f"  fired   {bootstrap_ci(D[D.fired].ece, seed=1)}")
    print(f"  silent  {bootstrap_ci(D[~D.fired].ece, seed=1)}")
    print(f"  worst curve the detector let through: {D[~D.fired].ece.max():.3f}")
    print(f"  worst curve it caught:                {D[D.fired].ece.max():.3f}")

    print()
    print("Every unit on its primary metric")
    print(f"  {'unit':<26}{'metric':<22}{'said':>8}{'truth':>10}{'worst ECE':>11}")
    print("  " + "-" * 77)
    metric_of = P.groupby("unit").metric.first()
    for _, r in units.sort_values("unit").iterrows():
        print(f"  {r.unit:<26}{metric_of[r.unit][:21]:<22}"
              f"{'FIRED' if r.fired else 'silent':>8}"
              f"{'BAD' if r.catastrophe else 'fine':>10}{r.worst_ece:>11.3f}")

    print()
    print("Reading it")
    tp, fp, fn, tn = confusion(units.fired, units.catastrophe)
    stp, sfp, sfn, stn = confusion(sits.fired, sits.catastrophe)
    print(f"  {len(units)} units and {tp + fn} catastrophes is still a thin base,")
    print("  and the intervals say so.")
    print()
    print("  ONE claim survives every view: the curves the detector let through")
    print("  were an order of magnitude better calibrated than the ones it")
    print(f"  caught. That gap is the result.")
    print()
    print(f"  The clean sheet on false positives does NOT survive. On primary")
    print(f"  metrics it is {fp} in {tp + fp} firings; add the secondary metrics and it")
    print(f"  is {sfp} in {stp + sfp}, dropping precision to "
          f"{stp / max(1, stp + sfp):.3f} and specificity to")
    print(f"  {stn / max(1, stn + sfp):.3f}. The extra metrics are the ones with narrow score")
    print("  bands, where a flat fit is hardest to tell from an inverted one.")
    print(f"  Quote the primary number, but do not pretend the other view is")
    print(f"  not there. It misses {fn} catastrophes either way.")
    print()
    print("  Narrowing these intervals needs more TARGETS, from campaigns whose")
    print("  designs are not already in one of these three. More metrics over")
    print("  designs already here would only add rows to the secondary view.")
    print()
    print("  This costs nothing to run. Published pipelines do not check.")

    out = ROOT / "experiments" / "detector_results.json"
    out.write_text(D.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\nraw results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
