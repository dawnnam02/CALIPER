"""정석 필터가 실제로 얼마나 걸러주는지, 실측 데이터로 확인한다.

    python validation/check_filters.py

=============================================================================
무엇을 확인하는가
=============================================================================
단백질 결합체 설계의 교과서 필터는 이 세 개다.

    pae_interaction < 10      계면을 AF2 가 확신하는가
    plddt_binder    > 80      결합체가 제대로 접히는가
    rmsd            < 2.0 Å   설계한 대로 접히는가

이 값들은 논문에서 왔다. 그런데 **논문의 값이 내 표적에서도 맞는지는
아무도 확인해 주지 않는다.** 이 스크립트가 그걸 확인한다.

쓰는 데이터는 실험 결과가 붙어 있는 공개 캠페인 넷이다.
그중 둘(Overath · Bennett)은 세 필터 열을 전부 갖고 있어서
정석 조합을 그대로 재현할 수 있다.

=============================================================================
왜 '농축 배수'로 보는가
=============================================================================
적중률만 보면 표적끼리 비교가 안 된다. 원래 쉬운 표적은 필터가 없어도
적중률이 높기 때문이다. 그래서 이렇게 잰다.

    농축 배수 = (필터 통과한 것들의 적중률) / (전체의 적중률)

    1.0배  = 필터가 아무 일도 안 했다
    2.0배  = 필터를 통과한 것이 두 배 더 잘 붙는다
    1.0 미만 = 필터가 **해롭다.** 안 쓰느니만 못하다

What this checks: whether the textbook filter thresholds actually enrich for
binders on real published campaigns, measured as fold-enrichment over the
unfiltered base rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

# 정석 임계값 — pipeline/step0_config.py 와 같은 값이어야 한다
PAE_MAX = 10.0
PLDDT_MIN = 80.0
RMSD_MAX = 2.0


# =============================================================================
# 캠페인마다 열 이름이 다르다. 여기서 이름만 맞춰준다.
# =============================================================================

def load_overath():
    f = ROOT / "data" / "overath" / "final_dataset.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, low_memory=False,
                    usecols=["target_id", "binder", "af2_pae_interaction",
                             "af2_plddt_binder", "af2_binder_aligned_rmsd"])
    d = d[d.binder.notna()]
    return pd.DataFrame({
        "campaign": "Overath",
        "target": d.target_id.astype(str),
        "bound": d.binder.astype(bool).astype(int),
        "pae": pd.to_numeric(d.af2_pae_interaction, errors="coerce"),
        "plddt": pd.to_numeric(d.af2_plddt_binder, errors="coerce"),
        "rmsd": pd.to_numeric(d.af2_binder_aligned_rmsd, errors="coerce"),
    })


def load_bennett():
    f = ROOT / "data" / "bennett" / "retrospective.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, low_memory=False,
                    usecols=["target", "avid_ub", "pAE_interaction",
                             "AF2_plddt_monomer", "AF2_complex_RMSD"])
    return pd.DataFrame({
        "campaign": "Bennett",
        "target": d.target.astype(str),
        # 결합 = 효모 표면제시에서 avidity Kd 가 실제로 측정된 것
        "bound": np.isfinite(pd.to_numeric(d.avid_ub, errors="coerce")).astype(int),
        "pae": pd.to_numeric(d.pAE_interaction, errors="coerce"),
        "plddt": pd.to_numeric(d.AF2_plddt_monomer, errors="coerce"),
        "rmsd": pd.to_numeric(d.AF2_complex_RMSD, errors="coerce"),
    })


def load_adaptyv():
    """pae 와 plddt 는 있고 rmsd 가 없다. 부분 검증만 된다."""
    f = ROOT / "data" / "adaptyv" / "round2.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.binding.astype(str).str.lower() != "unknown"]
    return pd.DataFrame({
        "campaign": "Adaptyv",
        "target": "EGFR",
        "bound": (d.binding.astype(str).str.lower() == "true").astype(int),
        "pae": pd.to_numeric(d.pae_interaction, errors="coerce"),
        "plddt": pd.to_numeric(d.plddt, errors="coerce"),
        "rmsd": np.nan,
    })


def load_bindcraft():
    f = ROOT / "data" / "bindcraft" / "screening.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, low_memory=False)
    y = pd.to_numeric(d.Binding, errors="coerce")
    keep = y.notna()
    return pd.DataFrame({
        "campaign": "BindCraft",
        "target": d.Target[keep].astype(str),
        "bound": y[keep].astype(int),
        "pae": pd.to_numeric(d.Average_i_pAE[keep], errors="coerce"),
        "plddt": pd.to_numeric(d.Average_pLDDT[keep], errors="coerce"),
        "rmsd": np.nan,
    })


def enrichment(mask, bound) -> tuple[int, float, float]:
    """통과 개수, 통과한 것들의 적중률, 농축 배수."""
    n = int(mask.sum())
    if n == 0:
        return 0, float("nan"), float("nan")
    base = float(bound.mean())
    hit = float(bound[mask].mean())
    return n, hit, (hit / base if base > 0 else float("nan"))


def main() -> int:
    parts = [p for p in (load_overath(), load_bennett(),
                         load_adaptyv(), load_bindcraft()) if p is not None]
    if not parts:
        print("데이터가 없다. 먼저 실행:  python scripts/get_data.py",
              file=sys.stderr)
        return 2
    D = pd.concat(parts, ignore_index=True)

    print("=" * 78)
    print("정석 필터 검증 — 실측 데이터")
    print("=" * 78)
    for c, g in D.groupby("campaign"):
        has = [n for n, col in (("pae", "pae"), ("plddt", "plddt"), ("rmsd", "rmsd"))
               if g[col].notna().any()]
        print(f"  {c:<12}{len(g):>9,}개   결합 {int(g.bound.sum()):>6,} "
              f"({100 * g.bound.mean():>5.2f}%)   열: {'+'.join(has)}")

    # -----------------------------------------------------------------
    # 1. 필터 하나씩 — 각각이 정말 일하고 있는가
    # -----------------------------------------------------------------
    print()
    print("[1] 필터 하나씩 따로 걸었을 때")
    print("    농축 1.0배 = 아무 일도 안 함 / 1.0 미만 = 해로움")
    print()
    print(f"    {'캠페인':<12}{'필터':<22}{'통과':>10}{'통과율':>8}"
          f"{'적중률':>9}{'농축':>8}")
    print("    " + "-" * 69)
    tests = [("pae < 10", lambda g: g.pae < PAE_MAX),
             ("plddt > 80", lambda g: g.plddt > PLDDT_MIN),
             ("rmsd < 2.0", lambda g: g.rmsd < RMSD_MAX)]
    for c, g in D.groupby("campaign"):
        for name, fn in tests:
            m = fn(g).fillna(False)
            if not m.any():
                continue
            n, hit, enr = enrichment(m, g.bound)
            flag = "  <- 해로움" if enr < 1.0 else ""
            print(f"    {c:<12}{name:<22}{n:>10,}{100 * m.mean():>7.1f}%"
                  f"{100 * hit:>8.1f}%{enr:>7.1f}x{flag}")
        print()

    # -----------------------------------------------------------------
    # 2. 임계값을 바꿔가며 — 10 이라는 숫자가 맞는가
    # -----------------------------------------------------------------
    print("[2] pae 임계값을 바꿔가며 (교과서 값은 10)")
    print(f"    {'캠페인':<12}{'임계값':>8}{'통과':>10}{'통과율':>8}"
          f"{'적중률':>9}{'농축':>8}")
    print("    " + "-" * 63)
    for c, g in D.groupby("campaign"):
        if not g.pae.notna().any():
            continue
        for t in (5, 7.5, 10, 15, 20):
            m = (g.pae < t).fillna(False)
            if not m.any():
                continue
            n, hit, enr = enrichment(m, g.bound)
            mark = "  <- 교과서" if t == 10 else ""
            print(f"    {c:<12}{t:>8}{n:>10,}{100 * m.mean():>7.1f}%"
                  f"{100 * hit:>8.1f}%{enr:>7.1f}x{mark}")
        print()

    # -----------------------------------------------------------------
    # 3. 세 개를 한꺼번에 — 정석 조합
    # -----------------------------------------------------------------
    print("[3] 정석 조합: pae<10 AND plddt>80 AND rmsd<2")
    print(f"    {'캠페인':<12}{'통과':>10}{'통과율':>8}{'적중률':>9}{'농축':>8}")
    print("    " + "-" * 47)
    for c, g in D.groupby("campaign"):
        if not g.rmsd.notna().any():
            print(f"    {c:<12}{'rmsd 열이 없어 조합을 못 만든다':>40}")
            continue
        m = ((g.pae < PAE_MAX) & (g.plddt > PLDDT_MIN)
             & (g.rmsd < RMSD_MAX)).fillna(False)
        n, hit, enr = enrichment(m, g.bound)
        print(f"    {c:<12}{n:>10,}{100 * m.mean():>7.1f}%"
              f"{100 * hit:>8.1f}%{enr:>7.1f}x")

    # -----------------------------------------------------------------
    # 4. 표적별 — 여기가 진짜 중요한 부분
    # -----------------------------------------------------------------
    print()
    print("[4] 표적별로 pae<10 이 몇 개를 남기나")
    print("    같은 임계값인데 표적마다 결과가 전혀 다르다. 이게 핵심이다.")
    print()
    print(f"    {'캠페인/표적':<28}{'전체':>10}{'통과':>9}{'통과율':>8}"
          f"{'적중률':>9}{'농축':>8}")
    print("    " + "-" * 72)
    wiped, useless = [], []
    for (c, t), g in D.groupby(["campaign", "target"]):
        if len(g) < 50 or not g.pae.notna().any():
            continue
        m = (g.pae < PAE_MAX).fillna(False)
        n, hit, enr = enrichment(m, g.bound)
        base = 100 * g.bound.mean()
        if n == 0:
            note = "  <- 라이브러리 전멸"
            wiped.append(f"{c}/{t}")
        elif enr < 1.0:
            note = "  <- 해로움"
            useless.append(f"{c}/{t}")
        else:
            note = ""
        hs = f"{100 * hit:>8.1f}%" if n else f"{'-':>9}"
        es = f"{enr:>7.1f}x" if n else f"{'-':>8}"
        print(f"    {c + '/' + t:<28}{len(g):>10,}{n:>9,}{100 * m.mean():>7.1f}%"
              f"{hs}{es}{note}   (원래 {base:.1f}%)")

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("읽는 법")
    print("=" * 78)
    if wiped:
        print(f"  ● 통과 0개인 표적: {', '.join(wiped)}")
        print("    교과서 임계값을 그대로 쓰면 이 표적들은 실험할 것이 남지 않는다.")
        print("    골격을 더 만들어도 소용없다. 임계값을 표적에 맞게 풀어야 한다.")
    if useless:
        print(f"  ● 필터가 해로운 표적: {', '.join(useless)}")
        print("    필터를 통과한 쪽이 오히려 덜 붙는다. 안 거르느니만 못하다.")
    print()
    print("  ● 그래서 파이프라인은 임계값을 조용히 적용하지 않는다.")
    print("    step5_filter.py 가 통과 개수를 먼저 보고하고,")
    print("    0에 가까우면 멈춰서 임계값을 풀지 물어본다.")
    print()
    print("  ● 이 검증을 네 표적에서 다시 하려면 실험 결과가 필요하다.")
    print("    1라운드를 돌린 뒤에 이 스크립트에 네 데이터를 넣어라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
