"""단계 5 — 거르고 순위를 매긴다.

    python pipeline/step5_filter.py

=============================================================================
이 단계는 임계값을 조용히 적용하지 않는다
=============================================================================
교과서 임계값은 이렇다.

    pae_interaction < 10 · plddt_binder > 80 · rmsd < 2.0 Å

그런데 실측 데이터 60만 개로 확인해 보니 **표적마다 결과가 전혀 달랐다.**

    표적            통과율      농축
    ------------------------------
    Bennett/Tie2      0.0%       -      ← 92,293개 중 0개. 라이브러리 전멸
    Overath/VirB8     0.0%       -      ← 99개 중 0개
    Bennett/H3        0.0%     0.0x     ← 2개 통과, 둘 다 안 붙음
    Bennett/SARS      0.1%    14.4x     ← 거의 다 죽이지만 남은 건 훌륭
    Overath/Pdl1     97.9%     1.0x     ← 거의 안 거른다. 필터가 무의미
    Bennett/InsulinR 23.0%     1.2x

같은 숫자인데 어떤 표적은 전멸하고 어떤 표적은 그냥 통과다.
**논문의 임계값이 네 표적에서도 맞다는 보장은 없다.**

그래서 이 스크립트는 순서를 이렇게 잡는다.

    1. 필터를 하나씩 걸어 몇 개가 죽는지 먼저 보고한다
    2. 남은 게 너무 적으면 **멈추고**, 데이터에서 계산한 대안 임계값을 제안한다
    3. 통과한 것들을 pae_interaction 순으로 줄 세운다

자세한 근거 → VALIDATION.md,  다시 재보려면 → validation/check_filters.py

Step 5 - filter and rank. Reports the funnel before applying, and refuses to
proceed silently when the textbook thresholds wipe out the library.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_config as cfg
from _shared import header

MIN_SURVIVORS = 20          # 이보다 적게 남으면 멈추고 물어본다


def main() -> int:
    header(5, "필터와 순위")
    cfg.ensure_dirs()

    try:
        import pandas as pd
    except ImportError:
        print("  pandas 가 필요하다:  pip install pandas")
        return 2

    sc = cfg.DIR_PREDICT / "scores.sc"
    if not sc.exists():
        print("  예측 점수가 없다. 먼저: python pipeline/step4_predict.py")
        return 2

    df = pd.read_csv(sc, sep=r"\s+")
    for c in ("pae_interaction", "plddt_binder", "binder_aligned_rmsd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    print(f"  들어온 설계: {n0:,}개")
    print()

    # -----------------------------------------------------------------
    # 1. 깔때기 — 각 필터가 몇 개를 죽이는지 하나씩 보여준다
    # -----------------------------------------------------------------
    print("  깔때기 (필터를 하나씩 누적해서 걸었을 때)")
    print(f"    {'필터':<34}{'남는 수':>10}{'남는 비율':>11}")
    print("    " + "-" * 55)

    checks = [
        ("pae_interaction", "<", cfg.FILTER_PAE_INTERACTION_MAX),
        ("plddt_binder", ">", cfg.FILTER_PLDDT_BINDER_MIN),
        ("binder_aligned_rmsd", "<", cfg.FILTER_RMSD_MAX),
    ]

    keep = pd.Series(True, index=df.index)
    per_filter = {}
    for col, op, thr in checks:
        if col not in df.columns:
            print(f"    {col + ' (열이 없음 — 건너뜀)':<34}{'':>10}{'':>11}")
            continue
        m = (df[col] < thr) if op == "<" else (df[col] > thr)
        m = m.fillna(False)
        per_filter[col] = int(m.sum())
        keep = keep & m
        label = f"{col} {op} {thr}"
        print(f"    {label:<34}{int(keep.sum()):>10,}{100*keep.mean():>10.1f}%")

    n_pass = int(keep.sum())
    print()
    print("  필터 하나씩 따로 걸었을 때 (누적 아님)")
    for col, n in per_filter.items():
        print(f"    {col:<34}{n:>10,}{100*n/max(1,n0):>10.1f}%")

    # -----------------------------------------------------------------
    # 2. 너무 많이 죽었으면 멈춘다
    # -----------------------------------------------------------------
    if n_pass < MIN_SURVIVORS:
        print()
        print("  " + "=" * 66)
        print(f"  멈춤: {n_pass}개만 살아남았다 (최소 {MIN_SURVIVORS}개 필요)")
        print("  " + "=" * 66)
        print()
        print("  이건 흔한 일이고, 대개 골격을 더 만들어도 해결되지 않는다.")
        print("  실측에서도 표적 두 개가 통과 0개였다 (Tie2, VirB8).")
        print()

        # 데이터 자체에서 대안 임계값을 계산해 준다
        print("  이 데이터에서 계산한 대안 임계값")
        print(f"    {'필터':<24}{'현재':>8}{'상위 100개가 되려면':>22}")
        print("    " + "-" * 54)
        for col, op, thr in checks:
            if col not in df.columns or df[col].isna().all():
                continue
            v = df[col].dropna()
            q = v.nsmallest(100).max() if op == "<" else v.nlargest(100).min()
            print(f"    {col:<24}{thr:>8.1f}{q:>22.2f}")
        print()
        print("  어느 쪽을 택할지는 판단이다:")
        print("    (가) 임계값을 푼다      → 실험할 것이 생기지만 적중률이 떨어진다")
        print("    (나) 단계 1로 돌아간다  → 핫스팟이나 표적 자체가 문제일 수 있다")
        print()
        print("    통과가 0에 가까우면 (나)를 먼저 의심해라. 필터가 전부 죽인다는 건")
        print("    보통 '설계가 부족하다'가 아니라 '그 자리에 붙는 모양이 없다'는 뜻이다.")
        print()
        print("  → PIPELINE.md 의 '필터가 전부 죽일 때' 절")
        print("  → 임계값을 바꾸려면 step0_config.py 의 FILTER_* 를 고쳐라")
        return 1

    # -----------------------------------------------------------------
    # 3. 순위
    # -----------------------------------------------------------------
    passed = df[keep].copy()
    sort_col = ("pae_interaction" if "pae_interaction" in passed.columns
                else passed.columns[0])
    passed = passed.sort_values(sort_col)          # 낮을수록 좋다

    out = cfg.DIR_FILTER / "passed.csv"
    passed.to_csv(out, index=False)

    print()
    print(f"  통과: {n_pass:,}개  ({100*n_pass/n0:.1f}%)  → {out}")
    print(f"  정렬 기준: {sort_col} (낮은 순)")
    print()
    print(f"  상위 10개")
    show = [c for c in ("description", "pae_interaction", "plddt_binder",
                        "binder_aligned_rmsd") if c in passed.columns]
    print("    " + "".join(f"{c[:20]:<22}" for c in show))
    print("    " + "-" * (22 * len(show)))
    for _, r in passed.head(10).iterrows():
        cells = []
        for c in show:
            v = r[c]
            cells.append(f"{v:<22.2f}" if isinstance(v, float) else f"{str(v)[:20]:<22}")
        print("    " + "".join(cells))

    print()
    print("  ※ 순위는 '이 순서로 좋다'가 아니라 '이 순서로 확신한다'이다.")
    print("    pae 가 낮다는 건 AF2 가 확신한다는 뜻이지 실제로 붙는다는 뜻이 아니다.")
    print("    실측 농축이 2배 남짓이라는 건 그런 의미다 — 도움은 되지만 보장은 아니다.")
    print()
    print("  ✓ 다음: python pipeline/step6_order.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
