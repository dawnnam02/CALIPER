"""단계 4 — 설계한 것이 정말 그렇게 접히는지 예측한다 (AF2 initial guess).

    python pipeline/step4_predict.py

=============================================================================
이 단계가 파이프라인 전체의 성패를 가른다
=============================================================================
지금까지는 "이런 모양이면 붙겠다"고 **가정**하고 만들었다.
이 단계는 그 가정을 독립적인 도구로 다시 확인한다.

설계할 때 쓴 것(RFdiffusion, ProteinMPNN)과 검증할 때 쓰는 것(AlphaFold2)이
서로 다른 모델이라는 게 중요하다. 같은 모델로 만들고 같은 모델로 채점하면
자기 실수를 못 잡는다. 시험 문제를 낸 사람이 자기 답안을 채점하는 셈이다.

=============================================================================
'initial guess' 가 뭔가 — 이 한 줄이 성공률을 10배 올렸다
=============================================================================
보통 AlphaFold2 는 **서열만 보고** 구조를 처음부터 예측한다.
결합체처럼 자연에 없는 단백질에는 이게 잘 안 통한다. 참고할 진화 정보
(비슷한 서열들의 목록, MSA)가 없기 때문이다.

initial guess 는 다르게 묻는다.
    "우리가 설계한 이 복합체 구조에서 출발해라.
     이 서열이 정말 이 모양이 맞다면, 너도 여기 머물 것이다."

설계가 맞으면 AF2 는 그 근처에 머문다. 틀리면 멀리 도망간다.
그 '머물렀는가'를 재는 것이다.

    근거: Bennett et al. 2023, Nature Communications 14:2625
          이 방법으로 실험 성공률이 약 10배 올랐다.

=============================================================================
나오는 값 네 개
=============================================================================
  pae_interaction   두 사슬 사이 위치를 AF2 가 얼마나 확신하나. **낮을수록 좋다**
                    이 파이프라인의 주 필터. 실측 농축 1.6~2.4배 (VALIDATION.md)

  plddt_binder      결합체 혼자서 잘 접히나. 높을수록 좋다.
                    ⚠ 실측에서 농축 효과가 거의 없었다. 참고용으로만 봐라

  rmsd              설계한 골격과 예측 구조가 얼마나 어긋나나. 낮을수록 좋다
                    실측에서 Bennett 기준 농축 2.6배 — pae 보다 셌다

  iptm              계면 전체의 신뢰도. 높을수록 좋다

Step 4 - validate designs with AF2 initial guess; the single highest-value step.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_config as cfg
from _shared import header, read_fasta, require_tool, run


def main() -> int:
    header(4, "구조 검증 (AlphaFold2 initial guess)")
    cfg.ensure_dirs()

    fasta = cfg.DIR_SEQUENCE / "designs.fasta"
    if not fasta.exists():
        print("  서열이 없다. 먼저: python pipeline/step3_sequence.py")
        return 2

    seqs = read_fasta(fasta)
    print(f"  예측할 서열 : {len(seqs):,}개")
    print(f"  initial guess: {'예' if cfg.AF2_USE_INITIAL_GUESS else '아니오'}")
    if not cfg.AF2_USE_INITIAL_GUESS:
        print("      ⚠ 끄면 성공률이 크게 떨어진다. 켜는 걸 강하게 권한다.")
    print(f"  recycle 횟수 : {cfg.AF2_NUM_RECYCLES}")
    print()
    print("  ※ 여기가 가장 오래 걸린다. 서열 하나에 GPU 로 수십 초 걸린다.")
    print(f"    {len(seqs):,}개면 대략 "
          f"{len(seqs) * 30 / 3600:.0f}~{len(seqs) * 90 / 3600:.0f}시간이다.")
    print("    중간에 끊겨도 이미 끝난 것은 다시 안 한다.")
    print()

    script = require_tool("af2ig")

    out_sc = cfg.DIR_PREDICT / "scores.sc"
    cmd = [
        sys.executable, str(script),
        "-pdbdir", str(cfg.DIR_BACKBONE),
        "-outpdbdir", str(cfg.DIR_PREDICT / "structures"),
        "-scorefilename", str(out_sc),
        "-recycle", str(cfg.AF2_NUM_RECYCLES),
    ]
    if not cfg.AF2_USE_INITIAL_GUESS:
        cmd += ["-no_initial_guess"]
    run(cmd)

    if not out_sc.exists():
        print(f"  점수 파일이 안 만들어졌다: {out_sc}")
        return 1

    # 결과를 훑어본다 — 값이 통째로 이상하면 여기서 알아채야 한다
    try:
        import pandas as pd
        df = pd.read_csv(out_sc, sep=r"\s+")
        print()
        print(f"  예측 완료: {len(df):,}개")
        print()
        print(f"  {'지표':<22}{'중앙값':>10}{'최선':>10}{'최악':>10}")
        print("  " + "-" * 52)
        for col, better in (("pae_interaction", "낮을수록"),
                            ("plddt_binder", "높을수록"),
                            ("binder_aligned_rmsd", "낮을수록")):
            if col not in df.columns:
                continue
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if v.empty:
                continue
            best, worst = (v.min(), v.max()) if better == "낮을수록" else (v.max(), v.min())
            print(f"  {col:<22}{v.median():>10.2f}{best:>10.2f}{worst:>10.2f}")

        if "pae_interaction" in df.columns:
            pae = pd.to_numeric(df.pae_interaction, errors="coerce")
            n_pass = int((pae < cfg.FILTER_PAE_INTERACTION_MAX).sum())
            print()
            print(f"  pae_interaction < {cfg.FILTER_PAE_INTERACTION_MAX} 통과: "
                  f"{n_pass:,} / {len(df):,}  ({100*n_pass/max(1,len(df)):.1f}%)")
            if n_pass == 0:
                print()
                print("  ✗ 통과가 0개다. 이건 흔한 일이고, 골격을 더 만들어도")
                print("    대개 해결되지 않는다. → PIPELINE.md '필터가 전부 죽일 때'")
    except ImportError:
        print("  (요약을 보려면 pandas 가 필요하다: pip install pandas)")

    print()
    print("  ✓ 다음: python pipeline/step5_filter.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
