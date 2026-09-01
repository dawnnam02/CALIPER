"""단계 2 — 골격을 만든다 (RFdiffusion).

    python pipeline/step2_backbone.py

=============================================================================
골격(backbone)이 뭔가
=============================================================================
단백질은 두 층으로 생각할 수 있다.

    골격  = 사슬이 공간에서 어떤 모양으로 접혀 있는가 (뼈대)
    서열  = 그 자리마다 어떤 아미노산이 놓이는가       (살)

이 단계는 **뼈대만** 만든다. 아직 어떤 아미노산인지는 정하지 않는다.
"표적의 이 자리에 딱 맞물리는 모양"을 먼저 찾는 것이다.

열쇠로 치면, 열쇠의 홈 모양을 먼저 깎고 재료는 나중에 정하는 셈이다.

=============================================================================
RFdiffusion 이 하는 일
=============================================================================
확산 모델(diffusion model)이다. 그림 생성 AI 와 같은 원리다.
완전한 노이즈에서 시작해서 조금씩 노이즈를 걷어내며 단백질 모양을 만든다.

결합체 설계에서는 조건을 하나 건다 —
**"표적의 이 핫스팟들에 닿는 모양이어야 한다."**

=============================================================================
자주 틀리는 설정 두 가지
=============================================================================
1) **노이즈 스케일을 0으로 안 둔다.**
   기본값은 1.0이다. 다양성을 위해 노이즈를 넣는 건데,
   결합체에서는 그 다양성이 성공률을 떨어뜨린다는 게 측정돼 있다.
   step0_config.py 의 RFDIFFUSION_NOISE_SCALE = 0.0 이 그것이다.

2) **골격을 너무 적게 만든다.**
   뒤 단계에서 90% 이상이 걸러진다. 골격 100개로 시작하면
   실험에 보낼 것이 한 자리 수로 남는다.

Step 2 - generate binder backbones with RFdiffusion, conditioned on hotspots.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_config as cfg
from _shared import header, require_tool, run


def main() -> int:
    header(2, "골격 생성 (RFdiffusion)")
    cfg.ensure_dirs()

    target = cfg.DIR_TARGET / "target_trimmed.pdb"
    if not target.exists():
        print("  손질한 표적이 없다. 먼저: python pipeline/step1_target.py")
        return 2

    # RFdiffusion 은 핫스팟을 "A56" 같은 형식으로 받는다 (사슬 + 번호)
    hotspots = ",".join(f"{cfg.TARGET_CHAIN}{n}" for n in cfg.TARGET_HOTSPOTS)

    # 길이는 범위로 준다. RFdiffusion 이 그 안에서 골라 만든다.
    length = f"{cfg.BINDER_LENGTH_MIN}-{cfg.BINDER_LENGTH_MAX}"

    print(f"  표적       : {target.name}")
    print(f"  핫스팟     : {hotspots}")
    print(f"  결합체 길이 : {length} 잔기")
    print(f"  만들 개수   : {cfg.N_BACKBONES}")
    print(f"  노이즈      : {cfg.RFDIFFUSION_NOISE_SCALE}  "
          f"(결합체에서는 0 이어야 한다)")
    print()

    script = require_tool("rfdiffusion")

    # contigmap.contigs 문법:
    #   [A1-150/0 55-100]  =  "표적 사슬 A 의 1~150번을 그대로 두고(/0 은 사슬 구분),
    #                          그 옆에 55~100 잔기짜리 새 사슬을 만들어라"
    contigs = f"[{cfg.TARGET_CHAIN}1-9999/0 {length}]"

    cmd = [
        sys.executable, str(script),
        f"inference.output_prefix={cfg.DIR_BACKBONE / 'backbone'}",
        f"inference.input_pdb={target}",
        f"inference.num_designs={cfg.N_BACKBONES}",
        f"contigmap.contigs={contigs}",
        f"ppi.hotspot_res=[{hotspots}]",
        f"denoiser.noise_scale_ca={cfg.RFDIFFUSION_NOISE_SCALE}",
        f"denoiser.noise_scale_frame={cfg.RFDIFFUSION_NOISE_SCALE}",
        f"diffuser.T={cfg.RFDIFFUSION_DIFFUSER_T}",
    ]
    run(cmd)

    made = sorted(cfg.DIR_BACKBONE.glob("backbone_*.pdb"))
    print()
    print(f"  만들어진 골격: {len(made)}개  → {cfg.DIR_BACKBONE}")

    if len(made) < cfg.N_BACKBONES * 0.5:
        print()
        print("  ⚠ 요청한 것보다 훨씬 적게 나왔다. 흔한 원인:")
        print("      - 핫스팟이 너무 많거나 서로 멀어서 다 닿는 모양이 없다")
        print("      - 결합체 길이 범위가 너무 좁다")
        print("      - 핫스팟이 파묻힌 자리다 (단계 1의 '이웃수'를 다시 봐라)")

    print()
    print("  ※ 지금 나온 것은 뼈대뿐이고 아미노산은 전부 글라이신(GLY)으로")
    print("    채워져 있다. 이건 정상이다. 다음 단계에서 진짜 서열을 넣는다.")
    print()
    print("  ✓ 다음: python pipeline/step3_sequence.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
