"""단계 1~6을 순서대로 돌린다.

    python pipeline/run_all.py           # 전부
    python pipeline/run_all.py --check   # 설정과 도구만 확인하고 끝
    python pipeline/run_all.py --from 4  # 4단계부터

각 단계는 따로 돌려도 된다. 오히려 처음에는 하나씩 돌리면서
결과를 눈으로 보는 걸 권한다. 특히 단계 1과 단계 5는
사람이 판단해야 하는 지점이 있다.

Run steps 1-6 in order. Any step can also be run on its own.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import step0_config as cfg
from _shared import tool_status

STEPS = [
    (1, "step1_target.py", "표적 준비", "사람 판단 필요"),
    (2, "step2_backbone.py", "골격 생성", "GPU · 수 시간"),
    (3, "step3_sequence.py", "서열 설계", "수 분"),
    (4, "step4_predict.py", "구조 검증", "GPU · 가장 오래"),
    (5, "step5_filter.py", "필터와 순위", "사람 판단 필요"),
    (6, "step6_order.py", "실험 발주", "수 초"),
]


def show_config() -> None:
    print("현재 설정  (바꾸려면 pipeline/step0_config.py)")
    print(f"  표적 파일   : {cfg.TARGET_PDB}"
          + ("" if cfg.TARGET_PDB.exists() else "   ← 파일이 없다"))
    print(f"  사슬        : {cfg.TARGET_CHAIN}")
    print(f"  핫스팟      : {cfg.TARGET_HOTSPOTS}")
    print(f"  결합체 길이 : {cfg.BINDER_LENGTH_MIN}~{cfg.BINDER_LENGTH_MAX}")
    print(f"  골격 수     : {cfg.N_BACKBONES:,}  "
          f"→ 서열 {cfg.N_BACKBONES * cfg.SEQS_PER_BACKBONE:,}개")
    print(f"  필터        : pae<{cfg.FILTER_PAE_INTERACTION_MAX} · "
          f"plddt>{cfg.FILTER_PLDDT_BINDER_MIN} · rmsd<{cfg.FILTER_RMSD_MAX}")
    print(f"  발주 개수   : {cfg.N_TO_ORDER}")


def main(argv: list[str]) -> int:
    print("=" * 74)
    print("  단백질 결합체 설계 — 정석 파이프라인")
    print("=" * 74)
    print()
    show_config()
    print()
    print("설치된 도구")
    tool_status()
    print()
    print("단계")
    for n, _, name, note in STEPS:
        print(f"  {n}. {name:<12} ({note})")

    if "--check" in argv:
        print()
        print("확인만 했다. 실제로 돌리려면 --check 를 빼라.")
        return 0

    start = 1
    if "--from" in argv:
        start = int(argv[argv.index("--from") + 1])

    for n, script, name, _ in STEPS:
        if n < start:
            continue
        r = subprocess.run([sys.executable, str(HERE / script)])
        if r.returncode != 0:
            print()
            print(f"단계 {n} ({name}) 에서 멈췄다. 위 메시지를 읽고 처리한 뒤")
            print(f"이어서 돌리려면:  python pipeline/run_all.py --from {n}")
            return r.returncode

    print()
    print("=" * 74)
    print(f"  끝. 발주 파일 → {cfg.DIR_ORDER}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
