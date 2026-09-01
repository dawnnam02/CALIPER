"""단계 6 — 실험에 보낼 96개를 확정한다.

    python pipeline/step6_order.py

=============================================================================
그냥 상위 96개를 보내면 안 되는 이유
=============================================================================
단계 5의 순위 상위 96개를 그대로 보내는 게 자연스러워 보인다. 그런데
그러면 대개 **골격 열댓 개에서 나온 형제 서열들**로 판이 채워진다.
골격 하나에서 서열 8개를 뽑았기 때문이다.

그 골격 하나가 틀린 가정이면 8개가 통째로 같이 실패한다.
96웰을 썼는데 실제로는 12가지만 시험한 셈이 된다.

이 단계가 하는 일 세 가지.

    1. 서로 너무 비슷한 것끼리는 하나만 남긴다 (다양성)
    2. 대조군을 넣는다
    3. 플레이트 배치를 만든다

=============================================================================
대조군을 반드시 넣어야 하는 이유
=============================================================================
설계 96개를 넣고 전부 안 붙었다고 하자. 원인이 뭔가?

    (가) 설계가 나빴다
    (나) 표적 단백질이 죽어 있었다
    (다) 실험 조건이 잘못됐다

**대조군이 없으면 이 셋을 구분할 방법이 없다.**

    양성 대조군 (이미 붙는 걸 아는 것 — 기존 항체, 천연 리간드)
        → 이게 안 붙으면 실험이 잘못된 것이다. 설계 탓하지 마라
    음성 대조군 (안 붙는 게 확실한 것 — 무관한 단백질)
        → 이게 붙으면 비특이 결합이다. 전체 결과를 의심해야 한다

웰 4개를 대조군에 쓰는 게 아깝게 느껴지지만, 이게 없으면
**실패했을 때 아무것도 배우지 못한다.** 그 판이 통째로 낭비가 된다.

Step 6 - assemble the plate: diversity pruning, controls, layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_config as cfg
from _shared import header, read_fasta


def identity(a: str, b: str) -> float:
    """두 서열이 얼마나 같은가. 0~1.

    길이가 다르면 짧은 쪽 기준으로 앞에서부터 맞춰 센다.
    거친 방법이지만 '형제 서열'을 골라내는 데는 충분하다.
    제대로 하려면 정렬(alignment)을 해야 한다.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def main() -> int:
    header(6, "실험 발주")
    cfg.ensure_dirs()

    try:
        import pandas as pd
    except ImportError:
        print("  pandas 가 필요하다:  pip install pandas")
        return 2

    passed_csv = cfg.DIR_FILTER / "passed.csv"
    fasta = cfg.DIR_SEQUENCE / "designs.fasta"
    if not passed_csv.exists():
        print("  통과 목록이 없다. 먼저: python pipeline/step5_filter.py")
        return 2

    df = pd.read_csv(passed_csv)
    seqs = read_fasta(fasta) if fasta.exists() else {}
    print(f"  후보: {len(df):,}개")

    # -----------------------------------------------------------------
    # 1. 다양성 — 형제 서열 솎아내기
    # -----------------------------------------------------------------
    n_slots = cfg.PLATE_SIZE - cfg.N_POSITIVE_CONTROLS - cfg.N_NEGATIVE_CONTROLS
    name_col = "description" if "description" in df.columns else df.columns[0]

    chosen, chosen_seqs, skipped = [], [], 0
    for _, row in df.iterrows():
        if len(chosen) >= n_slots:
            break
        s = seqs.get(str(row[name_col]))
        if s is None:
            chosen.append(row)               # 서열을 못 찾으면 그냥 넣는다
            continue
        if any(identity(s, t) > cfg.MAX_SEQUENCE_IDENTITY for t in chosen_seqs):
            skipped += 1
            continue
        chosen.append(row)
        chosen_seqs.append(s)

    print(f"  다양성 정리: {skipped}개 제외 "
          f"(서열 유사도 {100*cfg.MAX_SEQUENCE_IDENTITY:.0f}% 초과)")
    print(f"  선택: {len(chosen)}개")

    if len(chosen) < n_slots:
        print(f"    ⚠ 자리 {n_slots}개를 다 못 채웠다. 후보가 부족하거나")
        print("      후보들이 서로 너무 비슷하다. 골격을 더 만들어야 한다.")

    # 골격 다양성도 확인한다 — 이게 진짜 보고 싶은 것
    if name_col in df.columns:
        backbones = {str(r[name_col]).split("__")[0] for r in chosen}
        print(f"  서로 다른 골격 수: {len(backbones)}")
        if len(backbones) < len(chosen) / 3:
            print("    ⚠ 골격 다양성이 낮다. 골격 하나가 틀리면 여러 개가 같이 죽는다.")

    # -----------------------------------------------------------------
    # 2. 플레이트 배치
    # -----------------------------------------------------------------
    rows = "ABCDEFGH"
    wells = [f"{r}{c}" for r in rows for c in range(1, 13)]

    plate = []
    # 대조군을 먼저, 서로 떨어진 자리에 놓는다.
    # 한 구석에 몰아놓으면 그 구석에 생긴 문제(가장자리 증발 등)와
    # 대조군 실패를 구분할 수 없다.
    control_wells = ["A1", "A12", "H1", "H12"]
    for i in range(cfg.N_POSITIVE_CONTROLS):
        plate.append({"well": control_wells[i], "name": f"POSITIVE_CONTROL_{i+1}",
                      "type": "양성대조군", "sequence": "<여기에 기존 결합체 서열>"})
    for i in range(cfg.N_NEGATIVE_CONTROLS):
        w = control_wells[cfg.N_POSITIVE_CONTROLS + i]
        plate.append({"well": w, "name": f"NEGATIVE_CONTROL_{i+1}",
                      "type": "음성대조군", "sequence": "<여기에 무관한 단백질 서열>"})

    used = {p["well"] for p in plate}
    free = [w for w in wells if w not in used]
    for w, row in zip(free, chosen):
        nm = str(row[name_col])
        plate.append({"well": w, "name": nm, "type": "설계",
                      "sequence": seqs.get(nm, "")})

    out = cfg.DIR_ORDER / "plate.csv"
    pd.DataFrame(plate).sort_values("well").to_csv(out, index=False)

    fa = cfg.DIR_ORDER / "order.fasta"
    with fa.open("w", encoding="utf-8") as fh:
        for p in plate:
            if p["type"] == "설계" and p["sequence"]:
                fh.write(f">{p['well']}_{p['name']}\n{p['sequence']}\n")

    print()
    print(f"  플레이트 배치 → {out}")
    print(f"  합성 발주용 FASTA → {fa}")
    print()
    print(f"  구성: 설계 {sum(1 for p in plate if p['type']=='설계')}개 + "
          f"양성 {cfg.N_POSITIVE_CONTROLS}개 + 음성 {cfg.N_NEGATIVE_CONTROLS}개")

    # -----------------------------------------------------------------
    # 3. 보내기 전 마지막 확인
    # -----------------------------------------------------------------
    print()
    print("  보내기 전에 사람이 확인할 것")
    print("    □ 대조군 서열을 실제로 채워 넣었는가 (지금은 자리만 잡아뒀다)")
    print("    □ 발현 태그를 붙일 것인가 (His-tag, Avi-tag 등)")
    print("    □ 발주처의 서열 제한을 확인했는가 (길이, 금지 패턴)")
    print("    □ 이 표적에 이미 알려진 결합체가 있다면 양성 대조군으로 넣었는가")
    print()
    print("  실험 결과가 나오면 (여기가 2라운드의 출발점이다):")
    print("    1. plate.csv 에 binding 열을 추가한다 (붙었으면 1, 아니면 0)")
    print("    2. python validation/check_filters.py --csv plate.csv --name 내표적")
    print()
    print("    그러면 네 표적에서 임계값이 맞았는지 처음으로 알 수 있다.")
    print("    문헌값은 1라운드까지만 쓰는 값이고, 여기서부터는 네 데이터가 기준이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
