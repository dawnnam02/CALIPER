"""단계 1 — 표적을 고르고 손질한다.

    python pipeline/step1_target.py

=============================================================================
이 단계가 왜 제일 중요한가
=============================================================================
뒤 단계들은 전부 자동이다. 버튼을 누르면 돌아간다.
**사람이 판단해야 하는 건 사실상 이 단계뿐이다.**

그리고 여기서 틀리면 뒤가 전부 헛수고다. 계산을 아무리 잘해도
"붙을 수 없는 자리"에 붙이려 하고 있으면 안 붙는다.
GPU 를 일주일 돌리고 나서야 알게 된다.

=============================================================================
핫스팟(hotspot) 고르는 법
=============================================================================
핫스팟 = 결합체가 반드시 닿았으면 하는 표적 잔기.

**좋은 자리의 조건**

  1) 오목한 곳(pocket, groove)      평평한 면은 붙잡을 데가 없다
  2) 소수성 잔기가 있는 곳          물을 밀어내며 붙는 힘이 결합의 주력이다
                                     (Leu, Ile, Val, Phe, Trp, Met, Tyr)
  3) 기능적으로 의미 있는 곳        붙어도 아무 일도 안 일어나면 소용없다
                                     - 리간드/기질이 붙는 자리
                                     - 이미 알려진 항체가 붙는 자리(에피토프)
                                     - 두 단백질이 만나는 계면
  4) 노출돼 있는 곳                 파묻힌 잔기는 접근할 수 없다

**피해야 할 자리**

  ✗ 평평하고 극성인 면            결합체가 붙을 이유가 없다
  ✗ 당사슬(glycan)이 붙는 자리     N-X-S/T 서열. 실제 단백질에는 당이 달려
                                   있어서 구조 파일만 보면 안 보인다
  ✗ 유연한 고리(loop)나 말단       구조가 흔들려서 예측이 안 맞는다
  ✗ 결정화 인공물                  결정 안에서만 생기는 접촉면

이 스크립트가 (1),(2),(4)는 계산해서 점수를 매겨준다.
**(3) 기능적 의미는 사람이 논문을 읽고 판단해야 한다.** 자동화 못 한다.

Step 1 - prepare the target: pick hotspots, trim, sanity-check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_config as cfg
from _shared import header, read_pdb_ca

# 소수성 아미노산 (3글자 표기). 물을 싫어해서 결합의 주력이 되는 잔기들.
HYDROPHOBIC = {"LEU", "ILE", "VAL", "PHE", "TRP", "MET", "TYR", "ALA", "PRO"}

# 당사슬이 붙을 수 있는 서열 패턴에 관여하는 잔기
GLYCOSYLATION_RISK = {"ASN"}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def read_residues(path: Path, chain: str):
    """PDB 에서 잔기 목록을 읽는다. (번호, 3글자이름, 원자좌표들)"""
    residues = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        num = int(line[22:26])
        name = line[17:20].strip()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        residues.setdefault(num, {"name": name, "atoms": []})["atoms"].append(xyz)
    return residues


def neighbour_count(residues, num, radius=10.0):
    """어떤 잔기 주변 radius Å 안에 다른 잔기가 몇 개나 있는가.

    많으면 파묻힌 것(접근 불가), 적으면 튀어나온 것,
    중간이면 오목한 곳일 가능성이 높다. 아주 거친 근사다.
    """
    cx, cy, cz = residues[num]["atoms"][0]
    n = 0
    for other, r in residues.items():
        if other == num:
            continue
        x, y, z = r["atoms"][0]
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < radius ** 2:
            n += 1
    return n


def main() -> int:
    header(1, "표적 준비")
    cfg.ensure_dirs()

    if not cfg.TARGET_PDB.exists():
        print(f"  표적 구조 파일이 없다: {cfg.TARGET_PDB}")
        print()
        print("  어디서 받나:")
        print("    실험 구조가 있으면  → https://www.rcsb.org  (제일 좋다)")
        print("    없으면              → https://alphafold.ebi.ac.uk")
        print()
        print("  받은 뒤 step0_config.py 의 TARGET_PDB 를 그 파일로 바꿔라.")
        print()
        print("  ※ 실험 구조가 있으면 반드시 그걸 써라. AlphaFold 예측 구조를")
        print("    표적으로 쓰면, 예측 오차 위에 설계를 쌓는 셈이 된다.")
        return 2

    residues = read_residues(cfg.TARGET_PDB, cfg.TARGET_CHAIN)
    if not residues:
        print(f"  사슬 '{cfg.TARGET_CHAIN}' 을 찾을 수 없다. "
              f"step0_config.py 의 TARGET_CHAIN 을 확인해라.")
        return 2

    seq = "".join(AA3_TO_1.get(r["name"], "X") for _, r in sorted(residues.items()))
    print(f"  파일   : {cfg.TARGET_PDB.name}")
    print(f"  사슬   : {cfg.TARGET_CHAIN}")
    print(f"  잔기수 : {len(residues)}")
    print(f"  범위   : {min(residues)} ~ {max(residues)}")

    # -----------------------------------------------------------------
    # 지정한 핫스팟이 말이 되는지 확인한다
    # -----------------------------------------------------------------
    print()
    print(f"  지정한 핫스팟: {cfg.TARGET_HOTSPOTS}")
    print()
    print(f"  {'잔기':>7}{'이름':>7}{'소수성':>8}{'이웃수':>8}   판정")
    print("  " + "-" * 52)

    problems = []
    for num in cfg.TARGET_HOTSPOTS:
        if num not in residues:
            print(f"  {num:>7}{'없음':>7}{'-':>8}{'-':>8}   구조에 이 번호가 없다")
            problems.append(f"{num}번 잔기가 구조에 없다")
            continue
        name = residues[num]["name"]
        hydro = "예" if name in HYDROPHOBIC else "아니오"
        nb = neighbour_count(residues, num)
        notes = []
        if name not in HYDROPHOBIC:
            notes.append("극성 잔기")
        if nb > 22:
            notes.append("파묻힘 — 접근 어려움")
            problems.append(f"{num}번({name})이 파묻혀 있다")
        elif nb < 6:
            notes.append("너무 튀어나옴 — 붙잡을 데 없음")
        if name in GLYCOSYLATION_RISK:
            notes.append("ASN — 당사슬 확인 필요")
        print(f"  {num:>7}{name:>7}{hydro:>8}{nb:>8}   {' / '.join(notes) or 'OK'}")

    n = len(cfg.TARGET_HOTSPOTS)
    print()
    if n < 3:
        print(f"  ⚠ 핫스팟이 {n}개다. 3~5개를 권한다.")
        print("    너무 적으면 RFdiffusion 이 방향을 못 잡아 아무 데나 붙인다.")
    elif n > 6:
        print(f"  ⚠ 핫스팟이 {n}개다. 3~5개를 권한다.")
        print("    너무 많으면 다 만족시키는 골격이 없어서 생성이 거의 실패한다.")

    hydro_frac = sum(1 for x in cfg.TARGET_HOTSPOTS
                     if x in residues and residues[x]["name"] in HYDROPHOBIC) / max(1, n)
    if hydro_frac < 0.4:
        print(f"  ⚠ 핫스팟 중 소수성 잔기가 {100*hydro_frac:.0f}% 뿐이다.")
        print("    극성 면만 노리면 결합력이 잘 안 나온다. 오목한 소수성 자리를 찾아봐라.")

    # -----------------------------------------------------------------
    # 사람이 확인해야 하는 것 — 자동으로 못 한다
    # -----------------------------------------------------------------
    print()
    print("  사람이 확인해야 할 것 (코드가 대신 못 한다)")
    print("    □ 이 자리에 붙으면 실제로 원하는 효과가 나는가?")
    print("       (기능 억제? 표지? 그냥 붙기만 하면 되는가?)")
    print("    □ 이 자리에 당사슬이 붙어 있지 않은가?  UniProt 의 PTM 항목 확인")
    print("    □ 이 자리가 결정 구조의 인공물은 아닌가?")
    print("    □ 이미 이 자리를 노린 연구가 있는가? 있으면 그 결과부터 읽어라")

    # -----------------------------------------------------------------
    # 표적 잘라내기
    # -----------------------------------------------------------------
    kept = []
    hot_coords = [residues[x]["atoms"][0] for x in cfg.TARGET_HOTSPOTS if x in residues]
    if hot_coords:
        for num, r in residues.items():
            x, y, z = r["atoms"][0]
            if any((x - hx) ** 2 + (y - hy) ** 2 + (z - hz) ** 2
                   < cfg.TARGET_TRIM_RADIUS ** 2 for hx, hy, hz in hot_coords):
                kept.append(num)

    out = cfg.DIR_TARGET / "target_trimmed.pdb"
    lines = [l for l in cfg.TARGET_PDB.read_text(encoding="utf-8").splitlines()
             if l.startswith("ATOM") and l[21] == cfg.TARGET_CHAIN
             and int(l[22:26]) in set(kept)]
    out.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")

    print()
    print(f"  핫스팟에서 {cfg.TARGET_TRIM_RADIUS:.0f}Å 안쪽만 남겼다: "
          f"{len(residues)} → {len(kept)} 잔기")
    print(f"  저장: {out}")
    print()
    print("    왜 자르나: 표적이 크면 RFdiffusion 이 느려지고, 핫스팟에서 먼")
    print("    부분은 어차피 설계에 영향을 주지 않는다. 다만 너무 많이 자르면")
    print("    표적 구조가 무너져서 예측이 이상해진다. 20Å 이 대체로 안전하다.")

    (cfg.DIR_TARGET / "target_sequence.txt").write_text(seq, encoding="utf-8")

    print()
    if problems:
        print("  ✗ 문제가 있다:")
        for p in problems:
            print(f"      - {p}")
        print("    고치고 다시 돌려라. 이대로 진행하면 뒤 단계가 전부 낭비다.")
        return 1
    print("  ✓ 다음: python pipeline/step2_backbone.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
