"""단계 3 — 골격에 아미노산 서열을 채운다 (ProteinMPNN).

    python pipeline/step3_sequence.py

=============================================================================
하는 일
=============================================================================
단계 2가 만든 것은 뼈대뿐이다. 아미노산 자리는 전부 비어 있다.
이 단계가 자리마다 "여기엔 어떤 아미노산이 와야 이 모양이 유지될까"를 채운다.

이걸 역접힘(inverse folding)이라고 한다.
    보통의 문제:  서열 → 구조   (AlphaFold 가 하는 일)
    역접힘:       구조 → 서열   (여기서 하는 일)

=============================================================================
골격 하나에 서열 8개를 뽑는 이유
=============================================================================
같은 모양을 만드는 서열은 여러 개다. 어떤 건 잘 발현되고 어떤 건 안 되고,
어떤 건 잘 녹고 어떤 건 뭉친다. **어느 것이 좋은지는 만들기 전엔 모른다.**
그래서 여러 개를 뽑아 두고 다음 단계에서 걸러낸다.

8개가 관례다. 더 뽑아도 되지만 다음 단계(AF2)가 느려진다.

=============================================================================
soluble 가중치를 꼭 써야 하는 이유
=============================================================================
기본 ProteinMPNN 은 PDB 구조들로 학습됐다. 그런데 PDB 에 등록된 구조는
대부분 **다른 단백질과 붙어 있거나 막에 박혀 있는 것**들이다.
그런 단백질은 표면에 기름 성질(소수성) 잔기가 많다.

그 편향을 그대로 배우면, 혼자 물에 떠 있어야 하는 우리 결합체에도
표면에 기름칠을 해버린다. 그러면 **물에 안 녹고 뭉친다.**
실험실에서 "발현은 됐는데 전부 침전됐다"가 되는 것이다.

soluble 가중치는 그 편향을 줄여 학습한 것이다. 반드시 쓴다.

Step 3 - inverse folding with ProteinMPNN, soluble weights, 8 seqs/backbone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step0_config as cfg
from _shared import header, read_fasta, require_tool, run, write_fasta


def main() -> int:
    header(3, "서열 설계 (ProteinMPNN)")
    cfg.ensure_dirs()

    backbones = sorted(cfg.DIR_BACKBONE.glob("backbone_*.pdb"))
    if not backbones:
        print("  골격이 없다. 먼저: python pipeline/step2_backbone.py")
        return 2

    print(f"  골격 개수        : {len(backbones)}")
    print(f"  골격당 서열      : {cfg.SEQS_PER_BACKBONE}")
    print(f"  나올 서열 총합   : {len(backbones) * cfg.SEQS_PER_BACKBONE:,}")
    print(f"  샘플링 온도      : {cfg.MPNN_TEMPERATURE}  "
          f"(낮을수록 안전하고 비슷한 서열)")
    print(f"  soluble 가중치   : {'예' if cfg.MPNN_USE_SOLUBLE_WEIGHTS else '아니오'}")
    if not cfg.MPNN_USE_SOLUBLE_WEIGHTS:
        print("      ⚠ 끄면 물에 안 녹는 설계가 늘어난다. 켜는 걸 권한다.")
    print()

    script = require_tool("proteinmpnn")

    cmd = [
        sys.executable, str(script),
        "--pdb_path_chains", "B",          # B = 새로 만든 결합체 사슬
        "--folder_with_pdbs", str(cfg.DIR_BACKBONE),
        "--out_folder", str(cfg.DIR_SEQUENCE),
        "--num_seq_per_target", str(cfg.SEQS_PER_BACKBONE),
        "--sampling_temp", str(cfg.MPNN_TEMPERATURE),
        "--seed", "37",
        "--batch_size", "1",
    ]
    if cfg.MPNN_USE_SOLUBLE_WEIGHTS:
        cmd += ["--use_soluble_model"]
    run(cmd)

    # -----------------------------------------------------------------
    # 나온 서열을 한 파일로 모으고, 뻔한 문제를 걸러낸다
    # -----------------------------------------------------------------
    seqs = {}
    for fa in sorted((cfg.DIR_SEQUENCE / "seqs").glob("*.fa")):
        for name, s in read_fasta(fa).items():
            # ProteinMPNN 은 원본 골격 서열도 같이 뱉는다. 그건 뺀다.
            if "score=" in name and "sample=" in name:
                seqs[f"{fa.stem}__{len(seqs)}"] = s.split("/")[-1]

    print()
    print(f"  모은 서열: {len(seqs):,}개")

    # 실험실에서 문제를 일으키는 서열 패턴을 미리 본다.
    # 여기서 거르지 않으면 합성·발현 단계에서 돈과 시간을 버린다.
    flagged = {"시스테인 홀수개": [], "너무 긺": [], "반복 서열": []}
    clean = {}
    for name, s in seqs.items():
        why = None
        # 시스테인(C)은 둘씩 짝지어 다리를 만든다. 홀수면 하나가 남아
        # 다른 분자와 엉겨붙어 뭉친다.
        if s.count("C") % 2 == 1:
            flagged["시스테인 홀수개"].append(name); why = 1
        if len(s) > cfg.BINDER_LENGTH_MAX + 5:
            flagged["너무 긺"].append(name); why = 1
        # 같은 글자가 5개 이상 연속되면 DNA 합성이 어렵고 발현도 나쁘다
        if any(c * 5 in s for c in set(s)):
            flagged["반복 서열"].append(name); why = 1
        if not why:
            clean[name] = s

    for reason, names in flagged.items():
        if names:
            print(f"    걸러냄 — {reason}: {len(names)}개")

    out = cfg.DIR_SEQUENCE / "designs.fasta"
    write_fasta(out, clean)
    print(f"  남은 서열: {len(clean):,}개  → {out}")

    if clean:
        lens = [len(s) for s in clean.values()]
        print(f"  길이 분포: {min(lens)} ~ {max(lens)} "
              f"(평균 {sum(lens)/len(lens):.0f})")

    print()
    print("  ✓ 다음: python pipeline/step4_predict.py")
    print("    ※ 다음 단계가 이 파이프라인에서 가장 오래 걸린다.")
    print(f"      서열 {len(clean):,}개를 AF2 로 예측한다. GPU 로도 몇 시간~며칠이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
