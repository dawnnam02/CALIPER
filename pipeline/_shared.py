"""단계들이 같이 쓰는 잡일 모음.

여기는 파이프라인의 '내용'이 아니라 '배관'이다.
처음 읽는다면 step1_target.py 부터 보는 게 낫다.

Shared plumbing: external tool discovery, subprocess running, FASTA/PDB I/O.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# 외부 도구 찾기
# =============================================================================
#
# 이 파이프라인은 무거운 계산을 직접 하지 않는다. 남이 만든 도구를 부른다.
# 도구가 없으면 **거짓 결과를 만들어내지 않고**, 무엇을 설치해야 하는지
# 알려주고 멈춘다.

TOOLS = {
    "rfdiffusion": {
        "env": "RFDIFFUSION_DIR",
        "run": "scripts/run_inference.py",
        "what": "골격 생성",
        "get": "https://github.com/RosettaCommons/RFdiffusion",
        "needs": "GPU (VRAM 8GB 이상), 모델 가중치 약 2GB",
    },
    "proteinmpnn": {
        "env": "PROTEINMPNN_DIR",
        "run": "protein_mpnn_run.py",
        "what": "서열 설계",
        "get": "https://github.com/dauparas/ProteinMPNN",
        "needs": "CPU 로도 돌지만 느리다. 가중치 약 100MB",
    },
    "af2ig": {
        "env": "AF2_INITIAL_GUESS_DIR",
        "run": "predict.py",
        "what": "구조 검증 (AF2 initial guess)",
        "get": "https://github.com/nrbennet/dl_binder_design",
        "needs": "GPU 필수. AF2 파라미터 약 4GB",
    },
}


def find_tool(name: str) -> Path | None:
    """도구가 설치돼 있으면 실행 스크립트 경로를, 없으면 None 을 준다."""
    spec = TOOLS[name]
    root = os.environ.get(spec["env"])
    if root:
        p = Path(root) / spec["run"]
        if p.exists():
            return p
    # PATH 에 그냥 올라와 있는 경우도 본다
    exe = shutil.which(Path(spec["run"]).stem)
    return Path(exe) if exe else None


def require_tool(name: str) -> Path:
    """도구가 없으면 무엇을 어떻게 깔아야 하는지 말하고 멈춘다."""
    p = find_tool(name)
    if p:
        return p
    spec = TOOLS[name]
    print(f"\n[{spec['what']}] 도구를 찾을 수 없다: {name}", file=sys.stderr)
    print(f"  받는 곳 : {spec['get']}", file=sys.stderr)
    print(f"  필요한 것: {spec['needs']}", file=sys.stderr)
    print(f"  설치했으면 위치를 알려줘라:", file=sys.stderr)
    print(f"      export {spec['env']}=/받은/경로      (Windows: set {spec['env']}=...)",
          file=sys.stderr)
    print(f"\n  * 이 단계는 건너뛸 수 없다. 가짜 결과를 만들지 않는다.",
          file=sys.stderr)
    raise SystemExit(3)


def tool_status() -> None:
    """지금 무엇이 설치돼 있는지 표로 보여준다."""
    print(f"  {'도구':<14}{'하는 일':<24}{'상태'}")
    print("  " + "-" * 56)
    for name, spec in TOOLS.items():
        p = find_tool(name)
        state = f"OK  {p}" if p else "없음"
        print(f"  {name:<14}{spec['what']:<24}{state}")


def run(cmd: list[str], where: Path | None = None) -> None:
    """명령을 실행하고, 실패하면 그 자리에서 멈춘다.

    조용히 실패하고 다음 단계로 넘어가는 것이 파이프라인에서 제일 위험하다.
    빈 폴더를 들고 세 단계 뒤에 가서야 이상을 알아채게 되기 때문이다.
    """
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], cwd=where)
    if r.returncode != 0:
        print(f"\n  실패 (종료코드 {r.returncode}). 위 명령을 직접 돌려보고 "
              f"오류를 확인해라.", file=sys.stderr)
        raise SystemExit(r.returncode)


# =============================================================================
# 파일 읽고 쓰기
# =============================================================================

def read_fasta(path: Path) -> dict[str, str]:
    """FASTA 를 {이름: 서열} 로 읽는다."""
    out, name, buf = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if name:
        out[name] = "".join(buf)
    return out


def write_fasta(path: Path, seqs: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def read_pdb_ca(path: Path, chain: str | None = None):
    """PDB 에서 CA 원자만 뽑는다. (잔기번호, x, y, z) 목록.

    CA 는 알파탄소(alpha carbon) — 아미노산 하나당 하나씩 있는 중심 원자다.
    단백질의 전체 모양만 볼 때는 이것만 있으면 충분하다.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        if chain and line[21] != chain:
            continue
        out.append((int(line[22:26]),
                    float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return out


def header(step: int, title: str) -> None:
    print()
    print("=" * 74)
    print(f"  단계 {step} — {title}")
    print("=" * 74)
