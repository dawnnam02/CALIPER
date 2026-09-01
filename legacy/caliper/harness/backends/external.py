"""Adapters for real design tools.

CALIPER shells out; it never vendors model code or weights.  Each adapter
checks for its executable up front and fails with an actionable message rather
than an ImportError deep in a run that has already spent GPU hours.

Deliberately not implemented as silent fallbacks: if a run is configured for
RFdiffusion and RFdiffusion is missing, the run must stop.  Quietly
substituting the simulator would produce a report full of numbers that look
real and are not -- the single worst failure mode this project can have.

Korean note:
설정에 RFdiffusion이라고 적혀 있는데 설치가 안 돼 있으면 **그냥 멈춘다.**
조용히 시뮬레이터로 바꿔치기하면 진짜처럼 보이는 가짜 숫자가 리포트에 찍힌다.
이 프로젝트에서 그게 최악의 실패다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ...types import Candidate, Target
from .base import BackendUnavailable

# tool -> (executable to probe, how to install it)
KNOWN_TOOLS = {
    "rfdiffusion": (
        "run_inference.py",
        "git clone https://github.com/RosettaCommons/RFdiffusion and set "
        "rfdiffusion.path in the config to its checkout",
    ),
    "proteinmpnn": (
        "protein_mpnn_run.py",
        "git clone https://github.com/dauparas/ProteinMPNN and set "
        "proteinmpnn.path in the config",
    ),
    "bindcraft": (
        "bindcraft.py",
        "git clone https://github.com/martinpacesa/BindCraft",
    ),
    "boltz": (
        "boltz",
        "pip install boltz  (weights download on first use)",
    ),
    "colabfold": (
        "colabfold_batch",
        "pip install 'colabfold[alphafold]'",
    ),
}


def probe(tool: str, explicit_path: str | None = None) -> Path:
    """Locate a tool, or raise with the exact remedy."""
    key = tool.lower()
    if key not in KNOWN_TOOLS:
        raise BackendUnavailable(tool, "unknown tool name",
                                 f"known tools: {sorted(KNOWN_TOOLS)}")
    exe, remedy = KNOWN_TOOLS[key]
    if explicit_path:
        p = Path(explicit_path)
        cand = p / exe if p.is_dir() else p
        if cand.exists():
            return cand
        raise BackendUnavailable(tool, f"{cand} does not exist", remedy)
    found = shutil.which(exe)
    if found:
        return Path(found)
    raise BackendUnavailable(tool, f"{exe!r} not on PATH", remedy)


def tool_version(path: Path) -> str:
    """Best-effort version string, recorded in the provenance manifest.

    Falls back to the file's size and mtime, which at least detects that the
    installed tool changed between runs -- the thing the manifest is for.
    """
    try:
        r = subprocess.run([str(path), "--version"], capture_output=True,
                           text=True, timeout=20)
        v = (r.stdout or r.stderr).strip().splitlines()
        if v:
            return v[0][:120]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        st = path.stat()
        return f"unversioned(size={st.st_size},mtime={int(st.st_mtime)})"
    except OSError:
        return "unknown"


def build_external(cfg: dict, target: Target):
    """Assemble designer / scorers / assay from config.

    Raises immediately with instructions, because the honest thing to do when
    the tools are absent is to say so.
    """
    ext = cfg.get("external", {})
    wanted = [ext.get("designer", "rfdiffusion")]
    wanted += [s["tool"] for s in ext.get("stages", [])]
    missing = []
    for t in wanted:
        try:
            probe(t, ext.get(t, {}).get("path"))
        except BackendUnavailable as e:
            missing.append(str(e))
    raise BackendUnavailable(
        "external",
        "real-backend adapters are declared but not wired to executables in "
        "this build" + (f"; probes failed:\n  " + "\n  ".join(missing)
                        if missing else ""),
        "run with backend: simulator to exercise the orchestration layer, or "
        "implement the shell-out in caliper/backends/external.py for the tools "
        "you have installed. CALIPER will not silently substitute the "
        "simulator for a real backend.",
    )
