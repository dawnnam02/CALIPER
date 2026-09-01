"""Backend protocol.

A backend is anything that can turn candidates into scores.  CALIPER never
imports RFdiffusion, ProteinMPNN or Boltz directly; it shells out to whatever
the user has installed, behind this interface.  That keeps the orchestration
layer -- which is the actual contribution -- testable on a laptop with no GPU.

Korean note:
CALIPER는 RFdiffusion이나 Boltz를 직접 import하지 않는다.  설치돼 있으면 불러 쓰고,
없으면 시뮬레이터로 돈다.  그래야 GPU 없는 노트북에서 로직 자체를 검증할 수 있다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...types import Candidate, Target


@runtime_checkable
class Designer(Protocol):
    """Produces candidate binder sequences for a target."""

    name: str
    version: str
    unit_cost: float

    def design(self, target: Target, n: int, seed: int) -> list[Candidate]:
        ...


@runtime_checkable
class Scorer(Protocol):
    """Assigns a score to each candidate.  Higher is better, always."""

    name: str
    version: str
    unit_cost: float
    stage: str

    def score(self, target: Target, candidates: list[Candidate],
              seed: int) -> list[float]:
        ...


class BackendUnavailable(RuntimeError):
    """Raised when a real backend is configured but not installed.

    Carries an explicit remedy so the failure is actionable rather than a
    bare ImportError three frames deep.
    """

    def __init__(self, backend: str, reason: str, remedy: str) -> None:
        super().__init__(f"backend {backend!r} unavailable: {reason}\n  remedy: {remedy}")
        self.backend = backend
        self.reason = reason
        self.remedy = remedy
