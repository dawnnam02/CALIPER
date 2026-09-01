"""Core data types for CALIPER.

CALIPER treats a design campaign as a *measurement* problem: every stage emits
a score, and every score is eventually compared against ground truth.  The
types here are deliberately small and immutable so that a candidate's full
history can be hashed and cached.

한국어 메모: 후보 하나가 파이프라인을 통과하며 남기는 기록을 전부 담는 그릇이다.
점수를 덮어쓰지 않고 단계별로 쌓아두는 것이 핵심 — 나중에 교정(calibration)에 쓴다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


def stable_hash(payload: Any) -> str:
    """Deterministic short hash of any JSON-serialisable payload.

    Used as the content address for cache entries.  ``sort_keys`` makes the
    hash independent of dict ordering, and ``default=str`` keeps it total.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Target:
    """The thing we are designing a binder against."""

    name: str
    sequence: str
    hotspots: tuple[int, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target.name must not be empty")
        if not self.sequence:
            raise ValueError(f"Target {self.name!r}: sequence must not be empty")
        bad = sorted(set(self.sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
        if bad:
            raise ValueError(
                f"Target {self.name!r}: non-standard residues {bad}. "
                "CALIPER expects a one-letter amino-acid string."
            )
        for h in self.hotspots:
            if not 0 <= h < len(self.sequence):
                raise ValueError(
                    f"Target {self.name!r}: hotspot {h} outside sequence "
                    f"of length {len(self.sequence)}"
                )

    @property
    def uid(self) -> str:
        return stable_hash({"n": self.name, "s": self.sequence, "h": list(self.hotspots)})


@dataclass(frozen=True, slots=True)
class Candidate:
    """A single designed binder, plus everything measured about it so far.

    ``scores`` accumulates one entry per stage.  Nothing is ever overwritten:
    a candidate that is killed at stage 1 keeps its stage-1 score, and those
    "loser" scores are exactly what the calibrator learns from.

    한국어 메모: 탈락한 후보의 점수도 지우지 않는다. HTS 다중충실도 연구가 지적한
    "1차 스크리닝 데이터를 버린다"는 문제를 여기서 막는다.
    """

    cid: str
    target_uid: str
    sequence: str
    origin: str                                  # which designer produced it
    scores: Mapping[str, float] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    alive: bool = True
    killed_at: str | None = None

    def with_score(self, stage: str, value: float, **meta: Any) -> "Candidate":
        """Return a copy carrying one more stage score (never mutates)."""
        if stage in self.scores:
            raise ValueError(
                f"Candidate {self.cid}: stage {stage!r} already scored "
                f"({self.scores[stage]}). Stages must be unique per run."
            )
        if value != value:  # NaN
            raise ValueError(f"Candidate {self.cid}: stage {stage!r} produced NaN")
        return replace(
            self,
            scores={**self.scores, stage: float(value)},
            meta={**self.meta, **meta},
        )

    def killed(self, stage: str) -> "Candidate":
        return replace(self, alive=False, killed_at=stage)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "cid": self.cid,
            "sequence": self.sequence,
            "length": len(self.sequence),
            "origin": self.origin,
            "alive": self.alive,
            "killed_at": self.killed_at,
        }
        row.update({f"score.{k}": v for k, v in self.scores.items()})
        row.update({f"meta.{k}": v for k, v in self.meta.items()
                    if isinstance(v, (int, float, str, bool)) or v is None})
        return row


@dataclass(frozen=True, slots=True)
class StageReport:
    """What one stage did, for the provenance manifest."""

    stage: str
    backend: str
    n_in: int
    n_out: int
    cost_units: float
    wall_seconds: float
    cache_hits: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kill_rate(self) -> float:
        return 0.0 if self.n_in == 0 else 1.0 - (self.n_out / self.n_in)
