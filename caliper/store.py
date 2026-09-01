"""Content-addressed cache and provenance manifest.

Borrowed from build systems (Bazel/ccache): a stage's output is keyed by the
hash of (stage name, backend version, parameters, inputs).  Re-running a
campaign after changing one late-stage threshold should not re-run the
expensive early stages.

한국어 메모: 빌드 시스템에서 가져온 생각이다. "입력이 같으면 결과도 같다"를 이용해
비싼 단계를 다시 돌리지 않는다. 뒤쪽 임계값만 바꿨을 때 앞쪽을 재계산하지 않는 게 목적.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .types import stable_hash


class Store:
    """Filesystem cache + append-only run manifest."""

    def __init__(self, root: str | os.PathLike[str], *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.cache_dir = self.root / "cache"
        self.enabled = enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ---- cache -----------------------------------------------------------
    def key(self, stage: str, backend: str, version: str,
            params: dict[str, Any], inputs: Any) -> str:
        return stable_hash({
            "stage": stage, "backend": backend, "version": version,
            "params": params, "inputs": inputs,
        })

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # A corrupt entry must never take the run down: drop and recompute.
            try:
                path.unlink()
            except OSError:
                pass
            self._misses += 1
            return None
        self._hits += 1
        return payload["value"]

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self.cache_dir / f"{key}.json"
        payload = {"key": key, "written": time.time(), "value": value}
        # Atomic write: a crash mid-write must not leave a half file behind.
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> int:
        n = len(list(self.cache_dir.glob("*.json")))
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return n


class RunDir:
    """One campaign's output directory, with an append-only manifest."""

    def __init__(self, root: str | os.PathLike[str], run_id: str) -> None:
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.jsonl"

    def log(self, kind: str, **fields: Any) -> None:
        record = {"t": time.time(), "kind": kind, **fields}
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        with p.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path / name
        p.write_text(text, encoding="utf-8")
        return p

    def read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        out = []
        with self.manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
