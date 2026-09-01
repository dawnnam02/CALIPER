"""Fetch the two Adaptyv competition result sets hosted on Proteinbase.

    python scripts/fetch_proteinbase.py            # both
    python scripts/fetch_proteinbase.py nipah      # just one

Proteinbase serves each collection as one CSV whose `evaluations` column holds
a JSON array per design, mixing computational scores with experimental readouts
and with several assay replicates of the same design. This flattens that into
one row per design with plain columns, which is what the experiments expect.

Two things this file decides, and says why:

* **A design counts as a binder when a majority of its replicates say so.**
  Designs carry 3 to 14 replicates. Taking "any replicate bound" instead moves
  the Nipah count by zero designs, so the choice does not drive the result --
  but it is the stricter rule and it is stated rather than left implicit.
* **Rows with no experimental replicate at all are dropped**, not counted as
  non-binders. A design that was never assayed is missing data, and quietly
  scoring it as a failure would invent labels.

Korean note:
Proteinbase는 컬렉션 하나를 CSV 하나로 준다. `evaluations` 열 안에 JSON 배열이
들어 있고, 계산 점수와 실험 결과와 반복 측정이 뒤섞여 있다. 이걸 설계 하나당
한 줄로 펼친다.

- **결합 판정**: 반복 측정의 과반이 결합이면 결합. Nipah에서는 "하나라도 결합"
  기준과 결과가 완전히 같았다. 결론을 좌우하지 않지만 더 엄격한 쪽을 골랐다.
- **실험이 아예 없는 행은 버린다.** 음성으로 세지 않는다. 측정을 안 한 것과
  결합을 안 한 것은 다르고, 후자로 처리하면 없는 라벨을 만들어내는 것이다.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://proteinbase.com/api/proteins/download"

COLLECTIONS = {
    "nipah": {
        "collection_id": "019be357-ae36-ec95-4bc6-9db0046b0600",
        "slug": "nipah-binder-competition-results",
        "dest": ROOT / "data" / "nipah" / "nipah.csv",
        "about": ("Adaptyv Nipah binder competition -- 1,201 designs, "
                  "target NiV-G (ODC-ODbL)"),
        # what the experiments read. Everything else in the JSON is dropped.
        "keep": ["boltz2_ipsae", "boltz2_min_ipsae", "boltz2_iptm",
                 "boltz2_plddt", "boltz2_complex_iplddt", "boltz2_pdockq",
                 "esmfold_plddt", "proteinmpnn_score"],
        "min_rows": 1000,
        "min_binders": 50,
    },
    "rbx1": {
        "collection_id": "03ec16ff-6665-40cb-b8de-18eff34a3933",
        "slug": "gem-x-adaptyv-rbx1-binder-design-competition-results",
        "dest": ROOT / "data" / "rbx1" / "rbx1.csv",
        "about": ("GEM x Adaptyv RBX1 binder competition -- 322 designs, "
                  "target RBX1 (ODC-ODbL)"),
        "keep": ["esmfold_plddt", "proteinmpnn_score", "molecular_weight",
                 "isoelectric_point"],
        "min_rows": 250,
        "min_binders": 5,
    },
}


def flatten(raw: bytes, keep: list[str]):
    """CSV with a JSON `evaluations` column -> one flat row per design."""
    import pandas as pd

    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", low_memory=False)
    for col in ("id", "name", "sequence", "evaluations"):
        if col not in df.columns:
            raise ValueError(f"column {col!r} missing; the API layout changed")

    rows = []
    for rec in df.itertuples(index=False):
        try:
            evals = json.loads(rec.evaluations) if isinstance(rec.evaluations, str) else []
        except (TypeError, ValueError):
            evals = []
        out = {"id": rec.id, "name": rec.name, "sequence": rec.sequence,
               "author": getattr(rec, "author", None),
               "design_method": getattr(rec, "designMethod", None)}
        replicates = []
        for e in evals:
            metric, value = e.get("metric"), e.get("value")
            if e.get("type") == "experimental" and metric == "binding":
                replicates.append(bool(value))
            elif metric in keep and e.get("valueType") == "numeric":
                out[metric] = value                     # last value wins
        out["n_replicates"] = len(replicates)
        out["n_bound"] = sum(replicates)
        rows.append(out)

    flat = pd.DataFrame(rows)
    assayed = flat[flat.n_replicates > 0].copy()
    # majority of replicates; see the module docstring for why not "any"
    assayed["binder"] = (assayed.n_bound >= (assayed.n_replicates + 1) // 2).astype(int)
    return flat, assayed


def fetch(name: str, spec: dict) -> bool:
    import pandas as pd

    dest: Path = spec["dest"]
    if dest.exists() and dest.stat().st_size > 10_000:
        n = sum(1 for _ in dest.open(encoding="utf-8")) - 1
        print(f"  {name}: already here ({n:,} designs)")
        return True

    url = f"{API}?collectionId={spec['collection_id']}&slug={spec['slug']}"
    print(f"  {name}: downloading ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=180) as r:
            raw = r.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  {name}: FAILED -- {e}", file=sys.stderr)
        print(f"         download it by hand from:\n         {url}", file=sys.stderr)
        return False

    try:
        flat, assayed = flatten(raw, spec["keep"])
    except Exception as e:                              # noqa: BLE001 - report and stop
        print(f"  {name}: could not parse what arrived -- {e}", file=sys.stderr)
        return False

    dropped = len(flat) - len(assayed)
    if len(flat) < spec["min_rows"]:
        print(f"  {name}: only {len(flat)} designs, expected at least "
              f"{spec['min_rows']}. Truncated?", file=sys.stderr)
        return False
    if int(assayed.binder.sum()) < spec["min_binders"]:
        print(f"  {name}: only {int(assayed.binder.sum())} binders, expected at "
              f"least {spec['min_binders']}. The layout may have changed.",
              file=sys.stderr)
        return False
    have = [c for c in spec["keep"] if c in assayed.columns]
    if not have:
        print(f"  {name}: none of the expected score columns are present "
              f"({spec['keep']}).", file=sys.stderr)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "name", "sequence", "author", "design_method",
            "n_replicates", "n_bound", "binder"] + have
    tmp = dest.with_suffix(".part")
    assayed[cols].to_csv(tmp, index=False)
    tmp.replace(dest)
    print(f"  {name}: {len(assayed):,} assayed designs "
          f"({int(assayed.binder.sum())} binders, "
          f"{100 * assayed.binder.mean():.1f}%)"
          + (f", {dropped} never assayed and dropped" if dropped else "")
          + f" -> {dest}")
    return True


def main(argv: list[str]) -> int:
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("this needs pandas:  pip install pandas", file=sys.stderr)
        return 2

    wanted = argv[1:] or list(COLLECTIONS)
    unknown = [w for w in wanted if w not in COLLECTIONS]
    if unknown:
        print(f"unknown collection(s): {unknown}. "
              f"Choose from {list(COLLECTIONS)}", file=sys.stderr)
        return 2

    for name in wanted:
        print(f"\n{COLLECTIONS[name]['about']}")
        if not fetch(name, COLLECTIONS[name]):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
